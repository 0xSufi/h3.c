/* CUDA backend for the h3_gpu.h device API: context lifecycle, tensor
 * storage, host<->device transfers, weight file loading, command-batch
 * bookkeeping, device probe, and the copy/cast primitives. Compute ops live
 * in the other h3_cuda_*.cu units. See h3_cuda_internal.h for the execution
 * model. */
/* h3_gpu.h is a C API: give the declarations C linkage so gcc-compiled
 * model-layer objects link against these nvcc-compiled definitions. */
extern "C" {
#include "h3_gpu.h"
#include "h3_cuda.h"
}
#include "h3_cuda_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

static int h3_cuda_profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0") != 0;
}

/* Defined next to h3_cuda_workspace below; used by h3_gpu_free. */
static void h3_cuda_cache_evict_all(struct h3_gpu *gpu);

static void h3_cuda_profile_emit(struct h3_gpu *gpu, const char *phase,
                                 const h3_gpu_stats *from, double wall_from) {
    if (!h3_cuda_profile_enabled()) return;
    double wall = h3_cuda_now() - wall_from;
    h3_gpu_stats delta;
    delta.allocated_bytes =
        gpu->stats.allocated_bytes - from->allocated_bytes;
    delta.tensor_allocations =
        gpu->stats.tensor_allocations - from->tensor_allocations;
    delta.command_wait_seconds =
        gpu->stats.command_wait_seconds - from->command_wait_seconds;
    fprintf(stderr, "h3 profile%s%s %-24s wall %.3fs wait %.3fs alloc %.1f MiB (%llu tensors)\n",
            gpu->profile_label[0] ? " " : "", gpu->profile_label, phase,
            wall, delta.command_wait_seconds,
            (double)delta.allocated_bytes / (1024.0 * 1024.0),
            (unsigned long long)delta.tensor_allocations);
}

int h3_cuda_probe(h3_device_info *info, char *error, size_t error_size) {
    if (!info) return 0;
    memset(info, 0, sizeof(*info));
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count < 1) {
        if (error && error_size)
            snprintf(error, error_size, "no CUDA device available: %s",
                     cudaGetErrorString(status));
        return 0;
    }
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, 0);
    if (status != cudaSuccess) {
        if (error && error_size)
            snprintf(error, error_size, "cannot query CUDA device: %s",
                     cudaGetErrorString(status));
        return 0;
    }
    snprintf(info->name, sizeof(info->name), "%.127s", properties.name);
    snprintf(info->architecture, sizeof(info->architecture), "sm_%d%d",
             properties.major, properties.minor);
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        info->physical_memory = (uint64_t)pages * (uint64_t)page_size;
    info->recommended_working_set = properties.totalGlobalMem;
    info->max_buffer_length = properties.totalGlobalMem;
    info->unified_memory = 0;
    info->apple_gpu_family = 0;
    info->metal4 = 0;
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    (void)shader_source_path;  /* CUDA kernels are compiled in, not loaded. */
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count < 1) {
        if (error && error_size)
            snprintf(error, error_size, "no CUDA device available: %s",
                     cudaGetErrorString(status));
        return NULL;
    }
    struct h3_gpu *gpu = (struct h3_gpu *)calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory");
        return NULL;
    }
    gpu->device = 0;
    status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaFree(NULL);  /* force ctx init */
    if (status != cudaSuccess) {
        if (error && error_size)
            snprintf(error, error_size, "cannot initialize CUDA device: %s",
                     cudaGetErrorString(status));
        free(gpu);
        return NULL;
    }
    if (getenv("H3_DEBUG_GPU_MEMORY")) {
        size_t available = 0, total = 0;
        if (cudaMemGetInfo(&available, &total) == cudaSuccess)
            fprintf(stderr, "h3: CUDA free memory at context creation: "
                    "%.3f GiB\n", (double)available / (1024.0 * 1024 * 1024));
    }
    {   /* High-water mark for the tensor block cache: ~20% of device RAM. */
        size_t available = 0, total = 0;
        if (cudaMemGetInfo(&available, &total) == cudaSuccess)
            gpu->cache_limit = total / 5;
    }
    /* cuBLASLt is an accelerator, not a requirement: when the handle or its
     * workspace cannot be created (or H3_CUDA_NO_CUBLAS is set) every op
     * falls back to the hand kernels. */
    const char *no_cublas = getenv("H3_CUDA_NO_CUBLAS");
    gpu->no_cublas = no_cublas && *no_cublas && strcmp(no_cublas, "0") != 0;
    if (!gpu->no_cublas &&
        cublasLtCreate(&gpu->lt) == CUBLAS_STATUS_SUCCESS) {
        gpu->lt_workspace_bytes = 32u * 1024 * 1024;
        if (cudaMalloc(&gpu->lt_workspace, gpu->lt_workspace_bytes) !=
            cudaSuccess) {
            cublasLtDestroy(gpu->lt);
            gpu->lt = NULL;
            gpu->lt_workspace = NULL;
            gpu->lt_workspace_bytes = 0;
        }
    } else {
        gpu->lt = NULL;
    }
    gpu->profile_start_wall = h3_cuda_now();
    gpu->profile_mark_wall = gpu->profile_start_wall;
    gpu->profile_start_stats = gpu->stats;
    gpu->profile_mark_stats = gpu->stats;
    return (h3_gpu *)gpu;
}

void h3_gpu_free(h3_gpu *opaque) {
    if (!opaque) return;
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    h3_cuda_profile_emit(gpu, "total", &gpu->profile_start_stats,
                         gpu->profile_start_wall);
    if (getenv("H3_DEBUG_GPU_MEMORY")) {
        size_t available = 0, total = 0;
        if (cudaMemGetInfo(&available, &total) == cudaSuccess)
            fprintf(stderr, "h3: CUDA free memory at context teardown: "
                    "%.3f GiB\n", (double)available / (1024.0 * 1024 * 1024));
    }
    if (gpu->lt) cublasLtDestroy(gpu->lt);
    if (gpu->lt_workspace) cudaFree(gpu->lt_workspace);
    if (gpu->workspace) cudaFree(gpu->workspace);
    h3_cuda_cache_evict_all(gpu);
    if (gpu->load_staging) cudaFreeHost(gpu->load_staging);
    h3_cuda_fp8_release(gpu);
    for (unsigned i = 0; i < H3_CUDA_H2D_SLOTS; i++) {
        struct h3_cuda_h2d_slot *slot = &gpu->h2d_slots[i];
        if (slot->host) cudaFreeHost(slot->host);
        if (slot->event_valid) cudaEventDestroy(slot->event);
    }
    cudaDeviceSynchronize();
    free(gpu);
}

void *h3_cuda_workspace(struct h3_gpu *gpu, size_t bytes) {
    if (!bytes) bytes = 1;
    if (gpu->workspace && bytes <= gpu->workspace_bytes)
        return gpu->workspace;
    size_t capacity = gpu->workspace_bytes ? gpu->workspace_bytes : 1 << 20;
    while (capacity < bytes) {
        if (capacity > SIZE_MAX / 2) { capacity = bytes; break; }
        capacity *= 2;
    }
    void *fresh = NULL;
    if (cudaMalloc(&fresh, capacity) != cudaSuccess)
        return NULL;  /* keep the old block; the caller falls back */
    /* In-flight kernels may still read the old block. */
    cudaDeviceSynchronize();
    cudaFree(gpu->workspace);
    gpu->workspace = fresh;
    gpu->workspace_bytes = capacity;
    return fresh;
}

/* ------------------------------------------------------------ tensor cache */
/* Exact-size free list of device blocks. Reuse is safe without a sync
 * because every op (kernels and the staged H2D copies below) is issued on
 * the legacy default stream: a block recycled into a new tensor is only
 * touched by work enqueued after the kernels that used its previous owner,
 * and the default stream serializes them. */

struct h3_cuda_cache_block {
    void *data;
    size_t bytes;
    struct h3_cuda_cache_block *next;
};

/* Release every cached block. cudaFree synchronizes the device, so this runs
 * only at allocation boundaries (or context teardown), never while building
 * the free list. */
static void h3_cuda_cache_evict_all(struct h3_gpu *gpu) {
    while (gpu->cache_blocks) {
        struct h3_cuda_cache_block *block = gpu->cache_blocks;
        gpu->cache_blocks = block->next;
        gpu->cache_bytes -= block->bytes;
        cudaFree(block->data);
        free(block);
    }
    gpu->cache_bytes = 0;
}

static cudaError_t h3_cuda_cache_alloc(struct h3_gpu *gpu, size_t bytes,
                                       void **data) {
    struct h3_cuda_cache_block **link = &gpu->cache_blocks;
    while (*link) {
        if ((*link)->bytes == bytes) {
            struct h3_cuda_cache_block *block = *link;
            *link = block->next;
            gpu->cache_bytes -= block->bytes;
            *data = block->data;
            free(block);
            return cudaSuccess;
        }
        link = &(*link)->next;
    }
    /* A fresh cudaMalloc device-synchronizes; if the cache is over its
     * high-water mark, release it first so it cannot pin memory the new
     * allocation needs. */
    if (gpu->cache_bytes > gpu->cache_limit)
        h3_cuda_cache_evict_all(gpu);
    cudaError_t status = cudaMalloc(data, bytes);
    if (status != cudaSuccess && gpu->cache_blocks) {
        /* The cache may be holding the memory this allocation needs. */
        h3_cuda_cache_evict_all(gpu);
        status = cudaMalloc(data, bytes);
    }
    return status;
}

static void h3_cuda_cache_release(struct h3_gpu *gpu, void *data,
                                  size_t bytes) {
    struct h3_cuda_cache_block *block =
        (struct h3_cuda_cache_block *)malloc(sizeof(*block));
    if (!block) {
        cudaFree(data);  /* out of host memory: fall back to a real free */
        return;
    }
    block->data = data;
    block->bytes = bytes;
    block->next = gpu->cache_blocks;
    gpu->cache_blocks = block;
    gpu->cache_bytes += bytes;
}

/* ------------------------------------------------------ staged H2D / D2H */
/* Small H2D writes go through a pinned staging slot and return without
 * syncing: the cudaMemcpyAsync is stream-ordered before any later kernel on
 * the default stream. The host source is consumed by the memcpy into the
 * pinned slot, so callers may free or reuse their buffer on return. Slot
 * reuse waits on the slot's event so a new write cannot overwrite pinned
 * bytes whose copy is still in flight. Falls back to a synchronous copy for
 * oversized writes or when staging cannot be set up. */
static int h3_cuda_h2d(struct h3_gpu *gpu, void *device, const void *host,
                       size_t bytes) {
    if (!bytes) return 1;
    if (gpu && bytes <= H3_CUDA_H2D_SLOT_MAX) {
        struct h3_cuda_h2d_slot *slot =
            &gpu->h2d_slots[gpu->h2d_next++ % H3_CUDA_H2D_SLOTS];
        if (bytes > slot->bytes) {
            /* Grow the slot: its last copy must finish before the old pinned
             * buffer is freed. */
            if (slot->event_valid &&
                cudaEventSynchronize(slot->event) != cudaSuccess)
                goto synchronous;
            if (slot->host) cudaFreeHost(slot->host);
            size_t capacity = slot->bytes ? slot->bytes : 1u << 20;
            while (capacity < bytes) {
                if (capacity > SIZE_MAX / 2) { capacity = bytes; break; }
                capacity *= 2;
            }
            void *pinned = NULL;
            if (cudaMallocHost(&pinned, capacity) != cudaSuccess) {
                slot->host = NULL;
                slot->bytes = 0;
                goto synchronous;
            }
            slot->host = pinned;
            slot->bytes = capacity;
        } else if (slot->event_valid) {
            cudaError_t query = cudaEventQuery(slot->event);
            if (query == cudaErrorNotReady) {
                if (cudaEventSynchronize(slot->event) != cudaSuccess)
                    goto synchronous;
            } else if (query != cudaSuccess) {
                goto synchronous;
            }
        }
        if (!slot->event_valid) {
            if (cudaEventCreateWithFlags(&slot->event,
                                         cudaEventDisableTiming) !=
                cudaSuccess)
                goto synchronous;
            slot->event_valid = 1;
        }
        memcpy(slot->host, host, bytes);
        cudaError_t status = cudaMemcpyAsync(device, slot->host, bytes,
                                             cudaMemcpyHostToDevice, 0);
        if (status == cudaSuccess)
            status = cudaEventRecord(slot->event, 0);
        if (status == cudaSuccess) return 1;
    }
synchronous:
    return cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) ==
           cudaSuccess;
}

/* Reads stay synchronous (callers need the data) but ride the default
 * stream explicitly. */
static int h3_cuda_d2h(void *host, const void *device, size_t bytes) {
    if (!bytes) return 1;
    if (cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, 0) !=
        cudaSuccess)
        return 0;
    return cudaStreamSynchronize(0) == cudaSuccess;
}

int h3_gpu_is_m5(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_prefers_gpu_sampler(const h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

static h3_gpu_tensor *h3_cuda_tensor_new(struct h3_gpu *gpu,
                                         const void *values, size_t elements,
                                         size_t item_size,
                                         h3_gpu_dtype dtype) {
    if (!gpu || elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    struct h3_gpu_tensor *tensor =
        (struct h3_gpu_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3_cuda_fail(gpu, "out of memory");
        return NULL;
    }
    size_t allocation = bytes ? bytes : 1;
    cudaError_t status = h3_cuda_cache_alloc(gpu, allocation, &tensor->data);
    if (status != cudaSuccess) {
        h3_cuda_fail_cuda(gpu, "tensor allocation", status);
        free(tensor);
        return NULL;
    }
    tensor->elements = elements;
    tensor->bytes = bytes;
    tensor->dtype = dtype;
    tensor->owner = gpu;
    if (values && bytes) {
        if (!h3_cuda_h2d(gpu, tensor->data, values, bytes)) {
            h3_cuda_fail(gpu, "tensor upload failed");
            cudaFree(tensor->data);
            free(tensor);
            return NULL;
        }
    }
    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    gpu->stats.tensor_allocations++;
    return (h3_gpu_tensor *)tensor;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_cuda_tensor_new((struct h3_gpu *)gpu, NULL, elements,
                              sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_cuda_tensor_new((struct h3_gpu *)gpu, NULL, elements,
                              sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_cuda_tensor_new((struct h3_gpu *)gpu, NULL, elements,
                              sizeof(int8_t), H3_GPU_I8);
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_cuda_tensor_new((struct h3_gpu *)gpu, values, elements,
                              sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_cuda_tensor_new((struct h3_gpu *)gpu, values, elements,
                              sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_cuda_tensor_new((struct h3_gpu *)gpu, values, elements,
                              sizeof(uint32_t), H3_GPU_U32);
}

/* ----------------------------------------------------- file payload reads */
/* Reads [file_offset, +bytes) into staging with EINTR retry and short-read
 * detection. Returns 0 on success, the failing errno on error, or -1 when
 * the file ended early. */
/* Read [file_offset, +bytes); `need` bytes must land (the rest may be cut
 * short by end of file — O_DIRECT spans are rounded up to the block size). */
static int h3_cuda_read_range_min(int descriptor, unsigned char *staging,
                                  uint64_t file_offset, size_t bytes,
                                  size_t need) {
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = bytes - completed;
        if (request > (size_t)SSIZE_MAX) request = (size_t)SSIZE_MAX;
        ssize_t count = pread(descriptor, staging + completed, request,
                              (off_t)(file_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) return errno;
        if (count == 0) return completed >= need ? 0 : -1;
        completed += (size_t)count;
    }
    return 0;
}

static int h3_cuda_read_range(int descriptor, unsigned char *staging,
                              uint64_t file_offset, size_t bytes) {
    return h3_cuda_read_range_min(descriptor, staging, file_offset, bytes,
                                  bytes);
}

/* The file->host stage of a weight load is disk-bound, so large payloads are
 * read by a small per-call worker pool: each thread preads a contiguous
 * chunk of the range into its own slice of the staging buffer. The (much
 * faster) H2D upload stays sequential on the calling thread. */
#define H3_CUDA_PARALLEL_READ_MIN ((size_t)16 << 20)
#define H3_CUDA_PARALLEL_READ_THREADS 8
#define H3_CUDA_DIRECT_READ_THREADS 16
#define H3_CUDA_DIRECT_ALIGN ((size_t)4096)

struct h3_cuda_read_chunk {
    int descriptor;
    unsigned char *staging;
    uint64_t file_offset;
    size_t bytes;
    size_t need;
    int status;  /* h3_cuda_read_range result */
};

static void *h3_cuda_read_worker(void *argument) {
    struct h3_cuda_read_chunk *chunk = (struct h3_cuda_read_chunk *)argument;
    chunk->status = h3_cuda_read_range_min(chunk->descriptor, chunk->staging,
                                           chunk->file_offset, chunk->bytes,
                                           chunk->need);
    return NULL;
}

/* Splits [file_offset, +bytes) into per-thread chunks, reads them
 * concurrently, and joins every worker before returning. Returns the first
 * chunk failure (errno or -1), 0 when the whole range landed. If thread
 * creation fails the remaining range is read inline, so a pthread problem
 * degrades to the sequential path instead of failing the load. */
static int h3_cuda_read_parallel_min(int descriptor, unsigned char *staging,
                                     uint64_t file_offset, size_t bytes,
                                     size_t need, int thread_count) {
    struct h3_cuda_read_chunk chunks[H3_CUDA_DIRECT_READ_THREADS];
    pthread_t threads[H3_CUDA_DIRECT_READ_THREADS];
    if (thread_count > H3_CUDA_DIRECT_READ_THREADS)
        thread_count = H3_CUDA_DIRECT_READ_THREADS;
    /* Chunk boundaries stay on 4 KiB so O_DIRECT reads remain aligned. */
    size_t chunk_bytes = (bytes + (size_t)thread_count - 1) /
                         (size_t)thread_count;
    chunk_bytes = (chunk_bytes + H3_CUDA_DIRECT_ALIGN - 1) &
                  ~(H3_CUDA_DIRECT_ALIGN - 1);
    int spawned = 0;
    int tail_status = 0;
    for (int i = 0; i < thread_count; i++) {
        size_t begin = (size_t)i * chunk_bytes;
        if (begin >= bytes) break;
        struct h3_cuda_read_chunk *chunk = &chunks[i];
        chunk->descriptor = descriptor;
        chunk->staging = staging + begin;
        chunk->file_offset = file_offset + begin;
        chunk->bytes = bytes - begin < chunk_bytes ? bytes - begin
                                                   : chunk_bytes;
        chunk->need = need > begin ? (need - begin < chunk->bytes
                                          ? need - begin : chunk->bytes)
                                   : 0;
        chunk->status = 0;
        if (pthread_create(&threads[i], NULL, h3_cuda_read_worker,
                           chunk) == 0) {
            spawned++;
            continue;
        }
        /* No thread: finish this and all later chunks inline. */
        tail_status = h3_cuda_read_range_min(
            descriptor, staging + begin, file_offset + begin, bytes - begin,
            need > begin ? need - begin : 0);
        break;
    }
    for (int i = 0; i < spawned; i++) pthread_join(threads[i], NULL);
    for (int i = 0; i < spawned; i++)
        if (chunks[i].status) return chunks[i].status;
    return tail_status;
}

static int h3_cuda_read_parallel(int descriptor, unsigned char *staging,
                                 uint64_t file_offset, size_t bytes) {
    return h3_cuda_read_parallel_min(descriptor, staging, file_offset, bytes,
                                     bytes, H3_CUDA_PARALLEL_READ_THREADS);
}

static int h3_cuda_direct_io_enabled(void) {
    const char *value = getenv("H3_CUDA_DIRECT_IO");
    return !(value && *value && strcmp(value, "0") == 0);
}

/* Unbuffered read of [file_offset, +bytes) into page-aligned pinned
 * staging (which must hold bytes + 2 * H3_CUDA_DIRECT_ALIGN): the 4 KiB
 * aligned superset is read with O_DIRECT, so the NVMe serves the payload at
 * its raw rate instead of through the page cache (measured 2.8 GB/s
 * buffered against 10 GB/s raw here). Returns the payload's offset inside
 * `staging`, or (size_t)-1 when direct I/O is unavailable, in which case
 * nothing was read and the caller uses the buffered path. */
static size_t h3_cuda_read_direct(const char *path, unsigned char *staging,
                                  uint64_t file_offset, size_t bytes,
                                  int *status) {
    if (!h3_cuda_direct_io_enabled() ||
        ((uintptr_t)staging & (H3_CUDA_DIRECT_ALIGN - 1)))
        return (size_t)-1;
    int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (descriptor < 0) return (size_t)-1;
    uint64_t start = file_offset & ~(uint64_t)(H3_CUDA_DIRECT_ALIGN - 1);
    uint64_t end = (file_offset + bytes + H3_CUDA_DIRECT_ALIGN - 1) &
                   ~(uint64_t)(H3_CUDA_DIRECT_ALIGN - 1);
    size_t span = (size_t)(end - start);
    size_t need = (size_t)(file_offset - start) + bytes;
    int rc = span >= H3_CUDA_PARALLEL_READ_MIN
        ? h3_cuda_read_parallel_min(descriptor, staging, start, span, need,
                                    H3_CUDA_DIRECT_READ_THREADS)
        : h3_cuda_read_range_min(descriptor, staging, start, span, need);
    close(descriptor);
    if (rc == EINVAL) return (size_t)-1; /* filesystem refused O_DIRECT */
    *status = rc;
    return (size_t)(file_offset - start);
}

/* H3_PROFILE=1 load accounting: where whole-tensor file loads spend their
 * time (printed by h3_gpu_profile_mark / at context free). */
static double h3_cuda_load_read_seconds, h3_cuda_load_upload_seconds,
    h3_cuda_load_alloc_seconds;
static size_t h3_cuda_load_bytes;
static unsigned h3_cuda_load_calls;

static void h3_cuda_load_report(void) {
    if (!h3_cuda_load_calls || !h3_cuda_profile_enabled()) return;
    fprintf(stderr,
            "h3 profile file loads: %u tensors, %.2f GiB, read %.2fs (%.1f "
            "GB/s), upload %.2fs, alloc %.2fs\n",
            h3_cuda_load_calls, (double)h3_cuda_load_bytes / 1073741824.0,
            h3_cuda_load_read_seconds,
            h3_cuda_load_read_seconds > 0
                ? (double)h3_cuda_load_bytes / h3_cuda_load_read_seconds / 1e9
                : 0.0,
            h3_cuda_load_upload_seconds, h3_cuda_load_alloc_seconds);
    h3_cuda_load_calls = 0;
    h3_cuda_load_bytes = 0;
    h3_cuda_load_read_seconds = h3_cuda_load_upload_seconds =
        h3_cuda_load_alloc_seconds = 0.0;
}

/* Pinned, grow-only staging for file loads; NULL when pinned memory is
 * unavailable (callers fall back to a pageable buffer). */
static unsigned char *h3_cuda_load_staging(struct h3_gpu *gpu, size_t bytes) {
    if (gpu->load_staging_bytes >= bytes && gpu->load_staging)
        return (unsigned char *)gpu->load_staging;
    if (gpu->load_staging) cudaFreeHost(gpu->load_staging);
    gpu->load_staging = NULL;
    gpu->load_staging_bytes = 0;
    void *fresh = NULL;
    if (cudaMallocHost(&fresh, bytes ? bytes : 1) != cudaSuccess) {
        (void)cudaGetLastError();
        return NULL;
    }
    gpu->load_staging = fresh;
    gpu->load_staging_bytes = bytes;
    return (unsigned char *)fresh;
}

static h3_gpu_tensor *h3_cuda_tensor_load_file(
        struct h3_gpu *gpu, const char *path, uint64_t file_offset,
        size_t elements, size_t item_size, h3_gpu_dtype dtype,
        const char *label) {
    if (!gpu || !path || !*path || file_offset > INT64_MAX ||
        elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) return NULL;
    double t_alloc = h3_cuda_now();
    h3_gpu_tensor *result =
        h3_cuda_tensor_new(gpu, NULL, elements, item_size, dtype);
    if (!result) return NULL;
    h3_cuda_load_alloc_seconds += h3_cuda_now() - t_alloc;
    struct h3_gpu_tensor *tensor = (struct h3_gpu_tensor *)result;
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        h3_cuda_fail(gpu, "cannot open %s: %s", path, strerror(errno));
        h3_gpu_tensor_free(result);
        return NULL;
    }
    unsigned char *staging =
        h3_cuda_load_staging(gpu, bytes + 2 * H3_CUDA_DIRECT_ALIGN);
    int pinned = staging != NULL;
    if (!staging) staging = (unsigned char *)malloc(bytes ? bytes : 1);
    if (!staging) {
        h3_cuda_fail(gpu, "out of memory staging %s payload", label);
        close(descriptor);
        h3_gpu_tensor_free(result);
        return NULL;
    }
    double t_read = h3_cuda_now();
    size_t payload_offset = 0;
    int read_status = 0;
    size_t direct = pinned ? h3_cuda_read_direct(path, staging, file_offset,
                                                 bytes, &read_status)
                           : (size_t)-1;
    if (direct != (size_t)-1) {
        payload_offset = direct;
    } else {
        read_status = bytes >= H3_CUDA_PARALLEL_READ_MIN
            ? h3_cuda_read_parallel(descriptor, staging, file_offset, bytes)
            : h3_cuda_read_range(descriptor, staging, file_offset, bytes);
    }
    close(descriptor);
    h3_cuda_load_read_seconds += h3_cuda_now() - t_read;
    h3_cuda_load_bytes += bytes;
    h3_cuda_load_calls++;
    if (read_status) {
        h3_cuda_fail(gpu, "cannot read %s payload from %s: %s", label, path,
                     read_status > 0 ? strerror(read_status)
                                     : "unexpected end of file");
        if (!pinned) free(staging);
        h3_gpu_tensor_free(result);
        return NULL;
    }
    if (bytes) {
        /* Pinned staging uploads with one synchronous DMA (the buffer is
         * reused by the next load); the pageable fallback goes through the
         * H2D ring. */
        double t_upload = h3_cuda_now();
        int uploaded = pinned
            ? cudaMemcpy(tensor->data, staging + payload_offset, bytes,
                         cudaMemcpyHostToDevice) == cudaSuccess
            : h3_cuda_h2d(gpu, tensor->data, staging, bytes);
        h3_cuda_load_upload_seconds += h3_cuda_now() - t_upload;
        if (!pinned) free(staging);
        if (!uploaded) {
            (void)cudaGetLastError();
            h3_cuda_fail(gpu, "tensor upload failed");
            h3_gpu_tensor_free(result);
            return NULL;
        }
    } else if (!pinned) {
        free(staging);
    }
    return result;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    return h3_cuda_tensor_load_file((struct h3_gpu *)gpu, path, file_offset,
                                    elements, sizeof(uint16_t), H3_GPU_BF16,
                                    "BF16");
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    return h3_cuda_tensor_load_file((struct h3_gpu *)gpu, path, file_offset,
                                    elements, sizeof(float), H3_GPU_F32,
                                    "F32");
}

static int h3_cuda_tensor_read_file_bf16_mode(
        h3_gpu_tensor *opaque, const char *path, uint64_t file_offset,
        size_t elements, int uncached, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    struct h3_gpu_tensor *tensor = (struct h3_gpu_tensor *)opaque;
    if (!tensor || !path || !*path || tensor->dtype != H3_GPU_BF16 ||
        elements != tensor->elements ||
        elements > SIZE_MAX / sizeof(uint16_t) || file_offset > INT64_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "invalid BF16 file read request");
        return 0;
    }
    size_t bytes = elements * sizeof(uint16_t);
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) {
        if (error && error_size)
            snprintf(error, error_size, "BF16 file read range overflows");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        return 0;
    }
    if (uncached)
        (void)posix_fadvise(descriptor, (off_t)file_offset, (off_t)bytes,
                            POSIX_FADV_NOREUSE);
    unsigned char *staging = (unsigned char *)malloc(bytes ? bytes : 1);
    if (!staging) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        close(descriptor);
        return 0;
    }
    int read_status = bytes >= H3_CUDA_PARALLEL_READ_MIN
        ? h3_cuda_read_parallel(descriptor, staging, file_offset, bytes)
        : h3_cuda_read_range(descriptor, staging, file_offset, bytes);
    close(descriptor);
    if (read_status) {
        if (error && error_size)
            snprintf(error, error_size,
                     "cannot read BF16 payload from %s: %s", path,
                     read_status > 0 ? strerror(read_status)
                                     : "unexpected end of file");
        free(staging);
        return 0;
    }
    int uploaded = bytes ? h3_cuda_h2d(tensor->owner, tensor->data, staging,
                                       bytes)
                         : 1;
    free(staging);
    if (!uploaded) {
        if (error && error_size)
            snprintf(error, error_size, "cannot upload BF16 payload");
        return 0;
    }
    return 1;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_cuda_tensor_read_file_bf16_mode(tensor, path, file_offset,
                                              elements, 0, error, error_size);
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_cuda_tensor_read_file_bf16_mode(tensor, path, file_offset,
                                              elements, 1, error, error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *opaque) {
    if (!opaque) return;
    struct h3_gpu_tensor *tensor = (struct h3_gpu_tensor *)opaque;
    struct h3_gpu *owner = tensor->owner;
    if (owner) {
        owner->stats.live_bytes = owner->stats.live_bytes >= tensor->bytes ?
            owner->stats.live_bytes - tensor->bytes : 0;
    }
    if (tensor->data) {
        if (owner) h3_cuda_fp8_forget(owner, tensor->data);
        /* Return the block to the context cache instead of cudaFree (which
         * device-synchronizes). Reuse is stream-ordered on the default
         * stream; the cache is evicted at allocation boundaries. */
        if (owner)
            h3_cuda_cache_release(owner, tensor->data,
                                  tensor->bytes ? tensor->bytes : 1);
        else
            cudaFree(tensor->data);
    }
    free(tensor);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? ((const struct h3_gpu_tensor *)tensor)->elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? ((const struct h3_gpu_tensor *)tensor)->dtype
                  : H3_GPU_F32;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    const struct h3_gpu_tensor *view = (const struct h3_gpu_tensor *)tensor;
    if (!view || !values || view->dtype != H3_GPU_F32 ||
        source_offset > view->elements ||
        elements > view->elements - source_offset) return 0;
    return h3_cuda_d2h(values,
                       (const unsigned char *)view->data +
                           source_offset * sizeof(float),
                       elements * sizeof(float));
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    const struct h3_gpu_tensor *view = (const struct h3_gpu_tensor *)tensor;
    if (!view || !values || view->dtype != H3_GPU_BF16 ||
        elements > view->elements) return 0;
    return h3_cuda_d2h(values, view->data, elements * sizeof(uint16_t));
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    struct h3_gpu_tensor *view = (struct h3_gpu_tensor *)tensor;
    if (!view || !values || view->dtype != H3_GPU_F32 ||
        destination_offset > view->elements ||
        elements > view->elements - destination_offset) return 0;
    return h3_cuda_h2d(view->owner,
                       (unsigned char *)view->data +
                           destination_offset * sizeof(float),
                       values, elements * sizeof(float));
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    struct h3_gpu_tensor *view = (struct h3_gpu_tensor *)tensor;
    if (!view || !values || view->dtype != H3_GPU_BF16 ||
        destination_offset > view->elements ||
        elements > view->elements - destination_offset) return 0;
    return h3_cuda_h2d(view->owner,
                       (unsigned char *)view->data +
                           destination_offset * sizeof(uint16_t),
                       values, elements * sizeof(uint16_t));
}

int h3_gpu_begin(h3_gpu *opaque) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    if (!gpu || gpu->active) return 0;
    gpu->error[0] = '\0';
    gpu->active = 1;
    gpu->command_start_wall = h3_cuda_now();
    return 1;
}

int h3_gpu_continue(h3_gpu *opaque) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    if (!gpu || !gpu->active) return 0;
    double now = h3_cuda_now();
    gpu->stats.submissions++;
    gpu->stats.command_encode_seconds += now - gpu->command_start_wall;
    gpu->command_start_wall = now;
    return 1;
}

int h3_gpu_submit(h3_gpu *opaque) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    if (!gpu || !gpu->active) return 0;
    gpu->active = 0;
    double commit_time = h3_cuda_now();
    cudaError_t status = cudaDeviceSynchronize();
    double complete_time = h3_cuda_now();
    gpu->stats.submissions++;
    gpu->stats.command_encode_seconds += commit_time -
                                         gpu->command_start_wall;
    gpu->stats.command_wait_seconds += complete_time - commit_time;
    gpu->stats.gpu_seconds += complete_time - commit_time;
    if (status != cudaSuccess)
        return h3_cuda_fail_cuda(gpu, "command execution", status);
    return 1;
}

const char *h3_gpu_error(const h3_gpu *opaque) {
    const struct h3_gpu *gpu = (const struct h3_gpu *)opaque;
    if (!gpu || !gpu->error[0]) return "unknown CUDA error";
    return gpu->error;
}

int h3_gpu_get_stats(const h3_gpu *opaque, h3_gpu_stats *stats) {
    if (!opaque || !stats) return 0;
    *stats = ((const struct h3_gpu *)opaque)->stats;
    return 1;
}

void h3_gpu_profile_set_label(h3_gpu *opaque, const char *label) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    if (!gpu || !label || !*label) return;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s", label);
}

void h3_gpu_profile_mark(h3_gpu *opaque, const char *phase) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    if (!gpu || !phase || !*phase || !h3_cuda_profile_enabled()) return;
    h3_cuda_load_report();
    h3_cuda_profile_emit(gpu, phase, &gpu->profile_mark_stats,
                         gpu->profile_mark_wall);
    gpu->profile_mark_stats = gpu->stats;
    gpu->profile_mark_wall = h3_cuda_now();
}

/* Casts and device copies. */

__global__ void h3_cuda_cast_f32_to_bf16_kernel(
        const float *input, uint16_t *output, uint32_t count) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride)
        output[index] = h3_f32_to_bf16(input[index]);
}

__global__ void h3_cuda_cast_bf16_to_f32_kernel(
        const uint16_t *input, float *output, uint32_t count) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride)
        output[index] = h3_bf16_to_f32(input[index]);
}

/* Enough blocks to fill the device twice over without launching one block
 * per 256 elements on multi-gigabyte tensors. */
static uint32_t h3_cuda_cast_blocks(uint32_t elements) {
    uint32_t blocks = (elements + 255) / 256;
    static int sm_count = 0;
    if (!sm_count) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                               device);
        if (sm_count <= 0) sm_count = 1;
    }
    uint32_t cap = (uint32_t)sm_count * 16;
    return blocks > cap ? cap : blocks;
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    struct h3_gpu_tensor *out = (struct h3_gpu_tensor *)output;
    const struct h3_gpu_tensor *in = (const struct h3_gpu_tensor *)input;
    if (!h3_cuda_require_elements(gpu, in, elements, "cast input") ||
        !h3_cuda_require_dtype(gpu, in, H3_GPU_F32, "cast input") ||
        !h3_cuda_require_elements(gpu, out, elements, "cast output") ||
        !h3_cuda_require_dtype(gpu, out, H3_GPU_BF16, "cast output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (!elements) return 1;
    h3_cuda_cast_f32_to_bf16_kernel<<<h3_cuda_cast_blocks(elements), 256>>>(
        (const float *)in->data, (uint16_t *)out->data, elements);
    return h3_cuda_launch_check(gpu, "cast f32->bf16");
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    struct h3_gpu_tensor *out = (struct h3_gpu_tensor *)output;
    const struct h3_gpu_tensor *in = (const struct h3_gpu_tensor *)input;
    if (!h3_cuda_require_elements(gpu, in, elements, "cast input") ||
        !h3_cuda_require_dtype(gpu, in, H3_GPU_BF16, "cast input") ||
        !h3_cuda_require_elements(gpu, out, elements, "cast output") ||
        !h3_cuda_require_dtype(gpu, out, H3_GPU_F32, "cast output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (!elements) return 1;
    h3_cuda_cast_bf16_to_f32_kernel<<<h3_cuda_cast_blocks(elements), 256>>>(
        (const uint16_t *)in->data, (float *)out->data, elements);
    return h3_cuda_launch_check(gpu, "cast bf16->f32");
}

static int h3_cuda_copy(h3_gpu *opaque, h3_gpu_tensor *destination,
                        size_t destination_offset,
                        const h3_gpu_tensor *source, size_t source_offset,
                        size_t elements, h3_gpu_dtype dtype,
                        const char *label) {
    struct h3_gpu *gpu = (struct h3_gpu *)opaque;
    struct h3_gpu_tensor *dst = (struct h3_gpu_tensor *)destination;
    const struct h3_gpu_tensor *src = (const struct h3_gpu_tensor *)source;
    size_t item_size = h3_cuda_item_size(dtype);
    if (!h3_cuda_require_dtype(gpu, src, dtype, label) ||
        !h3_cuda_require_dtype(gpu, dst, dtype, label) ||
        source_offset > (src ? src->elements : 0) ||
        elements > (src ? src->elements : 0) - source_offset ||
        destination_offset > (dst ? dst->elements : 0) ||
        elements > (dst ? dst->elements : 0) - destination_offset) {
        return h3_cuda_fail(gpu, "invalid %s copy range", label);
    }
    if (!h3_cuda_require_command(gpu)) return 0;
    if (!elements) return 1;
    /* Async on the default stream: ordered after prior work like the old
     * synchronous copy, but without draining the pipeline. */
    cudaError_t status = cudaMemcpyAsync(
        (unsigned char *)dst->data + destination_offset * item_size,
        (const unsigned char *)src->data + source_offset * item_size,
        elements * item_size, cudaMemcpyDeviceToDevice, 0);
    if (status != cudaSuccess)
        return h3_cuda_fail_cuda(gpu, "tensor copy", status);
    gpu->stats.blit_copies++;
    return 1;
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    return h3_cuda_copy(gpu, destination, destination_offset, source,
                        source_offset, elements, H3_GPU_BF16, "BF16");
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    return h3_cuda_copy(gpu, destination, destination_offset, source,
                        source_offset, elements, H3_GPU_F32, "F32");
}
