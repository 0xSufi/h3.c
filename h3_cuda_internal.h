/* Shared internals for the CUDA backend translation units. Mirrors the
 * semantics of h3_gpu.m: tensors are device allocations, every op launches
 * kernels eagerly on the legacy default stream (which serializes execution
 * exactly like the ordered Metal command queue), and h3_gpu_submit()
 * synchronizes the device. Not part of the public API. */
#ifndef H3_CUDA_INTERNAL_H
#define H3_CUDA_INTERNAL_H

#include "h3_gpu.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

struct h3_gpu_tensor {
    void *data;
    size_t elements;
    size_t bytes;
    h3_gpu_dtype dtype;
    struct h3_gpu *owner;
};

/* Pinned H2D staging ring geometry (h3_gpu_cuda.cu). */
#define H3_CUDA_H2D_SLOTS 8
#define H3_CUDA_H2D_SLOT_MAX ((size_t)64 * 1024 * 1024)

struct h3_cuda_h2d_slot {
    void *host;             /* cudaMallocHost pinned staging, grow-only */
    size_t bytes;
    cudaEvent_t event;      /* recorded after the slot's last H2D copy */
    int event_valid;
};

struct h3_gpu {
    int device;
    int active;                 /* inside h3_gpu_begin..h3_gpu_submit */
    char error[512];
    h3_gpu_stats stats;
    double command_start_wall;  /* monotonic seconds */
    char profile_label[128];
    h3_gpu_stats profile_start_stats;
    h3_gpu_stats profile_mark_stats;
    double profile_start_wall;
    double profile_mark_wall;
    cublasLtHandle_t lt;        /* NULL when cuBLASLt is unavailable */
    void *lt_workspace;         /* fixed cublasLt workspace (32 MiB) */
    size_t lt_workspace_bytes;
    void *workspace;            /* grow-only scratch for GEMM intermediates */
    size_t workspace_bytes;
    int no_cublas;              /* H3_CUDA_NO_CUBLAS: force hand kernels */
    /* Caching allocator: exact-size free list of device blocks returned by
     * h3_gpu_tensor_free (definition lives in h3_gpu_cuda.cu). Cached bytes
     * are evicted with cudaFree at allocation boundaries once cache_bytes
     * exceeds cache_limit, or when a fresh cudaMalloc fails. */
    struct h3_cuda_cache_block *cache_blocks;
    size_t cache_bytes;
    size_t cache_limit;
    /* Pinned H2D staging ring: writes up to H3_CUDA_H2D_SLOT_MAX go
     * host->pinned->device via cudaMemcpyAsync on the default stream and
     * return without syncing; a slot's event (recorded after its copy)
     * guards reuse so a new write cannot clobber a copy still in flight. */
    struct h3_cuda_h2d_slot h2d_slots[H3_CUDA_H2D_SLOTS];
    unsigned h2d_next;          /* round-robin slot cursor */
    /* Grow-only pinned staging for whole-tensor file loads
     * (h3_cuda_tensor_load_file): preads land in it directly and upload
     * with one DMA, instead of a fresh pageable malloc per tensor whose
     * page faults plus the extra copy into the H2D ring capped the DiT
     * load at ~0.6 GB/s on a 10 GB/s NVMe. */
    void *load_staging;
    size_t load_staging_bytes;
};

static inline size_t h3_cuda_item_size(h3_gpu_dtype dtype) {
    return dtype == H3_GPU_BF16 ? sizeof(uint16_t) :
           dtype == H3_GPU_I8 ? sizeof(int8_t) : sizeof(uint32_t);
}

static inline int h3_cuda_fail(struct h3_gpu *gpu, const char *format, ...) {
    if (gpu) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(gpu->error, sizeof(gpu->error), format, arguments);
        va_end(arguments);
    }
    return 0;
}

static inline int h3_cuda_fail_cuda(struct h3_gpu *gpu, const char *operation,
                                    cudaError_t status) {
    return h3_cuda_fail(gpu, "CUDA %s failed: %s", operation,
                        cudaGetErrorString(status));
}

/* Capture a kernel-launch failure as the context error. kind selects the
 * stats counter to bump: 0 = direct_dispatches, 1 = mps_linear_dispatches,
 * 2 = mps_conv_dispatches, 3 = mps_sdpa_dispatches. The mps_* kinds mark ops
 * the Metal backend routes through MPS/MPSGraph so that stats-visible
 * behavior (checked by several tests) matches. */
static inline int h3_cuda_launch_check_kind(struct h3_gpu *gpu,
                                            const char *operation, int kind) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return h3_cuda_fail_cuda(gpu, operation, status);
    if (kind == 1) gpu->stats.mps_linear_dispatches++;
    else if (kind == 2) gpu->stats.mps_conv_dispatches++;
    else if (kind == 3) gpu->stats.mps_sdpa_dispatches++;
    else gpu->stats.direct_dispatches++;
    return 1;
}

static inline int h3_cuda_launch_check(struct h3_gpu *gpu,
                                       const char *operation) {
    return h3_cuda_launch_check_kind(gpu, operation, 0);
}

static inline int h3_cuda_require_command(struct h3_gpu *gpu) {
    if (!gpu || !gpu->active)
        return h3_cuda_fail(gpu, "no active command buffer");
    return 1;
}

static inline int h3_cuda_require_elements(struct h3_gpu *gpu,
                                           const h3_gpu_tensor *tensor,
                                           size_t elements,
                                           const char *label) {
    if (!tensor || tensor->elements < elements)
        return h3_cuda_fail(gpu, "%s tensor is too small", label);
    return 1;
}

static inline int h3_cuda_require_dtype(struct h3_gpu *gpu,
                                        const h3_gpu_tensor *tensor,
                                        h3_gpu_dtype dtype,
                                        const char *label) {
    if (!tensor || tensor->dtype != dtype)
        return h3_cuda_fail(gpu, "%s tensor has the wrong dtype", label);
    return 1;
}

/* One CUDA block per row, matching Metal's dispatchThreadgroups row kernels. */
static inline int h3_cuda_row_threads(uint32_t width) {
    int threads = 256;
    while (threads > 32 && threads / 2 >= (int)width) threads /= 2;
    return threads;
}

static inline double h3_cuda_now(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1e-9;
}

/* Grow-only scratch buffer for GEMM intermediates (defined in
 * h3_gpu_cuda.cu). Returns NULL when the allocation fails; callers must fall
 * back to their hand kernels. */
void *h3_cuda_workspace(struct h3_gpu *gpu, size_t bytes);

/* Row-major C[rows,n] = X[rows,k] * W[n,k]^T (+ optional bias[n]) through
 * cublasLt with F32 accumulation (h3_cuda_gemm.cu). ab_type is the X/W
 * element type, cd_type the C element type; bias, when present, has the C
 * element type. Returns 0 when cuBLASLt cannot serve the call (no heuristic,
 * unsupported dtype/alignment on this device); the caller then falls back to
 * its hand kernel. */
int h3_cuda_gemm_xwt(struct h3_gpu *gpu, const void *x, const void *weight,
                     const void *bias, void *c, cudaDataType ab_type,
                     cudaDataType cd_type, uint32_t rows, uint32_t input_dim,
                     uint32_t output_dim);

/* BF16 helpers: raw uint16 storage, exact widening on load and
 * round-to-nearest-even on store, matching h3_shaders.metal. */
__device__ __forceinline__ float h3_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint16_t h3_f32_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

#endif
