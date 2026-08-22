/* cublasLt GEMM helper for the CUDA backend: row-major
 * C[rows,n] = X[rows,k] * W[n,k]^T (+ optional bias[n]) with F32
 * accumulation. cublasLt is column-major, so the call is expressed through
 * the transpose duality: the row-major buffers
 *
 *   X (rows x k), W (n x k), C (rows x n)
 *
 * are the same memory as the column-major matrices
 *
 *   X' = X^T (k x rows, ld k), W' = W^T (k x n, ld k), C' = C^T (n x rows,
 *   ld n),
 *
 * and C = X W^T  <=>  C' = W X^T = (W')^T X', so a single column-major
 * GEMM C' = op_T(W') * op_N(X') computes exactly C. The bias epilogue
 * broadcasts along the rows of C' = output channels of C, which is the
 * per-output-channel bias we want.
 *
 * Alignment staging: cublasLt serves misaligned operands (pointers not
 * 16-byte aligned, e.g. element-offset views into tensors, or leading
 * dimensions whose byte stride is not a multiple of 16) only through its
 * slow align1 SIMT kernels. h3_cuda_gemm_xwt detects that case and stages
 * the misaligned operands through the grow-only workspace: a pitched
 * device-to-device copy into an aligned, ld-padded slice, the GEMM on the
 * staged buffers, and a pitched copy back when the output was staged. All
 * of it runs on the legacy default stream, so copies serialize with the
 * GEMM and the surrounding kernels exactly like any other launch.
 *
 * Any failure (no heuristic for the dtype/shape/alignment on this device,
 * unsupported epilogue, workspace allocation or copy failure, launch error)
 * makes the helper return 0 and the caller falls back to its hand kernel;
 * nothing is left half-written except the C buffer, which the hand kernel
 * fully overwrites. */
/* h3_gpu.h is a C API: give the declarations C linkage so gcc-compiled
 * model-layer objects link against these nvcc-compiled definitions. */
extern "C" {
#include "h3_gpu.h"
}
#include "h3_cuda_internal.h"

/* Bias add for when cublasLt serves the GEMM but not the bias epilogue: the
 * GEMM runs with beta=0 and the bias is added in F32, rounding to BF16 once
 * at the end (the hand kernels seed the accumulator with the bias instead;
 * both keep F32 accumulation). */
__global__ void h3k_gemm_bias_f32(float *c, const float *bias, uint32_t rows,
                                  uint32_t n) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= n) return;
    c[(size_t)row * n + column] += bias[column];
}

__global__ void h3k_gemm_bias_bf16(uint16_t *c, const uint16_t *bias,
                                   uint32_t rows, uint32_t n) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= n) return;
    size_t index = (size_t)row * n + column;
    c[index] = h3_f32_to_bf16(h3_bf16_to_f32(c[index]) +
                              h3_bf16_to_f32(bias[column]));
}

static dim3 h3_cuda_gemm_grid_2d(uint32_t width, uint32_t height) {
    return dim3((width + 15) / 16, (height + 15) / 16, 1);
}

/* Per-shape cuBLASLt algorithm cache. cublasLtMatmulAlgoGetHeuristic's
 * single top result is the generic "most likely" pick; asking for several
 * candidates and timing them once per shape reliably finds faster kernels
 * for the VAE decoder's F32 projection shapes. The engine issues GPU calls
 * from a single thread, so the table needs no locking. Entries key on the
 * full GEMM descriptor signature (shape, dtypes, bias epilogue, leading
 * dimensions and pointer alignments as actually handed to cuBLASLt after
 * staging), so staging choices stay consistent with what was timed. */
#define H3_CUDA_GEMM_ALGO_CACHE_MAX 64
#define H3_CUDA_GEMM_ALGO_CANDIDATES 8
#define H3_CUDA_GEMM_ALGO_REPS 3
/* GEMMs whose single timed run takes at least this long are measured once;
 * shorter ones get H3_CUDA_GEMM_ALGO_REPS runs to average out launch
 * jitter. At 38k DiT rows every candidate is 30-150 ms, and the old
 * 16 x (1 + 3) full-size probes per signature cost minutes per run. */
#define H3_CUDA_GEMM_ALGO_LONG_MS 4.0f

typedef struct {
    uint32_t rows, input_dim, output_dim;
    uint32_t ld_x, ld_w, ld_c;
    uint32_t align_a, align_b, align_c;
    int ab_type, cd_type, bias_epilogue;
    size_t workspace_bytes;     /* heuristic's required workspace */
    cublasLtMatmulAlgo_t algo;
    int valid;
} h3_cuda_gemm_algo_entry;

static h3_cuda_gemm_algo_entry
    h3_cuda_gemm_algo_cache[H3_CUDA_GEMM_ALGO_CACHE_MAX];
static unsigned h3_cuda_gemm_algo_cache_clock; /* round-robin eviction */

static h3_cuda_gemm_algo_entry *h3_cuda_gemm_algo_find(
    uint32_t rows, uint32_t input_dim, uint32_t output_dim, uint32_t ld_x,
    uint32_t ld_w, uint32_t ld_c, uint32_t align_a, uint32_t align_b,
    uint32_t align_c, cudaDataType ab_type, cudaDataType cd_type,
    int bias_epilogue) {
    for (int i = 0; i < H3_CUDA_GEMM_ALGO_CACHE_MAX; i++) {
        h3_cuda_gemm_algo_entry *entry = &h3_cuda_gemm_algo_cache[i];
        if (entry->valid && entry->rows == rows &&
            entry->input_dim == input_dim && entry->output_dim == output_dim &&
            entry->ld_x == ld_x && entry->ld_w == ld_w &&
            entry->ld_c == ld_c && entry->align_a == align_a &&
            entry->align_b == align_b && entry->align_c == align_c &&
            entry->ab_type == (int)ab_type && entry->cd_type == (int)cd_type &&
            entry->bias_epilogue == bias_epilogue)
            return entry;
    }
    return NULL;
}

static h3_cuda_gemm_algo_entry *h3_cuda_gemm_algo_slot(void) {
    for (int i = 0; i < H3_CUDA_GEMM_ALGO_CACHE_MAX; i++)
        if (!h3_cuda_gemm_algo_cache[i].valid)
            return &h3_cuda_gemm_algo_cache[i];
    h3_cuda_gemm_algo_entry *victim =
        &h3_cuda_gemm_algo_cache[h3_cuda_gemm_algo_cache_clock++ %
                                 H3_CUDA_GEMM_ALGO_CACHE_MAX];
    return victim;
}

/* One warmup plus H3_CUDA_GEMM_ALGO_REPS timed runs of a candidate on the
 * legacy default stream, measured with events. Every run fully overwrites C
 * (beta = 0; the bias epilogue is idempotent), so probing leaves a valid
 * result behind. Returns the average milliseconds per run, or -1 when the
 * candidate cannot run or cannot be timed. A failed candidate's sticky
 * launch error is drained so the final run and the caller's error check see
 * a clean state. */
static float h3_cuda_gemm_time_algo(struct h3_gpu *gpu,
                                    cublasLtMatmulDesc_t desc,
                                    const void *weight,
                                    cublasLtMatrixLayout_t a_desc,
                                    const void *x,
                                    cublasLtMatrixLayout_t b_desc, void *c,
                                    cublasLtMatrixLayout_t c_desc,
                                    const float *alpha, const float *beta,
                                    const cublasLtMatmulHeuristicResult_t
                                        *candidate, cudaEvent_t start,
                                    cudaEvent_t stop) {
    if (candidate->state != CUBLAS_STATUS_SUCCESS ||
        candidate->workspaceSize > gpu->lt_workspace_bytes)
        return -1.0f;
    /* One warmup run, then one timed run; only when that run is short
     * (launch jitter matters) are H3_CUDA_GEMM_ALGO_REPS - 1 more timed. */
    float ms = 0.0f;
    int timed = 0;
    for (int rep = 0; rep <= H3_CUDA_GEMM_ALGO_REPS; rep++) {
        if (rep == 1 && cudaEventRecord(start, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            return -1.0f;
        }
        if (cublasLtMatmul(gpu->lt, desc, alpha, weight, a_desc, x, b_desc,
                           beta, c, c_desc, c, c_desc, &candidate->algo,
                           gpu->lt_workspace, gpu->lt_workspace_bytes,
                           0 /* legacy default stream */) !=
            CUBLAS_STATUS_SUCCESS) {
            (void)cudaGetLastError();
            return -1.0f;
        }
        if (rep == 0) continue;
        timed = rep;
        if (rep == 1) {
            if (cudaEventRecord(stop, 0) != cudaSuccess ||
                cudaEventSynchronize(stop) != cudaSuccess ||
                cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
                (void)cudaGetLastError();
                return -1.0f;
            }
            if (ms >= H3_CUDA_GEMM_ALGO_LONG_MS) return ms;
        }
    }
    if (cudaEventRecord(stop, 0) != cudaSuccess ||
        cudaEventSynchronize(stop) != cudaSuccess ||
        cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1.0f;
    }
    return ms / (float)timed;
}

/* Largest power-of-two alignment of a device pointer, capped at 256, so the
 * heuristic is only offered alignments the buffers actually have (offset
 * views into tensors are not necessarily 256-byte aligned). */
static uint32_t h3_cuda_gemm_alignment(const void *pointer) {
    uintptr_t address = (uintptr_t)pointer;
    uint32_t align = 256;
    while (align > 2 && (address & (align - 1))) align >>= 1;
    return align;
}

/* Element size of a cublasLt matrix type used here. */
static size_t h3_cuda_gemm_elem_size(cudaDataType type) {
    return type == CUDA_R_16BF || type == CUDA_R_16F ? sizeof(uint16_t) :
           type == CUDA_R_8I ? sizeof(int8_t) : sizeof(float);
}

/* Round a leading dimension (in elements) up so the row stride in bytes is
 * a multiple of 16, keeping whole elements. */
static uint32_t h3_cuda_gemm_pad_ld(uint32_t ld, size_t elem_size) {
    size_t bytes = ((size_t)ld * elem_size + 15) & ~(size_t)15;
    return (uint32_t)(bytes / elem_size);
}

static int h3_cuda_gemm_run(struct h3_gpu *gpu, const void *x,
                            const void *weight, const void *bias, void *c,
                            cudaDataType ab_type, cudaDataType cd_type,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim, uint32_t ld_x, uint32_t ld_w,
                            uint32_t ld_c, int bias_epilogue) {
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t a_desc = NULL, b_desc = NULL, c_desc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    int served = 0;
    int results = 0;
    float alpha = 1.0f, beta = 0.0f;
    uint32_t align_a = 0, align_b = 0, align_c = 0;
    cublasLtMatmulAlgo_t tuned_algo;
    const cublasLtMatmulAlgo_t *algo = NULL;
    h3_cuda_gemm_algo_entry *entry = NULL;
    memset(&tuned_algo, 0, sizeof(tuned_algo));
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasOperation_t identity = CUBLAS_OP_N;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
        CUBLAS_STATUS_SUCCESS) return 0;
    if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transpose, sizeof(transpose)) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &identity, sizeof(identity)) !=
            CUBLAS_STATUS_SUCCESS) goto done;
    if (bias_epilogue) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue)) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(desc,
                                           CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                           &bias, sizeof(bias)) !=
                CUBLAS_STATUS_SUCCESS) goto done;
    }
    /* A = W' stored ld_w x n column-major, op_T -> n x k.
     * B = X' stored ld_x x rows column-major, op_N -> k x rows.
     * C/D = C' stored ld_c x rows column-major. */
    if (cublasLtMatrixLayoutCreate(&a_desc, ab_type, input_dim, output_dim,
                                   ld_w) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&b_desc, ab_type, input_dim, rows,
                                   ld_x) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&c_desc, cd_type, output_dim, rows,
                                   ld_c) != CUBLAS_STATUS_SUCCESS)
        goto done;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
        goto done;
    if (cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &gpu->lt_workspace_bytes, sizeof(gpu->lt_workspace_bytes)) !=
        CUBLAS_STATUS_SUCCESS) goto done;
    /* Split-K candidates may reduce their partial sums in the output type
     * (bf16 for the DiT projections), adding rounding the F32-accumulate
     * contract does not allow — seen as 2-ulp bf16 misses once the cheaper
     * probing below started picking different winners. Allow only schemes
     * that reduce in the compute type (or do not split at all). */
    {
        uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_INPLACE |
                                  CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        if (cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                &reduction_mask, sizeof(reduction_mask)) !=
            CUBLAS_STATUS_SUCCESS) goto done;
    }
    /* Alignment attributes describe the buffers actually handed to the
     * GEMM: staged slices are 256-byte aligned, so staged calls are
     * offered the vectorized kernels instead of the align1 fallback. */
    align_a = h3_cuda_gemm_alignment(weight);
    align_b = h3_cuda_gemm_alignment(x);
    align_c = h3_cuda_gemm_alignment(c);
    if (cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            &align_a, sizeof(align_a)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            &align_b, sizeof(align_b)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            &align_c, sizeof(align_c)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            &align_c, sizeof(align_c)) != CUBLAS_STATUS_SUCCESS)
        goto done;
    /* Algorithm choice: a cached per-shape winner when this exact GEMM
     * signature was tuned before, otherwise a one-time empirical pick over
     * the heuristic candidates (the first heuristic result alone is kept as
     * the fallback when timing is impossible, preserving the old behavior).
     * The probing runs fully overwrite C, so after tuning one final run
     * with the winning algorithm leaves the definitive result. */
    entry = h3_cuda_gemm_algo_find(
        rows, input_dim, output_dim, ld_x, ld_w, ld_c, align_a, align_b,
        align_c, ab_type, cd_type, bias_epilogue);
    if (entry && entry->workspace_bytes <= gpu->lt_workspace_bytes) {
        algo = &entry->algo;
    } else {
        cublasLtMatmulHeuristicResult_t
            candidates[H3_CUDA_GEMM_ALGO_CANDIDATES];
        float best_ms = -1.0f, first_ms = -1.0f;
        int best = 0;
        cudaEvent_t start = NULL, stop = NULL;
        if (cublasLtMatmulAlgoGetHeuristic(
                gpu->lt, desc, a_desc, b_desc, c_desc, c_desc, preference,
                H3_CUDA_GEMM_ALGO_CANDIDATES, candidates, &results) !=
                CUBLAS_STATUS_SUCCESS ||
            results < 1)
            goto done;
        if (cudaEventCreate(&start) == cudaSuccess &&
            cudaEventCreate(&stop) == cudaSuccess) {
            for (int i = 0; i < results; i++) {
                float ms = h3_cuda_gemm_time_algo(
                    gpu, desc, weight, a_desc, x, b_desc, c, c_desc, &alpha,
                    &beta, &candidates[i], start, stop);
                if (i == 0) first_ms = ms;
                if (ms >= 0.0f && (best_ms < 0.0f || ms < best_ms)) {
                    best_ms = ms;
                    best = i;
                }
            }
        } else {
            (void)cudaGetLastError();
        }
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (best_ms >= 0.0f && getenv("H3_CUDA_GEMM_TUNE_LOG"))
            fprintf(stderr,
                    "h3 gemm tune rows=%u k=%u n=%u ab=%d cd=%d: %d/%d "
                    "candidates, best #%d %.3f ms (heuristic #0 %.3f ms)\n",
                    rows, input_dim, output_dim, (int)ab_type, (int)cd_type,
                    results, H3_CUDA_GEMM_ALGO_CANDIDATES, best, best_ms,
                    first_ms);
        if (best_ms >= 0.0f) {
            entry = h3_cuda_gemm_algo_slot();
            *entry = (h3_cuda_gemm_algo_entry){
                rows, input_dim, output_dim, ld_x, ld_w, ld_c,
                align_a, align_b, align_c,
                (int)ab_type, (int)cd_type, bias_epilogue,
                candidates[best].workspaceSize, candidates[best].algo, 1};
            tuned_algo = candidates[best].algo;
            algo = &tuned_algo;
        } else {
            algo = &candidates[0].algo;
        }
    }
    if (cublasLtMatmul(gpu->lt, desc, &alpha, weight, a_desc, x, b_desc,
                       &beta, c, c_desc, c, c_desc, algo,
                       gpu->lt_workspace, gpu->lt_workspace_bytes,
                       0 /* legacy default stream */) !=
        CUBLAS_STATUS_SUCCESS) goto done;
    served = 1;
done:
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatmulDescDestroy(desc);
    return served;
}

/* Carve a 256-byte-aligned slice of `bytes` from the back of the grow-only
 * workspace (callers carve their own slices from the front), never
 * overlapping the occupied intervals (operands that live in the workspace).
 * Returns NULL when the slice does not fit or would overlap. */
static void *h3_cuda_gemm_carve(struct h3_gpu *gpu, size_t *top, size_t bytes,
                                const char *const *occupied,
                                const size_t *occupied_bytes, int occupied_n) {
    if (!bytes || *top < bytes) return NULL;
    size_t at = (*top - bytes) & ~(size_t)255;
    char *slice = (char *)gpu->workspace + at;
    char *end = slice + bytes;
    for (int i = 0; i < occupied_n; i++)
        if (slice < occupied[i] + occupied_bytes[i] && occupied[i] < end)
            return NULL;
    *top = at;
    return slice;
}

int h3_cuda_gemm_xwt(struct h3_gpu *gpu, const void *x, const void *weight,
                     const void *bias, void *c, cudaDataType ab_type,
                     cudaDataType cd_type, uint32_t rows, uint32_t input_dim,
                     uint32_t output_dim) {
    if (!gpu || !gpu->lt || gpu->no_cublas || !rows || !input_dim ||
        !output_dim) return 0;
    size_t ab_elem = h3_cuda_gemm_elem_size(ab_type);
    size_t cd_elem = h3_cuda_gemm_elem_size(cd_type);
    int ld_ab_bad = (((size_t)input_dim * ab_elem) & 15) != 0;
    int ld_cd_bad = (((size_t)output_dim * cd_elem) & 15) != 0;
    int stage_x = (((uintptr_t)x & 15) != 0) || ld_ab_bad;
    int stage_w = (((uintptr_t)weight & 15) != 0) || ld_ab_bad;
    int stage_c = (((uintptr_t)c & 15) != 0) || ld_cd_bad;
    const void *gemm_x = x, *gemm_w = weight;
    void *gemm_c = c;
    uint32_t ld_x = input_dim, ld_w = input_dim, ld_c = output_dim;
    if (stage_x || stage_w || stage_c) {
        /* Dense byte spans, used to keep staged slices clear of operands
         * that already live in the workspace. */
        size_t x_span = (size_t)rows * input_dim * ab_elem;
        size_t w_span = (size_t)output_dim * input_dim * ab_elem;
        size_t c_span = (size_t)rows * output_dim * cd_elem;
        char *base = (char *)gpu->workspace;
        int x_alias = 0, w_alias = 0, c_alias = 0;
        if (base) {
            char *limit = base + gpu->workspace_bytes;
            x_alias = (char *)x >= base && (char *)x < limit;
            w_alias = (char *)weight >= base && (char *)weight < limit;
            c_alias = (char *)c >= base && (char *)c < limit;
        }
        /* A staged operand needs a copy-in; that copy is pitched, so a
         * source inside the workspace could overlap the fresh slice. These
         * shapes essentially never occur (workspace-resident inputs always
         * come from aligned carve sites), so fall back to the hand kernel. */
        if ((stage_x && x_alias) || (stage_w && w_alias)) return 0;
        uint32_t new_ld_x = h3_cuda_gemm_pad_ld(input_dim, ab_elem);
        uint32_t new_ld_w = h3_cuda_gemm_pad_ld(input_dim, ab_elem);
        uint32_t new_ld_c = h3_cuda_gemm_pad_ld(output_dim, cd_elem);
        size_t stage_x_bytes = stage_x ? (size_t)new_ld_x * rows * ab_elem : 0;
        size_t stage_w_bytes = stage_w ? (size_t)new_ld_w * output_dim * ab_elem : 0;
        size_t stage_c_bytes = stage_c ? (size_t)new_ld_c * rows * cd_elem : 0;
        size_t needed = stage_x_bytes + stage_w_bytes + stage_c_bytes + 3 * 256;
        if (needed > gpu->workspace_bytes) {
            /* Growth frees the old block; that would invalidate slices the
             * caller carved from it and still uses after this call. */
            if (x_alias || w_alias || c_alias) return 0;
            if (!h3_cuda_workspace(gpu, needed)) return 0;
        }
        const char *occupied[3];
        size_t occupied_bytes[3];
        int occupied_n = 0;
        if (x_alias) { occupied[occupied_n] = (const char *)x; occupied_bytes[occupied_n++] = x_span; }
        if (w_alias) { occupied[occupied_n] = (const char *)weight; occupied_bytes[occupied_n++] = w_span; }
        if (c_alias) { occupied[occupied_n] = (const char *)c; occupied_bytes[occupied_n++] = c_span; }
        size_t top = gpu->workspace_bytes;
        void *staged_c = NULL, *staged_w = NULL, *staged_x = NULL;
        if (stage_c) {
            staged_c = h3_cuda_gemm_carve(gpu, &top, stage_c_bytes, occupied,
                                          occupied_bytes, occupied_n);
            if (!staged_c) return 0;
        }
        if (stage_w) {
            staged_w = h3_cuda_gemm_carve(gpu, &top, stage_w_bytes, occupied,
                                          occupied_bytes, occupied_n);
            if (!staged_w) return 0;
        }
        if (stage_x) {
            staged_x = h3_cuda_gemm_carve(gpu, &top, stage_x_bytes, occupied,
                                          occupied_bytes, occupied_n);
            if (!staged_x) return 0;
        }
        /* Pitched device-to-device copies on the default stream; cublasLt
         * then runs behind them in stream order. The output is not copied
         * in: the GEMM overwrites it completely (beta = 0). */
        if (stage_x &&
            cudaMemcpy2DAsync(staged_x, (size_t)new_ld_x * ab_elem, x,
                              (size_t)input_dim * ab_elem,
                              (size_t)input_dim * ab_elem, rows,
                              cudaMemcpyDeviceToDevice,
                              0 /* legacy default stream */) != cudaSuccess)
            goto copy_failed;
        if (stage_w &&
            cudaMemcpy2DAsync(staged_w, (size_t)new_ld_w * ab_elem, weight,
                              (size_t)input_dim * ab_elem,
                              (size_t)input_dim * ab_elem, output_dim,
                              cudaMemcpyDeviceToDevice,
                              0 /* legacy default stream */) != cudaSuccess)
            goto copy_failed;
        gemm_x = stage_x ? staged_x : x;
        gemm_w = stage_w ? staged_w : weight;
        gemm_c = stage_c ? staged_c : c;
        ld_x = stage_x ? new_ld_x : ld_x;
        ld_w = stage_w ? new_ld_w : ld_w;
        ld_c = stage_c ? new_ld_c : ld_c;
    }
    {
        /* The bias epilogue needs an aligned bias vector; without one the
         * plain GEMM runs and the bias is added by the fallback kernel. */
        int served = bias && ((uintptr_t)bias & 15) == 0 &&
                     h3_cuda_gemm_run(gpu, gemm_x, gemm_w, bias, gemm_c,
                                      ab_type, cd_type, rows, input_dim,
                                      output_dim, ld_x, ld_w, ld_c, 1);
        if (!served &&
            !h3_cuda_gemm_run(gpu, gemm_x, gemm_w, NULL, gemm_c, ab_type,
                              cd_type, rows, input_dim, output_dim, ld_x,
                              ld_w, ld_c, 0))
            return 0;
        if (stage_c &&
            cudaMemcpy2DAsync(c, (size_t)output_dim * cd_elem, gemm_c,
                              (size_t)ld_c * cd_elem,
                              (size_t)output_dim * cd_elem, rows,
                              cudaMemcpyDeviceToDevice,
                              0 /* legacy default stream */) != cudaSuccess)
            goto copy_failed;
        if (served || !bias) return 1;
        /* The plain GEMM ran; add the bias ourselves (on the real output,
         * after the staged copy-back when there was one). */
        if (cd_type == CUDA_R_16BF)
            h3k_gemm_bias_bf16<<<h3_cuda_gemm_grid_2d(output_dim, rows),
                                 dim3(16, 16)>>>(
                (uint16_t *)c, (const uint16_t *)bias, rows, output_dim);
        else
            h3k_gemm_bias_f32<<<h3_cuda_gemm_grid_2d(output_dim, rows),
                                dim3(16, 16)>>>(
                (float *)c, (const float *)bias, rows, output_dim);
        return cudaGetLastError() == cudaSuccess;
    }
copy_failed:
    /* Drain the copy error so the caller's hand-kernel fallback launches
     * into a clean error state. */
    (void)cudaGetLastError();
    return 0;
}
