/* CUDA port of the h3_gpu.h attention ops: QKV split + RMS norm + RoPE
 * variants, scaled-dot-product attention (tiled flash-attention-style
 * kernels with online softmax that replace Metal's MPSGraph path), causal
 * GQA, and the audio attention helpers. Kernel math mirrors
 * h3_shaders.metal; host-side validation and stats-counter behavior mirror
 * h3_gpu.m. F32 accumulation for BF16 storage, safe libm (no fast-math
 * intrinsics). The original naive block-per-row SDPA kernels are kept as
 * h3k_*_naive and selected with H3_CUDA_SDPA_NAIVE=1 (also the automatic
 * fallback for head_dim > 128 or tight shared memory). For bf16,
 * head_dim == 128, non-causal SDPA on sm_80+, h3k_sdpa_mma_bf16 replaces the
 * scalar inner loops with bf16 tensor-core mma (see its comment for the one
 * extra P-rounding boundary). Dispatch order: H3_CUDA_SDPA_NAIVE -> naive;
 * else bf16 && head_dim == 128 && cap >= 8 -> mma; else f32 ->
 * h3k_sdpa_flash_f32 (KVT=64 with >48KB smem opt-in where supported, else
 * 32; H3_CUDA_SDPA_FLASH_OLD=1 selects the original h3k_sdpa_flash<float>
 * for A/B); else scalar flash; smem overflow -> naive.
 * h3_gpu_head_rms_norm_bf16 lives in h3_cuda_core.cu and is not repeated
 * here. */
/* h3_gpu.h is a C API: give the declarations C linkage so gcc-compiled
 * model-layer objects link against these nvcc-compiled definitions. */
extern "C" {
#include "h3_gpu.h"
}
#include "h3_cuda_internal.h"

#include <stdlib.h>

/* ---------------------------------------------------------------- helpers */

static dim3 h3_attn_grid_1d(size_t count) {
    return dim3(count ? (unsigned)((count + 255) / 256) : 1, 1, 1);
}

static int h3_attn_require_bf16(struct h3_gpu *gpu,
                                const h3_gpu_tensor *tensor, size_t elements,
                                const char *label) {
    return h3_cuda_require_elements(gpu, tensor, elements, label) &&
           h3_cuda_require_dtype(gpu, tensor, H3_GPU_BF16, label);
}

static int h3_attn_require_f32(struct h3_gpu *gpu,
                               const h3_gpu_tensor *tensor, size_t elements,
                               const char *label) {
    return h3_cuda_require_elements(gpu, tensor, elements, label) &&
           h3_cuda_require_dtype(gpu, tensor, H3_GPU_F32, label);
}

static int h3_attn_max_shared(struct h3_gpu *gpu) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock,
                               gpu->device) != cudaSuccess || value <= 0)
        value = 48 * 1024;
    return value;
}

__device__ __forceinline__ float h3_attn_load(const float *p) { return *p; }
__device__ __forceinline__ float h3_attn_load(const uint16_t *p) {
    return h3_bf16_to_f32(*p);
}
__device__ __forceinline__ void h3_attn_store(float *p, float v) { *p = v; }
__device__ __forceinline__ void h3_attn_store(uint16_t *p, float v) {
    *p = h3_f32_to_bf16(v);
}

/* ------------------------------------------------------- QKV/RoPE kernels */

/* h3_qkv_rope_f32: fused QKV rows are [q|k|v] x [head, dim]; Q/K get a
 * weighted per-head RMS norm before half-pair RoPE; V is a plain split. */
__global__ void h3k_qkv_rope_f32(const float *qkv, const float *q_weight,
                                 const float *k_weight, const float *rope_cos,
                                 const float *rope_sin, float *query,
                                 float *key, float *value, uint32_t sequence,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint64_t)gid >= (uint64_t)head_dim * heads * sequence) return;
    uint32_t dimension = gid % head_dim;
    uint32_t head = (gid / head_dim) % heads;
    uint32_t row = gid / (head_dim * heads);
    size_t inner = (size_t)heads * head_dim;
    size_t q_base = (size_t)row * inner * 3 + (size_t)head * head_dim;
    size_t k_base = q_base + inner;
    size_t v_base = q_base + inner * 2;
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = qkv[q_base + d];
        float k = qkv[k_base + d];
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
    float q0 = qkv[q_base + dimension] * q_inverse * q_weight[dimension];
    float k0 = qkv[k_base + dimension] * k_inverse * k_weight[dimension];
    if (dimension < rope_half) {
        uint32_t pair = dimension + rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * rope_half + dimension];
        float s = rope_sin[row * rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < rope_half * 2) {
        uint32_t pair = dimension - rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * rope_half + pair];
        float s = rope_sin[row * rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    size_t output_index = ((size_t)row * heads + head) * head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[v_base + dimension];
}

/* h3_video_qkv_rope_f32: fused rows are [head, q/k/v, dim] and Q/K get an
 * unweighted per-head RMS norm before RoPE; V is a plain split. */
__global__ void h3k_video_qkv_rope_f32(const float *qkv, const float *rope_cos,
                                       const float *rope_sin, float *query,
                                       float *key, float *value,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, uint32_t rope_half,
                                       float epsilon) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint64_t)gid >= (uint64_t)head_dim * heads * sequence) return;
    uint32_t dimension = gid % head_dim;
    uint32_t head = (gid / head_dim) % heads;
    uint32_t row = gid / (head_dim * heads);
    size_t base = ((size_t)row * heads + head) * head_dim * 3;
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = qkv[base + d];
        float k = qkv[base + head_dim + d];
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
    float q0 = qkv[base + dimension] * q_inverse;
    float k0 = qkv[base + head_dim + dimension] * k_inverse;
    if (dimension < rope_half) {
        uint32_t pair = dimension + rope_half;
        float q1 = qkv[base + pair] * q_inverse;
        float k1 = qkv[base + head_dim + pair] * k_inverse;
        float c = rope_cos[row * rope_half + dimension];
        float s = rope_sin[row * rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < rope_half * 2) {
        uint32_t pair = dimension - rope_half;
        float q1 = qkv[base + pair] * q_inverse;
        float k1 = qkv[base + head_dim + pair] * k_inverse;
        float c = rope_cos[row * rope_half + pair];
        float s = rope_sin[row * rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    size_t output_index = ((size_t)row * heads + head) * head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[base + head_dim * 2 + dimension];
}

/* h3_qkv_rope_bf16 with the grouped flag: ungrouped rows are
 * [q/k/v, head, dim], grouped (H3 checkpoint) rows are [head, q/k/v, dim]. */
__global__ void h3k_qkv_rope_bf16(const uint16_t *qkv,
                                  const uint16_t *q_weight,
                                  const uint16_t *k_weight,
                                  const uint16_t *rope_cos,
                                  const uint16_t *rope_sin, uint16_t *query,
                                  uint16_t *key, uint16_t *value,
                                  uint32_t sequence, uint32_t heads,
                                  uint32_t head_dim, uint32_t rope_half,
                                  uint32_t grouped, float epsilon) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint64_t)gid >= (uint64_t)head_dim * heads * sequence) return;
    uint32_t dimension = gid % head_dim;
    uint32_t head = (gid / head_dim) % heads;
    uint32_t row = gid / (head_dim * heads);
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim;
    size_t k_base = q_base + inner;
    size_t v_base = q_base + inner * 2;
    if (grouped) {
        q_base = row_base + (size_t)head * head_dim * 3;
        k_base = q_base + head_dim;
        v_base = k_base + head_dim;
    }
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = h3_bf16_to_f32(qkv[q_base + d]);
        float k = h3_bf16_to_f32(qkv[k_base + d]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]) * k_inverse *
               h3_bf16_to_f32(k_weight[dimension]);
    if (dimension < rope_half) {
        uint32_t pair = dimension + rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * rope_half + dimension]);
        float s = h3_bf16_to_f32(rope_sin[row * rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < rope_half * 2) {
        uint32_t pair = dimension - rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    size_t output_index = ((size_t)row * heads + head) * head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(q0);
    key[output_index] = h3_f32_to_bf16(k0);
    value[output_index] = qkv[v_base + dimension];
}

/* h3_vision_qkv_rope_bf16: no norms; RoPE pairs wrap across the whole head
 * (rope_half * 2 == head_dim is enforced host-side). */
__global__ void h3k_vision_qkv_rope_bf16(const uint16_t *qkv,
                                         const uint16_t *rope_cos,
                                         const uint16_t *rope_sin,
                                         uint16_t *query, uint16_t *key,
                                         uint16_t *value, uint32_t sequence,
                                         uint32_t heads, uint32_t head_dim,
                                         uint32_t rope_half) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint64_t)gid >= (uint64_t)head_dim * heads * sequence) return;
    uint32_t dimension = gid % head_dim;
    uint32_t head = (gid / head_dim) % heads;
    uint32_t row = gid / (head_dim * heads);
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim;
    size_t k_base = row_base + inner + (size_t)head * head_dim;
    size_t v_base = row_base + inner * 2 + (size_t)head * head_dim;
    uint32_t rope_index = row * rope_half + dimension % rope_half;
    float c = h3_bf16_to_f32(rope_cos[rope_index]);
    float s = h3_bf16_to_f32(rope_sin[rope_index]);
    uint32_t pair = dimension < rope_half ? dimension + rope_half
                                          : dimension - rope_half;
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]);
    float q1 = h3_bf16_to_f32(qkv[q_base + pair]);
    float k1 = h3_bf16_to_f32(qkv[k_base + pair]);
    float qr = dimension < rope_half ? q0 * c - q1 * s : q0 * c + q1 * s;
    float kr = dimension < rope_half ? k0 * c - k1 * s : k0 * c + k1 * s;
    size_t output_index = ((size_t)row * heads + head) * head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(qr);
    key[output_index] = h3_f32_to_bf16(kr);
    value[output_index] = qkv[v_base + dimension];
}

/* h3_text_qk_rope_bf16: separate Q/K inputs, weighted per-head RMS norms,
 * half-pair RoPE over the full head. */
__global__ void h3k_text_qk_rope_bf16(const uint16_t *query_input,
                                      const uint16_t *key_input,
                                      const uint16_t *q_weight,
                                      const uint16_t *k_weight,
                                      const uint16_t *rope_cos,
                                      const uint16_t *rope_sin,
                                      uint16_t *query_output,
                                      uint16_t *key_output, uint32_t sequence,
                                      uint32_t query_heads,
                                      uint32_t kv_heads, uint32_t head_dim,
                                      float epsilon) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint64_t)gid >= (uint64_t)head_dim * query_heads * sequence) return;
    uint32_t dimension = gid % head_dim;
    uint32_t head = (gid / head_dim) % query_heads;
    uint32_t row = gid / (head_dim * query_heads);
    uint32_t half_dim = head_dim / 2;
    uint32_t pair = dimension < half_dim ? dimension + half_dim
                                         : dimension - half_dim;
    float c = h3_bf16_to_f32(rope_cos[row * half_dim + dimension % half_dim]);
    float s = h3_bf16_to_f32(rope_sin[row * half_dim + dimension % half_dim]);
    size_t q_base = ((size_t)row * query_heads + head) * head_dim;
    float q_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float value = h3_bf16_to_f32(query_input[q_base + d]);
        q_sum = fmaf(value, value, q_sum);
    }
    float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
    float q0 = h3_bf16_to_f32(query_input[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float q1 = h3_bf16_to_f32(query_input[q_base + pair]) * q_inverse *
               h3_bf16_to_f32(q_weight[pair]);
    float q_rotated = dimension < half_dim ? q0 * c - q1 * s
                                           : q0 * c + q1 * s;
    query_output[q_base + dimension] = h3_f32_to_bf16(q_rotated);
    if (head < kv_heads) {
        size_t k_base = ((size_t)row * kv_heads + head) * head_dim;
        float k_sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float value = h3_bf16_to_f32(key_input[k_base + d]);
            k_sum = fmaf(value, value, k_sum);
        }
        float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
        float k0 = h3_bf16_to_f32(key_input[k_base + dimension]) * k_inverse *
                   h3_bf16_to_f32(k_weight[dimension]);
        float k1 = h3_bf16_to_f32(key_input[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float k_rotated = dimension < half_dim ? k0 * c - k1 * s
                                               : k0 * c + k1 * s;
        key_output[k_base + dimension] = h3_f32_to_bf16(k_rotated);
    }
}

/* h3_rope_text_bf16: in-place RoPE on existing Q/K; tables stay F32. */
__global__ void h3k_rope_text_bf16(uint16_t *query, uint16_t *key,
                                   const float *rope_cos,
                                   const float *rope_sin, uint32_t sequence,
                                   uint32_t query_heads, uint32_t kv_heads,
                                   uint32_t head_dim, uint32_t max_heads) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint64_t)gid >= (uint64_t)sequence * max_heads) return;
    uint32_t row = gid / max_heads;
    uint32_t head = gid % max_heads;
    uint32_t half_dim = head_dim / 2;
    if (head < query_heads) {
        size_t base = ((size_t)row * query_heads + head) * head_dim;
        for (uint32_t d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(query[base + d]);
            float second = h3_bf16_to_f32(query[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            query[base + d] = h3_f32_to_bf16(first * c - second * s);
            query[base + half_dim + d] =
                h3_f32_to_bf16(second * c + first * s);
        }
    }
    if (head < kv_heads) {
        size_t base = ((size_t)row * kv_heads + head) * head_dim;
        for (uint32_t d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(key[base + d]);
            float second = h3_bf16_to_f32(key[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            key[base + d] = h3_f32_to_bf16(first * c - second * s);
            key[base + half_dim + d] =
                h3_f32_to_bf16(second * c + first * s);
        }
    }
}

/* ------------------------------------------------------------ SDPA kernels */

/* Original naive path (H3_CUDA_SDPA_NAIVE=1 / fallback): replaces Metal's
 * MPSGraph SDPA with one block per (row, head, batch), scores cached in
 * dynamic shared memory, numerically stable softmax. Inputs are
 * [batch, row, head, dim]; output is the same, or [batch, head, row, dim]
 * when HEAD_MAJOR. The scale multiplies each QK dot. */
template <typename T, int CAUSAL, int HEAD_MAJOR>
__global__ void h3k_sdpa_naive(const T *query, const T *key, const T *value,
                               T *output, uint32_t sequence, uint32_t heads,
                               uint32_t head_dim, float scale) {
    extern __shared__ float h3_sdpa_scores[];
    __shared__ float reductions[128];
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    size_t batch_base = (size_t)blockIdx.z * sequence * heads * head_dim;
    size_t q_base = batch_base + ((size_t)row * heads + head) * head_dim;
    uint32_t key_count = CAUSAL ? row + 1 : sequence;
    float local_max = -INFINITY;
    for (uint32_t key_row = tid; key_row < key_count; key_row += threads) {
        size_t k_base = batch_base +
                        ((size_t)key_row * heads + head) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            dot = fmaf(h3_attn_load(query + q_base + d),
                       h3_attn_load(key + k_base + d), dot);
        dot *= scale;
        h3_sdpa_scores[key_row] = dot;
        local_max = fmaxf(local_max, dot);
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride)
            reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
        __syncthreads();
    }
    float maximum = reductions[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (uint32_t key_row = tid; key_row < key_count; key_row += threads) {
        float probability = expf(h3_sdpa_scores[key_row] - maximum);
        h3_sdpa_scores[key_row] = probability;
        local_sum += probability;
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse_sum = 1.0f / reductions[0];
    for (uint32_t d = tid; d < head_dim; d += threads) {
        float sum = 0.0f;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            size_t v_index = batch_base +
                             ((size_t)key_row * heads + head) * head_dim + d;
            sum = fmaf(h3_sdpa_scores[key_row] * inverse_sum,
                       h3_attn_load(value + v_index), sum);
        }
        size_t out_index =
            HEAD_MAJOR
                ? batch_base + ((size_t)head * sequence + row) * head_dim + d
                : q_base + d;
        h3_attn_store(output + out_index, sum);
    }
}

/* Naive h3_gqa_causal_bf16 (H3_CUDA_SDPA_NAIVE=1 fallback): one block per
 * (query row, query head). Matches MLX's ordering: Q is pre-scaled and
 * rounded to BF16 before the QK dots. */
__global__ void h3k_gqa_causal_bf16_naive(const uint16_t *query,
                                          const uint16_t *key,
                                          const uint16_t *value,
                                          uint16_t *output,
                                          uint32_t sequence,
                                          uint32_t query_heads,
                                          uint32_t kv_heads,
                                          uint32_t head_dim,
                                          float scale) {
    extern __shared__ float h3_gqa_scores[];
    __shared__ float reductions[128];
    __shared__ float shared_query[128];
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t query_row = blockIdx.x;
    uint32_t query_head = blockIdx.y;
    uint32_t kv_head = query_head / (query_heads / kv_heads);
    size_t q_base = ((size_t)query_row * query_heads + query_head) * head_dim;
    uint32_t key_count = query_row + 1;
    for (uint32_t d = tid; d < head_dim; d += threads)
        shared_query[d] = h3_bf16_to_f32(h3_f32_to_bf16(
            h3_bf16_to_f32(query[q_base + d]) * scale));
    __syncthreads();
    float local_max = -INFINITY;
    for (uint32_t key_row = tid; key_row < key_count; key_row += threads) {
        size_t k_base = ((size_t)key_row * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            dot = fmaf(shared_query[d], h3_bf16_to_f32(key[k_base + d]), dot);
        h3_gqa_scores[key_row] = dot;
        local_max = fmaxf(local_max, dot);
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride)
            reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
        __syncthreads();
    }
    float maximum = reductions[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (uint32_t key_row = tid; key_row < key_count; key_row += threads) {
        float probability = expf(h3_gqa_scores[key_row] - maximum);
        h3_gqa_scores[key_row] = probability;
        local_sum += probability;
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse_sum = 1.0f / reductions[0];
    for (uint32_t d = tid; d < head_dim; d += threads) {
        float sum = 0.0f;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            size_t v_index =
                ((size_t)key_row * kv_heads + kv_head) * head_dim + d;
            sum = fmaf(h3_gqa_scores[key_row] * inverse_sum,
                       h3_bf16_to_f32(value[v_index]), sum);
        }
        output[q_base + d] = h3_f32_to_bf16(sum);
    }
}

/* ----------------------------------------------------- flash SDPA kernels */

/* Tiled flash-attention-style SDPA: one block per (H3_FLASH_QROWS query
 * rows, head, batch); K/V stream through dynamic shared memory in KVT-row
 * tiles (64 for BF16, 32 for F32) with an online softmax (running max/sum
 * per query row) and F32 accumulators for O. Layouts and the scale
 * application point match h3k_sdpa_naive exactly (the scale multiplies each
 * QK dot); with PRESCALE_Q the Q tile is instead scaled and rounded to BF16
 * at load time and the per-dot scale is skipped, matching
 * h3k_gqa_causal_bf16_naive (kv_heads < heads selects the KV head group).
 * head_dim <= 128 is required; the host falls back to the naive kernels
 * otherwise. */
#define H3_FLASH_QROWS 16
#define H3_FLASH_THREADS 256
/* Each thread accumulates one contiguous dim chunk of one query row:
 * head_dim*QROWS/THREADS <= 8 accumulators per thread at head_dim <= 128. */
#define H3_FLASH_ACC (128 * H3_FLASH_QROWS / H3_FLASH_THREADS)

template <typename T, int CAUSAL, int HEAD_MAJOR, int PRESCALE_Q>
__global__ void h3k_sdpa_flash(const T *query, const T *key, const T *value,
                               T *output, uint32_t sequence, uint32_t heads,
                               uint32_t kv_heads, uint32_t head_dim,
                               float scale) {
    constexpr uint32_t KVT = sizeof(T) == 2 ? 64u : 32u;
    /* Pad the K/V row stride by one 4-byte bank so the score loop (threads
     * reading the same dim across different keys) does not hit 32-way shared
     * memory bank conflicts at head_dim == 128. */
    const uint32_t ld = head_dim + (sizeof(T) == 2 ? 2u : 1u);
    extern __shared__ float h3_flash_smem[];
    float *q_tile = h3_flash_smem;                      /* QROWS * head_dim */
    float *scores = q_tile + H3_FLASH_QROWS * head_dim; /* QROWS * KVT */
    float *m_run = scores + H3_FLASH_QROWS * KVT;       /* QROWS */
    float *l_run = m_run + H3_FLASH_QROWS;              /* QROWS */
    float *corr = l_run + H3_FLASH_QROWS;               /* QROWS */
    T *k_tile = (T *)(corr + H3_FLASH_QROWS);           /* KVT * ld */
    T *v_tile = k_tile + KVT * ld;                      /* KVT * ld */

    const uint32_t tid = threadIdx.x;
    const uint32_t q0 = blockIdx.x * H3_FLASH_QROWS;
    const uint32_t head = blockIdx.y;
    const uint32_t kv_head =
        kv_heads == heads ? head : head / (heads / kv_heads);
    const size_t q_batch = (size_t)blockIdx.z * sequence * heads * head_dim;
    const size_t kv_batch = (size_t)blockIdx.z * sequence * kv_heads * head_dim;

    /* Stage the query tile (zero-filled past sequence). */
    for (uint32_t idx = tid; idx < H3_FLASH_QROWS * head_dim;
         idx += H3_FLASH_THREADS) {
        uint32_t r = idx / head_dim, d = idx % head_dim;
        uint32_t row = q0 + r;
        float q = 0.0f;
        if (row < sequence)
            q = h3_attn_load(query + q_batch +
                             ((size_t)row * heads + head) * head_dim + d);
        if (PRESCALE_Q) q = h3_bf16_to_f32(h3_f32_to_bf16(q * scale));
        q_tile[idx] = q;
    }
    if (tid < H3_FLASH_QROWS) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.0f;
    }
    __syncthreads();

    /* Thread (tid) owns output row (tid % QROWS) and the contiguous dim
     * chunk [(tid/QROWS)*ACC, (tid/QROWS)*ACC + ACC). */
    const uint32_t o_row = tid % H3_FLASH_QROWS;
    const uint32_t o_base = (tid / H3_FLASH_QROWS) * H3_FLASH_ACC;
    float acc[H3_FLASH_ACC];
#pragma unroll
    for (int k = 0; k < H3_FLASH_ACC; k++) acc[k] = 0.0f;

    const uint32_t lane = tid & 31, warp = tid >> 5;
    const uint32_t key_limit =
        CAUSAL ? min(sequence, q0 + H3_FLASH_QROWS) : sequence;
    for (uint32_t kb = 0; kb < key_limit; kb += KVT) {
        /* Stage the K/V tiles (zero-filled past sequence; those columns get
         * p == 0 below and never contribute). Coalesced: adjacent threads
         * read adjacent key elements. */
        for (uint32_t idx = tid; idx < KVT * head_dim;
             idx += H3_FLASH_THREADS) {
            uint32_t j = idx / head_dim, d = idx % head_dim;
            T k0 = T(0), v0 = T(0);
            if (kb + j < sequence) {
                size_t base = kv_batch +
                              ((size_t)(kb + j) * kv_heads + kv_head) *
                                  head_dim + d;
                k0 = key[base];
                v0 = value[base];
            }
            k_tile[j * ld + d] = k0;
            v_tile[j * ld + d] = v0;
        }
        __syncthreads();
        /* Scores for this tile; masked keys stay -INFINITY. */
        for (uint32_t idx = tid; idx < H3_FLASH_QROWS * KVT;
             idx += H3_FLASH_THREADS) {
            uint32_t r = idx / KVT, j = idx % KVT;
            uint32_t krow = kb + j;
            float s = -INFINITY;
            if (krow < sequence && (!CAUSAL || krow <= q0 + r)) {
                float dot = 0.0f;
                const float *qp = q_tile + r * head_dim;
                if (sizeof(T) == 2 && !(head_dim & 1)) {
                    /* Paired 32-bit loads halve shared-memory transactions;
                     * ld stays even, so even dims are 4-byte aligned. */
                    const uint16_t *kp = (const uint16_t *)k_tile + j * ld;
                    for (uint32_t d = 0; d < head_dim; d += 2) {
                        uint32_t pair = *(const uint32_t *)(kp + d);
                        dot = fmaf(qp[d], h3_bf16_to_f32((uint16_t)pair),
                                   dot);
                        dot = fmaf(qp[d + 1],
                                   h3_bf16_to_f32((uint16_t)(pair >> 16)),
                                   dot);
                    }
                } else {
                    for (uint32_t d = 0; d < head_dim; d++)
                        dot = fmaf(qp[d],
                                   h3_attn_load(k_tile + j * ld + d), dot);
                }
                s = PRESCALE_Q ? dot : dot * scale;
            }
            scores[idx] = s;
        }
        __syncthreads();
        /* Per-row online-softmax update; warp w owns rows w, w+4, ... */
        for (uint32_t r = warp; r < H3_FLASH_QROWS;
             r += H3_FLASH_THREADS / 32) {
            float tile_max = -INFINITY;
            for (uint32_t j = lane; j < KVT; j += 32)
                tile_max = fmaxf(tile_max, scores[r * KVT + j]);
#pragma unroll
            for (int stride = 16; stride; stride >>= 1)
                tile_max = fmaxf(
                    tile_max, __shfl_xor_sync(0xffffffffu, tile_max, stride));
            float m_new = fmaxf(m_run[r], tile_max);
            /* m_new stays -INFINITY only when the row has no valid key yet;
             * its accumulators are still zero, so any finite factor works. */
            float c = m_new == -INFINITY ? 0.0f : expf(m_run[r] - m_new);
            float p_sum = 0.0f;
            for (uint32_t j = lane; j < KVT; j += 32) {
                float p = m_new == -INFINITY
                              ? 0.0f
                              : expf(scores[r * KVT + j] - m_new);
                scores[r * KVT + j] = p;
                p_sum += p;
            }
#pragma unroll
            for (int stride = 16; stride; stride >>= 1)
                p_sum += __shfl_xor_sync(0xffffffffu, p_sum, stride);
            if (lane == 0) {
                l_run[r] = l_run[r] * c + p_sum;
                m_run[r] = m_new;
                corr[r] = c;
            }
        }
        __syncthreads();
        /* Rescale O by this tile's correction, then accumulate P*V. */
        float c = corr[o_row];
#pragma unroll
        for (int k = 0; k < H3_FLASH_ACC; k++) acc[k] *= c;
        for (uint32_t j = 0; j < KVT; j++) {
            float p = scores[o_row * KVT + j];
            if (p == 0.0f) continue;
            const T *vp = v_tile + j * ld + o_base;
            if (sizeof(T) == 2 && !(head_dim & 1)) {
                const uint16_t *vb = (const uint16_t *)vp;
#pragma unroll
                for (int k = 0; k < H3_FLASH_ACC; k += 2) {
                    if (o_base + k < head_dim) {
                        uint32_t pair = *(const uint32_t *)(vb + k);
                        acc[k] = fmaf(p, h3_bf16_to_f32((uint16_t)pair),
                                      acc[k]);
                        acc[k + 1] =
                            fmaf(p, h3_bf16_to_f32((uint16_t)(pair >> 16)),
                                 acc[k + 1]);
                    }
                }
            } else {
#pragma unroll
                for (int k = 0; k < H3_FLASH_ACC; k++) {
                    uint32_t d = o_base + k;
                    if (d < head_dim)
                        acc[k] = fmaf(p, h3_attn_load(vp + k), acc[k]);
                }
            }
        }
        __syncthreads();
    }

    uint32_t row = q0 + o_row;
    if (row < sequence) {
        float inv = 1.0f / l_run[o_row];
#pragma unroll
        for (int k = 0; k < H3_FLASH_ACC; k++) {
            uint32_t d = o_base + k;
            if (d < head_dim) {
                size_t out_index =
                    HEAD_MAJOR
                        ? q_batch +
                              ((size_t)head * sequence + row) * head_dim + d
                        : q_batch + ((size_t)row * heads + head) * head_dim + d;
                h3_attn_store(output + out_index, acc[k] * inv);
            }
        }
    }
}

/* Dynamic shared-memory bytes for one h3k_sdpa_flash launch (includes the
 * one-bank K/V row padding). */
static size_t h3_flash_shared(uint32_t head_dim, int bf16) {
    uint32_t kvt = bf16 ? 64 : 32;
    uint32_t ld = head_dim + (bf16 ? 2 : 1);
    return (size_t)(H3_FLASH_QROWS * head_dim + H3_FLASH_QROWS * kvt +
                    3 * H3_FLASH_QROWS) * sizeof(float) +
           (size_t)2 * kvt * ld * (bf16 ? sizeof(uint16_t) : sizeof(float));
}

/* ------------------------------------------- tuned f32 flash SDPA kernel */

/* h3k_sdpa_flash_f32: f32-only variant of h3k_sdpa_flash with the same
 * skeleton (one block per QROWS query rows x head x batch, K/V stream
 * through shared memory in KVT-row tiles, identical online softmax, masking,
 * scale application point and output writeback), but with a register-
 * blocked, vectorized score phase:
 *  - each thread owns an RPT(=KVT/16) x 2 micro-tile of the scores, so every
 *    K row read from shared memory feeds RPT dots and every Q row feeds 2:
 *    the f32 kernel is shared-memory-bandwidth bound and this moves ~2 bytes
 *    of shared memory per FMA instead of ~4;
 *  - the QK dot and the P*V accumulate use float4 shared-memory loads when
 *    head_dim % 4 == 0; the K/V row pitch is then padded by 4 floats instead
 *    of 1, keeping rows 16-byte aligned and the across-key lane stride
 *    4-bank-interleaved (conflict-free for float4 phases);
 *  - Q/K/V staging from global memory uses float4 copies when head_dim % 4
 *    == 0 (coalesced, 16B-aligned: heads*head_dim keeps the row base
 *    aligned);
 *  - the dot accumulates the even/odd lanes of each float4 into two
 *    accumulators combined at the end (ILP). This, the KVT tile grouping
 *    and QROWS are summation-order/tiling changes only, well inside the
 *    1e-4 flash tolerance; the softmax math (running max, per-tile rescale,
 *    expf) is unchanged.
 * The host picks (QROWS, KVT) per shape from shared-memory fit and the
 * resident-blocks/SM count (opting in to >48KB dynamic shared memory on
 * archs that support it); H3_CUDA_SDPA_KVT / H3_CUDA_SDPA_QROWS force a
 * choice. kv_heads == heads and no prescaling: the f32 entry points never
 * use GQA or bf16 rounding. */
#define H3_FLASH_F32_THREADS 256

template <int CAUSAL, int KVT, int QROWS>
__global__ void h3k_sdpa_flash_f32(const float *__restrict__ query,
                                   const float *__restrict__ key,
                                   const float *__restrict__ value,
                                   float *__restrict__ output,
                                   uint32_t sequence, uint32_t heads,
                                   uint32_t head_dim, float scale) {
    constexpr uint32_t ACC = QROWS * 128 / H3_FLASH_F32_THREADS;
    const uint32_t vec = !(head_dim & 3u);
    /* 4-float padding keeps 16B alignment and conflict-free float4 lanes;
     * scalar path keeps the one-bank pad. */
    const uint32_t ld = head_dim + (vec ? 4u : 1u);
    extern __shared__ __align__(16) float h3_flash_f32_smem[];
    float *q_tile = h3_flash_f32_smem; /* QROWS * head_dim */
    float *scores = q_tile + QROWS * head_dim; /* QROWS * KVT */
    float *m_run = scores + QROWS * KVT;       /* QROWS */
    float *l_run = m_run + QROWS;              /* QROWS */
    float *corr = l_run + QROWS;               /* QROWS */
    float *k_tile = corr + QROWS;              /* KVT * ld */
    float *v_tile = k_tile + KVT * ld;                      /* KVT * ld */

    const uint32_t tid = threadIdx.x;
    const uint32_t q0 = blockIdx.x * QROWS;
    const uint32_t head = blockIdx.y;
    const size_t q_batch = (size_t)blockIdx.z * sequence * heads * head_dim;

    /* Stage the query tile (zero-filled past sequence). */
    if (vec) {
        uint32_t hd4 = head_dim / 4;
        for (uint32_t idx = tid; idx < QROWS * hd4;
             idx += H3_FLASH_F32_THREADS) {
            uint32_t r = idx / hd4, d = (idx % hd4) * 4;
            uint32_t row = q0 + r;
            float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < sequence)
                qv = *(const float4 *)(query + q_batch +
                                       ((size_t)row * heads + head) *
                                           head_dim + d);
            *(float4 *)(q_tile + r * head_dim + d) = qv;
        }
    } else {
        for (uint32_t idx = tid; idx < QROWS * head_dim;
             idx += H3_FLASH_F32_THREADS) {
            uint32_t r = idx / head_dim, d = idx % head_dim;
            uint32_t row = q0 + r;
            q_tile[idx] =
                row < sequence
                    ? query[q_batch +
                            ((size_t)row * heads + head) * head_dim + d]
                    : 0.0f;
        }
    }
    if (tid < QROWS) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.0f;
    }
    __syncthreads();

    /* Output ownership identical to h3k_sdpa_flash: thread tid owns row
     * (tid % QROWS) and the dim chunk [(tid/QROWS)*ACC, ...+ACC). */
    const uint32_t o_row = tid % QROWS;
    const uint32_t o_base = (tid / QROWS) * ACC;
    float acc[ACC];
#pragma unroll
    for (int k = 0; k < ACC; k++) acc[k] = 0.0f;

    /* Score micro-tile: RPT x 2 per thread, rows [RPT*rp, RPT*rp+RPT), keys
     * [2*kp, 2*kp+2). KVT=64 uses RPT=4 on the first 128 threads, KVT=32
     * RPT=2 on the first 128 (the phase is shared-memory-bound; the K-row
     * reuse of the taller micro-tile wins over spreading over more warps).
     * A warp covers one row group x all KVT keys, so its Q-row float4 loads
     * are warp-wide broadcasts. */
    constexpr uint32_t RPT = KVT / 16; /* 4 at KVT=64, 2 at KVT=32 */
    const uint32_t rp = tid / (KVT / 2);
    const uint32_t kp = tid % (KVT / 2);
    const uint32_t lane = tid & 31, warp = tid >> 5;
    const uint32_t key_limit =
        CAUSAL ? min(sequence, q0 + QROWS) : sequence;
    for (uint32_t kb = 0; kb < key_limit; kb += KVT) {
        /* Stage the K/V tiles (zero-filled past sequence; those columns get
         * p == 0 below and never contribute). Coalesced gmem reads. */
        if (vec) {
            uint32_t hd4 = head_dim / 4;
            for (uint32_t idx = tid; idx < (uint32_t)KVT * hd4;
                 idx += H3_FLASH_F32_THREADS) {
                uint32_t j = idx / hd4, d = (idx % hd4) * 4;
                float4 k0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), v0 = k0;
                if (kb + j < sequence) {
                    size_t base = q_batch + /* kv_heads == heads */
                                  ((size_t)(kb + j) * heads + head) *
                                      head_dim + d;
                    k0 = *(const float4 *)(key + base);
                    v0 = *(const float4 *)(value + base);
                }
                *(float4 *)(k_tile + j * ld + d) = k0;
                *(float4 *)(v_tile + j * ld + d) = v0;
            }
        } else {
            for (uint32_t idx = tid; idx < (uint32_t)KVT * head_dim;
                 idx += H3_FLASH_F32_THREADS) {
                uint32_t j = idx / head_dim, d = idx % head_dim;
                float k0 = 0.0f, v0 = 0.0f;
                if (kb + j < sequence) {
                    size_t base = q_batch + /* kv_heads == heads */
                                  ((size_t)(kb + j) * heads + head) *
                                      head_dim + d;
                    k0 = key[base];
                    v0 = value[base];
                }
                k_tile[j * ld + d] = k0;
                v_tile[j * ld + d] = v0;
            }
        }
        __syncthreads();
        /* Scores for this tile; masked keys stay -INFINITY. Active threads:
         * (QROWS/RPT) row groups x (KVT/2) key pairs = 8*QROWS. */
        if (tid < 8 * QROWS) {
            float dot[RPT][2][2]; /* [row][key][even/odd float4 lanes] */
#pragma unroll
            for (uint32_t i = 0; i < RPT; i++) {
#pragma unroll
                for (int j = 0; j < 2; j++)
                    dot[i][j][0] = dot[i][j][1] = 0.0f;
            }
            const float *qp = q_tile + RPT * rp * head_dim;
            const float *kp0 = k_tile + 2 * kp * ld;
            if (vec) {
                for (uint32_t d = 0; d < head_dim; d += 4) {
                    float4 qv[RPT];
#pragma unroll
                    for (uint32_t i = 0; i < RPT; i++)
                        qv[i] = *(const float4 *)(qp + i * head_dim + d);
                    float4 kv0 = *(const float4 *)(kp0 + d);
                    float4 kv1 = *(const float4 *)(kp0 + ld + d);
#pragma unroll
                    for (uint32_t i = 0; i < RPT; i++) {
                        dot[i][0][0] = fmaf(qv[i].x, kv0.x, dot[i][0][0]);
                        dot[i][0][1] = fmaf(qv[i].y, kv0.y, dot[i][0][1]);
                        dot[i][0][0] = fmaf(qv[i].z, kv0.z, dot[i][0][0]);
                        dot[i][0][1] = fmaf(qv[i].w, kv0.w, dot[i][0][1]);
                        dot[i][1][0] = fmaf(qv[i].x, kv1.x, dot[i][1][0]);
                        dot[i][1][1] = fmaf(qv[i].y, kv1.y, dot[i][1][1]);
                        dot[i][1][0] = fmaf(qv[i].z, kv1.z, dot[i][1][0]);
                        dot[i][1][1] = fmaf(qv[i].w, kv1.w, dot[i][1][1]);
                    }
                }
            } else {
                for (uint32_t d = 0; d < head_dim; d++) {
                    float qv[RPT];
#pragma unroll
                    for (uint32_t i = 0; i < RPT; i++)
                        qv[i] = qp[i * head_dim + d];
                    float kv0 = kp0[d], kv1 = kp0[ld + d];
#pragma unroll
                    for (uint32_t i = 0; i < RPT; i++) {
                        dot[i][0][0] = fmaf(qv[i], kv0, dot[i][0][0]);
                        dot[i][1][0] = fmaf(qv[i], kv1, dot[i][1][0]);
                    }
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < RPT; i++) {
                uint32_t r = RPT * rp + i;
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    uint32_t krow = kb + 2 * kp + (uint32_t)j;
                    float s = -INFINITY;
                    if (krow < sequence && (!CAUSAL || krow <= q0 + r))
                        s = (dot[i][j][0] + dot[i][j][1]) * scale;
                    scores[r * KVT + 2 * kp + (uint32_t)j] = s;
                }
            }
        }
        __syncthreads();
        /* Per-row online-softmax update, identical math to h3k_sdpa_flash;
         * warp w owns rows w, w+8, ... */
        for (uint32_t r = warp; r < QROWS;
             r += H3_FLASH_F32_THREADS / 32) {
            float tile_max = -INFINITY;
            for (uint32_t j = lane; j < (uint32_t)KVT; j += 32)
                tile_max = fmaxf(tile_max, scores[r * KVT + j]);
#pragma unroll
            for (int stride = 16; stride; stride >>= 1)
                tile_max = fmaxf(
                    tile_max, __shfl_xor_sync(0xffffffffu, tile_max, stride));
            float m_new = fmaxf(m_run[r], tile_max);
            float c = m_new == -INFINITY ? 0.0f : expf(m_run[r] - m_new);
            float p_sum = 0.0f;
            for (uint32_t j = lane; j < (uint32_t)KVT; j += 32) {
                float p = m_new == -INFINITY
                              ? 0.0f
                              : expf(scores[r * KVT + j] - m_new);
                scores[r * KVT + j] = p;
                p_sum += p;
            }
#pragma unroll
            for (int stride = 16; stride; stride >>= 1)
                p_sum += __shfl_xor_sync(0xffffffffu, p_sum, stride);
            if (lane == 0) {
                l_run[r] = l_run[r] * c + p_sum;
                m_run[r] = m_new;
                corr[r] = c;
            }
        }
        __syncthreads();
        /* Rescale O by this tile's correction, then accumulate P*V (serial
         * per-accumulator key order, as in h3k_sdpa_flash). */
        float c = corr[o_row];
#pragma unroll
        for (int k = 0; k < ACC; k++) acc[k] *= c;
        for (uint32_t j = 0; j < (uint32_t)KVT; j++) {
            float p = scores[o_row * KVT + j];
            if (p == 0.0f) continue;
            const float *vp = v_tile + j * ld + o_base;
            if (vec) {
                /* o_base + k is a multiple of 4 and head_dim is too, so an
                 * in-range chunk start means a full float4 is in range. */
#pragma unroll
                for (int k = 0; k < ACC; k += 4) {
                    if (o_base + k < head_dim) {
                        float4 vv = *(const float4 *)(vp + k);
                        acc[k] = fmaf(p, vv.x, acc[k]);
                        acc[k + 1] = fmaf(p, vv.y, acc[k + 1]);
                        acc[k + 2] = fmaf(p, vv.z, acc[k + 2]);
                        acc[k + 3] = fmaf(p, vv.w, acc[k + 3]);
                    }
                }
            } else {
#pragma unroll
                for (int k = 0; k < ACC; k++) {
                    uint32_t d = o_base + k;
                    if (d < head_dim) acc[k] = fmaf(p, vp[k], acc[k]);
                }
            }
        }
        __syncthreads();
    }

    uint32_t row = q0 + o_row;
    if (row < sequence) {
        float inv = 1.0f / l_run[o_row];
#pragma unroll
        for (int k = 0; k < ACC; k++) {
            uint32_t d = o_base + k;
            if (d < head_dim)
                output[q_batch + ((size_t)row * heads + head) * head_dim +
                       d] = acc[k] * inv;
        }
    }
}

/* Dynamic shared-memory bytes for one h3k_sdpa_flash_f32 launch. */
static size_t h3_flash_shared_f32(uint32_t head_dim, int kvt, int qrows) {
    uint32_t ld = head_dim + ((head_dim & 3) ? 1 : 4);
    return (size_t)((uint32_t)qrows * head_dim +
                    (uint32_t)qrows * (uint32_t)kvt + 3 * (uint32_t)qrows) *
               sizeof(float) +
           (size_t)2 * (uint32_t)kvt * ld * sizeof(float);
}

/* One-time opt-in to >48KB dynamic shared memory for a kernel
 * instantiation; 1 on success/already-done, 0 if unsupported. */
template <int CAUSAL, int KVT, int QROWS>
static int h3_f32_flash_attr(size_t shared) {
    static int state = 0; /* 0 unknown, 1 ok, -1 failed */
    if (!state)
        state = cudaFuncSetAttribute(
                    (const void *)h3k_sdpa_flash_f32<CAUSAL, KVT, QROWS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)shared) == cudaSuccess
                    ? 1
                    : -1;
    return state > 0;
}

/* Does the (KVT, QROWS) f32 config fit this device (with the >48KB opt-in
 * where needed)? CAUSAL selects the instantiation for the attribute. */
template <int CAUSAL>
static int h3_f32_flash_fits(h3_gpu *gpu, uint32_t head_dim, int kvt,
                             int qrows) {
    size_t shared = h3_flash_shared_f32(head_dim, kvt, qrows);
    if (shared <= (size_t)h3_attn_max_shared(gpu)) return 1;
    int optin = 0;
    if (cudaDeviceGetAttribute(&optin,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               gpu->device) != cudaSuccess ||
        (size_t)optin < shared)
        return 0;
    if (kvt == 64 && qrows == 32) return h3_f32_flash_attr<CAUSAL, 64, 32>(shared);
    if (kvt == 64) return h3_f32_flash_attr<CAUSAL, 64, 16>(shared);
    if (qrows == 32) return h3_f32_flash_attr<CAUSAL, 32, 32>(shared);
    return h3_f32_flash_attr<CAUSAL, 32, 16>(shared);
}

/* ------------------------------------------------- tensor-core SDPA (bf16) */

/* h3k_sdpa_mma_bf16: tensor-core flash SDPA for the DiT shape (bf16,
 * head_dim == 128, non-causal, kv_heads == heads). Same tiling skeleton as
 * h3k_sdpa_flash — one block per (16 query rows, head, batch), K/V stream
 * through shared memory in 64-row tiles, online softmax in f32 — but the
 * inner dot loops are mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
 * 4 warps (128 threads): for S = Q*K^T each warp owns a 16x16 slice of the
 * KV tile (two m16n8 tiles, K loop over the 128 dims); for O += P*V each
 * warp owns a 16x32 slice of the output (four m16n8 tiles, K loop over the
 * 64 key rows).
 *
 * Fragment and staging notes (tuned path): every mma operand that is a raw
 * copy out of a shared-memory tile loads via ldmatrix.sync.aligned.m8n8.x4
 * — the Q A-fragments and the K B-fragments of QK^T (K smem rows double as
 * the col-major B operand), and the V B-fragments of P*V via the .trans
 * variant (V is stored row-major but the fragment wants same-column pairs
 * of adjacent key rows). The 136-bf16 row pitch is 17 16-byte units (odd),
 * so each 8-row ldmatrix phase lands on distinct banks. K/V tiles stage
 * with cp.async 16-byte copies (one async copy per 8 bf16; out-of-range
 * rows zero-fill through the src-size operand) and Q stages with vectorized
 * 16-byte copies, replacing the earlier scalar per-element loads. The P
 * A-fragments still pack by hand from the f32 scores tile — that is the
 * bf16 RNE rounding boundary below — but the scores row pitch is padded to
 * 65 floats: at pitch 64 the same-column threads of the eight fragment rows
 * shared a bank (8-way conflicts) in both the score stores and the pack
 * loads.
 *
 * EXTRA ROUNDING BOUNDARY: the P operand of P*V is the f32 softmax
 * probabilities rounded to bf16 (RNE) before the mma — the one rounding the
 * scalar flash kernel does not have. P is in [0,1], so each element carries
 * <= 2^-9 relative error and the f32-accumulated result stays well inside
 * the 1e-2-relative test tolerance. Scores and O accumulate in f32; the
 * output is bf16 RNE via the usual helpers.
 *
 * Requires sm_80+: the body is compiled out below that (and on sm_61 the
 * host wrapper never selects it — it checks the runtime compute capability,
 * and the Makefile builds for the native arch). */
#define H3_MMA_THREADS 128
#define H3_MMA_QROWS 16
#define H3_MMA_KVT 64
#define H3_MMA_HD 128
/* K/V/Q row pitch: 128 dims + 8 bf16 (16 bytes) of padding. */
#define H3_MMA_LD 136
/* Scores row pitch in floats: 64 keys + 1 float of padding against bank
 * conflicts in the fragment-pattern accesses (see above). */
#define H3_MMA_SLD 65

/* Pack two f32 values as an RNE bf16 pair (lo in the low half). */
__device__ __forceinline__ uint32_t h3_pack_bf16(float lo, float hi) {
    return (uint32_t)h3_f32_to_bf16(lo) | ((uint32_t)h3_f32_to_bf16(hi) << 16);
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void h3_mma_bf16_16x8x16(float *c, uint32_t a0,
                                                    uint32_t a1, uint32_t a2,
                                                    uint32_t a3, uint32_t b0,
                                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* ldmatrix.m8n8.x4: loads four 8x8 bf16 tiles; thread `lane` supplies the
 * address of row (lane % 8) of tile (lane / 8) and receives two adjacent
 * bf16 of row (lane / 4), columns 2*(lane % 4), of each tile in order —
 * exactly the m16n8k16 A-fragment order, and with K rows as tiles also the
 * B-fragment order of QK^T. */
__device__ __forceinline__ void h3_ldmatrix_x4(uint32_t r[4],
                                               const uint16_t *addr) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

/* .trans variant: same addressing, but the fragment is transposed, giving
 * same-column pairs of adjacent rows — the B-fragment order P*V needs from
 * the row-major V tile. */
__device__ __forceinline__ void h3_ldmatrix_x4_trans(uint32_t r[4],
                                                     const uint16_t *addr) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, "
        "[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

/* cp.async 16-byte global->shared copy; src_size 0 zero-fills instead of
 * reading (out-of-range rows), so staged tiles keep the old zero-fill
 * contract. */
__device__ __forceinline__ void h3_cp_async_16(uint16_t *smem,
                                               const uint16_t *gmem,
                                               int src_size) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
                 "l"(gmem), "r"(src_size));
}

__device__ __forceinline__ void h3_cp_async_wait_all(void) {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

template <int HEAD_MAJOR>
__global__ void h3k_sdpa_mma_bf16(const uint16_t *query, const uint16_t *key,
                                  const uint16_t *value, uint16_t *output,
                                  uint32_t sequence, uint32_t heads,
                                  float scale) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ float h3_mma_smem[];
    uint16_t *q_tile = (uint16_t *)h3_mma_smem;           /* QROWS x LD */
    uint16_t *k_tile = q_tile + H3_MMA_QROWS * H3_MMA_LD; /* KVT x LD */
    uint16_t *v_tile = k_tile + H3_MMA_KVT * H3_MMA_LD;   /* KVT x LD */
    float *scores = (float *)(v_tile + H3_MMA_KVT * H3_MMA_LD); /* 16 x SLD */
    float *m_run = scores + H3_MMA_QROWS * H3_MMA_SLD;    /* QROWS */
    float *l_run = m_run + H3_MMA_QROWS;                  /* QROWS */
    float *corr = l_run + H3_MMA_QROWS;                   /* QROWS */

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31, warp = tid >> 5;
    const uint32_t q0 = blockIdx.x * H3_MMA_QROWS;
    const uint32_t head = blockIdx.y;
    const size_t q_batch = (size_t)blockIdx.z * sequence * heads * H3_MMA_HD;

    /* Stage the query tile as bf16 (zero-filled past sequence) in 16-byte
     * chunks: 8 bf16 per copy, rows start 256-byte aligned in global memory
     * and the padded pitch keeps smem rows 16-byte aligned. */
    for (uint32_t c = tid; c < H3_MMA_QROWS * (H3_MMA_HD / 8);
         c += H3_MMA_THREADS) {
        uint32_t r = c >> 4, d = (c & 15) * 8;
        uint32_t row = q0 + r;
        uint4 q = {0u, 0u, 0u, 0u};
        if (row < sequence)
            q = *(const uint4 *)(query + q_batch +
                                 ((size_t)row * heads + head) * H3_MMA_HD +
                                 d);
        *(uint4 *)(q_tile + r * H3_MMA_LD + d) = q;
    }
    if (tid < H3_MMA_QROWS) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.0f;
    }
    __syncthreads();

    /* m16n8 fragment indices held by this thread: rows a_row / a_row + 8,
     * columns a_col / a_col + 1 within each n-tile. */
    const uint32_t a_row = lane >> 2, a_col = (lane & 3) * 2;

    /* O fragments: warp w owns dims [w*32, w*32+32) as four m16n8 tiles. */
    float o_acc[4][4];
#pragma unroll
    for (int t = 0; t < 4; t++)
#pragma unroll
        for (int k = 0; k < 4; k++) o_acc[t][k] = 0.0f;

    for (uint32_t kb = 0; kb < sequence; kb += H3_MMA_KVT) {
        /* Stage the K/V tiles with cp.async 16-byte copies (zero-filled
         * past sequence via src_size; those columns are masked to -INFINITY
         * below and never contribute). Coalesced: adjacent threads read
         * adjacent 16-byte key chunks. */
        for (uint32_t c = tid; c < H3_MMA_KVT * (H3_MMA_HD / 8);
             c += H3_MMA_THREADS) {
            uint32_t j = c >> 4, d = (c & 15) * 8;
            size_t base = q_batch +
                          ((size_t)(kb + j) * heads + head) * H3_MMA_HD + d;
            int live = kb + j < sequence ? 16 : 0;
            h3_cp_async_16(k_tile + j * H3_MMA_LD + d, key + base, live);
            h3_cp_async_16(v_tile + j * H3_MMA_LD + d, value + base, live);
        }
        h3_cp_async_wait_all();
        __syncthreads();

        /* S = Q*K^T for this tile: warp w covers key columns [w*16, w*16+16)
         * as two m16n8 tiles; the K loop runs over 8 m16k16 steps. A (Q) and
         * B (K) fragments load via one ldmatrix.x4 each per step: thread
         * lane addresses row (lane % 8) of tile (lane / 8), where the tiles
         * are {rows 0-7, 8-15} x {k, k+8} for A and {keys n, n+8} x {k, k+8}
         * for B — exactly the fragment registers in mma order. */
        float s_acc[2][4];
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
            for (int k = 0; k < 4; k++) s_acc[t][k] = 0.0f;
        const uint16_t *qa = q_tile + (lane & 15) * H3_MMA_LD +
                             ((lane & 16) ? 8 : 0);
        const uint16_t *kbp = k_tile +
                              (warp * 16 + (lane & 7) +
                               ((lane & 16) ? 8 : 0)) *
                                  H3_MMA_LD +
                              ((lane & 8) ? 8 : 0);
#pragma unroll
        for (int ks = 0; ks < H3_MMA_HD / 16; ks++) {
            uint32_t a[4], b[4];
            h3_ldmatrix_x4(a, qa + ks * 16);
            h3_ldmatrix_x4(b, kbp + ks * 16);
            h3_mma_bf16_16x8x16(s_acc[0], a[0], a[1], a[2], a[3], b[0], b[1]);
            h3_mma_bf16_16x8x16(s_acc[1], a[0], a[1], a[2], a[3], b[2], b[3]);
        }
        /* Scale (matching h3k_sdpa_flash: the scale multiplies each QK dot),
         * mask keys past the sequence end, publish to shared memory. */
#pragma unroll
        for (int t = 0; t < 2; t++) {
            uint32_t j = warp * 16 + t * 8 + a_col;
            float *row0 = scores + a_row * H3_MMA_SLD;
            float *row1 = scores + (a_row + 8) * H3_MMA_SLD;
            row0[j] = kb + j < sequence ? s_acc[t][0] * scale : -INFINITY;
            row0[j + 1] =
                kb + j + 1 < sequence ? s_acc[t][1] * scale : -INFINITY;
            row1[j] = kb + j < sequence ? s_acc[t][2] * scale : -INFINITY;
            row1[j + 1] =
                kb + j + 1 < sequence ? s_acc[t][3] * scale : -INFINITY;
        }
        __syncthreads();

        /* Per-row online-softmax update, identical math to h3k_sdpa_flash;
         * warp w owns rows w, w+4, w+8, w+12. */
        for (uint32_t r = warp; r < H3_MMA_QROWS; r += H3_MMA_THREADS / 32) {
            float tile_max = -INFINITY;
            for (uint32_t j = lane; j < H3_MMA_KVT; j += 32)
                tile_max = fmaxf(tile_max, scores[r * H3_MMA_SLD + j]);
#pragma unroll
            for (int stride = 16; stride; stride >>= 1)
                tile_max = fmaxf(
                    tile_max, __shfl_xor_sync(0xffffffffu, tile_max, stride));
            float m_new = fmaxf(m_run[r], tile_max);
            /* m_new stays -INFINITY only when the row has no valid key yet;
             * its accumulators are still zero, so any finite factor works. */
            float c = m_new == -INFINITY ? 0.0f : expf(m_run[r] - m_new);
            float p_sum = 0.0f;
            for (uint32_t j = lane; j < H3_MMA_KVT; j += 32) {
                float p = m_new == -INFINITY
                              ? 0.0f
                              : expf(scores[r * H3_MMA_SLD + j] - m_new);
                scores[r * H3_MMA_SLD + j] = p;
                p_sum += p;
            }
#pragma unroll
            for (int stride = 16; stride; stride >>= 1)
                p_sum += __shfl_xor_sync(0xffffffffu, p_sum, stride);
            if (lane == 0) {
                l_run[r] = l_run[r] * c + p_sum;
                m_run[r] = m_new;
                corr[r] = c;
            }
        }
        __syncthreads();

        /* Rescale O by this tile's correction, then accumulate P*V. The P
         * fragments are the f32 probabilities rounded to bf16 (RNE) right
         * here — the extra rounding boundary documented above. The V
         * B-fragments load via ldmatrix.x4.trans: thread lane addresses key
         * row (kc + lane % 8 + (lane & 8)) of n-tile pair (lane / 16), and
         * the transposed fragment delivers the same-column key-row pairs. */
        float c_lo = corr[a_row], c_hi = corr[a_row + 8];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            o_acc[t][0] *= c_lo;
            o_acc[t][1] *= c_lo;
            o_acc[t][2] *= c_hi;
            o_acc[t][3] *= c_hi;
        }
        const uint16_t *vp = v_tile +
                             ((lane & 7) + ((lane & 8) ? 8 : 0)) *
                                 H3_MMA_LD +
                             warp * 32 + ((lane & 16) ? 8 : 0);
#pragma unroll
        for (int ks = 0; ks < H3_MMA_KVT / 16; ks++) {
            uint32_t kc = ks * 16 + a_col;
            uint32_t a0 = h3_pack_bf16(scores[a_row * H3_MMA_SLD + kc],
                                       scores[a_row * H3_MMA_SLD + kc + 1]);
            uint32_t a1 =
                h3_pack_bf16(scores[(a_row + 8) * H3_MMA_SLD + kc],
                             scores[(a_row + 8) * H3_MMA_SLD + kc + 1]);
            uint32_t a2 = h3_pack_bf16(scores[a_row * H3_MMA_SLD + kc + 8],
                                       scores[a_row * H3_MMA_SLD + kc + 9]);
            uint32_t a3 =
                h3_pack_bf16(scores[(a_row + 8) * H3_MMA_SLD + kc + 8],
                             scores[(a_row + 8) * H3_MMA_SLD + kc + 9]);
            uint32_t b01[4], b23[4];
            h3_ldmatrix_x4_trans(b01, vp + ks * 16 * H3_MMA_LD);
            h3_ldmatrix_x4_trans(b23, vp + ks * 16 * H3_MMA_LD + 16);
            h3_mma_bf16_16x8x16(o_acc[0], a0, a1, a2, a3, b01[0], b01[1]);
            h3_mma_bf16_16x8x16(o_acc[1], a0, a1, a2, a3, b01[2], b01[3]);
            h3_mma_bf16_16x8x16(o_acc[2], a0, a1, a2, a3, b23[0], b23[1]);
            h3_mma_bf16_16x8x16(o_acc[3], a0, a1, a2, a3, b23[2], b23[3]);
        }
        __syncthreads();
    }

    /* Write O (f32 accumulators / l_run, bf16 RNE). */
    uint32_t row0 = q0 + a_row, row1 = row0 + 8;
    if (row0 < sequence) {
        float inv = 1.0f / l_run[a_row];
        size_t base =
            HEAD_MAJOR
                ? q_batch + ((size_t)head * sequence + row0) * H3_MMA_HD
                : q_batch + ((size_t)row0 * heads + head) * H3_MMA_HD;
#pragma unroll
        for (int t = 0; t < 4; t++) {
            uint32_t d = warp * 32 + t * 8 + a_col;
            h3_attn_store(output + base + d, o_acc[t][0] * inv);
            h3_attn_store(output + base + d + 1, o_acc[t][1] * inv);
        }
    }
    if (row1 < sequence) {
        float inv = 1.0f / l_run[a_row + 8];
        size_t base =
            HEAD_MAJOR
                ? q_batch + ((size_t)head * sequence + row1) * H3_MMA_HD
                : q_batch + ((size_t)row1 * heads + head) * H3_MMA_HD;
#pragma unroll
        for (int t = 0; t < 4; t++) {
            uint32_t d = warp * 32 + t * 8 + a_col;
            h3_attn_store(output + base + d, o_acc[t][2] * inv);
            h3_attn_store(output + base + d + 1, o_acc[t][3] * inv);
        }
    }
#endif
}

/* Dynamic shared-memory bytes for one h3k_sdpa_mma_bf16 launch. */
static size_t h3_mma_shared(void) {
    return (size_t)(H3_MMA_QROWS + 2 * H3_MMA_KVT) * H3_MMA_LD *
               sizeof(uint16_t) +
           (size_t)(H3_MMA_QROWS * H3_MMA_SLD + 3 * H3_MMA_QROWS) *
               sizeof(float);
}

/* Runtime gate for the tensor-core path: bf16 mma needs sm_80+. Cached per
 * device; paired with the __CUDA_ARCH__ >= 800 guard on the kernel body (the
 * Makefile builds for the native arch, so cap >= 8 implies the mma code is
 * in the binary). */
static int h3_attn_has_bf16_mma(struct h3_gpu *gpu) {
    static int cached_device = -1;
    static int cached_result = 0;
    if (cached_device != gpu->device) {
        int major = 0;
        cached_result =
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   gpu->device) == cudaSuccess &&
            major >= 8;
        cached_device = gpu->device;
    }
    return cached_result;
}

/* ----------------------------------------------------------- audio kernels */

__global__ void h3k_audio_qkv_split_f32(const float *qkv, const float *q_bias,
                                        const float *k_bias,
                                        const float *v_bias, float *query,
                                        float *key, float *value,
                                        uint32_t batch, uint32_t length,
                                        uint32_t heads, uint32_t head_dim) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (gid >= count) return;
    uint32_t column = (uint32_t)(gid % width);
    size_t base = (size_t)(gid / width) * width * 3;
    query[gid] = qkv[base + column] + q_bias[column];
    key[gid] = qkv[base + width + column] + k_bias[column];
    value[gid] = qkv[base + width * 2 + column] + v_bias[column];
}

__global__ void h3k_audio_attention_pool_f32(const float *attended,
                                             float *output, uint32_t batch,
                                             uint32_t length, uint32_t heads,
                                             uint32_t head_dim,
                                             uint32_t output_dim) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)batch * length * output_dim;
    if (gid >= count) return;
    uint32_t column = (uint32_t)(gid % output_dim);
    size_t row = gid / output_dim;
    uint32_t pool = head_dim / output_dim;
    float sum = 0.0f;
    for (uint32_t head = 0; head < heads; head++) {
        size_t base = (row * heads + head) * head_dim + (size_t)column * pool;
        for (uint32_t item = 0; item < pool; item++)
            sum += attended[base + item];
    }
    output[gid] = sum / (float)(heads * pool);
}

/* ---------------------------------------------------------- host wrappers */

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_attn_require_f32(gpu, qkv, count * 3, "QKV input") ||
        !h3_attn_require_f32(gpu, q_norm, head_dim, "Q norm") ||
        !h3_attn_require_f32(gpu, k_norm, head_dim, "K norm") ||
        !h3_attn_require_f32(gpu, rope_cos, rope_count, "RoPE cosine") ||
        !h3_attn_require_f32(gpu, rope_sin, rope_count, "RoPE sine") ||
        !h3_attn_require_f32(gpu, query, count, "query") ||
        !h3_attn_require_f32(gpu, key, count, "key") ||
        !h3_attn_require_f32(gpu, value, count, "value") ||
        rope_half * 2 > head_dim ||
        !h3_cuda_require_command(gpu)) return 0;
    if (count)
        h3k_qkv_rope_f32<<<h3_attn_grid_1d(count), 256>>>(
            (const float *)qkv->data, (const float *)q_norm->data,
            (const float *)k_norm->data, (const float *)rope_cos->data,
            (const float *)rope_sin->data, (float *)query->data,
            (float *)key->data, (float *)value->data, sequence, heads,
            head_dim, rope_half, epsilon);
    return h3_cuda_launch_check(gpu, "h3_qkv_rope_f32");
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_attn_require_f32(gpu, qkv, count * 3, "video QKV") ||
        !h3_attn_require_f32(gpu, rope_cos, rope_count, "video RoPE cosine") ||
        !h3_attn_require_f32(gpu, rope_sin, rope_count, "video RoPE sine") ||
        !h3_attn_require_f32(gpu, query, count, "video query") ||
        !h3_attn_require_f32(gpu, key, count, "video key") ||
        !h3_attn_require_f32(gpu, value, count, "video value") ||
        rope_half * 2 > head_dim ||
        !h3_cuda_require_command(gpu)) return 0;
    if (count)
        h3k_video_qkv_rope_f32<<<h3_attn_grid_1d(count), 256>>>(
            (const float *)qkv->data, (const float *)rope_cos->data,
            (const float *)rope_sin->data, (float *)query->data,
            (float *)key->data, (float *)value->data, sequence, heads,
            head_dim, rope_half, epsilon);
    return h3_cuda_launch_check(gpu, "h3_video_qkv_rope_f32");
}

static int h3_cuda_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 uint32_t grouped, float epsilon) {
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_attn_require_bf16(gpu, qkv, count * 3, "QKV input") ||
        !h3_attn_require_bf16(gpu, q_norm, head_dim, "Q norm") ||
        !h3_attn_require_bf16(gpu, k_norm, head_dim, "K norm") ||
        !h3_attn_require_bf16(gpu, rope_cos, rope_count, "RoPE cosine") ||
        !h3_attn_require_bf16(gpu, rope_sin, rope_count, "RoPE sine") ||
        !h3_attn_require_bf16(gpu, query, count, "query") ||
        !h3_attn_require_bf16(gpu, key, count, "key") ||
        !h3_attn_require_bf16(gpu, value, count, "value") ||
        rope_half * 2 > head_dim ||
        !h3_cuda_require_command(gpu)) return 0;
    if (count)
        h3k_qkv_rope_bf16<<<h3_attn_grid_1d(count), 256>>>(
            (const uint16_t *)qkv->data, (const uint16_t *)q_norm->data,
            (const uint16_t *)k_norm->data, (const uint16_t *)rope_cos->data,
            (const uint16_t *)rope_sin->data, (uint16_t *)query->data,
            (uint16_t *)key->data, (uint16_t *)value->data, sequence, heads,
            head_dim, rope_half, grouped, epsilon);
    return h3_cuda_launch_check(gpu, "h3_qkv_rope_bf16");
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_cuda_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm, k_norm,
                                 rope_cos, rope_sin, sequence, heads, head_dim,
                                 rope_half, 0, epsilon);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    return h3_cuda_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm, k_norm,
                                 rope_cos, rope_sin, sequence, heads, head_dim,
                                 rope_half, 1, epsilon);
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    /* CUDA has no TensorOps path: mirror the Metal fallback, which is the
     * ordinary linear into the caller-provided QKV scratch followed by the
     * grouped QKV/RoPE kernel. Each inner call bumps its own counter. */
    uint32_t inner = heads * head_dim;
    return h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                              inner * 3) &&
           h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm,
                                        k_norm, rope_cos, rope_sin, rows,
                                        heads, head_dim, rope_half, epsilon);
}

int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_attn_require_bf16(gpu, qkv, count * 3, "vision QKV") ||
        !h3_attn_require_bf16(gpu, rope_cos, rope_count,
                              "vision RoPE cosine") ||
        !h3_attn_require_bf16(gpu, rope_sin, rope_count,
                              "vision RoPE sine") ||
        !h3_attn_require_bf16(gpu, query, count, "vision query") ||
        !h3_attn_require_bf16(gpu, key, count, "vision key") ||
        !h3_attn_require_bf16(gpu, value, count, "vision value") ||
        rope_half * 2 != head_dim ||
        !h3_cuda_require_command(gpu)) return 0;
    if (count)
        h3k_vision_qkv_rope_bf16<<<h3_attn_grid_1d(count), 256>>>(
            (const uint16_t *)qkv->data, (const uint16_t *)rope_cos->data,
            (const uint16_t *)rope_sin->data, (uint16_t *)query->data,
            (uint16_t *)key->data, (uint16_t *)value->data, sequence, heads,
            head_dim, rope_half);
    return h3_cuda_launch_check(gpu, "h3_vision_qkv_rope_bf16");
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_attn_require_bf16(gpu, query_input, query_count, "text query") ||
        !h3_attn_require_bf16(gpu, key_input, key_count, "text key") ||
        !h3_attn_require_bf16(gpu, q_norm, head_dim, "text Q norm") ||
        !h3_attn_require_bf16(gpu, k_norm, head_dim, "text K norm") ||
        !h3_attn_require_bf16(gpu, rope_cos, rope_count, "text RoPE cosine") ||
        !h3_attn_require_bf16(gpu, rope_sin, rope_count, "text RoPE sine") ||
        !h3_attn_require_bf16(gpu, query_output, query_count,
                              "text query output") ||
        !h3_attn_require_bf16(gpu, key_output, key_count,
                              "text key output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (query_count)
        h3k_text_qk_rope_bf16<<<h3_attn_grid_1d(query_count), 256>>>(
            (const uint16_t *)query_input->data,
            (const uint16_t *)key_input->data,
            (const uint16_t *)q_norm->data, (const uint16_t *)k_norm->data,
            (const uint16_t *)rope_cos->data,
            (const uint16_t *)rope_sin->data, (uint16_t *)query_output->data,
            (uint16_t *)key_output->data, sequence, query_heads, kv_heads,
            head_dim, epsilon);
    return h3_cuda_launch_check(gpu, "h3_text_qk_rope_bf16");
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_attn_require_bf16(gpu, query, query_count, "RoPE query") ||
        !h3_attn_require_bf16(gpu, key, key_count, "RoPE key") ||
        !h3_attn_require_f32(gpu, rope_cos_f32, rope_count, "RoPE cosine") ||
        !h3_attn_require_f32(gpu, rope_sin_f32, rope_count, "RoPE sine") ||
        !h3_cuda_require_command(gpu)) return 0;
    uint32_t max_heads = query_heads > kv_heads ? query_heads : kv_heads;
    size_t count = (size_t)sequence * max_heads;
    if (count)
        h3k_rope_text_bf16<<<h3_attn_grid_1d(count), 256>>>(
            (uint16_t *)query->data, (uint16_t *)key->data,
            (const float *)rope_cos_f32->data,
            (const float *)rope_sin_f32->data, sequence, query_heads,
            kv_heads, head_dim, max_heads);
    return h3_cuda_launch_check(gpu, "h3_rope_text_bf16");
}

/* H3_CUDA_SDPA_NAIVE=1 selects the original block-per-row kernels. Read per
 * call so tests can toggle it within one process. */
static int h3_cuda_sdpa_naive(void) {
    const char *value = getenv("H3_CUDA_SDPA_NAIVE");
    return value && *value && strcmp(value, "0") != 0;
}

/* H3_CUDA_SDPA_FLASH_OLD=1 routes f32 SDPA to the original (pre-tuning)
 * h3k_sdpa_flash<float> instantiation instead of h3k_sdpa_flash_f32, for
 * A/B measurements. */
static int h3_cuda_sdpa_flash_old(void) {
    const char *value = getenv("H3_CUDA_SDPA_FLASH_OLD");
    return value && *value && strcmp(value, "0") != 0;
}

/* Pick the f32 flash (QROWS, KVT). Larger tiles amortize barriers and K/V
 * gmem traffic over more query rows; smaller tiles raise occupancy. Rule:
 * among the configs that fit (with the >48KB opt-in where supported),
 * maximize resident blocks/SM (capped at the 2048-thread/SM limit of 8),
 * tie-breaking to the larger QROWS then KVT. H3_CUDA_SDPA_QROWS /
 * H3_CUDA_SDPA_KVT force a choice (tuning knobs). */
static void h3_f32_flash_config(h3_gpu *gpu, uint32_t head_dim, int causal,
                                int *out_qrows, int *out_kvt) {
    const char *fq = getenv("H3_CUDA_SDPA_QROWS");
    const char *fk = getenv("H3_CUDA_SDPA_KVT");
    int want_q = fq && *fq ? atoi(fq) : 0;
    int want_k = fk && *fk ? atoi(fk) : 0;
    int per_sm = 0;
    if (cudaDeviceGetAttribute(&per_sm,
                               cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                               gpu->device) != cudaSuccess ||
        per_sm <= 0)
        per_sm = h3_attn_max_shared(gpu);
    static const int cand[4][2] = {{32, 64}, {32, 32}, {16, 64}, {16, 32}};
    int best = -1, best_blocks = 0;
    for (int i = 0; i < 4; i++) {
        int q = cand[i][0], k = cand[i][1];
        if (want_q && q != want_q) continue;
        if (want_k && k != want_k) continue;
        int fits = causal ? h3_f32_flash_fits<1>(gpu, head_dim, k, q)
                          : h3_f32_flash_fits<0>(gpu, head_dim, k, q);
        if (!fits) continue;
        int blocks =
            (int)((size_t)per_sm / h3_flash_shared_f32(head_dim, k, q));
        if (blocks > 8) blocks = 8;
        if (blocks > best_blocks) {
            best = i;
            best_blocks = blocks;
        }
    }
    if (best < 0) best = 3; /* 16/32 always fits (checked by the caller) */
    *out_qrows = cand[best][0];
    *out_kvt = cand[best][1];
}

/* The Metal host routes SDPA through MPSGraph; the CUDA kernels replace it,
 * but the stats counter must still read as mps_sdpa_dispatches. */
static int h3_cuda_sdpa(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *query,
                        const h3_gpu_tensor *key,
                        const h3_gpu_tensor *value, uint32_t batch,
                        uint32_t sequence, uint32_t heads, uint32_t head_dim,
                        float scale, h3_gpu_dtype dtype, int causal,
                        int head_major_output) {
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (!batch || !sequence || !heads || !head_dim ||
        !h3_cuda_require_command(gpu) ||
        !h3_cuda_require_elements(gpu, query, count, "SDPA query") ||
        !h3_cuda_require_elements(gpu, key, count, "SDPA key") ||
        !h3_cuda_require_elements(gpu, value, count, "SDPA value") ||
        !h3_cuda_require_elements(gpu, output, count, "SDPA output"))
        return 0;
    if (query->dtype != dtype || key->dtype != dtype ||
        value->dtype != dtype || output->dtype != dtype)
        return h3_cuda_fail(gpu, "SDPA tensor dtype mismatch");
    int bf16 = dtype == H3_GPU_BF16;
    size_t shared = (size_t)sequence * sizeof(float);
    /* Flash path needs head_dim <= 128 and its fixed tile footprint; fall
     * back to the naive kernel otherwise. For f32 the footprint is that of
     * the tuned kernel's smallest tile (KVT=32, QROWS=16), which every
     * config choice falls back to. */
    int naive = h3_cuda_sdpa_naive() || head_dim > 128 ||
                (bf16 ? h3_flash_shared(head_dim, 1)
                      : h3_flash_shared_f32(head_dim, 32, 16)) >
                    (size_t)h3_attn_max_shared(gpu);
    if (naive && shared + 512 > (size_t)h3_attn_max_shared(gpu))
        return h3_cuda_fail(gpu, "SDPA sequence exceeds shared memory");
    /* Tensor-core path: bf16, head_dim == 128, non-causal, sm_80+. */
    if (!naive && bf16 && !causal && head_dim == 128 &&
        h3_attn_has_bf16_mma(gpu) &&
        h3_mma_shared() <= (size_t)h3_attn_max_shared(gpu)) {
        dim3 grid((sequence + H3_MMA_QROWS - 1) / H3_MMA_QROWS, heads, batch);
        if (head_major_output)
            h3k_sdpa_mma_bf16<1><<<grid, H3_MMA_THREADS, h3_mma_shared()>>>(
                (const uint16_t *)query->data, (const uint16_t *)key->data,
                (const uint16_t *)value->data, (uint16_t *)output->data,
                sequence, heads, scale);
        else
            h3k_sdpa_mma_bf16<0><<<grid, H3_MMA_THREADS, h3_mma_shared()>>>(
                (const uint16_t *)query->data, (const uint16_t *)key->data,
                (const uint16_t *)value->data, (uint16_t *)output->data,
                sequence, heads, scale);
        return h3_cuda_launch_check_kind(gpu, "h3_sdpa", 3);
    }
    if (!naive) {
        dim3 grid((sequence + H3_FLASH_QROWS - 1) / H3_FLASH_QROWS, heads,
                  batch);
        size_t flash_shared = h3_flash_shared(head_dim, bf16);
        if (!bf16) {
            if (h3_cuda_sdpa_flash_old()) {
                size_t flash_shared = h3_flash_shared(head_dim, 0);
                if (causal)
                    h3k_sdpa_flash<float, 1, 0, 0><<<grid, H3_FLASH_THREADS,
                                                     flash_shared>>>(
                        (const float *)query->data, (const float *)key->data,
                        (const float *)value->data, (float *)output->data,
                        sequence, heads, heads, head_dim, scale);
                else
                    h3k_sdpa_flash<float, 0, 0, 0><<<grid, H3_FLASH_THREADS,
                                                     flash_shared>>>(
                        (const float *)query->data, (const float *)key->data,
                        (const float *)value->data, (float *)output->data,
                        sequence, heads, heads, head_dim, scale);
            } else {
                int qrows, kvt;
                h3_f32_flash_config(gpu, head_dim, causal, &qrows, &kvt);
                dim3 fgrid((sequence + (uint32_t)qrows - 1) / (uint32_t)qrows,
                           heads, batch);
                size_t fshared = h3_flash_shared_f32(head_dim, kvt, qrows);
#define H3_FLASH_F32_LAUNCH(C, K, Q)                                     \
    h3k_sdpa_flash_f32<C, K, Q>                                          \
        <<<fgrid, H3_FLASH_F32_THREADS, fshared>>>(                      \
            (const float *)query->data, (const float *)key->data,        \
            (const float *)value->data, (float *)output->data, sequence, \
            heads, head_dim, scale)
                if (causal) {
                    if (kvt == 64 && qrows == 32)
                        H3_FLASH_F32_LAUNCH(1, 64, 32);
                    else if (kvt == 64)
                        H3_FLASH_F32_LAUNCH(1, 64, 16);
                    else if (qrows == 32)
                        H3_FLASH_F32_LAUNCH(1, 32, 32);
                    else
                        H3_FLASH_F32_LAUNCH(1, 32, 16);
                } else {
                    if (kvt == 64 && qrows == 32)
                        H3_FLASH_F32_LAUNCH(0, 64, 32);
                    else if (kvt == 64)
                        H3_FLASH_F32_LAUNCH(0, 64, 16);
                    else if (qrows == 32)
                        H3_FLASH_F32_LAUNCH(0, 32, 32);
                    else
                        H3_FLASH_F32_LAUNCH(0, 32, 16);
                }
#undef H3_FLASH_F32_LAUNCH
            }
        } else if (head_major_output) {
            h3k_sdpa_flash<uint16_t, 0, 1, 0><<<grid, H3_FLASH_THREADS,
                                                flash_shared>>>(
                (const uint16_t *)query->data, (const uint16_t *)key->data,
                (const uint16_t *)value->data, (uint16_t *)output->data,
                sequence, heads, heads, head_dim, scale);
        } else {
            h3k_sdpa_flash<uint16_t, 0, 0, 0><<<grid, H3_FLASH_THREADS,
                                                flash_shared>>>(
                (const uint16_t *)query->data, (const uint16_t *)key->data,
                (const uint16_t *)value->data, (uint16_t *)output->data,
                sequence, heads, heads, head_dim, scale);
        }
        return h3_cuda_launch_check_kind(gpu, "h3_sdpa", 3);
    }
    dim3 grid(sequence, heads, batch);
    if (dtype == H3_GPU_F32) {
        if (causal)
            h3k_sdpa_naive<float, 1, 0><<<grid, 128, shared>>>(
                (const float *)query->data, (const float *)key->data,
                (const float *)value->data, (float *)output->data, sequence,
                heads, head_dim, scale);
        else
            h3k_sdpa_naive<float, 0, 0><<<grid, 128, shared>>>(
                (const float *)query->data, (const float *)key->data,
                (const float *)value->data, (float *)output->data, sequence,
                heads, head_dim, scale);
    } else if (head_major_output) {
        h3k_sdpa_naive<uint16_t, 0, 1><<<grid, 128, shared>>>(
            (const uint16_t *)query->data, (const uint16_t *)key->data,
            (const uint16_t *)value->data, (uint16_t *)output->data,
            sequence, heads, head_dim, scale);
    } else {
        h3k_sdpa_naive<uint16_t, 0, 0><<<grid, 128, shared>>>(
            (const uint16_t *)query->data, (const uint16_t *)key->data,
            (const uint16_t *)value->data, (uint16_t *)output->data,
            sequence, heads, head_dim, scale);
    }
    return h3_cuda_launch_check_kind(gpu, "h3_sdpa", 3);
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    return h3_cuda_sdpa(gpu, output, query, key, value, 1, sequence, heads,
                        head_dim, scale, H3_GPU_F32, 0, 0);
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale) {
    return h3_cuda_sdpa(gpu, output, query, key, value, batch, sequence,
                        heads, head_dim, scale, H3_GPU_F32, 1, 0);
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_cuda_sdpa(gpu, output, query, key, value, 1, sequence, heads,
                        head_dim, scale, H3_GPU_BF16, 0, 0);
}

int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_cuda_sdpa(gpu, output, query, key, value, 1, sequence, heads,
                        head_dim, scale, H3_GPU_BF16, 0, 1);
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale) {
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads || head_dim > 128 ||
        !h3_attn_require_bf16(gpu, query, query_count, "GQA query") ||
        !h3_attn_require_bf16(gpu, key, kv_count, "GQA key") ||
        !h3_attn_require_bf16(gpu, value, kv_count, "GQA value") ||
        !h3_attn_require_bf16(gpu, output, query_count, "GQA output") ||
        !h3_cuda_require_command(gpu)) return 0;
    /* CUDA ignores H3_MPS_GQA: the portable kernel is the only path, and
     * like Metal's portable path it bumps direct_dispatches. */
    if (!h3_cuda_sdpa_naive()) {
        /* head_dim <= 128 was validated above, so the tile footprint always
         * fits the default shared-memory budget. */
        dim3 grid((sequence + H3_FLASH_QROWS - 1) / H3_FLASH_QROWS,
                  query_heads, 1);
        h3k_sdpa_flash<uint16_t, 1, 0, 1><<<grid, H3_FLASH_THREADS,
                                            h3_flash_shared(head_dim, 1)>>>(
            (const uint16_t *)query->data, (const uint16_t *)key->data,
            (const uint16_t *)value->data, (uint16_t *)output->data,
            sequence, query_heads, kv_heads, head_dim, scale);
        return h3_cuda_launch_check(gpu, "h3_gqa_causal_bf16");
    }
    size_t shared = (size_t)sequence * sizeof(float);
    if (shared + 2 * 128 * sizeof(float) > (size_t)h3_attn_max_shared(gpu))
        return h3_cuda_fail(gpu,
            "causal attention sequence exceeds threadgroup memory");
    h3k_gqa_causal_bf16_naive<<<dim3(sequence, query_heads), 128, shared>>>(
        (const uint16_t *)query->data, (const uint16_t *)key->data,
        (const uint16_t *)value->data, (uint16_t *)output->data, sequence,
        query_heads, kv_heads, head_dim, scale);
    return h3_cuda_launch_check(gpu, "h3_gqa_causal_bf16");
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!batch || !length || !heads || !head_dim || count > UINT32_MAX ||
        !h3_attn_require_f32(gpu, qkv, count * 3, "audio QKV") ||
        !h3_attn_require_f32(gpu, q_bias, width, "audio Q bias") ||
        !h3_attn_require_f32(gpu, k_bias, width, "audio K bias") ||
        !h3_attn_require_f32(gpu, v_bias, width, "audio V bias") ||
        !h3_attn_require_f32(gpu, query, count, "audio query") ||
        !h3_attn_require_f32(gpu, key, count, "audio key") ||
        !h3_attn_require_f32(gpu, value, count, "audio value") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_audio_qkv_split_f32<<<h3_attn_grid_1d(count), 256>>>(
        (const float *)qkv->data, (const float *)q_bias->data,
        (const float *)k_bias->data, (const float *)v_bias->data,
        (float *)query->data, (float *)key->data, (float *)value->data,
        batch, length, heads, head_dim);
    return h3_cuda_launch_check(gpu, "h3_audio_qkv_split_f32");
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_attn_require_f32(gpu, attended, input_count,
                             "audio attended values") ||
        !h3_attn_require_f32(gpu, output, output_count,
                             "audio pooled values") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_audio_attention_pool_f32<<<h3_attn_grid_1d(output_count), 256>>>(
        (const float *)attended->data, (float *)output->data, batch, length,
        heads, head_dim, output_dim);
    return h3_cuda_launch_check(gpu, "h3_audio_attention_pool_f32");
}
