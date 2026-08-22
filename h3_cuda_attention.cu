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

#include <cuda_fp8.h>

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

/* Pack two f32 values as an RNE bf16 pair (lo in the low half). */
__device__ __forceinline__ uint32_t h3_pack_bf16(float lo, float hi) {
    return (uint32_t)h3_f32_to_bf16(lo) | ((uint32_t)h3_f32_to_bf16(hi) << 16);
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

/* One warp per (row, head) for the DiT shape (head_dim 128, rope_half a
 * multiple of 4): lane l owns dims 4l..4l+3 of q, k and v, so each head's
 * sums of squares are one 8-byte load per lane plus a warp reduction (the
 * per-element kernel above recomputed them in all 128 threads of a head),
 * RoPE pairs (d, d +/- rope_half) move between lanes l and l +/- rope_half/4
 * with shuffles, and the stores are 8-byte groups. At 44.5k rows x 56 heads
 * this turns a 48 ms launch into a bandwidth-bound ~13 ms one. Sums reduce
 * in tree order (within the 1-ulp bf16 contract of the tests). */
__global__ void h3k_qkv_rope_bf16_warp(const uint16_t *__restrict__ qkv,
                                       const uint16_t *__restrict__ q_weight,
                                       const uint16_t *__restrict__ k_weight,
                                       const uint16_t *__restrict__ rope_cos,
                                       const uint16_t *__restrict__ rope_sin,
                                       uint16_t *__restrict__ query,
                                       uint16_t *__restrict__ key,
                                       uint16_t *__restrict__ value,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t rope_half, uint32_t grouped,
                                       float epsilon) {
    const uint32_t HD = 128;
    const uint32_t gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31;
    if (gw >= sequence * heads) return;
    const uint32_t row = gw / heads, head = gw % heads;
    const size_t inner = (size_t)heads * HD;
    const size_t row_base = (size_t)row * inner * 3;
    size_t q_base, k_base, v_base;
    if (grouped) {
        q_base = row_base + (size_t)head * HD * 3;
        k_base = q_base + HD;
        v_base = k_base + HD;
    } else {
        q_base = row_base + (size_t)head * HD;
        k_base = q_base + inner;
        v_base = q_base + inner * 2;
    }
    const uint32_t d0 = lane * 4;
    uint2 qr = *(const uint2 *)(qkv + q_base + d0);
    uint2 kr = *(const uint2 *)(qkv + k_base + d0);
    uint2 vr = *(const uint2 *)(qkv + v_base + d0);
    uint2 qw = *(const uint2 *)(q_weight + d0);
    uint2 kw = *(const uint2 *)(k_weight + d0);
    float q[4] = {h3_bf16_to_f32((uint16_t)(qr.x & 0xffffu)),
                  h3_bf16_to_f32((uint16_t)(qr.x >> 16)),
                  h3_bf16_to_f32((uint16_t)(qr.y & 0xffffu)),
                  h3_bf16_to_f32((uint16_t)(qr.y >> 16))};
    float k[4] = {h3_bf16_to_f32((uint16_t)(kr.x & 0xffffu)),
                  h3_bf16_to_f32((uint16_t)(kr.x >> 16)),
                  h3_bf16_to_f32((uint16_t)(kr.y & 0xffffu)),
                  h3_bf16_to_f32((uint16_t)(kr.y >> 16))};
    float wq[4] = {h3_bf16_to_f32((uint16_t)(qw.x & 0xffffu)),
                   h3_bf16_to_f32((uint16_t)(qw.x >> 16)),
                   h3_bf16_to_f32((uint16_t)(qw.y & 0xffffu)),
                   h3_bf16_to_f32((uint16_t)(qw.y >> 16))};
    float wk[4] = {h3_bf16_to_f32((uint16_t)(kw.x & 0xffffu)),
                   h3_bf16_to_f32((uint16_t)(kw.x >> 16)),
                   h3_bf16_to_f32((uint16_t)(kw.y & 0xffffu)),
                   h3_bf16_to_f32((uint16_t)(kw.y >> 16))};
    float q_sum = fmaf(q[0], q[0], fmaf(q[1], q[1], fmaf(q[2], q[2], q[3] * q[3])));
    float k_sum = fmaf(k[0], k[0], fmaf(k[1], k[1], fmaf(k[2], k[2], k[3] * k[3])));
#pragma unroll
    for (int off = 16; off; off >>= 1) {
        q_sum += __shfl_xor_sync(0xffffffffu, q_sum, off);
        k_sum += __shfl_xor_sync(0xffffffffu, k_sum, off);
    }
    const float q_inv = 1.0f / sqrtf(q_sum / (float)HD + epsilon);
    const float k_inv = 1.0f / sqrtf(k_sum / (float)HD + epsilon);
    float qn[4], kn[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        qn[i] = q[i] * q_inv * wq[i];
        kn[i] = k[i] * k_inv * wk[i];
    }
    /* Partner values for the RoPE pairs: dims below rope_half pair with
     * +rope_half, the next rope_half dims with -rope_half, the rest have no
     * partner (source = self, unused). */
    const uint32_t shift = rope_half / 4;
    int src = (int)lane;
    if (d0 < rope_half) src = (int)(lane + shift);
    else if (d0 < rope_half * 2) src = (int)(lane - shift);
    float qp[4], kp[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        qp[i] = __shfl_sync(0xffffffffu, qn[i], src);
        kp[i] = __shfl_sync(0xffffffffu, kn[i], src);
    }
    float qo[4], ko[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t d = d0 + (uint32_t)i;
        if (d < rope_half) {
            float c = h3_bf16_to_f32(rope_cos[row * rope_half + d]);
            float sn = h3_bf16_to_f32(rope_sin[row * rope_half + d]);
            qo[i] = qn[i] * c - qp[i] * sn;
            ko[i] = kn[i] * c - kp[i] * sn;
        } else if (d < rope_half * 2) {
            uint32_t pair = d - rope_half;
            float c = h3_bf16_to_f32(rope_cos[row * rope_half + pair]);
            float sn = h3_bf16_to_f32(rope_sin[row * rope_half + pair]);
            qo[i] = qn[i] * c + qp[i] * sn;
            ko[i] = kn[i] * c + kp[i] * sn;
        } else {
            qo[i] = qn[i];
            ko[i] = kn[i];
        }
    }
    const size_t out = ((size_t)row * heads + head) * HD + d0;
    uint2 qs = {h3_pack_bf16(qo[0], qo[1]), h3_pack_bf16(qo[2], qo[3])};
    uint2 kss = {h3_pack_bf16(ko[0], ko[1]), h3_pack_bf16(ko[2], ko[3])};
    *(uint2 *)(query + out) = qs;
    *(uint2 *)(key + out) = kss;
    *(uint2 *)(value + out) = vr;
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

/* ------------------------------------------- FA2-class tensor-core SDPA */

/* h3k_sdpa_fa2_bf16: FlashAttention-2-style forward for the DiT shape (bf16,
 * head_dim 128, non-causal, kv_heads == heads) on sm_80+. What differs from
 * h3k_sdpa_mma_bf16 — and matters at N ~ 38k rows, where that kernel ran at
 * ~14 TFLOPS on GB10 — is the ratio of mma work to staged bytes and the
 * number of shared-memory round trips per tile:
 *   - 128 query rows per block (8 warps x 16 rows), so every staged 64-row
 *     K/V tile feeds 8x more mma work than the 16-row kernel (per-head K/V
 *     traffic drops from 2376 to 297 passes at N=38016);
 *   - K/V double-buffered through cp.async: tile t+1 streams into the other
 *     stage while tile t computes; one barrier pair per tile;
 *   - the Q A-fragments are loaded into registers once and reused for every
 *     tile; the S accumulators, the running max/sum, and the P operand all
 *     stay in registers: the m16n8 C-fragments of two adjacent key n-tiles
 *     are exactly the m16n8k16 A-fragment of P for the next mma, so P never
 *     goes through shared memory (the old kernel published f32 scores to
 *     smem, ran a warp-per-row softmax, then re-packed from smem).
 * Softmax is exp2-based with scale*log2(e) folded into the exponent (the
 * same math on a different rounding path, within the flash tolerance); P is
 * rounded to bf16 (RNE) before P*V exactly like the existing mma kernel.
 * Rows past the sequence read zero Q and are never stored; keys past the
 * sequence are zero-filled by cp.async and masked to -inf. Inputs are
 * [batch, row, head, dim]; output is the same or [batch, head, row, dim]
 * when HEAD_MAJOR. Requires the >48KB dynamic shared-memory opt-in (2 stages
 * x K+V x 64 x 136 bf16 = 68 KiB); the host checks and sets it. */
#define H3_FA2_THREADS 256
#define H3_FA2_QROWS 128
#define H3_FA2_KVT 64
#define H3_FA2_STAGES 2
/* Row pitch in bf16: head_dim + 8 (16 bytes) of padding keeps the
 * ldmatrix phases on distinct banks (an odd number of 16-byte units). */
#define H3_FA2_LD(hd) ((hd) + 8)
#define H3_FA2_STAGE_ELEMS(hd) (2 * H3_FA2_KVT * H3_FA2_LD(hd))

/* Dynamic shared memory for one launch: HD 128 -> 68 KiB (needs the >48KB
 * opt-in, one block per SM), HD 64 -> 36 KiB (two blocks per SM). */
static size_t h3_fa2_shared(uint32_t head_dim) {
    return (size_t)H3_FA2_STAGES * H3_FA2_STAGE_ELEMS(head_dim) *
           sizeof(uint16_t);
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void h3_cp_async_commit(void) {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void h3_cp_async_wait(void) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/* Softmax exponent for the FA2/FP8 kernels: FEXP=1 uses ex2.approx.f32
 * (__exp2f) — a single MUFU op, and ex2(-inf) = +0 exactly like the libm
 * path needs; FEXP=0 keeps the libm exp2f rounding. Selected once per
 * process (H3_CUDA_SDPA_EXACT_EXP=1 restores libm). */
template <int FEXP>
__device__ __forceinline__ float h3_fa2_exp2(float x) {
    if (FEXP) {
        float r;
        asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
        return r;
    }
    return exp2f(x);
}

/* Stage K/V tile `tile` into `stage`: 2 x 64 rows x 16 chunks of 16 bytes,
 * eight cp.async per thread; rows past the sequence zero-fill (src-size 0)
 * and read from row 0 so the address stays valid. */
template <int HD>
__device__ __forceinline__ void h3_fa2_issue(uint16_t *smem,
                                             const uint16_t *k_src,
                                             const uint16_t *v_src,
                                             size_t row_stride,
                                             uint32_t sequence, uint32_t tile,
                                             uint32_t stage, uint32_t tid) {
    constexpr int LD = H3_FA2_LD(HD);
    constexpr int CHUNKS = HD / 8; /* 16-byte chunks per row */
    uint16_t *ks = smem + stage * H3_FA2_STAGE_ELEMS(HD);
    uint16_t *vs = ks + H3_FA2_KVT * LD;
    uint32_t kb = tile * H3_FA2_KVT;
#pragma unroll
    for (uint32_t i = 0; i < (H3_FA2_KVT * CHUNKS) / H3_FA2_THREADS; i++) {
        uint32_t c = tid + i * H3_FA2_THREADS;
        uint32_t j = c / CHUNKS, d = (c % CHUNKS) * 8;
        uint32_t row = kb + j;
        int live = row < sequence;
        size_t off = (size_t)(live ? row : 0u) * row_stride + d;
        h3_cp_async_16(ks + j * LD + d, k_src + off, live ? 16 : 0);
        h3_cp_async_16(vs + j * LD + d, v_src + off, live ? 16 : 0);
    }
}
#endif

template <int HEAD_MAJOR, int HD, int FEXP>
__global__ void __launch_bounds__(H3_FA2_THREADS, HD == 64 ? 2 : 1)
    h3k_sdpa_fa2_bf16(const uint16_t *__restrict__ query,
                      const uint16_t *__restrict__ key,
                      const uint16_t *__restrict__ value,
                      uint16_t *__restrict__ output, uint32_t sequence,
                      uint32_t heads, float scale) {
#if __CUDA_ARCH__ >= 800
    constexpr int LD = H3_FA2_LD(HD);
    constexpr int STAGE = H3_FA2_STAGE_ELEMS(HD);
    extern __shared__ __align__(16) uint16_t h3_fa2_smem[];
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t head = blockIdx.y;
    const uint32_t q0 = blockIdx.x * H3_FA2_QROWS;
    const size_t row_stride = (size_t)heads * HD;
    const size_t batch_off = (size_t)blockIdx.z * sequence * row_stride;
    const size_t head_off = (size_t)head * HD;
    const uint32_t a_row = lane >> 2, a_col = (lane & 3) * 2;

    /* Q A-fragments for this thread's two rows, all eight k-steps, loaded
     * once. Rows past the sequence read zeros. */
    const uint32_t r0 = q0 + warp * 16 + a_row, r1 = r0 + 8;
    const int ok0 = r0 < sequence, ok1 = r1 < sequence;
    uint32_t qf[HD / 16][4];
    {
        const uint16_t *p0 = query + batch_off + head_off +
                             (size_t)(ok0 ? r0 : 0u) * row_stride;
        const uint16_t *p1 = query + batch_off + head_off +
                             (size_t)(ok1 ? r1 : 0u) * row_stride;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            uint32_t c = (uint32_t)ks * 16 + a_col;
            qf[ks][0] = ok0 ? *(const uint32_t *)(p0 + c) : 0u;
            qf[ks][1] = ok1 ? *(const uint32_t *)(p1 + c) : 0u;
            qf[ks][2] = ok0 ? *(const uint32_t *)(p0 + c + 8) : 0u;
            qf[ks][3] = ok1 ? *(const uint32_t *)(p1 + c + 8) : 0u;
        }
    }

    float o_acc[HD / 8][4];
#pragma unroll
    for (int t = 0; t < HD / 8; t++)
#pragma unroll
        for (int k = 0; k < 4; k++) o_acc[t][k] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float scale_log2 = scale * 1.4426950408889634f;

    const uint16_t *k_src = key + batch_off + head_off;
    const uint16_t *v_src = value + batch_off + head_off;
    const uint32_t tiles = (sequence + H3_FA2_KVT - 1) / H3_FA2_KVT;

    h3_fa2_issue<HD>(h3_fa2_smem, k_src, v_src, row_stride, sequence, 0, 0,
                     tid);
    h3_cp_async_commit();

    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t stage = t & 1u;
        if (t + 1 < tiles)
            h3_fa2_issue<HD>(h3_fa2_smem, k_src, v_src, row_stride,
                             sequence, t + 1, stage ^ 1u, tid);
        h3_cp_async_commit(); /* always commit so wait_group<1> is uniform */
        h3_cp_async_wait<1>();
        __syncthreads();
        const uint16_t *k_tile = h3_fa2_smem + stage * STAGE;
        const uint16_t *v_tile = k_tile + H3_FA2_KVT * LD;

        /* S = Q K^T: eight key n-tiles, eight k-steps over the 128 dims. One
         * ldmatrix.x4 per (n-tile pair, k-step) gives both B fragments. */
        float s[H3_FA2_KVT / 8][4];
#pragma unroll
        for (int n = 0; n < H3_FA2_KVT / 8; n++)
#pragma unroll
            for (int k = 0; k < 4; k++) s[n][k] = 0.0f;
#pragma unroll
        for (int p = 0; p < H3_FA2_KVT / 16; p++) {
            const uint16_t *kp = k_tile +
                                 (p * 16 + (lane & 7) + ((lane & 16) ? 8 : 0)) *
                                     LD +
                                 ((lane & 8) ? 8 : 0);
#pragma unroll
            for (int ks = 0; ks < HD / 16; ks++) {
                uint32_t b[4];
                h3_ldmatrix_x4(b, kp + ks * 16);
                h3_mma_bf16_16x8x16(s[2 * p], qf[ks][0], qf[ks][1], qf[ks][2],
                                    qf[ks][3], b[0], b[1]);
                h3_mma_bf16_16x8x16(s[2 * p + 1], qf[ks][0], qf[ks][1],
                                    qf[ks][2], qf[ks][3], b[2], b[3]);
            }
        }

        /* Online softmax in registers: scale into the exp2 domain, mask the
         * tail tile, row max over this thread's 16 values then across the 4
         * lanes sharing the row, rescale, exponentiate, row sums. */
        const uint32_t kb = t * H3_FA2_KVT;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < H3_FA2_KVT / 8; n++) {
            s[n][0] *= scale_log2;
            s[n][1] *= scale_log2;
            s[n][2] *= scale_log2;
            s[n][3] *= scale_log2;
            if (kb + H3_FA2_KVT > sequence) {
                uint32_t kc = kb + (uint32_t)n * 8 + a_col;
                if (kc >= sequence) { s[n][0] = -INFINITY; s[n][2] = -INFINITY; }
                if (kc + 1 >= sequence) { s[n][1] = -INFINITY; s[n][3] = -INFINITY; }
            }
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        /* A row with no valid key yet has mn == -inf and zero accumulators;
         * reference it to 0 so exp2f(-inf - 0) = 0 instead of NaN. */
        const float ref0 = mn0 == -INFINITY ? 0.0f : mn0;
        const float ref1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float c0 = h3_fa2_exp2<FEXP>(m0 - ref0);
        const float c1 = h3_fa2_exp2<FEXP>(m1 - ref1);
        float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
        for (int n = 0; n < H3_FA2_KVT / 8; n++) {
            s[n][0] = h3_fa2_exp2<FEXP>(s[n][0] - ref0);
            s[n][1] = h3_fa2_exp2<FEXP>(s[n][1] - ref0);
            s[n][2] = h3_fa2_exp2<FEXP>(s[n][2] - ref1);
            s[n][3] = h3_fa2_exp2<FEXP>(s[n][3] - ref1);
            rs0 += s[n][0] + s[n][1];
            rs1 += s[n][2] + s[n][3];
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * c0 + rs0;
        l1 = l1 * c1 + rs1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int d = 0; d < HD / 8; d++) {
            o_acc[d][0] *= c0;
            o_acc[d][1] *= c0;
            o_acc[d][2] *= c1;
            o_acc[d][3] *= c1;
        }

        /* O += P V: P A-fragments packed straight from the S C-fragments
         * (n-tiles 2ks, 2ks+1 form k-step ks), V B-fragments via
         * ldmatrix.x4.trans over the row-major V tile. */
#pragma unroll
        for (int ks = 0; ks < H3_FA2_KVT / 16; ks++) {
            uint32_t a0 = h3_pack_bf16(s[2 * ks][0], s[2 * ks][1]);
            uint32_t a1 = h3_pack_bf16(s[2 * ks][2], s[2 * ks][3]);
            uint32_t a2 = h3_pack_bf16(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            uint32_t a3 = h3_pack_bf16(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            const uint16_t *vp = v_tile +
                                 ((lane & 7) + ((lane & 8) ? 8 : 0) + ks * 16) *
                                     LD +
                                 ((lane & 16) ? 8 : 0);
#pragma unroll
            for (int dp = 0; dp < HD / 16; dp++) {
                uint32_t b[4];
                h3_ldmatrix_x4_trans(b, vp + dp * 16);
                h3_mma_bf16_16x8x16(o_acc[2 * dp], a0, a1, a2, a3, b[0], b[1]);
                h3_mma_bf16_16x8x16(o_acc[2 * dp + 1], a0, a1, a2, a3, b[2],
                                    b[3]);
            }
        }
        __syncthreads(); /* all warps done with this stage before it refills */
    }

    /* Normalize and store bf16 pairs (4-byte aligned: a_col is even). */
    if (ok0) {
        float inv = 1.0f / l0;
        size_t base = HEAD_MAJOR
                          ? batch_off + ((size_t)head * sequence + r0) * HD
                          : batch_off + (size_t)r0 * row_stride + head_off;
#pragma unroll
        for (int d = 0; d < HD / 8; d++)
            *(uint32_t *)(output + base + d * 8 + a_col) =
                h3_pack_bf16(o_acc[d][0] * inv, o_acc[d][1] * inv);
    }
    if (ok1) {
        float inv = 1.0f / l1;
        size_t base = HEAD_MAJOR
                          ? batch_off + ((size_t)head * sequence + r1) * HD
                          : batch_off + (size_t)r1 * row_stride + head_off;
#pragma unroll
        for (int d = 0; d < HD / 8; d++)
            *(uint32_t *)(output + base + d * 8 + a_col) =
                h3_pack_bf16(o_acc[d][2] * inv, o_acc[d][3] * inv);
    }
#endif
}

/* H3_CUDA_SDPA_EXACT_EXP=1 restores the libm exp2f softmax rounding in the
 * FA2/FP8 kernels; the default uses ex2.approx.f32 (see h3_fa2_exp2). */
static int h3_cuda_sdpa_fast_exp(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("H3_CUDA_SDPA_EXACT_EXP");
        cached = !(value && *value && strcmp(value, "0") != 0);
    }
    return cached;
}

/* One-time >48KB dynamic shared-memory opt-in per instantiation. */
template <int HEAD_MAJOR, int HD, int FEXP>
static int h3_fa2_attr(void) {
    static int state = 0; /* 0 unknown, 1 ok, -1 failed */
    if (!state)
        state = cudaFuncSetAttribute(
                    (const void *)h3k_sdpa_fa2_bf16<HEAD_MAJOR, HD, FEXP>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)h3_fa2_shared(HD)) == cudaSuccess
                    ? 1
                    : -1;
    return state > 0;
}

template <int HEAD_MAJOR, int HD>
static int h3_fa2_attr_exp(void) {
    return h3_cuda_sdpa_fast_exp() ? h3_fa2_attr<HEAD_MAJOR, HD, 1>()
                                   : h3_fa2_attr<HEAD_MAJOR, HD, 0>();
}

/* H3_CUDA_SDPA_FA2=0 routes bf16/hd128 SDPA back to h3k_sdpa_mma_bf16 for
 * A/B measurements. */
static int h3_cuda_sdpa_fa2_enabled(void) {
    const char *value = getenv("H3_CUDA_SDPA_FA2");
    return !(value && *value && strcmp(value, "0") == 0);
}

static int h3_fa2_fits(struct h3_gpu *gpu, int head_major, uint32_t head_dim) {
    int optin = 0;
    if ((head_dim != 64 && head_dim != 128) ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               gpu->device) != cudaSuccess ||
        (size_t)optin < h3_fa2_shared(head_dim))
        return 0;
    if (head_dim == 64)
        return head_major ? h3_fa2_attr_exp<1, 64>() : h3_fa2_attr_exp<0, 64>();
    return head_major ? h3_fa2_attr_exp<1, 128>() : h3_fa2_attr_exp<0, 128>();
}

/* Launch the FA2 kernel for a validated (bf16, non-causal, hd 64/128)
 * problem. */
static void h3_fa2_launch(const uint16_t *query, const uint16_t *key,
                          const uint16_t *value, uint16_t *output,
                          uint32_t batch, uint32_t sequence, uint32_t heads,
                          uint32_t head_dim, float scale, int head_major) {
    dim3 grid((sequence + H3_FA2_QROWS - 1) / H3_FA2_QROWS, heads, batch);
    size_t shared = h3_fa2_shared(head_dim);
    const int fexp = h3_cuda_sdpa_fast_exp();
#define H3_FA2_GO(HM, HD, FE)                                              \
    h3k_sdpa_fa2_bf16<HM, HD, FE><<<grid, H3_FA2_THREADS, shared>>>(       \
        query, key, value, output, sequence, heads, scale)
#define H3_FA2_GO_EXP(HM, HD)                                              \
    do { if (fexp) H3_FA2_GO(HM, HD, 1); else H3_FA2_GO(HM, HD, 0); } while (0)
    if (head_dim == 64) {
        if (head_major) H3_FA2_GO_EXP(1, 64); else H3_FA2_GO_EXP(0, 64);
    } else {
        if (head_major) H3_FA2_GO_EXP(1, 128); else H3_FA2_GO_EXP(0, 128);
    }
#undef H3_FA2_GO_EXP
#undef H3_FA2_GO
}

/* f32 SDPA through the bf16 FA2 kernel: Q/K/V are rounded to bf16 into the
 * workspace, the kernel runs, and the bf16 result widens back to f32. This
 * is the video VAE decoder's attention (hd 64, ~2k rows per tile): the
 * scalar f32 flash kernel ran it at ~2 TFLOPS. H3_CUDA_SDPA_F32_EXACT=1
 * keeps the exact f32 kernels. */
__global__ void h3k_fa2_cast_f32_to_bf16(const float *__restrict__ in,
                                         uint16_t *__restrict__ out,
                                         size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += stride)
        out[i] = h3_f32_to_bf16(in[i]);
}
__global__ void h3k_fa2_cast_bf16_to_f32(const uint16_t *__restrict__ in,
                                         float *__restrict__ out,
                                         size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += stride)
        out[i] = h3_bf16_to_f32(in[i]);
}

static int h3_cuda_sdpa_f32_exact(void) {
    const char *value = getenv("H3_CUDA_SDPA_F32_EXACT");
    return value && *value && strcmp(value, "0") != 0;
}

/* ------------------------------------------------- FP8 (e4m3) flash SDPA */

/* h3k_sdpa_fa2_fp8: the FA2 kernel above on the e4m3 tensor cores
 * (mma.m16n8k32, sm_89+; 2x the bf16 rate), opt-in with H3_CUDA_SDPA_FP8=1.
 * A pre-pass (h3_cuda_sdpa_fp8_pack) measures per-tensor absmax for Q, K
 * and V, quantizes Q and K to e4m3 in head-major layout, and writes V
 * transposed ([head][dim][key], keys zero-padded to the 64-key tile) with
 * the keys of every 32-block permuted so that the softmax C-fragments of
 * four adjacent 8-key n-tiles are exactly the 32-key A-fragment of P*V
 * (FA3's trick; P is scaled by 448 before e4m3). Scores carry s_q*s_k, the
 * output s_v/448, all folded into the existing scale / normalization.
 * Numerics: e4m3 keeps 3 mantissa bits, so expect a few percent of
 * relative error against the bf16 kernel (see the attention test). */
#define H3_FP8_THREADS 256
#define H3_FP8_QROWS 128
#define H3_FP8_KVT 64
#define H3_FP8_HD 128
#define H3_FP8_LDK (H3_FP8_HD + 16)   /* bytes per K row: 9 x 16 B (odd) */
#define H3_FP8_LDV (H3_FP8_KVT + 16)  /* bytes per V^T row: 5 x 16 B (odd) */
#define H3_FP8_STAGE_BYTES (H3_FP8_KVT * H3_FP8_LDK + H3_FP8_HD * H3_FP8_LDV)
#define H3_FP8_P_SCALE 448.0f

/* Key held at position p of a 32-key block in the permuted V^T layout:
 * the m16n8k32 A-fragment byte (4q + j) / (16 + 4q + j) of lane group q
 * must carry the probability the C-fragments hold there — keys 2q, 2q+1 of
 * n-tile 0 and 8+2q, 9+2q of n-tile 1 (same for the upper 16). */
__host__ __device__ __forceinline__ uint32_t h3_fp8_perm_key(uint32_t p) {
    uint32_t base = p & ~31u, r = p & 31u;
    uint32_t hi = r & 16u, q = (r & 15u) >> 2, j = r & 3u;
    return base + hi + (j < 2u ? 2u * q + j : 8u + 2u * q + (j - 2u));
}

__global__ void h3k_fp8_attn_absmax(const uint16_t *__restrict__ x, size_t n,
                                    unsigned *__restrict__ result) {
    __shared__ unsigned partial[256];
    float m = 0.0f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t n8 = n / 8;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8;
         i += stride) {
        uint4 v = ((const uint4 *)x)[i];
        uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            m = fmaxf(m, fabsf(h3_bf16_to_f32((uint16_t)(w[j] & 0xffffu))));
            m = fmaxf(m, fabsf(h3_bf16_to_f32((uint16_t)(w[j] >> 16))));
        }
    }
    for (size_t i = n8 * 8 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride)
        m = fmaxf(m, fabsf(h3_bf16_to_f32(x[i])));
    partial[threadIdx.x] = __float_as_uint(m);
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s; s >>= 1) {
        if (threadIdx.x < s)
            partial[threadIdx.x] = max(partial[threadIdx.x],
                                       partial[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(result, partial[0]);
}

/* scales[0..2] = s_q, s_k, s_v; [3..5] = their inverses. */
__global__ void h3k_fp8_attn_scales(const unsigned *__restrict__ absmax,
                                    float *__restrict__ scales) {
    int i = threadIdx.x;
    if (i >= 3) return;
    float m = __uint_as_float(absmax[i]);
    float s = m > 0.0f ? m / H3_FP8_P_SCALE : 1.0f;
    scales[i] = s;
    scales[3 + i] = 1.0f / s;
}

/* Delayed scaling: scales for this call come from the running absmax the
 * pack kernels have accumulated over every previous call, with 6% headroom.
 * The accumulator is monotone (never reset): e4m3 is a float format, so a
 * larger-than-necessary scale costs no relative precision, while a monotone
 * max means only a first sighting of a new global maximum can transiently
 * saturate (to +-448, as in e4m3 training recipes). */
__global__ void h3k_fp8_attn_delayed_scales(const unsigned *__restrict__ amax,
                                            float *__restrict__ scales) {
    int i = threadIdx.x;
    if (i >= 3) return;
    float m = __uint_as_float(amax[i]);
    float s = m > 0.0f ? m * 1.06f / H3_FP8_P_SCALE : 1.0f;
    scales[i] = s;
    scales[3 + i] = 1.0f / s;
}

/* Block-reduce a thread-local |x| max and fold it into amax[slot]. */
__device__ __forceinline__ void h3_fp8_amax_fold(unsigned *__restrict__ amax,
                                                 float m) {
    __shared__ unsigned partial[256];
    partial[threadIdx.x] = __float_as_uint(m);
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s; s >>= 1) {
        if (threadIdx.x < s)
            partial[threadIdx.x] = max(partial[threadIdx.x],
                                       partial[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(amax, partial[0]);
}

__device__ __forceinline__ uint32_t h3_fp8_pack4(float a, float b, float c,
                                                 float d) {
    __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b),
                                                       __NV_SATFINITE,
                                                       __NV_E4M3);
    __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d),
                                                       __NV_SATFINITE,
                                                       __NV_E4M3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

/* Q or K: [seq][heads][128] bf16 -> [heads][seq][128] e4m3 (8 per thread).
 * The raw (pre-scale) absmax folds into amax for the next call's delayed
 * scale. */
__global__ void h3k_fp8_attn_pack_qk(const uint16_t *__restrict__ in,
                                     uint8_t *__restrict__ out,
                                     uint32_t sequence, uint32_t heads,
                                     const float *__restrict__ inverse,
                                     unsigned *__restrict__ amax) {
    const float inv = *inverse;
    float mx = 0.0f;
    size_t chunks = (size_t)sequence * heads * (H3_FP8_HD / 8);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x; c < chunks;
         c += stride) {
        uint32_t d = (uint32_t)(c % (H3_FP8_HD / 8)) * 8;
        size_t rh = c / (H3_FP8_HD / 8);
        uint32_t h = (uint32_t)(rh % heads), r = (uint32_t)(rh / heads);
        uint4 v = *(const uint4 *)(in + rh * H3_FP8_HD + d);
        uint32_t w[4] = {v.x, v.y, v.z, v.w};
        float f[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float lo = h3_bf16_to_f32((uint16_t)(w[j] & 0xffffu));
            float hi = h3_bf16_to_f32((uint16_t)(w[j] >> 16));
            mx = fmaxf(mx, fmaxf(fabsf(lo), fabsf(hi)));
            f[2 * j] = lo * inv;
            f[2 * j + 1] = hi * inv;
        }
        uint2 o = {h3_fp8_pack4(f[0], f[1], f[2], f[3]),
                   h3_fp8_pack4(f[4], f[5], f[6], f[7])};
        *(uint2 *)(out + ((size_t)h * sequence + r) * H3_FP8_HD + d) = o;
    }
    if (amax) h3_fp8_amax_fold(amax, mx);
}

/* V: [seq][heads][128] bf16 -> [heads][128][seq_pad] e4m3, transposed
 * through shared memory one (head, 64-key tile) per block, keys permuted
 * inside each 32-block (h3_fp8_perm_key) and zero past the sequence. */
__global__ void h3k_fp8_attn_pack_vt(const uint16_t *__restrict__ in,
                                     uint8_t *__restrict__ out,
                                     uint32_t sequence, uint32_t seq_pad,
                                     uint32_t heads,
                                     const float *__restrict__ inverse,
                                     unsigned *__restrict__ amax) {
    __shared__ uint8_t tile[H3_FP8_KVT][H3_FP8_HD + 4];
    const float inv = *inverse;
    float mx = 0.0f;
    const uint32_t head = blockIdx.y, kb = blockIdx.x * H3_FP8_KVT;
    const uint32_t tid = threadIdx.x;
    /* load 64 keys x 128 dims (8 bf16 per thread per step), quantize */
    for (uint32_t c = tid; c < H3_FP8_KVT * (H3_FP8_HD / 8); c += blockDim.x) {
        uint32_t j = c >> 4, d = (c & 15) * 8;
        uint32_t key = kb + j;
        uint8_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (key < sequence) {
            uint4 v = *(const uint4 *)(in + ((size_t)key * heads + head) *
                                                H3_FP8_HD + d);
            uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int t = 0; t < 4; t++) {
                float lo = h3_bf16_to_f32((uint16_t)(w[t] & 0xffffu));
                float hi = h3_bf16_to_f32((uint16_t)(w[t] >> 16));
                mx = fmaxf(mx, fmaxf(fabsf(lo), fabsf(hi)));
                __nv_fp8x2_storage_t p = __nv_cvt_float2_to_fp8x2(
                    make_float2(lo * inv, hi * inv), __NV_SATFINITE,
                    __NV_E4M3);
                q[2 * t] = (uint8_t)(p & 0xffu);
                q[2 * t + 1] = (uint8_t)(p >> 8);
            }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) tile[j][d + t] = q[t];
    }
    if (amax) h3_fp8_amax_fold(amax, mx);
    __syncthreads();
    /* write 128 dim rows x 64 keys (4 permuted keys = 4 bytes per thread) */
    for (uint32_t c = tid; c < H3_FP8_HD * (H3_FP8_KVT / 4); c += blockDim.x) {
        uint32_t d = c >> 4, p0 = (c & 15) * 4;
        uint32_t b0 = tile[h3_fp8_perm_key(p0)][d];
        uint32_t b1 = tile[h3_fp8_perm_key(p0 + 1)][d];
        uint32_t b2 = tile[h3_fp8_perm_key(p0 + 2)][d];
        uint32_t b3 = tile[h3_fp8_perm_key(p0 + 3)][d];
        *(uint32_t *)(out + ((size_t)head * H3_FP8_HD + d) * seq_pad + kb +
                      p0) = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

#if __CUDA_ARCH__ >= 890
__device__ __forceinline__ void h3_mma_e4m3_16x8x32(float *c, uint32_t a0,
                                                    uint32_t a1, uint32_t a2,
                                                    uint32_t a3, uint32_t b0,
                                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void h3_ldmatrix_x4_b(uint32_t r[4],
                                                 const uint8_t *addr) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void h3_cp_async_16_b(uint8_t *smem,
                                                 const uint8_t *gmem,
                                                 int src_size) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
                 "l"(gmem), "r"(src_size));
}

/* Stage K tile (64 x 128 B) and V^T tile (128 x 64 B) of key block `tile`. */
__device__ __forceinline__ void h3_fp8_issue(uint8_t *smem,
                                             const uint8_t *k_src,
                                             const uint8_t *vt_src,
                                             uint32_t sequence,
                                             uint32_t seq_pad, uint32_t tile,
                                             uint32_t stage, uint32_t tid) {
    uint8_t *ks = smem + stage * H3_FP8_STAGE_BYTES;
    uint8_t *vs = ks + H3_FP8_KVT * H3_FP8_LDK;
    uint32_t kb = tile * H3_FP8_KVT;
    /* K: 64 rows x 8 chunks = 512; V^T: 128 rows x 4 chunks = 512 */
#pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t c = tid + i * H3_FP8_THREADS;
        uint32_t j = c >> 3, d = (c & 7) * 16;
        uint32_t row = kb + j;
        int live = row < sequence;
        h3_cp_async_16_b(ks + j * H3_FP8_LDK + d,
                         k_src + (size_t)(live ? row : 0u) * H3_FP8_HD + d,
                         live ? 16 : 0);
    }
#pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t c = tid + i * H3_FP8_THREADS;
        uint32_t d = c >> 2, kc = (c & 3) * 16;
        h3_cp_async_16_b(vs + d * H3_FP8_LDV + kc,
                         vt_src + (size_t)d * seq_pad + kb + kc, 16);
    }
}
#endif

/* Blocks per SM the register allocation targets: 2 caps the kernel at 128
 * registers (a little spilling), 1 lets it breathe; -DH3_FP8_MIN_BLOCKS=1
 * builds the other variant for A/B timing. */
#ifndef H3_FP8_MIN_BLOCKS
#define H3_FP8_MIN_BLOCKS 2
#endif
template <int HEAD_MAJOR, int FEXP>
__global__ void __launch_bounds__(H3_FP8_THREADS, H3_FP8_MIN_BLOCKS)
    h3k_sdpa_fa2_fp8(const uint8_t *__restrict__ q8,
                     const uint8_t *__restrict__ k8,
                     const uint8_t *__restrict__ vt8,
                     const float *__restrict__ scales,
                     uint16_t *__restrict__ output, uint32_t sequence,
                     uint32_t seq_pad, uint32_t heads, float scale) {
#if __CUDA_ARCH__ >= 890
    extern __shared__ __align__(16) uint8_t h3_fp8_smem[];
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t head = blockIdx.y;
    const uint32_t q0 = blockIdx.x * H3_FP8_QROWS;
    const size_t head_rows = (size_t)head * sequence;
    const uint32_t a_row = lane >> 2, a_col = (lane & 3) * 4;
    const float s_q = scales[0], s_k = scales[1], s_v = scales[2];

    const uint32_t r0 = q0 + warp * 16 + a_row, r1 = r0 + 8;
    const int ok0 = r0 < sequence, ok1 = r1 < sequence;
    uint32_t qf[H3_FP8_HD / 32][4];
    {
        const uint8_t *p0 = q8 + (head_rows + (ok0 ? r0 : 0u)) * H3_FP8_HD;
        const uint8_t *p1 = q8 + (head_rows + (ok1 ? r1 : 0u)) * H3_FP8_HD;
#pragma unroll
        for (int ks = 0; ks < H3_FP8_HD / 32; ks++) {
            uint32_t c = (uint32_t)ks * 32 + a_col;
            qf[ks][0] = ok0 ? *(const uint32_t *)(p0 + c) : 0u;
            qf[ks][1] = ok1 ? *(const uint32_t *)(p1 + c) : 0u;
            qf[ks][2] = ok0 ? *(const uint32_t *)(p0 + c + 16) : 0u;
            qf[ks][3] = ok1 ? *(const uint32_t *)(p1 + c + 16) : 0u;
        }
    }
    float o_acc[H3_FP8_HD / 8][4];
#pragma unroll
    for (int t = 0; t < H3_FP8_HD / 8; t++)
#pragma unroll
        for (int k = 0; k < 4; k++) o_acc[t][k] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float scale_log2 = scale * s_q * s_k * 1.4426950408889634f;

    const uint8_t *k_src = k8 + head_rows * H3_FP8_HD;
    const uint8_t *vt_src = vt8 + (size_t)head * H3_FP8_HD * seq_pad;
    const uint32_t tiles = seq_pad / H3_FP8_KVT;

    h3_fp8_issue(h3_fp8_smem, k_src, vt_src, sequence, seq_pad, 0, 0, tid);
    h3_cp_async_commit();
    /* C-fragment column (within an 8-key n-tile) this thread holds. */
    const uint32_t c_col = (lane & 3) * 2;
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t stage = t & 1u;
        if (t + 1 < tiles)
            h3_fp8_issue(h3_fp8_smem, k_src, vt_src, sequence, seq_pad, t + 1,
                         stage ^ 1u, tid);
        h3_cp_async_commit();
        h3_cp_async_wait<1>();
        __syncthreads();
        const uint8_t *k_tile = h3_fp8_smem + stage * H3_FP8_STAGE_BYTES;
        const uint8_t *v_tile = k_tile + H3_FP8_KVT * H3_FP8_LDK;

        float s[H3_FP8_KVT / 8][4];
#pragma unroll
        for (int n = 0; n < H3_FP8_KVT / 8; n++)
#pragma unroll
            for (int k = 0; k < 4; k++) s[n][k] = 0.0f;
#pragma unroll
        for (int p = 0; p < H3_FP8_KVT / 16; p++) {
            const uint8_t *kp = k_tile +
                                (p * 16 + (lane & 7) + ((lane & 16) ? 8 : 0)) *
                                    H3_FP8_LDK +
                                ((lane & 8) ? 16 : 0);
#pragma unroll
            for (int ks = 0; ks < H3_FP8_HD / 32; ks++) {
                uint32_t b[4];
                h3_ldmatrix_x4_b(b, kp + ks * 32);
                h3_mma_e4m3_16x8x32(s[2 * p], qf[ks][0], qf[ks][1], qf[ks][2],
                                    qf[ks][3], b[0], b[1]);
                h3_mma_e4m3_16x8x32(s[2 * p + 1], qf[ks][0], qf[ks][1],
                                    qf[ks][2], qf[ks][3], b[2], b[3]);
            }
        }
        const uint32_t kb = t * H3_FP8_KVT;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < H3_FP8_KVT / 8; n++) {
            s[n][0] *= scale_log2;
            s[n][1] *= scale_log2;
            s[n][2] *= scale_log2;
            s[n][3] *= scale_log2;
            if (kb + H3_FP8_KVT > sequence) {
                uint32_t kc = kb + (uint32_t)n * 8 + c_col;
                if (kc >= sequence) { s[n][0] = -INFINITY; s[n][2] = -INFINITY; }
                if (kc + 1 >= sequence) { s[n][1] = -INFINITY; s[n][3] = -INFINITY; }
            }
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float ref0 = mn0 == -INFINITY ? 0.0f : mn0;
        const float ref1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float c0 = h3_fa2_exp2<FEXP>(m0 - ref0);
        const float c1 = h3_fa2_exp2<FEXP>(m1 - ref1);
        float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
        for (int n = 0; n < H3_FP8_KVT / 8; n++) {
            s[n][0] = h3_fa2_exp2<FEXP>(s[n][0] - ref0);
            s[n][1] = h3_fa2_exp2<FEXP>(s[n][1] - ref0);
            s[n][2] = h3_fa2_exp2<FEXP>(s[n][2] - ref1);
            s[n][3] = h3_fa2_exp2<FEXP>(s[n][3] - ref1);
            rs0 += s[n][0] + s[n][1];
            rs1 += s[n][2] + s[n][3];
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * c0 + rs0;
        l1 = l1 * c1 + rs1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int d = 0; d < H3_FP8_HD / 8; d++) {
            o_acc[d][0] *= c0;
            o_acc[d][1] *= c0;
            o_acc[d][2] *= c1;
            o_acc[d][3] *= c1;
        }
        /* O += P V: 32-key A-fragments from four n-tiles (scaled by 448),
         * V^T B-fragments via ldmatrix over the permuted key-major rows. */
#pragma unroll
        for (int ks = 0; ks < H3_FP8_KVT / 32; ks++) {
            const int n0 = ks * 4;
            uint32_t a0 = h3_fp8_pack4(s[n0][0] * H3_FP8_P_SCALE,
                                       s[n0][1] * H3_FP8_P_SCALE,
                                       s[n0 + 1][0] * H3_FP8_P_SCALE,
                                       s[n0 + 1][1] * H3_FP8_P_SCALE);
            uint32_t a1 = h3_fp8_pack4(s[n0][2] * H3_FP8_P_SCALE,
                                       s[n0][3] * H3_FP8_P_SCALE,
                                       s[n0 + 1][2] * H3_FP8_P_SCALE,
                                       s[n0 + 1][3] * H3_FP8_P_SCALE);
            uint32_t a2 = h3_fp8_pack4(s[n0 + 2][0] * H3_FP8_P_SCALE,
                                       s[n0 + 2][1] * H3_FP8_P_SCALE,
                                       s[n0 + 3][0] * H3_FP8_P_SCALE,
                                       s[n0 + 3][1] * H3_FP8_P_SCALE);
            uint32_t a3 = h3_fp8_pack4(s[n0 + 2][2] * H3_FP8_P_SCALE,
                                       s[n0 + 2][3] * H3_FP8_P_SCALE,
                                       s[n0 + 3][2] * H3_FP8_P_SCALE,
                                       s[n0 + 3][3] * H3_FP8_P_SCALE);
#pragma unroll
            for (int dp = 0; dp < H3_FP8_HD / 16; dp++) {
                const uint8_t *vp = v_tile +
                                    (dp * 16 + (lane & 7) + ((lane & 16) ? 8 : 0)) *
                                        H3_FP8_LDV +
                                    ((lane & 8) ? 16 : 0) + ks * 32;
                uint32_t b[4];
                h3_ldmatrix_x4_b(b, vp);
                h3_mma_e4m3_16x8x32(o_acc[2 * dp], a0, a1, a2, a3, b[0], b[1]);
                h3_mma_e4m3_16x8x32(o_acc[2 * dp + 1], a0, a1, a2, a3, b[2],
                                    b[3]);
            }
        }
        __syncthreads();
    }
    const float unscale = s_v / H3_FP8_P_SCALE;
    const size_t row_stride = (size_t)heads * H3_FP8_HD;
    const size_t head_off = (size_t)head * H3_FP8_HD;
    if (ok0) {
        float inv = unscale / l0;
        size_t base = HEAD_MAJOR ? (head_rows + r0) * H3_FP8_HD
                                 : (size_t)r0 * row_stride + head_off;
#pragma unroll
        for (int d = 0; d < H3_FP8_HD / 8; d++)
            *(uint32_t *)(output + base + d * 8 + c_col) =
                h3_pack_bf16(o_acc[d][0] * inv, o_acc[d][1] * inv);
    }
    if (ok1) {
        float inv = unscale / l1;
        size_t base = HEAD_MAJOR ? (head_rows + r1) * H3_FP8_HD
                                 : (size_t)r1 * row_stride + head_off;
#pragma unroll
        for (int d = 0; d < H3_FP8_HD / 8; d++)
            *(uint32_t *)(output + base + d * 8 + c_col) =
                h3_pack_bf16(o_acc[d][2] * inv, o_acc[d][3] * inv);
    }
#endif
}

static float __uint_as_float_host(unsigned u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static int h3_cuda_sdpa_fp8_enabled(void) {
    const char *value = getenv("H3_CUDA_SDPA_FP8");
    return value && *value && strcmp(value, "0") != 0;
}

/* H3_CUDA_SDPA_FP8_EXACT_SCALE=1 restores the per-call absmax pre-passes;
 * the default quantizes with the previous call's accumulated absmax
 * (delayed scaling), removing three full Q/K/V reads per call. */
static int h3_cuda_sdpa_fp8_exact_scale(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("H3_CUDA_SDPA_FP8_EXACT_SCALE");
        cached = value && *value && strcmp(value, "0") != 0;
    }
    return cached;
}

static int h3_attn_has_fp8_mma(struct h3_gpu *gpu) {
    static int cached_device = -1;
    static int cached_result = 0;
    if (cached_device != gpu->device) {
        int major = 0, minor = 0;
        cached_result =
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   gpu->device) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   gpu->device) == cudaSuccess &&
            (major > 8 || (major == 8 && minor >= 9));
        cached_device = gpu->device;
    }
    return cached_result;
}

/* Whole FP8 SDPA: pre-pass into the workspace, then the kernel. Returns 0
 * (nothing consequential launched) when the workspace is unavailable. */
static int h3_cuda_sdpa_fp8(struct h3_gpu *gpu, const uint16_t *query,
                            const uint16_t *key, const uint16_t *value,
                            uint16_t *output, uint32_t sequence,
                            uint32_t heads, float scale, int head_major) {
    uint32_t seq_pad = (sequence + H3_FP8_KVT - 1) / H3_FP8_KVT * H3_FP8_KVT;
    size_t qk_bytes = (size_t)sequence * heads * H3_FP8_HD;
    size_t vt_bytes = (size_t)seq_pad * heads * H3_FP8_HD;
    size_t q_off = 0, k_off = (qk_bytes + 255) & ~(size_t)255;
    size_t v_off = k_off + ((qk_bytes + 255) & ~(size_t)255);
    size_t s_off = v_off + ((vt_bytes + 255) & ~(size_t)255);
    size_t total = s_off + 256;
    uint8_t *ws = (uint8_t *)h3_cuda_workspace(gpu, total);
    if (!ws) return 0;
    uint8_t *q8 = ws + q_off, *k8 = ws + k_off, *vt8 = ws + v_off;
    float *scales = (float *)(ws + s_off);
    unsigned *absmax = (unsigned *)(ws + s_off + 64);
    size_t count = (size_t)sequence * heads * H3_FP8_HD;
    if (!gpu->sdpa_fp8_amax) {
        if (cudaMalloc((void **)&gpu->sdpa_fp8_amax,
                       4 * sizeof(unsigned)) != cudaSuccess)
            gpu->sdpa_fp8_amax = NULL;
        else if (cudaMemsetAsync(gpu->sdpa_fp8_amax, 0,
                                 4 * sizeof(unsigned), 0) != cudaSuccess) {
            cudaFree(gpu->sdpa_fp8_amax);
            gpu->sdpa_fp8_amax = NULL;
        }
    }
    int exact = h3_cuda_sdpa_fp8_exact_scale() || !gpu->sdpa_fp8_amax ||
                !gpu->sdpa_fp8_primed;
    if (exact) {
        if (cudaMemsetAsync(absmax, 0, 4 * sizeof(unsigned), 0) !=
            cudaSuccess)
            return 0;
        unsigned blocks = (unsigned)((count / 8 + 255) / 256);
        if (blocks > 2048u) blocks = 2048u;
        h3k_fp8_attn_absmax<<<blocks, 256>>>(query, count, absmax);
        h3k_fp8_attn_absmax<<<blocks, 256>>>(key, count, absmax + 1);
        h3k_fp8_attn_absmax<<<blocks, 256>>>(value, count, absmax + 2);
        h3k_fp8_attn_scales<<<1, 32>>>(absmax, scales);
        /* The packs below fold this call's absmax into the running
         * accumulator, priming the delayed path. */
        if (gpu->sdpa_fp8_amax) gpu->sdpa_fp8_primed = 1;
    } else {
        h3k_fp8_attn_delayed_scales<<<1, 32>>>(gpu->sdpa_fp8_amax, scales);
    }
    unsigned *amax_q = gpu->sdpa_fp8_amax;
    unsigned *amax_k = amax_q ? amax_q + 1 : NULL;
    unsigned *amax_v = amax_q ? amax_q + 2 : NULL;
    unsigned pblocks = (unsigned)((count / 8 + 255) / 256);
    if (pblocks > 4096u) pblocks = 4096u;
    h3k_fp8_attn_pack_qk<<<pblocks, 256>>>(query, q8, sequence, heads,
                                           scales + 3, amax_q);
    h3k_fp8_attn_pack_qk<<<pblocks, 256>>>(key, k8, sequence, heads,
                                           scales + 4, amax_k);
    dim3 vgrid(seq_pad / H3_FP8_KVT, heads);
    h3k_fp8_attn_pack_vt<<<vgrid, 256>>>(value, vt8, sequence, seq_pad, heads,
                                         scales + 5, amax_v);
    dim3 grid((sequence + H3_FP8_QROWS - 1) / H3_FP8_QROWS, heads, 1);
    size_t shared = (size_t)2 * H3_FP8_STAGE_BYTES;
    if (getenv("H3_CUDA_SDPA_FP8_DEBUG")) {
        /* Debug: dump the pre-pass products (synchronous). */
        float hs[6];
        unsigned hm[4];
        uint8_t hq[16], hk[16], hv[16];
        cudaDeviceSynchronize();
        cudaMemcpy(hs, scales, sizeof(hs), cudaMemcpyDeviceToHost);
        cudaMemcpy(hm, absmax, sizeof(hm), cudaMemcpyDeviceToHost);
        cudaMemcpy(hq, q8, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(hk, k8, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(hv, vt8, 16, cudaMemcpyDeviceToHost);
        fprintf(stderr, "fp8 sdpa debug: absmax %g %g %g scales %g %g %g inv %g %g %g\n",
                __uint_as_float_host(hm[0]), __uint_as_float_host(hm[1]),
                __uint_as_float_host(hm[2]), hs[0], hs[1], hs[2], hs[3], hs[4],
                hs[5]);
        fprintf(stderr, "  q8[0..15]:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %02x", hq[i]);
        fprintf(stderr, "\n  k8[0..15]:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %02x", hk[i]);
        fprintf(stderr, "\n  vt8[0..15]:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %02x", hv[i]);
        fprintf(stderr, "\n  status after pre-pass: %s\n",
                cudaGetErrorString(cudaGetLastError()));
    }
#define H3_FP8_GO(HM, FE)                                                  \
    h3k_sdpa_fa2_fp8<HM, FE><<<grid, H3_FP8_THREADS, shared>>>(            \
        q8, k8, vt8, scales, output, sequence, seq_pad, heads, scale)
    if (h3_cuda_sdpa_fast_exp()) {
        if (head_major) H3_FP8_GO(1, 1); else H3_FP8_GO(0, 1);
    } else {
        if (head_major) H3_FP8_GO(1, 0); else H3_FP8_GO(0, 0);
    }
#undef H3_FP8_GO
    if (getenv("H3_CUDA_SDPA_FP8_DEBUG")) {
        cudaError_t e = cudaDeviceSynchronize();
        uint16_t ho[8];
        cudaMemcpy(ho, output, 16, cudaMemcpyDeviceToHost);
        fprintf(stderr, "  kernel status: %s; out[0..7]:", cudaGetErrorString(e));
        for (int i = 0; i < 8; i++) fprintf(stderr, " %04x", ho[i]);
        fprintf(stderr, "\n");
    }
    return 1;
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
    if (count && head_dim == 128 && rope_half % 4 == 0) {
        size_t warps = (size_t)sequence * heads;
        h3k_qkv_rope_bf16_warp<<<h3_attn_grid_1d(warps * 32), 256>>>(
            (const uint16_t *)qkv->data, (const uint16_t *)q_norm->data,
            (const uint16_t *)k_norm->data, (const uint16_t *)rope_cos->data,
            (const uint16_t *)rope_sin->data, (uint16_t *)query->data,
            (uint16_t *)key->data, (uint16_t *)value->data, sequence, heads,
            rope_half, grouped, epsilon);
    } else if (count)
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
    /* FA2-class tensor-core path: bf16, head_dim == 128, non-causal,
     * sm_80+, >48KB smem opt-in available. */
    /* FP8 (e4m3) flash path: bf16 hd128 non-causal, batch 1, opt-in. */
    if (!naive && bf16 && !causal && head_dim == 128 && batch == 1 &&
        h3_cuda_sdpa_fp8_enabled() && h3_attn_has_fp8_mma(gpu) &&
        h3_cuda_sdpa_fp8(gpu, (const uint16_t *)query->data,
                         (const uint16_t *)key->data,
                         (const uint16_t *)value->data,
                         (uint16_t *)output->data, sequence, heads, scale,
                         head_major_output))
        return h3_cuda_launch_check_kind(gpu, "h3_sdpa", 3);
    if (!naive && bf16 && !causal && h3_attn_has_bf16_mma(gpu) &&
        h3_cuda_sdpa_fa2_enabled() &&
        h3_fa2_fits(gpu, head_major_output, head_dim)) {
        h3_fa2_launch((const uint16_t *)query->data,
                      (const uint16_t *)key->data,
                      (const uint16_t *)value->data, (uint16_t *)output->data,
                      batch, sequence, heads, head_dim, scale,
                      head_major_output);
        return h3_cuda_launch_check_kind(gpu, "h3_sdpa", 3);
    }
    /* f32 through the bf16 FA2 kernel (see h3k_fa2_cast_*). */
    if (!naive && !bf16 && !causal && !head_major_output &&
        h3_attn_has_bf16_mma(gpu) && h3_cuda_sdpa_fa2_enabled() &&
        !h3_cuda_sdpa_f32_exact() && h3_fa2_fits(gpu, 0, head_dim)) {
        size_t bytes = count * sizeof(uint16_t);
        uint16_t *ws = (uint16_t *)h3_cuda_workspace(gpu, 4 * bytes);
        if (ws) {
            uint16_t *q16 = ws, *k16 = ws + count, *v16 = ws + 2 * count,
                     *o16 = ws + 3 * count;
            unsigned blocks = (unsigned)((count + 255) / 256);
            if (blocks > 4096u) blocks = 4096u;
            h3k_fa2_cast_f32_to_bf16<<<blocks, 256>>>(
                (const float *)query->data, q16, count);
            h3k_fa2_cast_f32_to_bf16<<<blocks, 256>>>(
                (const float *)key->data, k16, count);
            h3k_fa2_cast_f32_to_bf16<<<blocks, 256>>>(
                (const float *)value->data, v16, count);
            h3_fa2_launch(q16, k16, v16, o16, batch, sequence, heads, head_dim,
                          scale, 0);
            h3k_fa2_cast_bf16_to_f32<<<blocks, 256>>>(
                o16, (float *)output->data, count);
            return h3_cuda_launch_check_kind(gpu, "h3_sdpa", 3);
        }
        (void)cudaGetLastError(); /* workspace unavailable: exact path */
    }
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
