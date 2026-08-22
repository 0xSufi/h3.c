/* CUDA port of the DiT fused h3_gpu.h ops: AdaLN, gated residuals, fused
 * gate+AdaLN, token pool/expand, and the Euler sampler step. Kernel math
 * mirrors h3_shaders.metal; host-side validation mirrors h3_gpu.m.
 * Correctness-first: naive kernels, F32 accumulation for BF16 storage,
 * round-to-nearest-even at every BF16 output boundary. The Metal 4 int8 /
 * NAX entry points exist here only as permanent error stubs so the model
 * layer links; h3_gpu_has_int8_mlp()/h3_gpu_has_nax_mlp() report 0 on CUDA,
 * so they are unreachable in practice. */
/* h3_gpu.h is a C API: give the declarations C linkage so gcc-compiled
 * model-layer objects link against these nvcc-compiled definitions. */
extern "C" {
#include "h3_gpu.h"
}
#include "h3_cuda_internal.h"

/* ---------------------------------------------------------------- helpers */

static dim3 h3_cuda_grid_1d(uint32_t count) {
    return dim3(count ? (count + 255) / 256 : 1, 1, 1);
}

static dim3 h3_cuda_grid_2d(uint32_t width, uint32_t height) {
    return dim3((width + 15) / 16, (height + 15) / 16, 1);
}

static int h3_cuda_require_bf16(struct h3_gpu *gpu,
                                const h3_gpu_tensor *tensor, size_t elements,
                                const char *label) {
    return h3_cuda_require_elements(gpu, tensor, elements, label) &&
           h3_cuda_require_dtype(gpu, tensor, H3_GPU_BF16, label);
}

static int h3_cuda_require_f32(struct h3_gpu *gpu,
                               const h3_gpu_tensor *tensor, size_t elements,
                               const char *label) {
    return h3_cuda_require_elements(gpu, tensor, elements, label) &&
           h3_cuda_require_dtype(gpu, tensor, H3_GPU_F32, label);
}

static int h3_cuda_require_u32(struct h3_gpu *gpu,
                               const h3_gpu_tensor *tensor, size_t elements,
                               const char *label) {
    return h3_cuda_require_elements(gpu, tensor, elements, label) &&
           h3_cuda_require_dtype(gpu, tensor, H3_GPU_U32, label);
}

/* ------------------------------------------------------- kernel arguments */

struct h3_cuda_adaln_args {
    uint32_t rows, width, slots, shift_slot, scale_slot;
    float epsilon;
};

struct h3_cuda_gate_args {
    uint32_t rows, width, slots, gate_slot;
};

struct h3_cuda_gate_adaln_args {
    uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot;
    float epsilon;
};

struct h3_cuda_adaln_linear_args {
    uint32_t rows, width, output_dim, slots, shift_slot, scale_slot;
    uint32_t has_bias;
};

struct h3_cuda_token_pool_args {
    uint32_t input_offset, original_offset, baseline_offset;
    uint32_t rows, width;
};

struct h3_cuda_token_pool_adaln_args {
    uint32_t input_offset, original_offset, baseline_offset;
    uint32_t rows, width, slots, shift_slot, scale_slot;
    float epsilon;
};

struct h3_cuda_token_expand_args {
    uint32_t original_offset, baseline_offset;
    uint32_t rows, width, exact_prefix_rows;
    float update_scale;
};

struct h3_cuda_token_expand_adaln_args {
    uint32_t original_offset, baseline_offset, rows, width;
    uint32_t exact_prefix_rows, slots, shift_slot, scale_slot;
    float update_scale, epsilon;
};

/* ---------------------------------------------------------------- kernels */

/* One block per row, matching Metal's dispatchThreadgroups row kernels. */
__global__ void h3k_adaln_f32(const float *input, const float *weight,
                              const float *modulation,
                              const uint32_t *row_map, float *output,
                              h3_cuda_adaln_args args) {
    uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const float *x = input + (size_t)row * args.width;
    float local = 0.0f;
    for (uint32_t k = threadIdx.x; k < args.width; k += blockDim.x)
        local = fmaf(x[k], x[k], local);
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)args.width +
                                 args.epsilon);
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        float normalized = x[column] * inverse * weight[column];
        float shift = modulation[base + (size_t)args.shift_slot * args.width +
                                 column];
        float scale = modulation[base + (size_t)args.scale_slot * args.width +
                                 column];
        output[(size_t)row * args.width + column] =
            normalized * (1.0f + scale) + shift;
    }
}

__global__ void h3k_gate_f32(const float *residual, const float *branch,
                             const float *modulation,
                             const uint32_t *row_map, float *output,
                             h3_cuda_gate_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float gate = modulation[base + (size_t)args.gate_slot * args.width +
                            column];
    size_t index = (size_t)row * args.width + column;
    output[index] = residual[index] + branch[index] * gate;
}

__global__ void h3k_adaln_bf16(const uint16_t *input, const uint16_t *weight,
                               const uint16_t *modulation,
                               const uint32_t *row_map, uint16_t *output,
                               h3_cuda_adaln_args args) {
    uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const uint16_t *x = input + (size_t)row * args.width;
    float local = 0.0f;
    for (uint32_t k = threadIdx.x; k < args.width; k += blockDim.x) {
        float value = h3_bf16_to_f32(x[k]);
        local = fmaf(value, value, local);
    }
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)args.width +
                                 args.epsilon);
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + (size_t)args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + (size_t)args.scale_slot * args.width + column]);
        output[(size_t)row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* Emit only the one F32 inverse RMS scalar per row needed by the fused final
 * projection (h3k_adaln_linear_bf16). */
__global__ void h3k_rms_inverse_bf16(const uint16_t *input, float *inverse,
                                     uint32_t rows, uint32_t width,
                                     float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float reductions[256];
    const uint16_t *x = input + (size_t)row * width;
    float local = 0.0f;
    for (uint32_t k = threadIdx.x; k < width; k += blockDim.x) {
        float value = h3_bf16_to_f32(x[k]);
        local = fmaf(value, value, local);
    }
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        inverse[row] = 1.0f / sqrtf(reductions[0] / (float)width + epsilon);
}

/* Naive one-thread-per-output AdaLN + linear fusion. The normalized value is
 * rounded to BF16 before the dot product, retaining the exact arithmetic
 * boundary of AdaLN + linear on Metal. */
__global__ void h3k_adaln_linear_bf16(const uint16_t *input,
                                      const float *inverse,
                                      const uint16_t *norm_weight,
                                      const uint16_t *modulation,
                                      const uint32_t *row_map,
                                      const uint16_t *weight,
                                      const uint16_t *bias,
                                      uint16_t *output,
                                      h3_cuda_adaln_linear_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.output_dim) return;
    float sum = args.has_bias ? h3_bf16_to_f32(bias[column]) : 0.0f;
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    const uint16_t *x = input + (size_t)row * args.width;
    const uint16_t *w = weight + (size_t)column * args.width;
    float row_inverse = inverse[row];
    for (uint32_t k = 0; k < args.width; k++) {
        float value = h3_bf16_to_f32(x[k]);
        float shift = h3_bf16_to_f32(
            modulation[base + (size_t)args.shift_slot * args.width + k]);
        float scale = h3_bf16_to_f32(
            modulation[base + (size_t)args.scale_slot * args.width + k]);
        float normed = value * row_inverse *
            h3_bf16_to_f32(norm_weight[k]);
        uint16_t normalized = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
        sum = fmaf(h3_bf16_to_f32(normalized), h3_bf16_to_f32(w[k]), sum);
    }
    output[(size_t)row * args.output_dim + column] = h3_f32_to_bf16(sum);
}

/* Materializes the AdaLN output for the cuBLAS path of
 * h3_gpu_adaln_linear_bf16. The arithmetic matches the inner loop of
 * h3k_adaln_linear_bf16 exactly, including the BF16 rounding before the dot
 * product, so the following GEMM sees the same inputs as the hand kernel. */
__global__ void h3k_adaln_out_bf16(const uint16_t *input,
                                   const float *inverse,
                                   const uint16_t *norm_weight,
                                   const uint16_t *modulation,
                                   const uint32_t *row_map,
                                   uint16_t *output,
                                   h3_cuda_adaln_linear_args args) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || k >= args.width) return;
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float value = h3_bf16_to_f32(input[(size_t)row * args.width + k]);
    float shift = h3_bf16_to_f32(
        modulation[base + (size_t)args.shift_slot * args.width + k]);
    float scale = h3_bf16_to_f32(
        modulation[base + (size_t)args.scale_slot * args.width + k]);
    float normed = value * inverse[row] * h3_bf16_to_f32(norm_weight[k]);
    output[(size_t)row * args.width + k] =
        h3_f32_to_bf16(normed * (1.0f + scale) + shift);
}

__global__ void h3k_gate_bf16(const uint16_t *residual,
                              const uint16_t *branch,
                              const uint16_t *modulation,
                              const uint32_t *row_map, uint16_t *output,
                              h3_cuda_gate_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float gate = h3_bf16_to_f32(
        modulation[base + (size_t)args.gate_slot * args.width + column]);
    size_t index = (size_t)row * args.width + column;
    float value = h3_bf16_to_f32(residual[index]) +
                  h3_bf16_to_f32(branch[index]) * gate;
    output[index] = h3_f32_to_bf16(value);
}

/* Fused gated residual + AdaLN, one block per row. The gate's BF16 rounding
 * boundary is preserved: the rounded gated row feeds both the gated_residual
 * output and the RMS reduction. width <= 5376 (host-validated). */
__global__ void h3k_gate_adaln_bf16(const uint16_t *residual,
                                    const uint16_t *branch,
                                    const uint16_t *gate_modulation,
                                    const uint32_t *row_map,
                                    const uint16_t *weight,
                                    const uint16_t *norm_modulation,
                                    uint16_t *gated_residual,
                                    uint16_t *output,
                                    h3_cuda_gate_adaln_args args) {
    uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    __shared__ uint16_t gated_values[5376];
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float local = 0.0f;
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        size_t index = (size_t)row * args.width + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + (size_t)args.gate_slot * args.width +
                            column]);
        uint16_t gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local = fmaf(value, value, local);
    }
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)args.width +
                                 args.epsilon);
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + (size_t)args.shift_slot * args.width +
                            column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + (size_t)args.scale_slot * args.width +
                            column]);
        output[(size_t)row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* The pair table is [output row, {first, second}]. Singleton rows repeat
 * their source index so non-video prefixes remain bit exact. */
__global__ void h3k_token_pool_bf16(const uint16_t *input,
                                    const uint2 *pairs, uint16_t *output,
                                    uint16_t *baseline,
                                    const uint32_t *baseline_indices,
                                    uint16_t *original,
                                    h3_cuda_token_pool_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint2 pair = pairs[row];
    uint16_t first = input[(size_t)args.input_offset +
                           (size_t)pair.x * args.width + column];
    original[(size_t)args.original_offset + (size_t)pair.x * args.width +
             column] = first;
    uint16_t pooled = first;
    if (pair.x != pair.y) {
        uint16_t second = input[(size_t)args.input_offset +
                                (size_t)pair.y * args.width + column];
        original[(size_t)args.original_offset + (size_t)pair.y * args.width +
                 column] = second;
        float average = (h3_bf16_to_f32(first) +
                         h3_bf16_to_f32(second)) * 0.5f;
        pooled = h3_f32_to_bf16(average);
    }
    output[(size_t)row * args.width + column] = pooled;
    uint32_t baseline_index = baseline_indices[row];
    if (baseline_index != 0xffffffffu)
        baseline[(size_t)args.baseline_offset +
                 (size_t)baseline_index * args.width + column] = pooled;
}

/* Pool and snapshot the full token grid while immediately producing the
 * first reduced block's attention AdaLN, one block per row. */
__global__ void h3k_token_pool_adaln_bf16(
                                    const uint16_t *input,
                                    const uint2 *pairs, uint16_t *residual,
                                    uint16_t *baseline,
                                    const uint32_t *baseline_indices,
                                    uint16_t *original,
                                    const uint16_t *weight,
                                    const uint16_t *modulation,
                                    const uint32_t *row_map,
                                    uint16_t *output,
                                    h3_cuda_token_pool_adaln_args args) {
    uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    __shared__ uint16_t pooled_values[5376];
    uint2 pair = pairs[row];
    uint32_t baseline_index = baseline_indices[row];
    float local = 0.0f;
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        uint16_t first = input[(size_t)args.input_offset +
                               (size_t)pair.x * args.width + column];
        original[(size_t)args.original_offset + (size_t)pair.x * args.width +
                 column] = first;
        uint16_t pooled = first;
        if (pair.x != pair.y) {
            uint16_t second = input[(size_t)args.input_offset +
                                    (size_t)pair.y * args.width + column];
            original[(size_t)args.original_offset +
                     (size_t)pair.y * args.width + column] = second;
            pooled = h3_f32_to_bf16((h3_bf16_to_f32(first) +
                                     h3_bf16_to_f32(second)) * 0.5f);
        }
        size_t destination = (size_t)row * args.width + column;
        residual[destination] = pooled;
        pooled_values[column] = pooled;
        if (baseline_index != 0xffffffffu)
            baseline[(size_t)args.baseline_offset +
                     (size_t)baseline_index * args.width + column] = pooled;
        float value = h3_bf16_to_f32(pooled);
        local = fmaf(value, value, local);
    }
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)args.width +
                                 args.epsilon);
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        float normalized = h3_bf16_to_f32(pooled_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + (size_t)args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + (size_t)args.scale_slot * args.width + column]);
        output[(size_t)row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* Restore the full grid while retaining the original within-pair detail. The
 * reduced stack contributes only its update relative to the pooled
 * baseline. */
__global__ void h3k_token_expand_delta_bf16(
                                    const uint16_t *original,
                                    const uint16_t *reduced,
                                    const uint16_t *baseline,
                                    const uint32_t *baseline_indices,
                                    const uint32_t *parents,
                                    uint16_t *output,
                                    h3_cuda_token_expand_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t parent = parents[row];
    size_t destination = (size_t)row * args.width + column;
    size_t reduced_index = (size_t)parent * args.width + column;
    if (row < args.exact_prefix_rows) {
        output[destination] = reduced[reduced_index];
        return;
    }
    uint32_t baseline_row = baseline_indices[parent];
    if (baseline_row == 0xffffffffu) {
        output[destination] = reduced[reduced_index];
        return;
    }
    size_t baseline_index = (size_t)args.baseline_offset +
        (size_t)baseline_row * args.width + column;
    float update = h3_bf16_to_f32(reduced[reduced_index]) -
                   h3_bf16_to_f32(baseline[baseline_index]);
    output[destination] = h3_f32_to_bf16(
        h3_bf16_to_f32(original[(size_t)args.original_offset + destination]) +
        args.update_scale * update);
}

/* Fused token expansion + AdaLN, one block per row. The restored BF16 row is
 * kept in shared memory across the reduction; width <= 5376 (host
 * validated). */
__global__ void h3k_token_expand_adaln_bf16(
                                    const uint16_t *original,
                                    const uint16_t *reduced,
                                    const uint16_t *baseline,
                                    const uint32_t *baseline_indices,
                                    const uint32_t *parents,
                                    uint16_t *residual,
                                    const uint16_t *weight,
                                    const uint16_t *modulation,
                                    const uint32_t *row_map,
                                    uint16_t *output,
                                    h3_cuda_token_expand_adaln_args args) {
    uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    __shared__ uint16_t restored_values[5376];
    uint32_t parent = parents[row];
    uint32_t baseline_row = baseline_indices[parent];
    int direct = row < args.exact_prefix_rows ||
                 baseline_row == 0xffffffffu;
    float local = 0.0f;
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        size_t destination = (size_t)row * args.width + column;
        size_t reduced_index = (size_t)parent * args.width + column;
        uint16_t restored = reduced[reduced_index];
        if (!direct) {
            size_t baseline_index = (size_t)args.baseline_offset +
                (size_t)baseline_row * args.width + column;
            float update = h3_bf16_to_f32(restored) -
                           h3_bf16_to_f32(baseline[baseline_index]);
            restored = h3_f32_to_bf16(
                h3_bf16_to_f32(
                    original[(size_t)args.original_offset + destination]) +
                args.update_scale * update);
        }
        restored_values[column] = restored;
        residual[destination] = restored;
        float value = h3_bf16_to_f32(restored);
        local = fmaf(value, value, local);
    }
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)args.width +
                                 args.epsilon);
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    for (uint32_t column = threadIdx.x; column < args.width;
         column += blockDim.x) {
        float normalized = h3_bf16_to_f32(restored_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + (size_t)args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + (size_t)args.scale_slot * args.width + column]);
        output[(size_t)row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

__global__ void h3k_euler_bf16(float *sample, const uint16_t *last,
                               const uint16_t *previous,
                               uint32_t sample_offset, uint32_t elements,
                               float delta, float ratio) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= elements) return;
    float last_value = h3_bf16_to_f32(last[gid]);
    float velocity = fmaf(ratio,
                          last_value - h3_bf16_to_f32(previous[gid]),
                          last_value);
    size_t sample_index = (size_t)sample_offset + gid;
    sample[sample_index] = fmaf(delta, velocity, sample[sample_index]);
}

/* ------------------------------------------------------------------- host */

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_f32(gpu, input, count, "AdaLN input") ||
        !h3_cuda_require_f32(gpu, norm_weight, width, "AdaLN norm") ||
        !h3_cuda_require_f32(gpu, modulation, 1, "AdaLN modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows, "AdaLN row map") ||
        !h3_cuda_require_f32(gpu, output, count, "AdaLN output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (shift_slot >= slots || scale_slot >= slots)
        return h3_cuda_fail(gpu, "AdaLN modulation slot is out of range");
    h3_cuda_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                               epsilon};
    if (count)
        h3k_adaln_f32<<<rows, h3_cuda_row_threads(width)>>>(
            (const float *)input->data, (const float *)norm_weight->data,
            (const float *)modulation->data, (const uint32_t *)row_map->data,
            (float *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_adaln_f32");
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_f32(gpu, residual, count, "gate residual") ||
        !h3_cuda_require_f32(gpu, branch, count, "gate branch") ||
        !h3_cuda_require_f32(gpu, modulation, 1, "gate modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows, "gate row map") ||
        !h3_cuda_require_f32(gpu, output, count, "gate output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (gate_slot >= slots)
        return h3_cuda_fail(gpu, "gate modulation slot is out of range");
    h3_cuda_gate_args args = {rows, width, slots, gate_slot};
    if (count)
        h3k_gate_f32<<<h3_cuda_grid_2d(width, rows), dim3(16, 16)>>>(
            (const float *)residual->data, (const float *)branch->data,
            (const float *)modulation->data, (const uint32_t *)row_map->data,
            (float *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_gate_f32");
}

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    size_t count = (size_t)rows * width;
    if (input_offset > SIZE_MAX - count) {
        h3_cuda_fail(gpu, "AdaLN input offset is out of range");
        return 0;
    }
    if (!h3_cuda_require_bf16(gpu, input, input_offset + count,
                              "AdaLN input") ||
        !h3_cuda_require_bf16(gpu, norm_weight, width, "AdaLN norm") ||
        !h3_cuda_require_bf16(gpu, modulation, 1, "AdaLN modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows, "AdaLN row map") ||
        !h3_cuda_require_bf16(gpu, output, count, "AdaLN output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (shift_slot >= slots || scale_slot >= slots)
        return h3_cuda_fail(gpu, "AdaLN modulation slot is out of range");
    h3_cuda_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                               epsilon};
    if (count)
        h3k_adaln_bf16<<<rows, h3_cuda_row_threads(width)>>>(
            (const uint16_t *)input->data + input_offset,
            (const uint16_t *)norm_weight->data,
            (const uint16_t *)modulation->data,
            (const uint32_t *)row_map->data, (uint16_t *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_adaln_bf16");
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(
        gpu, output, input, 0, norm_weight, modulation, row_map, rows,
        width, slots, shift_slot, scale_slot, epsilon);
}

int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    size_t input_count = (size_t)rows * width;
    size_t weight_count = (size_t)output_dim * width;
    size_t output_count = (size_t)rows * output_dim;
    if (input_offset > SIZE_MAX - input_count) {
        h3_cuda_fail(gpu, "fused final head input offset is out of range");
        return 0;
    }
    if (!h3_cuda_require_bf16(gpu, input, input_offset + input_count,
                              "fused final head input") ||
        !h3_cuda_require_f32(gpu, inverse, rows,
                             "fused final head inverse RMS") ||
        !h3_cuda_require_bf16(gpu, norm_weight, width,
                              "fused final head norm") ||
        !h3_cuda_require_bf16(gpu, modulation, 1,
                              "fused final head modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows,
                             "fused final head row map") ||
        !h3_cuda_require_bf16(gpu, weight, weight_count,
                              "fused final head weight") ||
        !h3_cuda_require_bf16(gpu, output, output_count,
                              "fused final head output") ||
        (bias && !h3_cuda_require_bf16(gpu, bias, output_dim,
                                       "fused final head bias")) ||
        !h3_cuda_require_command(gpu)) return 0;
    if (shift_slot >= slots || scale_slot >= slots)
        return h3_cuda_fail(gpu,
                            "fused final head modulation slot out of range");
    const uint16_t *input_data = (const uint16_t *)input->data + input_offset;
    if (rows && width)
        h3k_rms_inverse_bf16<<<rows, h3_cuda_row_threads(width)>>>(
            input_data, (float *)inverse->data, rows, width, epsilon);
    if (!h3_cuda_launch_check(gpu, "h3_rms_inverse_bf16")) return 0;
    h3_cuda_adaln_linear_args args = {
        rows, width, output_dim, slots, shift_slot, scale_slot,
        bias ? 1u : 0u
    };
    /* cuBLAS path: materialize the BF16-rounded AdaLN output into the
     * workspace, then a BF16 GEMM with bias. */
    if (rows && width && output_dim && gpu->lt && !gpu->no_cublas) {
        uint16_t *normed = (uint16_t *)h3_cuda_workspace(
            gpu, (size_t)rows * width * sizeof(uint16_t));
        if (normed) {
            h3k_adaln_out_bf16<<<h3_cuda_grid_2d(width, rows),
                                 dim3(16, 16)>>>(
                input_data, (const float *)inverse->data,
                (const uint16_t *)norm_weight->data,
                (const uint16_t *)modulation->data,
                (const uint32_t *)row_map->data, normed, args);
            if (cudaGetLastError() == cudaSuccess &&
                h3_cuda_gemm_xwt(gpu, normed, weight->data,
                                 bias ? bias->data : NULL, output->data,
                                 CUDA_R_16BF, CUDA_R_16BF, rows, width,
                                 output_dim))
                return h3_cuda_launch_check(gpu, "h3_adaln_linear_bf16");
        }
    }
    if (output_count)
        h3k_adaln_linear_bf16<<<h3_cuda_grid_2d(output_dim, rows),
                                dim3(16, 16)>>>(
            input_data, (const float *)inverse->data,
            (const uint16_t *)norm_weight->data,
            (const uint16_t *)modulation->data,
            (const uint32_t *)row_map->data, (const uint16_t *)weight->data,
            (const uint16_t *)(bias ? bias->data : input->data),
            (uint16_t *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_adaln_linear_bf16");
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_bf16(gpu, residual, count, "gate residual") ||
        !h3_cuda_require_bf16(gpu, branch, count, "gate branch") ||
        !h3_cuda_require_bf16(gpu, modulation, 1, "gate modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows, "gate row map") ||
        !h3_cuda_require_bf16(gpu, output, count, "gate output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (gate_slot >= slots)
        return h3_cuda_fail(gpu, "gate modulation slot is out of range");
    h3_cuda_gate_args args = {rows, width, slots, gate_slot};
    if (count)
        h3k_gate_bf16<<<h3_cuda_grid_2d(width, rows), dim3(16, 16)>>>(
            (const uint16_t *)residual->data, (const uint16_t *)branch->data,
            (const uint16_t *)modulation->data,
            (const uint32_t *)row_map->data, (uint16_t *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_gate_bf16");
}

int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) {
    size_t elements = (size_t)rows * width;
    if (!rows || !width || width > 5376 || elements > UINT32_MAX)
        return h3_cuda_fail(gpu, "fused gate AdaLN shape is out of range");
    if (gate_slot >= slots || shift_slot >= slots || scale_slot >= slots)
        return h3_cuda_fail(gpu,
                            "fused gate AdaLN modulation slot out of range");
    if (!h3_cuda_require_bf16(gpu, residual, elements,
                              "fused gate AdaLN residual") ||
        !h3_cuda_require_bf16(gpu, branch, elements,
                              "fused gate AdaLN branch") ||
        !h3_cuda_require_bf16(gpu, norm_weight, width,
                              "fused gate AdaLN norm") ||
        !h3_cuda_require_bf16(gpu, gate_modulation, 1,
                              "fused gate AdaLN gate modulation") ||
        !h3_cuda_require_bf16(gpu, norm_modulation, 1,
                              "fused gate AdaLN norm modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows,
                             "fused gate AdaLN row map") ||
        !h3_cuda_require_bf16(gpu, gated_residual, elements,
                              "fused gate AdaLN gated residual") ||
        !h3_cuda_require_bf16(gpu, output, elements,
                              "fused gate AdaLN output") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3_cuda_gate_adaln_args args = {
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon
    };
    h3k_gate_adaln_bf16<<<rows, 256>>>(
        (const uint16_t *)residual->data, (const uint16_t *)branch->data,
        (const uint16_t *)gate_modulation->data,
        (const uint32_t *)row_map->data, (const uint16_t *)norm_weight->data,
        (const uint16_t *)norm_modulation->data,
        (uint16_t *)gated_residual->data, (uint16_t *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_gate_adaln_bf16");
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows || baseline_rows > rows ||
        !width || elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset)
        return h3_cuda_fail(gpu, "token pool shape is out of range");
    if (!input || input->dtype != H3_GPU_BF16 ||
        input_offset > input->elements ||
        input_elements > input->elements - input_offset)
        return h3_cuda_fail(gpu, "token pool input tensor is too small");
    if (!original || original->dtype != H3_GPU_BF16 ||
        original_offset > original->elements ||
        input_elements > original->elements - original_offset)
        return h3_cuda_fail(gpu, "token pool original tensor is too small");
    if (!baseline || baseline->dtype != H3_GPU_BF16 ||
        baseline_offset > baseline->elements ||
        baseline_elements > baseline->elements - baseline_offset)
        return h3_cuda_fail(gpu, "token pool baseline tensor is too small");
    if (!h3_cuda_require_bf16(gpu, output, elements, "token pool output") ||
        !h3_cuda_require_u32(gpu, baseline_indices, rows,
                             "token pool baseline indices") ||
        !h3_cuda_require_u32(gpu, pairs, (size_t)rows * 2,
                             "token pool pairs") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3_cuda_token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width
    };
    h3k_token_pool_bf16<<<h3_cuda_grid_2d(width, rows), dim3(16, 16)>>>(
        (const uint16_t *)input->data, (const uint2 *)pairs->data,
        (uint16_t *)output->data, (uint16_t *)baseline->data,
        (const uint32_t *)baseline_indices->data,
        (uint16_t *)original->data, args);
    return h3_cuda_launch_check(gpu, "h3_token_pool_bf16");
}

int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows || !width || width > 5376 ||
        baseline_rows > rows || elements > UINT32_MAX ||
        input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset)
        return h3_cuda_fail(gpu, "fused token pool shape is out of range");
    if (shift_slot >= slots || scale_slot >= slots)
        return h3_cuda_fail(gpu,
                            "fused token pool modulation slot out of range");
    if (!input || input->dtype != H3_GPU_BF16 ||
        input_offset > input->elements ||
        input_elements > input->elements - input_offset)
        return h3_cuda_fail(gpu, "fused token pool input tensor is too small");
    if (!original || original->dtype != H3_GPU_BF16 ||
        original_offset > original->elements ||
        input_elements > original->elements - original_offset)
        return h3_cuda_fail(gpu,
                            "fused token pool original tensor is too small");
    if (!baseline || baseline->dtype != H3_GPU_BF16 ||
        baseline_offset > baseline->elements ||
        baseline_elements > baseline->elements - baseline_offset)
        return h3_cuda_fail(gpu,
                            "fused token pool baseline tensor is too small");
    if (!h3_cuda_require_bf16(gpu, residual, elements,
                              "fused token pool residual") ||
        !h3_cuda_require_u32(gpu, baseline_indices, rows,
                             "fused token pool baseline indices") ||
        !h3_cuda_require_u32(gpu, pairs, (size_t)rows * 2,
                             "fused token pool pairs") ||
        !h3_cuda_require_bf16(gpu, norm_weight, width,
                              "fused token pool norm") ||
        !h3_cuda_require_bf16(gpu, modulation, 1,
                              "fused token pool modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows,
                             "fused token pool row map") ||
        !h3_cuda_require_bf16(gpu, output, elements,
                              "fused token pool AdaLN output") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3_cuda_token_pool_adaln_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width, slots, shift_slot,
        scale_slot, epsilon
    };
    h3k_token_pool_adaln_bf16<<<rows, 256>>>(
        (const uint16_t *)input->data, (const uint2 *)pairs->data,
        (uint16_t *)residual->data, (uint16_t *)baseline->data,
        (const uint32_t *)baseline_indices->data,
        (uint16_t *)original->data, (const uint16_t *)norm_weight->data,
        (const uint16_t *)modulation->data, (const uint32_t *)row_map->data,
        (uint16_t *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_token_pool_adaln_bf16");
}

int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale) {
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows ||
        baseline_rows > reduced_rows || !width ||
        exact_prefix_rows > reduced_rows || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset)
        return h3_cuda_fail(gpu, "token expand shape is out of range");
    if (!original || original->dtype != H3_GPU_BF16 ||
        original_offset > original->elements ||
        elements > original->elements - original_offset)
        return h3_cuda_fail(gpu, "token expand original tensor is too small");
    if (!baseline || baseline->dtype != H3_GPU_BF16 ||
        baseline_offset > baseline->elements ||
        baseline_elements > baseline->elements - baseline_offset)
        return h3_cuda_fail(gpu, "token expand baseline tensor is too small");
    if (!h3_cuda_require_bf16(gpu, output, elements,
                              "token expand output") ||
        !h3_cuda_require_bf16(gpu, reduced, reduced_elements,
                              "token expand reduced") ||
        !h3_cuda_require_u32(gpu, baseline_indices, reduced_rows,
                             "token expand baseline indices") ||
        !h3_cuda_require_u32(gpu, parents, rows, "token expand parents") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3_cuda_token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale
    };
    h3k_token_expand_delta_bf16<<<h3_cuda_grid_2d(width, rows),
                                  dim3(16, 16)>>>(
        (const uint16_t *)original->data, (const uint16_t *)reduced->data,
        (const uint16_t *)baseline->data,
        (const uint32_t *)baseline_indices->data,
        (const uint32_t *)parents->data, (uint16_t *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_token_expand_delta_bf16");
}

int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows || !width ||
        width > 5376 || baseline_rows > reduced_rows ||
        exact_prefix_rows > reduced_rows || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset)
        return h3_cuda_fail(gpu, "fused token AdaLN shape is out of range");
    if (shift_slot >= slots || scale_slot >= slots)
        return h3_cuda_fail(gpu,
                            "fused token AdaLN modulation slot out of range");
    if (!original || original->dtype != H3_GPU_BF16 ||
        original_offset > original->elements ||
        elements > original->elements - original_offset)
        return h3_cuda_fail(gpu,
                            "fused token AdaLN original tensor is too small");
    if (!baseline || baseline->dtype != H3_GPU_BF16 ||
        baseline_offset > baseline->elements ||
        baseline_elements > baseline->elements - baseline_offset)
        return h3_cuda_fail(gpu,
                            "fused token AdaLN baseline tensor is too small");
    if (!h3_cuda_require_bf16(gpu, reduced, reduced_elements,
                              "fused token AdaLN reduced input") ||
        !h3_cuda_require_u32(gpu, baseline_indices, reduced_rows,
                             "fused token AdaLN baseline indices") ||
        !h3_cuda_require_u32(gpu, parents, rows,
                             "fused token AdaLN parents") ||
        !h3_cuda_require_bf16(gpu, residual, elements,
                              "fused token AdaLN residual") ||
        !h3_cuda_require_bf16(gpu, norm_weight, width,
                              "fused token AdaLN norm") ||
        !h3_cuda_require_bf16(gpu, modulation, 1,
                              "fused token AdaLN modulation") ||
        !h3_cuda_require_u32(gpu, row_map, rows,
                             "fused token AdaLN row map") ||
        !h3_cuda_require_bf16(gpu, output, elements,
                              "fused token AdaLN output") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3_cuda_token_expand_adaln_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset, rows, width,
        exact_prefix_rows, slots, shift_slot, scale_slot,
        update_scale, epsilon
    };
    h3k_token_expand_adaln_bf16<<<rows, 256>>>(
        (const uint16_t *)original->data, (const uint16_t *)reduced->data,
        (const uint16_t *)baseline->data,
        (const uint32_t *)baseline_indices->data,
        (const uint32_t *)parents->data, (uint16_t *)residual->data,
        (const uint16_t *)norm_weight->data,
        (const uint16_t *)modulation->data, (const uint32_t *)row_map->data,
        (uint16_t *)output->data, args);
    return h3_cuda_launch_check(gpu, "h3_token_expand_adaln_bf16");
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    if (!sample || sample->dtype != H3_GPU_F32 ||
        sample_offset > sample->elements ||
        elements > sample->elements - sample_offset ||
        sample_offset > UINT32_MAX || elements > UINT32_MAX - sample_offset)
        return h3_cuda_fail(gpu, "Euler sample range is out of range");
    if (!h3_cuda_require_bf16(gpu, last, elements, "Euler last velocity") ||
        !h3_cuda_require_bf16(gpu, previous, elements,
                              "Euler previous velocity") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_euler_bf16<<<h3_cuda_grid_1d(elements), 256>>>(
            (float *)sample->data, (const uint16_t *)last->data,
            (const uint16_t *)previous->data, (uint32_t)sample_offset,
            elements, delta, ratio);
    return h3_cuda_launch_check(gpu, "h3_euler_bf16");
}

/* --------------------------------- Metal 4 TensorOps / NAX error stubs */
/* Unreachable on CUDA (h3_gpu_has_int8_mlp()/h3_gpu_has_nax_mlp() return 0)
 * but required for linking. */

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    (void)output; (void)scales; (void)input; (void)rows; (void)columns;
    return h3_cuda_fail((struct h3_gpu *)gpu,
        "h3_gpu_quantize_weight_int8 requires Metal 4 TensorOps and is not "
        "available in the CUDA backend");
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    (void)output; (void)quantized_input; (void)input_scales; (void)input;
    (void)weight; (void)weight_scales; (void)rows; (void)input_dim;
    (void)output_dim; (void)use_slower_uncached_int8_scales;
    return h3_cuda_fail((struct h3_gpu *)gpu,
        "h3_gpu_linear_int8_bf16 requires Metal 4 TensorOps and is not "
        "available in the CUDA backend");
}

int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) {
    (void)output; (void)quantized_input; (void)input_scales; (void)input;
    (void)weight; (void)weight_scales; (void)rows; (void)heads;
    (void)head_dim; (void)output_dim;
    return h3_cuda_fail((struct h3_gpu *)gpu,
        "h3_gpu_linear_int8_head_major_bf16 requires Metal 4 TensorOps and "
        "is not available in the CUDA backend");
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized) {
    (void)output; (void)activated; (void)quantized_activation;
    (void)activation_scales; (void)input; (void)fc1_weight; (void)fc1_scales;
    (void)fc2_weight; (void)fc2_scales; (void)fc1_bf16; (void)fc2_bf16;
    (void)rows; (void)input_dim; (void)hidden_dim; (void)output_dim;
    (void)use_slower_grouped_quantizer; (void)use_slower_dynamic_fc1_k;
    (void)use_int8_row_fc2; (void)input_is_quantized;
    return h3_cuda_fail((struct h3_gpu *)gpu,
        "h3_gpu_mlp_int8_bf16 requires Metal 4 TensorOps and is not "
        "available in the CUDA backend");
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    (void)output; (void)activated; (void)input; (void)fc1_weight;
    (void)fc2_weight; (void)rows; (void)input_dim; (void)hidden_dim;
    (void)output_dim;
    return h3_cuda_fail((struct h3_gpu *)gpu,
        "h3_gpu_mlp_nax_bf16 requires Metal 4 TensorOps and is not "
        "available in the CUDA backend");
}

int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales) {
    (void)query; (void)key; (void)value; (void)quantized_input;
    (void)input_scales; (void)input; (void)weight; (void)weight_scales;
    (void)q_norm; (void)k_norm; (void)rope_cos; (void)rope_sin; (void)rows;
    (void)input_dim; (void)heads; (void)head_dim; (void)rope_half;
    (void)epsilon; (void)input_is_quantized; (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms; (void)use_slower_uncached_int8_scales;
    return h3_cuda_fail((struct h3_gpu *)gpu,
        "h3_gpu_grouped_qkv_linear_rope_int8 requires Metal 4 TensorOps and "
        "is not available in the CUDA backend");
}

int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    (void)gated_residual; (void)quantized_output; (void)quantized_scales;
    (void)residual; (void)branch; (void)norm_weight; (void)gate_modulation;
    (void)norm_modulation; (void)row_map; (void)rows; (void)padded_rows;
    (void)width; (void)slots; (void)gate_slot; (void)shift_slot;
    (void)scale_slot; (void)epsilon;
    return h3_cuda_fail((struct h3_gpu *)gpu,
        "h3_gpu_gate_adaln_quantize_int8 requires Metal 4 TensorOps and is "
        "not available in the CUDA backend");
}
