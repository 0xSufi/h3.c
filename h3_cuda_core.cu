/* CUDA port of the core h3_gpu.h ops: elementwise activations, norms, and
 * linear layers. Kernel math mirrors h3_shaders.metal; host-side validation
 * and stats-counter behavior mirror h3_gpu.m. Correctness-first: naive
 * kernels, F32 accumulation for BF16 storage, safe libm (no fast-math
 * intrinsics). */
/* h3_gpu.h is a C API: give the declarations C linkage so gcc-compiled
 * model-layer objects link against these nvcc-compiled definitions. */
extern "C" {
#include "h3_gpu.h"
}
#include "h3_cuda_internal.h"

/* ---------------------------------------------------------------- kernels */

__global__ void h3k_silu_f32(const float *input, float *output,
                             uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float value = input[gid];
    output[gid] = value / (1.0f + expf(-value));
}

__global__ void h3k_silu_bf16(const uint16_t *input, uint16_t *output,
                              uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float value = h3_bf16_to_f32(input[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + expf(-value)));
}

__global__ void h3k_geglu_f32(const float *gate, const float *linear,
                              float *output, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = gate[gid];
    float cube = x * x * x;
    float gelu = 0.5f * x *
        (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
    output[gid] = gelu * linear[gid];
}

__global__ void h3k_clip_f32(const float *input, float *output,
                             uint32_t count, float minimum, float maximum) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = fminf(fmaxf(input[gid], minimum), maximum);
}

__global__ void h3k_add_scaled_f32(const float *left, const float *right,
                                   float *output, uint32_t count,
                                   float left_scale, float right_scale) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = left[gid] * left_scale + right[gid] * right_scale;
}

__global__ void h3k_scale_add_f32(const float *residual, const float *branch,
                                  const float *scale, float *output,
                                  uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= width) return;
    size_t index = (size_t)row * width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

__global__ void h3k_add_bf16(const uint16_t *left, const uint16_t *right,
                             uint16_t *output, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) +
                                 h3_bf16_to_f32(right[gid]));
}

__global__ void h3k_sub_bf16(const uint16_t *left, const uint16_t *right,
                             uint16_t *output, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) -
                                 h3_bf16_to_f32(right[gid]));
}

__global__ void h3k_silu_mul_bf16(const uint16_t *gate, const uint16_t *up,
                                  uint16_t *output, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float value = h3_bf16_to_f32(gate[gid]);
    float other = h3_bf16_to_f32(up[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + expf(-value)) * other);
}

__global__ void h3k_swiglu_f32(const float *fused, float *output,
                               uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= width) return;
    size_t base = (size_t)row * width * 2;
    float gate = fused[base + column];
    float up = fused[base + width + column];
    output[(size_t)row * width + column] =
        gate / (1.0f + expf(-gate)) * up;
}

__global__ void h3k_swiglu_bf16(const uint16_t *fused, uint16_t *output,
                                uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= width) return;
    size_t base = (size_t)row * width * 2;
    float gate = h3_bf16_to_f32(fused[base + column]);
    float up = h3_bf16_to_f32(fused[base + width + column]);
    output[(size_t)row * width + column] =
        h3_f32_to_bf16(gate / (1.0f + expf(-gate)) * up);
}

__global__ void h3k_embedding_bf16(const uint16_t *weight,
                                   const uint32_t *token_ids,
                                   uint16_t *output, uint32_t tokens,
                                   uint32_t vocab_size, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t token = blockIdx.y * blockDim.y + threadIdx.y;
    if (token >= tokens || column >= width) return;
    uint32_t identifier = token_ids[token];
    output[(size_t)token * width + column] = identifier < vocab_size ?
        weight[(size_t)identifier * width + column] : (uint16_t)0;
}

/* Polynomial erf from h3_shaders.metal (h3_erf_approx). */
__device__ __forceinline__ float h3_cuda_erf_approx(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = fabsf(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t) +
                           1.421413741f) * t - 0.284496736f) * t +
                           0.254829592f) * t;
    return sign * (1.0f - polynomial * expf(-x * x));
}

__global__ void h3k_gelu_bf16(const uint16_t *input, uint16_t *output,
                              uint32_t count, uint32_t approximate) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float value = h3_bf16_to_f32(input[gid]);
    float activated;
    if (approximate) {
        float inner = 0.7978845608028654f *
            (value + 0.044715f * value * value * value);
        activated = inner <= -10.0f ? 0.0f :
                    inner >= 10.0f ? value :
                    0.5f * value * (1.0f + tanhf(inner));
    } else {
        activated = value <= -10.0f ? 0.0f :
                    value >= 10.0f ? value :
                    0.5f * value *
                    (1.0f + h3_cuda_erf_approx(value * 0.7071067811865475f));
    }
    output[gid] = h3_f32_to_bf16(activated);
}

/* One block per row, matching Metal's dispatchThreadgroups row kernels. */
__global__ void h3k_rms_norm_f32(const float *input, const float *weight,
                                 float *output, uint32_t rows, uint32_t width,
                                 float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float reductions[256];
    const float *x = input + (size_t)row * width;
    float local = 0.0f;
    for (uint32_t k = threadIdx.x; k < width; k += blockDim.x)
        local = fmaf(x[k], x[k], local);
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x)
        output[(size_t)row * width + column] =
            x[column] * inverse * weight[column];
}

__global__ void h3k_rms_norm_bf16(const uint16_t *input,
                                  const uint16_t *weight, uint16_t *output,
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
    float inverse = 1.0f / sqrtf(reductions[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse;
        output[(size_t)row * width + column] =
            h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
    }
}

__global__ void h3k_layer_norm_f32(const float *input, const float *weight,
                                   const float *bias, float *output,
                                   uint32_t rows, uint32_t width,
                                   float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float reductions[256];
    const float *x = input + (size_t)row * width;
    float local = 0.0f;
    for (uint32_t k = threadIdx.x; k < width; k += blockDim.x)
        local += x[k];
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = reductions[0] / (float)width;
    __syncthreads();
    local = 0.0f;
    for (uint32_t k = threadIdx.x; k < width; k += blockDim.x) {
        float centered = x[k] - mean;
        local = fmaf(centered, centered, local);
    }
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x)
        output[(size_t)row * width + column] =
            (x[column] - mean) * inverse * weight[column] + bias[column];
}

__global__ void h3k_layer_norm_bf16(const uint16_t *input,
                                    const uint16_t *weight,
                                    const uint16_t *bias, uint16_t *output,
                                    uint32_t rows, uint32_t width,
                                    float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float reductions[256];
    const uint16_t *x = input + (size_t)row * width;
    float local = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x)
        local += h3_bf16_to_f32(x[column]);
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = reductions[0] / (float)width;
    __syncthreads();
    float local_square = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float centered = h3_bf16_to_f32(x[column]) - mean;
        local_square = fmaf(centered, centered, local_square);
    }
    reductions[threadIdx.x] = local_square;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = (h3_bf16_to_f32(x[column]) - mean) * inverse;
        float value = fmaf(normalized, h3_bf16_to_f32(weight[column]),
                           h3_bf16_to_f32(bias[column]));
        output[(size_t)row * width + column] = h3_f32_to_bf16(value);
    }
}

/* One block per outer row: shared-memory tree reduction over inner for
 * ||v||, then a parallel scale pass (the rms_norm kernel shape). The math is
 * the serial kernel's, reordered: out = v * g/||v|| with ||v|| = sqrt(sum of
 * squares), no epsilon. */
__global__ void h3k_weight_norm_f32(const float *vector,
                                    const float *magnitude, float *output,
                                    uint32_t outer, uint32_t inner) {
    uint32_t row = blockIdx.x;
    if (row >= outer) return;
    __shared__ float reductions[256];
    size_t base = (size_t)row * inner;
    const float *v = vector + base;
    float local = 0.0f;
    for (uint32_t index = threadIdx.x; index < inner; index += blockDim.x) {
        float value = v[index];
        local = fmaf(value, value, local);
    }
    reductions[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = magnitude[row] / sqrtf(reductions[0]);
    for (uint32_t index = threadIdx.x; index < inner; index += blockDim.x)
        output[base + index] = v[index] * scale;
}

/* One thread owns a complete head, in-place, matching Metal. */
__global__ void h3k_head_rms_norm_bf16(uint16_t *tensor,
                                       const uint16_t *weight,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, float epsilon) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= sequence || head >= heads) return;
    size_t base = ((size_t)row * heads + head) * head_dim;
    float sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        sum = fmaf(value, value, sum);
    }
    float inverse = 1.0f / sqrtf(sum / (float)head_dim + epsilon);
    for (uint32_t d = 0; d < head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        tensor[base + d] = h3_f32_to_bf16(
            value * inverse * h3_bf16_to_f32(weight[d]));
    }
}

/* Naive one-thread-per-output matmuls (correctness first). */
__global__ void h3k_linear_f32(const float *input, const float *weight,
                               const float *bias, float *output,
                               uint32_t rows, uint32_t input_dim,
                               uint32_t output_dim, uint32_t has_bias) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= output_dim) return;
    float sum = has_bias ? bias[column] : 0.0f;
    const float *x = input + (size_t)row * input_dim;
    const float *w = weight + (size_t)column * input_dim;
    for (uint32_t k = 0; k < input_dim; k++) sum = fmaf(x[k], w[k], sum);
    output[(size_t)row * output_dim + column] = sum;
}

__global__ void h3k_linear_bf16(const uint16_t *input, const uint16_t *weight,
                                const uint16_t *bias, uint16_t *output,
                                uint32_t rows, uint32_t input_dim,
                                uint32_t output_dim, uint32_t has_bias) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= output_dim) return;
    float sum = has_bias ? h3_bf16_to_f32(bias[column]) : 0.0f;
    const uint16_t *x = input + (size_t)row * input_dim;
    const uint16_t *w = weight + (size_t)column * input_dim;
    for (uint32_t k = 0; k < input_dim; k++)
        sum = fmaf(h3_bf16_to_f32(x[k]), h3_bf16_to_f32(w[k]), sum);
    output[(size_t)row * output_dim + column] = h3_f32_to_bf16(sum);
}

/* F32 input/weight/bias, BF16 output: the fused patch projection. */
__global__ void h3k_linear_f32_bf16(const float *input, const float *weight,
                                    const float *bias, uint16_t *output,
                                    uint32_t rows, uint32_t input_dim,
                                    uint32_t output_dim, uint32_t has_bias) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= output_dim) return;
    float sum = has_bias ? bias[column] : 0.0f;
    const float *x = input + (size_t)row * input_dim;
    const float *w = weight + (size_t)column * input_dim;
    for (uint32_t k = 0; k < input_dim; k++) sum = fmaf(x[k], w[k], sum);
    output[(size_t)row * output_dim + column] = h3_f32_to_bf16(sum);
}

__global__ void h3k_linear_f32_bf16_map(const float *input,
                                        const float *weight,
                                        const float *bias, uint16_t *output,
                                        const uint32_t *row_map,
                                        uint32_t rows, uint32_t input_dim,
                                        uint32_t output_dim,
                                        uint32_t has_bias) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= output_dim) return;
    float sum = has_bias ? bias[column] : 0.0f;
    const float *x = input + (size_t)row * input_dim;
    const float *w = weight + (size_t)column * input_dim;
    for (uint32_t k = 0; k < input_dim; k++) sum = fmaf(x[k], w[k], sum);
    output[(size_t)row_map[row] * output_dim + column] = h3_f32_to_bf16(sum);
}

/* MLP (h3_gpu_mlp_bf16): fused = input @ fc1^T is [rows, 2*hidden]; the
 * first hidden columns are the gate, the second hidden the up projection;
 * activated = silu(gate) * up; output = bf16(activated @ fc2^T). No biases.
 * The hand path keeps the intermediate F32, matching MPSGraph (which only
 * casts the final result to BF16); the cuBLAS path in h3_gpu_mlp_bf16 rounds
 * it to bf16 once, see the note there. */
__global__ void h3k_mlp_fc1_swiglu_bf16(const uint16_t *input,
                                        const uint16_t *fc1_weight,
                                        float *activated, uint32_t rows,
                                        uint32_t input_dim,
                                        uint32_t hidden_dim) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= hidden_dim) return;
    const uint16_t *x = input + (size_t)row * input_dim;
    const uint16_t *gate_w = fc1_weight + (size_t)column * input_dim;
    const uint16_t *up_w = fc1_weight + (size_t)(hidden_dim + column) *
                                         input_dim;
    float gate = 0.0f, up = 0.0f;
    for (uint32_t k = 0; k < input_dim; k++) {
        float value = h3_bf16_to_f32(x[k]);
        gate = fmaf(value, h3_bf16_to_f32(gate_w[k]), gate);
        up = fmaf(value, h3_bf16_to_f32(up_w[k]), up);
    }
    activated[(size_t)row * hidden_dim + column] =
        gate / (1.0f + expf(-gate)) * up;
}

__global__ void h3k_mlp_fc2_bf16(const float *activated,
                                 const uint16_t *fc2_weight, uint16_t *output,
                                 uint32_t rows, uint32_t hidden_dim,
                                 uint32_t output_dim) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= output_dim) return;
    const float *x = activated + (size_t)row * hidden_dim;
    const uint16_t *w = fc2_weight + (size_t)column * hidden_dim;
    float sum = 0.0f;
    for (uint32_t k = 0; k < hidden_dim; k++)
        sum = fmaf(x[k], h3_bf16_to_f32(w[k]), sum);
    output[(size_t)row * output_dim + column] = h3_f32_to_bf16(sum);
}

/* Elementwise casts around the cuBLAS GEMM paths (grow-only workspace holds
 * the F32 intermediates). Grid-stride loops: the launch caps the block count
 * at what fills the GPU (see h3_cuda_cast_grid) instead of scheduling one
 * block per 256 elements, which means tens of millions of blocks on the
 * multi-hundred-megabyte VAE feature maps. */
__global__ void h3k_cast_bf16_to_f32(const uint16_t *input, float *output,
                                     size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride)
        output[index] = h3_bf16_to_f32(input[index]);
}

__global__ void h3k_cast_f32_to_bf16(const float *input, uint16_t *output,
                                     size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride)
        output[index] = h3_f32_to_bf16(input[index]);
}

/* Cast with a row scatter, for h3_gpu_patch_linear_bf16_map: input row r is
 * written to output row row_map[r]. */
__global__ void h3k_cast_f32_to_bf16_map(const float *input, uint16_t *output,
                                         const uint32_t *row_map,
                                         uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= width) return;
    output[(size_t)row_map[row] * width + column] =
        h3_f32_to_bf16(input[(size_t)row * width + column]);
}

/* ------------------------------------------------------------ host launch */

static dim3 h3_cuda_grid_1d(uint32_t count) {
    return dim3(count ? (count + 255) / 256 : 1, 1, 1);
}

/* Grid for the grid-stride cast kernels: enough 256-thread blocks to fill
 * the device (16 blocks = 4096 threads per SM, two full waves), capped there
 * so huge tensors do not pay the scheduling cost of one block per 256
 * elements. The SM count is queried once and cached. */
static dim3 h3_cuda_cast_grid(size_t count) {
    static int fill_blocks;
    if (!fill_blocks) {
        int device = 0, sms = 0;
        cudaGetDevice(&device);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                   device) != cudaSuccess || sms < 1)
            sms = 1;
        fill_blocks = sms * 16;
    }
    size_t wanted = count ? (count + 255) / 256 : 1;
    return dim3(wanted < (size_t)fill_blocks ? (unsigned)wanted
                                             : (unsigned)fill_blocks, 1, 1);
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

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_cuda_require_elements(gpu, input, elements, "SiLU input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "SiLU input") ||
        !h3_cuda_require_elements(gpu, output, elements, "SiLU output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "SiLU output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_silu_f32<<<h3_cuda_grid_1d(elements), 256>>>(
            (const float *)input->data, (float *)output->data, elements);
    return h3_cuda_launch_check(gpu, "h3_silu_f32");
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_cuda_require_bf16(gpu, input, elements, "SiLU input") ||
        !h3_cuda_require_bf16(gpu, output, elements, "SiLU output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_silu_bf16<<<h3_cuda_grid_1d(elements), 256>>>(
            (const uint16_t *)input->data, (uint16_t *)output->data,
            elements);
    return h3_cuda_launch_check(gpu, "h3_silu_bf16");
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    if (!h3_cuda_require_bf16(gpu, input, elements, "GELU input") ||
        !h3_cuda_require_bf16(gpu, output, elements, "GELU output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_gelu_bf16<<<h3_cuda_grid_1d(elements), 256>>>(
            (const uint16_t *)input->data, (uint16_t *)output->data,
            elements, approximate ? 1u : 0u);
    return h3_cuda_launch_check(gpu, "h3_gelu_bf16");
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    if (!h3_cuda_require_elements(gpu, gate, elements, "GeGLU gate") ||
        !h3_cuda_require_elements(gpu, linear, elements, "GeGLU linear") ||
        !h3_cuda_require_elements(gpu, output, elements, "GeGLU output") ||
        !h3_cuda_require_dtype(gpu, gate, H3_GPU_F32, "GeGLU gate") ||
        !h3_cuda_require_dtype(gpu, linear, H3_GPU_F32, "GeGLU linear") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "GeGLU output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_geglu_f32<<<h3_cuda_grid_1d(elements), 256>>>(
            (const float *)gate->data, (const float *)linear->data,
            (float *)output->data, elements);
    return h3_cuda_launch_check(gpu, "h3_geglu_f32");
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    if (!(minimum <= maximum)) {
        h3_cuda_fail(gpu, "clip minimum exceeds maximum");
        return 0;
    }
    if (!h3_cuda_require_elements(gpu, input, elements, "clip input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "clip input") ||
        !h3_cuda_require_elements(gpu, output, elements, "clip output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "clip output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_clip_f32<<<h3_cuda_grid_1d(elements), 256>>>(
            (const float *)input->data, (float *)output->data, elements,
            minimum, maximum);
    return h3_cuda_launch_check(gpu, "h3_clip_f32");
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    if (!h3_cuda_require_elements(gpu, left, elements, "scaled-add left") ||
        !h3_cuda_require_dtype(gpu, left, H3_GPU_F32, "scaled-add left") ||
        !h3_cuda_require_elements(gpu, right, elements, "scaled-add right") ||
        !h3_cuda_require_dtype(gpu, right, H3_GPU_F32, "scaled-add right") ||
        !h3_cuda_require_elements(gpu, output, elements, "scaled-add output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "scaled-add output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_add_scaled_f32<<<h3_cuda_grid_1d(elements), 256>>>(
            (const float *)left->data, (const float *)right->data,
            (float *)output->data, elements, left_scale, right_scale);
    return h3_cuda_launch_check(gpu, "h3_add_scaled_f32");
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_elements(gpu, residual, count, "scale-add residual") ||
        !h3_cuda_require_dtype(gpu, residual, H3_GPU_F32, "scale-add residual") ||
        !h3_cuda_require_elements(gpu, branch, count, "scale-add branch") ||
        !h3_cuda_require_dtype(gpu, branch, H3_GPU_F32, "scale-add branch") ||
        !h3_cuda_require_elements(gpu, scale, width, "scale-add scale") ||
        !h3_cuda_require_dtype(gpu, scale, H3_GPU_F32, "scale-add scale") ||
        !h3_cuda_require_elements(gpu, output, count, "scale-add output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "scale-add output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (count)
        h3k_scale_add_f32<<<h3_cuda_grid_2d(width, rows), dim3(16, 16)>>>(
            (const float *)residual->data, (const float *)branch->data,
            (const float *)scale->data, (float *)output->data, rows, width);
    return h3_cuda_launch_check(gpu, "h3_scale_add_f32");
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_cuda_require_bf16(gpu, left, elements, "add left") ||
        !h3_cuda_require_bf16(gpu, right, elements, "add right") ||
        !h3_cuda_require_bf16(gpu, output, elements, "add output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_add_bf16<<<h3_cuda_grid_1d(elements), 256>>>(
            (const uint16_t *)left->data, (const uint16_t *)right->data,
            (uint16_t *)output->data, elements);
    return h3_cuda_launch_check(gpu, "h3_add_bf16");
}

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_cuda_require_bf16(gpu, left, elements, "subtract left") ||
        !h3_cuda_require_bf16(gpu, right, elements, "subtract right") ||
        !h3_cuda_require_bf16(gpu, output, elements, "subtract output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_sub_bf16<<<h3_cuda_grid_1d(elements), 256>>>(
            (const uint16_t *)left->data, (const uint16_t *)right->data,
            (uint16_t *)output->data, elements);
    return h3_cuda_launch_check(gpu, "h3_sub_bf16");
}

int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    if (!h3_cuda_require_bf16(gpu, gate, elements, "SiLU gate") ||
        !h3_cuda_require_bf16(gpu, up, elements, "SiLU up") ||
        !h3_cuda_require_bf16(gpu, output, elements, "SiLU product") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (elements)
        h3k_silu_mul_bf16<<<h3_cuda_grid_1d(elements), 256>>>(
            (const uint16_t *)gate->data, (const uint16_t *)up->data,
            (uint16_t *)output->data, elements);
    return h3_cuda_launch_check(gpu, "h3_silu_mul_bf16");
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    if (!h3_cuda_require_elements(gpu, fused, (size_t)rows * width * 2,
                                  "SwiGLU input") ||
        !h3_cuda_require_elements(gpu, output, (size_t)rows * width,
                                  "SwiGLU output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (rows && width)
        h3k_swiglu_f32<<<h3_cuda_grid_2d(width, rows), dim3(16, 16)>>>(
            (const float *)fused->data, (float *)output->data, rows, width);
    return h3_cuda_launch_check(gpu, "h3_swiglu_f32");
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    if (!h3_cuda_require_bf16(gpu, fused, (size_t)rows * width * 2,
                              "SwiGLU input") ||
        !h3_cuda_require_bf16(gpu, output, (size_t)rows * width,
                              "SwiGLU output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (rows && width)
        h3k_swiglu_bf16<<<h3_cuda_grid_2d(width, rows), dim3(16, 16)>>>(
            (const uint16_t *)fused->data, (uint16_t *)output->data,
            rows, width);
    return h3_cuda_launch_check(gpu, "h3_swiglu_bf16");
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    if (!h3_cuda_require_bf16(gpu, weight, (size_t)vocab_size * width,
                              "embedding weight") ||
        !h3_cuda_require_elements(gpu, token_ids, tokens, "token IDs") ||
        !h3_cuda_require_dtype(gpu, token_ids, H3_GPU_U32, "token IDs") ||
        !h3_cuda_require_bf16(gpu, output, (size_t)tokens * width,
                              "embedding output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (tokens && width)
        h3k_embedding_bf16<<<h3_cuda_grid_2d(width, tokens), dim3(16, 16)>>>(
            (const uint16_t *)weight->data, (const uint32_t *)token_ids->data,
            (uint16_t *)output->data, tokens, vocab_size, width);
    return h3_cuda_launch_check(gpu, "h3_embedding_bf16");
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_elements(gpu, input, count, "RMSNorm input") ||
        !h3_cuda_require_elements(gpu, weight, width, "RMSNorm weight") ||
        !h3_cuda_require_elements(gpu, output, count, "RMSNorm output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (rows && width)
        h3k_rms_norm_f32<<<rows, h3_cuda_row_threads(width)>>>(
            (const float *)input->data, (const float *)weight->data,
            (float *)output->data, rows, width, epsilon);
    return h3_cuda_launch_check(gpu, "h3_rms_norm_f32");
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_bf16(gpu, input, count, "RMSNorm input") ||
        !h3_cuda_require_bf16(gpu, weight, width, "RMSNorm weight") ||
        !h3_cuda_require_bf16(gpu, output, count, "RMSNorm output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (rows && width)
        h3k_rms_norm_bf16<<<rows, h3_cuda_row_threads(width)>>>(
            (const uint16_t *)input->data, (const uint16_t *)weight->data,
            (uint16_t *)output->data, rows, width, epsilon);
    return h3_cuda_launch_check(gpu, "h3_rms_norm_bf16");
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_elements(gpu, input, count, "LayerNorm input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "LayerNorm input") ||
        !h3_cuda_require_elements(gpu, weight, width, "LayerNorm weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32, "LayerNorm weight") ||
        !h3_cuda_require_elements(gpu, bias, width, "LayerNorm bias") ||
        !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32, "LayerNorm bias") ||
        !h3_cuda_require_elements(gpu, output, count, "LayerNorm output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "LayerNorm output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (rows && width)
        h3k_layer_norm_f32<<<rows, h3_cuda_row_threads(width)>>>(
            (const float *)input->data, (const float *)weight->data,
            (const float *)bias->data, (float *)output->data, rows, width,
            epsilon);
    return h3_cuda_launch_check(gpu, "h3_layer_norm_f32");
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!h3_cuda_require_bf16(gpu, input, count, "LayerNorm input") ||
        !h3_cuda_require_bf16(gpu, weight, width, "LayerNorm weight") ||
        !h3_cuda_require_bf16(gpu, bias, width, "LayerNorm bias") ||
        !h3_cuda_require_bf16(gpu, output, count, "LayerNorm output") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (rows && width)
        h3k_layer_norm_bf16<<<rows, h3_cuda_row_threads(width)>>>(
            (const uint16_t *)input->data, (const uint16_t *)weight->data,
            (const uint16_t *)bias->data, (uint16_t *)output->data, rows,
            width, epsilon);
    return h3_cuda_launch_check(gpu, "h3_layer_norm_bf16");
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    size_t count = (size_t)outer * inner;
    if (!outer || !inner) {
        h3_cuda_fail(gpu, "weight-norm dimensions must be nonzero");
        return 0;
    }
    if (!h3_cuda_require_elements(gpu, vector, count, "weight-norm vector") ||
        !h3_cuda_require_dtype(gpu, vector, H3_GPU_F32, "weight-norm vector") ||
        !h3_cuda_require_elements(gpu, magnitude, outer,
                                  "weight-norm magnitude") ||
        !h3_cuda_require_dtype(gpu, magnitude, H3_GPU_F32,
                               "weight-norm magnitude") ||
        !h3_cuda_require_elements(gpu, output, count, "weight-norm output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "weight-norm output") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_weight_norm_f32<<<outer, h3_cuda_row_threads(inner)>>>(
        (const float *)vector->data, (const float *)magnitude->data,
        (float *)output->data, outer, inner);
    return h3_cuda_launch_check(gpu, "h3_weight_norm_f32");
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    size_t count = (size_t)sequence * heads * head_dim;
    if (!h3_cuda_require_bf16(gpu, tensor, count, "head norm tensor") ||
        !h3_cuda_require_bf16(gpu, weight, head_dim, "head norm weight") ||
        !h3_cuda_require_command(gpu)) return 0;
    if (sequence && heads && head_dim)
        h3k_head_rms_norm_bf16<<<h3_cuda_grid_2d(sequence, heads),
                                 dim3(16, 16)>>>(
            (uint16_t *)tensor->data, (const uint16_t *)weight->data,
            sequence, heads, head_dim, epsilon);
    return h3_cuda_launch_check(gpu, "h3_head_rms_norm_bf16");
}

/* The Metal host routes large matmuls through MPSGraph; the CUDA backend
 * routes them through cuBLASLt (falling back to the naive kernel on any
 * cuBLAS failure). The stats counter reflects the routing shape rule either
 * way, matching Metal. */
static int h3_cuda_linear_kind(uint32_t rows, uint32_t input_dim,
                               uint32_t output_dim) {
    return rows >= 32 && input_dim >= 256 && output_dim >= 256 ? 1 : 0;
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_cuda_require_elements(gpu, input, input_count, "linear input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "linear input") ||
        !h3_cuda_require_elements(gpu, weight, weight_count, "linear weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32, "linear weight") ||
        !h3_cuda_require_elements(gpu, output, output_count, "linear output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "linear output") ||
        (bias && (!h3_cuda_require_elements(gpu, bias, output_dim,
                                            "linear bias") ||
                  !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32,
                                         "linear bias"))) ||
        !h3_cuda_require_command(gpu)) return 0;
    int kind = h3_cuda_linear_kind(rows, input_dim, output_dim);
    /* cuBLAS path (the MPSGraph analog): row-major F32 C = X*W^T + bias.
     * h3_cuda_gemm_xwt returns 0 when cuBLASLt is disabled or cannot serve
     * the shape, and the naive kernel below then rewrites the output. */
    if (rows && output_dim &&
        h3_cuda_gemm_xwt(gpu, input->data, weight->data,
                         bias ? bias->data : NULL, output->data,
                         CUDA_R_32F, CUDA_R_32F, rows, input_dim, output_dim))
        return h3_cuda_launch_check_kind(gpu, "h3_linear_f32", kind);
    if (rows && output_dim)
        h3k_linear_f32<<<h3_cuda_grid_2d(output_dim, rows), dim3(16, 16)>>>(
            (const float *)input->data, (const float *)weight->data,
            bias ? (const float *)bias->data : NULL, (float *)output->data,
            rows, input_dim, output_dim, bias ? 1u : 0u);
    return h3_cuda_launch_check_kind(gpu, "h3_linear_f32", kind);
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_cuda_require_bf16(gpu, input, input_count, "linear input") ||
        !h3_cuda_require_bf16(gpu, weight, weight_count, "linear weight") ||
        !h3_cuda_require_bf16(gpu, output, output_count, "linear output") ||
        (bias && !h3_cuda_require_bf16(gpu, bias, output_dim,
                                       "linear bias")) ||
        !h3_cuda_require_command(gpu)) return 0;
    int kind = h3_cuda_linear_kind(rows, input_dim, output_dim);
    if (rows && output_dim &&
        h3_cuda_gemm_xwt(gpu, input->data, weight->data,
                         bias ? bias->data : NULL, output->data,
                         CUDA_R_16BF, CUDA_R_16BF, rows, input_dim,
                         output_dim))
        return h3_cuda_launch_check_kind(gpu, "h3_linear_bf16", kind);
    if (rows && output_dim)
        h3k_linear_bf16<<<h3_cuda_grid_2d(output_dim, rows), dim3(16, 16)>>>(
            (const uint16_t *)input->data, (const uint16_t *)weight->data,
            bias ? (const uint16_t *)bias->data : NULL,
            (uint16_t *)output->data, rows, input_dim, output_dim,
            bias ? 1u : 0u);
    return h3_cuda_launch_check_kind(gpu, "h3_linear_bf16", kind);
}

int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96))
        return h3_cuda_fail(gpu, "unsupported fused patch projection shape");
    if (input_offset > SIZE_MAX - input_count ||
        output_offset > SIZE_MAX - output_count ||
        input_offset > SIZE_MAX / sizeof(float) ||
        output_offset > SIZE_MAX / sizeof(uint16_t))
        return h3_cuda_fail(gpu,
                            "fused patch projection offset is out of range");
    if (!h3_cuda_require_elements(gpu, input, input_offset + input_count,
                                  "patch projection input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32,
                               "patch projection input") ||
        !h3_cuda_require_elements(gpu, weight, weight_count,
                                  "patch projection weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32,
                               "patch projection weight") ||
        !h3_cuda_require_elements(gpu, output, output_offset + output_count,
                                  "patch projection output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_BF16,
                               "patch projection output") ||
        (bias && (!h3_cuda_require_elements(gpu, bias, output_dim,
                                            "patch projection bias") ||
                  !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32,
                                         "patch projection bias"))) ||
        !h3_cuda_require_command(gpu)) return 0;
    /* cuBLAS path: F32 GEMM into the workspace, then cast to BF16 at the
     * output offset. The weight is already F32 (validated above), so no
     * weight cast is needed. */
    if (rows && output_dim && gpu->lt && !gpu->no_cublas) {
        size_t count = (size_t)rows * output_dim;
        float *sums = (float *)h3_cuda_workspace(gpu, count * sizeof(float));
        if (sums &&
            h3_cuda_gemm_xwt(gpu, (const float *)input->data + input_offset,
                             (const float *)weight->data,
                             bias ? (const float *)bias->data : NULL, sums,
                             CUDA_R_32F, CUDA_R_32F, rows, input_dim,
                             output_dim)) {
            h3k_cast_f32_to_bf16<<<h3_cuda_cast_grid(count), 256>>>(
                sums, (uint16_t *)output->data + output_offset, count);
            return h3_cuda_launch_check(gpu, "h3_linear_f32_tiled_bf16");
        }
    }
    if (rows && output_dim)
        h3k_linear_f32_bf16<<<h3_cuda_grid_2d(output_dim, rows),
                              dim3(16, 16)>>>(
            (const float *)input->data + input_offset,
            (const float *)weight->data,
            bias ? (const float *)bias->data : NULL,
            (uint16_t *)output->data + output_offset,
            rows, input_dim, output_dim, bias ? 1u : 0u);
    return h3_cuda_launch_check(gpu, "h3_linear_f32_tiled_bf16");
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(
        gpu, output, 0, input, 0, weight, bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)output_rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96))
        return h3_cuda_fail(gpu, "unsupported mapped patch projection shape");
    if (!h3_cuda_require_elements(gpu, input, input_count,
                                  "mapped patch input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "mapped patch input") ||
        !h3_cuda_require_elements(gpu, weight, weight_count,
                                  "mapped patch weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32,
                               "mapped patch weight") ||
        !h3_cuda_require_elements(gpu, output, output_count,
                                  "mapped patch output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_BF16,
                               "mapped patch output") ||
        !h3_cuda_require_elements(gpu, row_map, rows, "mapped patch row map") ||
        !h3_cuda_require_dtype(gpu, row_map, H3_GPU_U32,
                               "mapped patch row map") ||
        (bias && (!h3_cuda_require_elements(gpu, bias, output_dim,
                                            "mapped patch bias") ||
                  !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32,
                                         "mapped patch bias"))) ||
        !h3_cuda_require_command(gpu)) return 0;
    /* cuBLAS path: F32 GEMM into the workspace, then a cast that scatters
     * input row r to output row row_map[r], like the hand kernel. */
    if (rows && output_dim && gpu->lt && !gpu->no_cublas) {
        size_t count = (size_t)rows * output_dim;
        float *sums = (float *)h3_cuda_workspace(gpu, count * sizeof(float));
        if (sums &&
            h3_cuda_gemm_xwt(gpu, input->data, weight->data,
                             bias ? (const float *)bias->data : NULL, sums,
                             CUDA_R_32F, CUDA_R_32F, rows, input_dim,
                             output_dim)) {
            h3k_cast_f32_to_bf16_map<<<h3_cuda_grid_2d(output_dim, rows),
                                       dim3(16, 16)>>>(
                sums, (uint16_t *)output->data,
                (const uint32_t *)row_map->data, rows, output_dim);
            return h3_cuda_launch_check(gpu, "h3_linear_f32_tiled_bf16_map");
        }
    }
    if (rows && output_dim)
        h3k_linear_f32_bf16_map<<<h3_cuda_grid_2d(output_dim, rows),
                                  dim3(16, 16)>>>(
            (const float *)input->data, (const float *)weight->data,
            bias ? (const float *)bias->data : NULL,
            (uint16_t *)output->data, (const uint32_t *)row_map->data,
            rows, input_dim, output_dim, bias ? 1u : 0u);
    return h3_cuda_launch_check(gpu, "h3_linear_f32_tiled_bf16_map");
}

int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    if (!h3_cuda_require_bf16(gpu, input, (size_t)rows * input_dim,
                              "MLP input") ||
        !h3_cuda_require_bf16(gpu, fc1_weight,
                              (size_t)hidden_dim * 2 * input_dim,
                              "MLP fc1 weight") ||
        !h3_cuda_require_bf16(gpu, fc2_weight,
                              (size_t)output_dim * hidden_dim,
                              "MLP fc2 weight") ||
        !h3_cuda_require_bf16(gpu, output, (size_t)rows * output_dim,
                              "MLP output") ||
        !h3_cuda_require_command(gpu)) return 0;
    /* cuBLAS path: fc1 bf16 GEMM (F32 accumulate, bf16 out) into the
     * workspace as fused [rows, 2*hidden], SwiGLU in f32 math with a bf16
     * result, fc2 bf16 GEMM straight into the bf16 output — both GEMMs on
     * the tensor cores. The activated intermediate is rounded to bf16 once
     * (the hand path below and MPSGraph keep it f32): that single extra
     * rounding boundary, the same class the int8/NAX MLP paths accept, is
     * what lets fc2 run as a bf16 GEMM instead of the F32 (CUDA-core rate)
     * GEMM the earlier path used; at 38k rows on GB10 that was the
     * difference between 31 and ~100 TFLOPS for the whole MLP. */
    if (rows && hidden_dim && output_dim && gpu->lt && !gpu->no_cublas) {
        size_t fused_count = (size_t)rows * 2 * hidden_dim;
        size_t activated_count = (size_t)rows * hidden_dim;
        size_t activated_off =
            (fused_count * sizeof(uint16_t) + 255) & ~(size_t)255;
        size_t total = activated_off + activated_count * sizeof(uint16_t);
        uint16_t *workspace = (uint16_t *)h3_cuda_workspace(gpu, total);
        int served = 0;
        if (workspace) {
            uint16_t *fused = workspace;
            uint16_t *activated_bf16 =
                (uint16_t *)((char *)workspace + activated_off);
            served = h3_cuda_gemm_xwt(gpu, input->data, fc1_weight->data,
                                      NULL, fused, CUDA_R_16BF, CUDA_R_16BF,
                                      rows, input_dim, 2 * hidden_dim);
            if (served) {
                h3k_swiglu_bf16<<<h3_cuda_grid_2d(hidden_dim, rows),
                                  dim3(16, 16)>>>(fused, activated_bf16, rows,
                                                  hidden_dim);
                served = cudaGetLastError() == cudaSuccess;
            }
            if (served)
                served = h3_cuda_gemm_xwt(gpu, activated_bf16,
                                          fc2_weight->data, NULL,
                                          output->data, CUDA_R_16BF,
                                          CUDA_R_16BF, rows, hidden_dim,
                                          output_dim);
        }
        if (served) {
            if (!h3_cuda_launch_check_kind(gpu, "h3_mlp_bf16 fc1", 1))
                return 0;
            return h3_cuda_launch_check_kind(gpu, "h3_mlp_bf16 fc2", 1);
        }
        /* fall through to the hand kernels */
    }
    float *activated = NULL;
    size_t activated_count = (size_t)rows * hidden_dim;
    cudaError_t status = cudaMalloc(&activated,
                                    activated_count * sizeof(float));
    if (status != cudaSuccess)
        return h3_cuda_fail_cuda(gpu, "MLP intermediate allocation", status);
    if (rows && hidden_dim)
        h3k_mlp_fc1_swiglu_bf16<<<h3_cuda_grid_2d(hidden_dim, rows),
                                  dim3(16, 16)>>>(
            (const uint16_t *)input->data, (const uint16_t *)fc1_weight->data,
            activated, rows, input_dim, hidden_dim);
    if (!h3_cuda_launch_check_kind(gpu, "h3_mlp_bf16 fc1", 1)) {
        cudaFree(activated);
        return 0;
    }
    if (rows && output_dim)
        h3k_mlp_fc2_bf16<<<h3_cuda_grid_2d(output_dim, rows), dim3(16, 16)>>>(
            activated, (const uint16_t *)fc2_weight->data,
            (uint16_t *)output->data, rows, hidden_dim, output_dim);
    int ok = h3_cuda_launch_check_kind(gpu, "h3_mlp_bf16 fc2", 1);
    cudaFree(activated);
    return ok;
}
