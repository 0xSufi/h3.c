/* CUDA port of the h3_gpu.h convolution family and audio/VAE encoder ops:
 * Conv1d/ConvTranspose1d/Conv3d (MPSGraph on Metal -> kind-2 dispatch stats),
 * Snake/SnakeBeta activations, and the VAE encoder pad + group-norm-SiLU
 * kernels. Kernel math mirrors h3_shaders.metal; host-side validation and
 * stats-counter behavior mirror h3_gpu.m. Correctness-first: naive kernels,
 * float accumulation, safe libm (no fast-math intrinsics).
 *
 * Note: h3_gpu_weight_norm_f32 is already implemented in h3_cuda_core.cu
 * (milestone A, norms) and is intentionally not redefined here. */
/* h3_gpu.h is a C API: give the declarations C linkage so gcc-compiled
 * model-layer objects link against these nvcc-compiled definitions. */
extern "C" {
#include "h3_gpu.h"
}
#include "h3_cuda_internal.h"

/* ---------------------------------------------------------------- kernels */

/* Activations are time-major [batch, length, channels] (channels-last);
 * weights are PyTorch OIK order [out, in, kernel]; symmetric padding. */
__global__ void h3k_conv1d_f32(const float *input, const float *weight,
                               const float *bias, float *output,
                               uint32_t batch, uint32_t length,
                               uint32_t input_channels,
                               uint32_t output_channels, uint32_t kernel,
                               uint32_t stride, uint32_t padding,
                               uint32_t dilation, uint32_t output_length) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = batch * output_length * output_channels;
    if (gid >= count) return;
    uint32_t oc = gid % output_channels;
    uint32_t time = (gid / output_channels) % output_length;
    uint32_t b = gid / (output_channels * output_length);
    float sum = bias ? bias[oc] : 0.0f;
    for (uint32_t k = 0; k < kernel; k++) {
        int64_t source = (int64_t)time * stride + (int64_t)k * dilation -
                         (int64_t)padding;
        if (source < 0 || source >= (int64_t)length) continue;
        const float *in_row =
            input + ((size_t)b * length + (uint32_t)source) * input_channels;
        const float *w_row =
            weight + ((size_t)oc * input_channels) * kernel + k;
        for (uint32_t ic = 0; ic < input_channels; ic++)
            sum = fmaf(in_row[ic], w_row[(size_t)ic * kernel], sum);
    }
    output[((size_t)b * output_length + time) * output_channels + oc] = sum;
}

/* ConvTranspose1d weights are IOK order [in, out, kernel]. */
__global__ void h3k_conv_transpose1d_f32(const float *input,
                                         const float *weight,
                                         const float *bias, float *output,
                                         uint32_t batch, uint32_t length,
                                         uint32_t input_channels,
                                         uint32_t output_channels,
                                         uint32_t kernel, uint32_t stride,
                                         uint32_t padding,
                                         uint32_t output_length) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = batch * output_length * output_channels;
    if (gid >= count) return;
    uint32_t oc = gid % output_channels;
    uint32_t time = (gid / output_channels) % output_length;
    uint32_t b = gid / (output_channels * output_length);
    float sum = bias ? bias[oc] : 0.0f;
    for (uint32_t k = 0; k < kernel; k++) {
        int64_t numerator = (int64_t)time + padding - k;
        if (numerator < 0 || numerator % stride) continue;
        int64_t source = numerator / stride;
        if (source >= (int64_t)length) continue;
        const float *in_row =
            input + ((size_t)b * length + (uint32_t)source) * input_channels;
        for (uint32_t ic = 0; ic < input_channels; ic++) {
            float w = weight[((size_t)ic * output_channels + oc) * kernel + k];
            sum = fmaf(in_row[ic], w, sum);
        }
    }
    output[((size_t)b * output_length + time) * output_channels + oc] = sum;
}

/* NDHWC channels-last activations, OIDHW weights, per-axis strides, no
 * padding and no dilation (matching the MPSGraph descriptor in h3_gpu.m). */
__global__ void h3k_conv3d_f32(const float *input, const float *weight,
                               const float *bias, float *output,
                               uint32_t batch, uint32_t depth, uint32_t height,
                               uint32_t width, uint32_t input_channels,
                               uint32_t output_channels, uint32_t kernel_depth,
                               uint32_t kernel_height, uint32_t kernel_width,
                               uint32_t stride_depth, uint32_t stride_height,
                               uint32_t stride_width, uint32_t output_depth,
                               uint32_t output_height, uint32_t output_width) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t spatial = output_depth * output_height * output_width;
    uint32_t count = batch * spatial * output_channels;
    if (gid >= count) return;
    uint32_t oc = gid % output_channels;
    uint32_t plane = gid / output_channels;
    uint32_t out_x = plane % output_width;
    uint32_t out_y = (plane / output_width) % output_height;
    uint32_t out_t = (plane / (output_width * output_height)) % output_depth;
    uint32_t b = plane / spatial;
    float sum = bias ? bias[oc] : 0.0f;
    uint32_t base_t = out_t * stride_depth;
    uint32_t base_y = out_y * stride_height;
    uint32_t base_x = out_x * stride_width;
    for (uint32_t kd = 0; kd < kernel_depth; kd++) {
        for (uint32_t kh = 0; kh < kernel_height; kh++) {
            for (uint32_t kw = 0; kw < kernel_width; kw++) {
                size_t in_base =
                    ((((size_t)b * depth + base_t + kd) * height + base_y +
                      kh) * width + base_x + kw) * input_channels;
                for (uint32_t ic = 0; ic < input_channels; ic++) {
                    float w = weight[((((size_t)oc * input_channels + ic) *
                                           kernel_depth + kd) *
                                          kernel_height + kh) *
                                         kernel_width + kw];
                    sum = fmaf(input[in_base + ic], w, sum);
                }
            }
        }
    }
    output[((size_t)plane) * output_channels + oc] = sum;
}

__global__ void h3k_snake1d_f32(const float *input, const float *alpha,
                                float *output, uint32_t count,
                                uint32_t channels) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float a = alpha[gid % channels];
    float x = input[gid];
    float wave = sinf(a * x);
    output[gid] = x + wave * wave / (a + 1e-9f);
}

/* BigVGAN's fixed 2x upsample -> SnakeBeta -> 2x low-pass downsample fused
 * at the original rate, ported verbatim from h3_alias_free_snake_f32. */
__global__ void h3k_alias_free_snake_f32(
        const float *input, const float *alpha_log, const float *beta_log,
        const float *upsample_filter, const float *downsample_filter,
        float *output, uint32_t batch, uint32_t length, uint32_t channels) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = batch * length * channels;
    if (gid >= count) return;
    uint32_t channel = gid % channels;
    uint32_t time = (gid / channels) % length;
    uint32_t b = gid / (channels * length);
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = (int)(time * 2) + down_k - 5;
        up_time = min(max(up_time, 0), (int)(length * 2) - 1);
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = min(max(padded_time - 5, 0), (int)length - 1);
            size_t source =
                ((size_t)b * length + (uint32_t)source_time) * channels +
                channel;
            upsampled = fmaf(input[source], 2.0f * upsample_filter[up_k],
                             upsampled);
        }
        float sine = sinf(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fmaf(activated, downsample_filter[down_k], result);
    }
    output[((size_t)b * length + time) * channels + channel] = result;
}

__device__ __forceinline__ int h3_cuda_reflect_coordinate(int coordinate,
                                                          int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

/* [B,T,H,W,C] channels-last padding: spatial reflect, temporal front
 * zero-fill. One thread per output element. */
__global__ void h3k_vae_encoder_pad_f32(const float *input, float *output,
                                        uint32_t batch, uint32_t depth,
                                        uint32_t height, uint32_t width,
                                        uint32_t channels,
                                        uint32_t depth_front,
                                        uint32_t height_before,
                                        uint32_t height_after,
                                        uint32_t width_before,
                                        uint32_t width_after) {
    uint32_t out_height = height + height_before + height_after;
    uint32_t out_width = width + width_before + width_after;
    uint32_t out_depth = depth + depth_front;
    size_t count = (size_t)batch * out_depth * out_height * out_width *
                   channels;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    uint32_t channel = (uint32_t)(gid % channels);
    size_t rem = gid / channels;
    uint32_t out_x = (uint32_t)(rem % out_width);
    rem /= out_width;
    uint32_t out_y = (uint32_t)(rem % out_height);
    rem /= out_height;
    uint32_t out_t = (uint32_t)(rem % out_depth);
    uint32_t b = (uint32_t)(rem / out_depth);
    if (out_t < depth_front) {
        output[gid] = 0.0f;
        return;
    }
    int source_y = h3_cuda_reflect_coordinate((int)out_y - (int)height_before,
                                              (int)height);
    int source_x = h3_cuda_reflect_coordinate((int)out_x - (int)width_before,
                                              (int)width);
    uint32_t source_t = out_t - depth_front;
    size_t source =
        ((((size_t)b * depth + source_t) * height + (uint32_t)source_y) *
             width + (uint32_t)source_x) * channels + channel;
    output[gid] = input[source];
}

/* Group norm over [B,D,H,W,C] (groups split channels; rows = B*D*groups,
 * each row spans H*W*(C/groups) elements), then SiLU. One block per row. */
__global__ void h3k_vae_encoder_group_norm_silu_f32(
        const float *input, const float *weight, const float *bias,
        float *output, uint32_t batch, uint32_t depth, uint32_t height,
        uint32_t width, uint32_t channels, uint32_t groups, float epsilon) {
    __shared__ float reductions[256];
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t rows = batch * depth * groups;
    if (row >= rows) return;
    uint32_t channels_per_group = channels / groups;
    uint32_t group_index = row % groups;
    uint32_t temporal_plane = row / groups;
    uint32_t elements = height * width * channels_per_group;
    float local = 0.0f;
    for (uint32_t index = tid; index < elements; index += blockDim.x) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group_index * channels_per_group +
                           index % channels_per_group;
        size_t source = ((size_t)temporal_plane * height * width + spatial) *
                            channels + channel;
        local += input[source];
    }
    reductions[tid] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float mean = reductions[0] / (float)elements;
    local = 0.0f;
    for (uint32_t index = tid; index < elements; index += blockDim.x) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group_index * channels_per_group +
                           index % channels_per_group;
        size_t source = ((size_t)temporal_plane * height * width + spatial) *
                            channels + channel;
        float centered = input[source] - mean;
        local = fmaf(centered, centered, local);
    }
    reductions[tid] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = 1.0f / sqrtf(reductions[0] / (float)elements + epsilon);
    for (uint32_t index = tid; index < elements; index += blockDim.x) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group_index * channels_per_group +
                           index % channels_per_group;
        size_t destination =
            ((size_t)temporal_plane * height * width + spatial) * channels +
            channel;
        float value = (input[destination] - mean) * inverse * weight[channel] +
                      bias[channel];
        output[destination] = value / (1.0f + expf(-value));
    }
}

/* ----------------------------------------------------------- host wrappers */

static dim3 h3_conv_grid_1d(size_t count) {
    return dim3((unsigned)((count + 255) / 256));
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) {
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || !dilation || (uint64_t)length + 2 * padding < effective)
        return 0;
    uint32_t output_length = (uint32_t)(((uint64_t)length + 2 * padding -
                                         effective) / stride + 1);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_cuda_require_elements(gpu, input, input_count, "Conv1d input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "Conv1d input") ||
        !h3_cuda_require_elements(gpu, weight, weight_count,
                                  "Conv1d weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32, "Conv1d weight") ||
        !h3_cuda_require_elements(gpu, output, output_count,
                                  "Conv1d output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "Conv1d output") ||
        (bias && (!h3_cuda_require_elements(gpu, bias, output_channels,
                                            "Conv1d bias") ||
                  !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32,
                                         "Conv1d bias"))) ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_conv1d_f32<<<h3_conv_grid_1d(output_count), 256>>>(
        (const float *)input->data, (const float *)weight->data,
        bias ? (const float *)bias->data : NULL, (float *)output->data,
        batch, length, input_channels, output_channels, kernel, stride,
        padding, dilation, output_length);
    return h3_cuda_launch_check_kind(gpu, "h3_conv1d_f32", 2);
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(gpu, output, input, weight, bias,
        batch, length, input_channels, output_channels, kernel, 1, padding,
        dilation);
}

int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) {
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding)
        return 0;
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_cuda_require_elements(gpu, input, input_count,
                                  "ConvTranspose1d input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32,
                               "ConvTranspose1d input") ||
        !h3_cuda_require_elements(gpu, weight, weight_count,
                                  "ConvTranspose1d weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32,
                               "ConvTranspose1d weight") ||
        !h3_cuda_require_elements(gpu, output, output_count,
                                  "ConvTranspose1d output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32,
                               "ConvTranspose1d output") ||
        (bias && (!h3_cuda_require_elements(gpu, bias, output_channels,
                                            "ConvTranspose1d bias") ||
                  !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32,
                                         "ConvTranspose1d bias"))) ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_conv_transpose1d_f32<<<h3_conv_grid_1d(output_count), 256>>>(
        (const float *)input->data, (const float *)weight->data,
        bias ? (const float *)bias->data : NULL, (float *)output->data,
        batch, length, input_channels, output_channels, kernel, stride,
        padding, output_length);
    return h3_cuda_launch_check_kind(gpu, "h3_conv_transpose1d_f32", 2);
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    if (!batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        depth < kernel_depth || height < kernel_height || width < kernel_width)
        return 0;
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count = (size_t)batch * depth * height * width *
                         input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (!h3_cuda_require_elements(gpu, input, input_count, "Conv3d input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "Conv3d input") ||
        !h3_cuda_require_elements(gpu, weight, weight_count,
                                  "Conv3d weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32, "Conv3d weight") ||
        !h3_cuda_require_elements(gpu, output, output_count,
                                  "Conv3d output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "Conv3d output") ||
        (bias && (!h3_cuda_require_elements(gpu, bias, output_channels,
                                            "Conv3d bias") ||
                  !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32,
                                         "Conv3d bias"))) ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_conv3d_f32<<<h3_conv_grid_1d(output_count), 256>>>(
        (const float *)input->data, (const float *)weight->data,
        bias ? (const float *)bias->data : NULL, (float *)output->data,
        batch, depth, height, width, input_channels, output_channels,
        kernel_depth, kernel_height, kernel_width, stride_depth,
        stride_height, stride_width, output_depth, output_height,
        output_width);
    return h3_cuda_launch_check_kind(gpu, "h3_conv3d_f32", 2);
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels || count > UINT32_MAX) return 0;
    if (!h3_cuda_require_elements(gpu, input, count, "Snake1d input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "Snake1d input") ||
        !h3_cuda_require_elements(gpu, alpha, channels, "Snake1d alpha") ||
        !h3_cuda_require_dtype(gpu, alpha, H3_GPU_F32, "Snake1d alpha") ||
        !h3_cuda_require_elements(gpu, output, count, "Snake1d output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "Snake1d output") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_snake1d_f32<<<h3_conv_grid_1d(count), 256>>>(
        (const float *)input->data, (const float *)alpha->data,
        (float *)output->data, (uint32_t)count, channels);
    return h3_cuda_launch_check(gpu, "h3_snake1d_f32");
}

int h3_gpu_alias_free_snake_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *alpha_log,
                      const h3_gpu_tensor *beta_log,
                      const h3_gpu_tensor *upsample_filter,
                      const h3_gpu_tensor *downsample_filter,
                      uint32_t batch, uint32_t length,
                      uint32_t channels) {
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels) return 0;
    if (!h3_cuda_require_elements(gpu, input, count, "Snake input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32, "Snake input") ||
        !h3_cuda_require_elements(gpu, output, count, "Snake output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32, "Snake output") ||
        !h3_cuda_require_elements(gpu, alpha_log, channels, "Snake alpha") ||
        !h3_cuda_require_dtype(gpu, alpha_log, H3_GPU_F32, "Snake alpha") ||
        !h3_cuda_require_elements(gpu, beta_log, channels, "Snake beta") ||
        !h3_cuda_require_dtype(gpu, beta_log, H3_GPU_F32, "Snake beta") ||
        !h3_cuda_require_elements(gpu, upsample_filter, 12,
                                  "Snake upsample filter") ||
        !h3_cuda_require_dtype(gpu, upsample_filter, H3_GPU_F32,
                               "Snake upsample filter") ||
        !h3_cuda_require_elements(gpu, downsample_filter, 12,
                                  "Snake downsample filter") ||
        !h3_cuda_require_dtype(gpu, downsample_filter, H3_GPU_F32,
                               "Snake downsample filter") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_alias_free_snake_f32<<<h3_conv_grid_1d(count), 256>>>(
        (const float *)input->data, (const float *)alpha_log->data,
        (const float *)beta_log->data, (const float *)upsample_filter->data,
        (const float *)downsample_filter->data, (float *)output->data,
        batch, length, channels);
    return h3_cuda_launch_check(gpu, "h3_alias_free_snake_f32");
}

int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    if (!batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) return 0;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_cuda_require_elements(gpu, input, input_count,
                                  "VAE encoder pad input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32,
                               "VAE encoder pad input") ||
        !h3_cuda_require_elements(gpu, output, output_count,
                                  "VAE encoder pad output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32,
                               "VAE encoder pad output") ||
        !h3_cuda_require_command(gpu)) return 0;
    h3k_vae_encoder_pad_f32<<<h3_conv_grid_1d(output_count), 256>>>(
        (const float *)input->data, (float *)output->data, batch, depth,
        height, width, channels, depth_front, height_before, height_after,
        width_before, width_after);
    return h3_cuda_launch_check(gpu, "h3_vae_encoder_pad_f32");
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!batch || !depth || !height || !width || !channels || !groups ||
        channels % groups || !(epsilon > 0.0f)) return 0;
    if (!h3_cuda_require_elements(gpu, input, count,
                                  "VAE encoder norm input") ||
        !h3_cuda_require_dtype(gpu, input, H3_GPU_F32,
                               "VAE encoder norm input") ||
        !h3_cuda_require_elements(gpu, weight, channels,
                                  "VAE encoder norm weight") ||
        !h3_cuda_require_dtype(gpu, weight, H3_GPU_F32,
                               "VAE encoder norm weight") ||
        !h3_cuda_require_elements(gpu, bias, channels,
                                  "VAE encoder norm bias") ||
        !h3_cuda_require_dtype(gpu, bias, H3_GPU_F32,
                               "VAE encoder norm bias") ||
        !h3_cuda_require_elements(gpu, output, count,
                                  "VAE encoder norm output") ||
        !h3_cuda_require_dtype(gpu, output, H3_GPU_F32,
                               "VAE encoder norm output") ||
        !h3_cuda_require_command(gpu)) return 0;
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    uint32_t elements = height * width * (channels / groups);
    h3k_vae_encoder_group_norm_silu_f32<<<(unsigned)rows,
                                          h3_cuda_row_threads(elements)>>>(
        (const float *)input->data, (const float *)weight->data,
        (const float *)bias->data, (float *)output->data, batch, depth,
        height, width, channels, groups, epsilon);
    return h3_cuda_launch_check(gpu, "h3_vae_encoder_group_norm_silu_f32");
}
