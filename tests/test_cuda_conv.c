/* Brute-force CPU-reference checks for the CUDA conv-family/audio ops in
 * h3_cuda_conv.cu: conv1d_stride, conv3d, snake1d, vae_encoder_pad, and
 * vae_encoder_group_norm_silu. Tolerance 1e-5 absolute. */
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static h3_gpu *gpu;

static void die(const char *message) {
    fprintf(stderr, "FAIL test_cuda_conv.c: %s\n", message);
    exit(1);
}

static void gpu_ok(int ok, const char *operation) {
    if (!ok) {
        fprintf(stderr, "FAIL test_cuda_conv.c: %s: %s\n", operation,
                h3_gpu_error(gpu));
        exit(1);
    }
}

static float frand(unsigned *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (float)(*seed >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
}

static void compare(h3_gpu_tensor *tensor, const float *expected, size_t count,
                    const char *label) {
    float *got = malloc(count * sizeof(*got));
    if (!got || !h3_gpu_tensor_read_f32(tensor, got, count))
        die("cannot read result");
    float maximum = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float error = fabsf(got[index] - expected[index]);
        if (error > maximum) maximum = error;
    }
    free(got);
    printf("%-24s max abs %.7g\n", label, maximum);
    if (maximum > 1e-5f) die(label);
}

static void test_conv1d_stride(void) {
    const int batch = 2, length = 12, in_c = 3, out_c = 4, kernel = 3;
    const int stride = 2, padding = 3, dilation = 2;
    int out_length = (length + 2 * padding - dilation * (kernel - 1) - 1) /
                         stride + 1;
    size_t in_count = (size_t)batch * length * in_c;
    size_t w_count = (size_t)out_c * in_c * kernel;
    size_t out_count = (size_t)batch * out_length * out_c;
    float *input = malloc(in_count * sizeof(float));
    float *weight = malloc(w_count * sizeof(float));
    float *bias = malloc(out_c * sizeof(float));
    float *expected = malloc(out_count * sizeof(float));
    unsigned seed = 7;
    for (size_t i = 0; i < in_count; i++) input[i] = frand(&seed);
    for (size_t i = 0; i < w_count; i++) weight[i] = frand(&seed) * 0.3f;
    for (int i = 0; i < out_c; i++) bias[i] = frand(&seed) * 0.1f;
    for (int b = 0; b < batch; b++)
        for (int t = 0; t < out_length; t++)
            for (int oc = 0; oc < out_c; oc++) {
                float sum = bias[oc];
                for (int k = 0; k < kernel; k++) {
                    int source = t * stride + k * dilation - padding;
                    if (source < 0 || source >= length) continue;
                    for (int ic = 0; ic < in_c; ic++)
                        sum += input[((size_t)b * length + source) * in_c + ic] *
                               weight[((size_t)oc * in_c + ic) * kernel + k];
                }
                expected[((size_t)b * out_length + t) * out_c + oc] = sum;
            }
    h3_gpu_tensor *d_in = h3_gpu_tensor_from_f32(gpu, input, in_count);
    h3_gpu_tensor *d_w = h3_gpu_tensor_from_f32(gpu, weight, w_count);
    h3_gpu_tensor *d_b = h3_gpu_tensor_from_f32(gpu, bias, out_c);
    h3_gpu_tensor *d_out = h3_gpu_tensor_new_f32(gpu, out_count);
    if (!d_in || !d_w || !d_b || !d_out) die("allocation failed");
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_conv1d_stride_f32(gpu, d_out, d_in, d_w, d_b, batch, length,
                                    in_c, out_c, kernel, stride, padding,
                                    dilation), "conv1d_stride");
    gpu_ok(h3_gpu_submit(gpu), "submit");
    compare(d_out, expected, out_count, "conv1d_stride s2 d2 p3");
    h3_gpu_tensor_free(d_in); h3_gpu_tensor_free(d_w);
    h3_gpu_tensor_free(d_b); h3_gpu_tensor_free(d_out);
    free(input); free(weight); free(bias); free(expected);
}

static void test_conv3d(void) {
    const int batch = 2, depth = 3, height = 4, width = 5;
    const int in_c = 2, out_c = 3;
    const int kd = 2, kh = 3, kw = 3, sd = 2, sh = 1, sw = 2;
    int od = (depth - kd) / sd + 1;
    int oh = (height - kh) / sh + 1;
    int ow = (width - kw) / sw + 1;
    size_t in_count = (size_t)batch * depth * height * width * in_c;
    size_t w_count = (size_t)out_c * in_c * kd * kh * kw;
    size_t out_count = (size_t)batch * od * oh * ow * out_c;
    float *input = malloc(in_count * sizeof(float));
    float *weight = malloc(w_count * sizeof(float));
    float *bias = malloc(out_c * sizeof(float));
    float *expected = malloc(out_count * sizeof(float));
    unsigned seed = 11;
    for (size_t i = 0; i < in_count; i++) input[i] = frand(&seed);
    for (size_t i = 0; i < w_count; i++) weight[i] = frand(&seed) * 0.2f;
    for (int i = 0; i < out_c; i++) bias[i] = frand(&seed) * 0.1f;
    for (int b = 0; b < batch; b++)
        for (int t = 0; t < od; t++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int oc = 0; oc < out_c; oc++) {
                        float sum = bias[oc];
                        for (int z = 0; z < kd; z++)
                            for (int j = 0; j < kh; j++)
                                for (int i = 0; i < kw; i++)
                                    for (int ic = 0; ic < in_c; ic++) {
                                        size_t src =
                                            ((((size_t)b * depth + t * sd + z) *
                                                  height + y * sh + j) * width +
                                             x * sw + i) * in_c + ic;
                                        size_t w =
                                            ((((size_t)oc * in_c + ic) * kd +
                                              z) * kh + j) * kw + i;
                                        sum += input[src] * weight[w];
                                    }
                        expected[(((size_t)b * od + t) * oh + y) * ow * out_c +
                                 x * out_c + oc] = sum;
                    }
    h3_gpu_tensor *d_in = h3_gpu_tensor_from_f32(gpu, input, in_count);
    h3_gpu_tensor *d_w = h3_gpu_tensor_from_f32(gpu, weight, w_count);
    h3_gpu_tensor *d_b = h3_gpu_tensor_from_f32(gpu, bias, out_c);
    h3_gpu_tensor *d_out = h3_gpu_tensor_new_f32(gpu, out_count);
    if (!d_in || !d_w || !d_b || !d_out) die("allocation failed");
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_conv3d_f32(gpu, d_out, d_in, d_w, d_b, batch, depth, height,
                             width, in_c, out_c, kd, kh, kw, sd, sh, sw),
           "conv3d");
    gpu_ok(h3_gpu_submit(gpu), "submit");
    compare(d_out, expected, out_count, "conv3d k2x3x3 s2x1x2");
    h3_gpu_tensor_free(d_in); h3_gpu_tensor_free(d_w);
    h3_gpu_tensor_free(d_b); h3_gpu_tensor_free(d_out);
    free(input); free(weight); free(bias); free(expected);
}

static void test_snake1d(void) {
    const int batch = 2, length = 5, channels = 3;
    size_t count = (size_t)batch * length * channels;
    float *input = malloc(count * sizeof(float));
    float *alpha = malloc(channels * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    unsigned seed = 23;
    for (size_t i = 0; i < count; i++) input[i] = frand(&seed) * 2.0f;
    for (int i = 0; i < channels; i++)
        alpha[i] = 0.2f + 0.3f * fabsf(frand(&seed));
    for (size_t i = 0; i < count; i++) {
        float a = alpha[i % channels];
        float x = input[i];
        float wave = sinf(a * x);
        expected[i] = x + wave * wave / (a + 1e-9f);
    }
    h3_gpu_tensor *d_in = h3_gpu_tensor_from_f32(gpu, input, count);
    h3_gpu_tensor *d_a = h3_gpu_tensor_from_f32(gpu, alpha, channels);
    h3_gpu_tensor *d_out = h3_gpu_tensor_new_f32(gpu, count);
    if (!d_in || !d_a || !d_out) die("allocation failed");
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_snake1d_f32(gpu, d_out, d_in, d_a, batch, length, channels),
           "snake1d");
    gpu_ok(h3_gpu_submit(gpu), "submit");
    compare(d_out, expected, count, "snake1d");
    h3_gpu_tensor_free(d_in); h3_gpu_tensor_free(d_a);
    h3_gpu_tensor_free(d_out);
    free(input); free(alpha); free(expected);
}

static int reflect(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

static void test_vae_encoder_pad(void) {
    const int batch = 2, depth = 3, height = 4, width = 5, channels = 2;
    const int depth_front = 2, h_before = 1, h_after = 2, w_before = 2,
              w_after = 1;
    int od = depth + depth_front;
    int oh = height + h_before + h_after;
    int ow = width + w_before + w_after;
    size_t in_count = (size_t)batch * depth * height * width * channels;
    size_t out_count = (size_t)batch * od * oh * ow * channels;
    float *input = malloc(in_count * sizeof(float));
    float *expected = malloc(out_count * sizeof(float));
    unsigned seed = 31;
    for (size_t i = 0; i < in_count; i++) input[i] = frand(&seed);
    for (int b = 0; b < batch; b++)
        for (int t = 0; t < od; t++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int c = 0; c < channels; c++) {
                        size_t dst = ((((size_t)b * od + t) * oh + y) * ow +
                                      x) * channels + c;
                        if (t < depth_front) {
                            expected[dst] = 0.0f;
                            continue;
                        }
                        int sy = reflect(y - h_before, height);
                        int sx = reflect(x - w_before, width);
                        int st = t - depth_front;
                        expected[dst] =
                            input[((((size_t)b * depth + st) * height + sy) *
                                       width + sx) * channels + c];
                    }
    h3_gpu_tensor *d_in = h3_gpu_tensor_from_f32(gpu, input, in_count);
    h3_gpu_tensor *d_out = h3_gpu_tensor_new_f32(gpu, out_count);
    if (!d_in || !d_out) die("allocation failed");
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_vae_encoder_pad_f32(gpu, d_out, d_in, batch, depth, height,
                                      width, channels, depth_front, h_before,
                                      h_after, w_before, w_after),
           "vae_encoder_pad");
    gpu_ok(h3_gpu_submit(gpu), "submit");
    compare(d_out, expected, out_count, "vae_encoder_pad reflect");
    h3_gpu_tensor_free(d_in); h3_gpu_tensor_free(d_out);
    free(input); free(expected);
}

static void test_group_norm_silu(void) {
    const int batch = 1, depth = 2, height = 3, width = 4;
    const int channels = 4, groups = 2;
    const float epsilon = 1e-5f;
    size_t count = (size_t)batch * depth * height * width * channels;
    float *input = malloc(count * sizeof(float));
    float *weight = malloc(channels * sizeof(float));
    float *bias = malloc(channels * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    unsigned seed = 47;
    for (size_t i = 0; i < count; i++) input[i] = frand(&seed) * 3.0f;
    for (int i = 0; i < channels; i++) weight[i] = 1.0f + frand(&seed) * 0.5f;
    for (int i = 0; i < channels; i++) bias[i] = frand(&seed) * 0.2f;
    int cpg = channels / groups;
    int elements = height * width * cpg;
    int rows = batch * depth * groups;
    for (int row = 0; row < rows; row++) {
        int group_index = row % groups;
        int plane = row / groups;
        double mean = 0.0;
        for (int index = 0; index < elements; index++) {
            int spatial = index / cpg;
            int channel = group_index * cpg + index % cpg;
            mean += input[((size_t)plane * height * width + spatial) *
                              channels + channel];
        }
        mean /= elements;
        double var = 0.0;
        for (int index = 0; index < elements; index++) {
            int spatial = index / cpg;
            int channel = group_index * cpg + index % cpg;
            double centered = input[((size_t)plane * height * width + spatial) *
                                        channels + channel] - mean;
            var += centered * centered;
        }
        var /= elements;
        float inverse = (float)(1.0 / sqrt(var + epsilon));
        for (int index = 0; index < elements; index++) {
            int spatial = index / cpg;
            int channel = group_index * cpg + index % cpg;
            size_t dst = ((size_t)plane * height * width + spatial) *
                             channels + channel;
            float value = (float)((input[dst] - mean) * inverse) *
                              weight[channel] + bias[channel];
            expected[dst] = value / (1.0f + expf(-value));
        }
    }
    h3_gpu_tensor *d_in = h3_gpu_tensor_from_f32(gpu, input, count);
    h3_gpu_tensor *d_w = h3_gpu_tensor_from_f32(gpu, weight, channels);
    h3_gpu_tensor *d_b = h3_gpu_tensor_from_f32(gpu, bias, channels);
    h3_gpu_tensor *d_out = h3_gpu_tensor_new_f32(gpu, count);
    if (!d_in || !d_w || !d_b || !d_out) die("allocation failed");
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_vae_encoder_group_norm_silu_f32(gpu, d_out, d_in, d_w, d_b,
                                                  batch, depth, height, width,
                                                  channels, groups, epsilon),
           "group_norm_silu");
    gpu_ok(h3_gpu_submit(gpu), "submit");
    compare(d_out, expected, count, "group_norm_silu g2 c4");
    h3_gpu_tensor_free(d_in); h3_gpu_tensor_free(d_w);
    h3_gpu_tensor_free(d_b); h3_gpu_tensor_free(d_out);
    free(input); free(weight); free(bias); free(expected);
}

int main(void) {
    char error[512];
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    test_conv1d_stride();
    test_conv3d();
    test_snake1d();
    test_vae_encoder_pad();
    test_group_norm_silu();
    h3_gpu_free(gpu);
    puts("ok: CUDA conv/audio ops match brute-force references");
    return 0;
}
