#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQUENCE = 32, HIDDEN = 256, HEADS = 4, HEAD_DIM = 32,
    INNER = HEADS * HEAD_DIM, FFN = 128, T_ROWS = 2, T_DIM = 32,
    MODALITIES = 3, MODULATION_SLOTS = 6, ROPE_HALF = 12,
    MAX_TENSORS = 64
};

typedef struct {
    h3_st_header fixture;
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t owned_count;
} test_context;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void require_gpu(test_context *test, int condition,
                        const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_bf16.c: %s: %s\n", operation,
            h3_gpu_error(test->gpu));
    exit(1);
}

static h3_gpu_tensor *own(test_context *test, h3_gpu_tensor *tensor) {
    require(tensor != NULL, "Metal tensor allocation failed");
    require(test->owned_count < MAX_TENSORS, "tensor registry is full");
    test->owned[test->owned_count++] = tensor;
    return tensor;
}

static void *load(test_context *test, const char *name, h3_dtype dtype,
                  size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(&test->fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_bf16.c: invalid or absent %s\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes ? bytes : 1);
    require(data != NULL, "host tensor allocation failed");
    char error[256];
    if (!h3_st_read_data(&test->fixture, tensor, data, bytes, error,
                         sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", error);
        exit(1);
    }
    return data;
}

static h3_gpu_tensor *upload(test_context *test, const char *name,
                             size_t elements) {
    uint16_t *values = load(test, name, H3_DTYPE_BF16, elements);
    h3_gpu_tensor *tensor = own(test, h3_gpu_tensor_from_bf16(test->gpu,
                                                              values, elements));
    free(values);
    return tensor;
}

static h3_gpu_tensor *fresh(test_context *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_bf16(test->gpu, elements));
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void test_token_reduction_kernels(test_context *test) {
    enum { FULL_ROWS = 5, REDUCED_ROWS = 3, WIDTH = 4 };
    const float input_values[FULL_ROWS * WIDTH] = {
         1.0f,  2.0f,  3.0f,  4.0f,
        10.0f, 12.0f, 14.0f, 16.0f,
        14.0f, 16.0f, 18.0f, 20.0f,
        -8.0f, -4.0f,  0.0f,  4.0f,
        -4.0f,  0.0f,  4.0f,  8.0f
    };
    const float pooled_values[REDUCED_ROWS * WIDTH] = {
         1.0f,  2.0f,  3.0f,  4.0f,
        12.0f, 14.0f, 16.0f, 18.0f,
        -6.0f, -2.0f,  2.0f,  6.0f
    };
    const float processed_values[REDUCED_ROWS * WIDTH] = {
         2.0f,  3.0f,  4.0f,  5.0f,
        14.0f, 12.0f, 20.0f, 14.0f,
        -5.0f,  0.0f,  5.0f, 10.0f
    };
    const float expanded_values[FULL_ROWS * WIDTH] = {
         2.0f,  3.0f,  4.0f,  5.0f,
        12.0f, 10.0f, 18.0f, 12.0f,
        16.0f, 14.0f, 22.0f, 16.0f,
        -7.0f, -2.0f,  3.0f,  8.0f,
        -3.0f,  2.0f,  7.0f, 12.0f
    };
    const uint32_t pairs[REDUCED_ROWS * 2] = {0, 0, 1, 2, 3, 4};
    const uint32_t parents[FULL_ROWS] = {0, 1, 1, 2, 2};
    uint16_t input_bf16[FULL_ROWS * WIDTH];
    uint16_t processed_bf16[REDUCED_ROWS * WIDTH];
    uint16_t expected_pooled[REDUCED_ROWS * WIDTH];
    uint16_t expected_expanded[FULL_ROWS * WIDTH];
    for (size_t index = 0; index < FULL_ROWS * WIDTH; index++) {
        input_bf16[index] = f32_to_bf16(input_values[index]);
        expected_expanded[index] = f32_to_bf16(expanded_values[index]);
    }
    for (size_t index = 0; index < REDUCED_ROWS * WIDTH; index++) {
        processed_bf16[index] = f32_to_bf16(processed_values[index]);
        expected_pooled[index] = f32_to_bf16(pooled_values[index]);
    }
    h3_gpu_tensor *input = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, input_bf16, FULL_ROWS * WIDTH));
    h3_gpu_tensor *gpu_pairs = own(test, h3_gpu_tensor_from_u32(
        test->gpu, pairs, REDUCED_ROWS * 2));
    h3_gpu_tensor *pooled = fresh(test, REDUCED_ROWS * WIDTH);
    h3_gpu_tensor *processed = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, processed_bf16, REDUCED_ROWS * WIDTH));
    h3_gpu_tensor *gpu_parents = own(test, h3_gpu_tensor_from_u32(
        test->gpu, parents, FULL_ROWS));
    h3_gpu_tensor *expanded = fresh(test, FULL_ROWS * WIDTH);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin token-pool command stream");
    require_gpu(test, h3_gpu_token_pool_bf16(
        test->gpu, pooled, input, gpu_pairs, REDUCED_ROWS, WIDTH),
        "encode token pooling");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit token-pool command stream");
    uint16_t got_pooled[REDUCED_ROWS * WIDTH];
    require(h3_gpu_tensor_read_bf16(
                pooled, got_pooled, REDUCED_ROWS * WIDTH),
            "cannot read pooled tokens");
    require(memcmp(got_pooled, expected_pooled, sizeof(got_pooled)) == 0,
            "horizontal token pooling differs from exact BF16 reference");

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin token-expand command stream");
    require_gpu(test, h3_gpu_token_expand_delta_bf16(
        test->gpu, expanded, input, processed, pooled, gpu_parents,
        FULL_ROWS, WIDTH, 1, 1.0f), "encode token delta expansion");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit token-expand command stream");
    uint16_t got_expanded[FULL_ROWS * WIDTH];
    require(h3_gpu_tensor_read_bf16(
                expanded, got_expanded, FULL_ROWS * WIDTH),
            "cannot read expanded tokens");
    require(memcmp(got_expanded, expected_expanded,
                   sizeof(got_expanded)) == 0,
            "token expansion lost the full-grid residual");
}

static void test_euler_update(test_context *test) {
    float sample_values[] = {-7.0f, -8.0f, 10.0f, 20.0f,
                             30.0f, 40.0f, -9.0f};
    const float last_values[] = {1.0f, -2.0f, 0.5f, 4.0f};
    const float previous_values[] = {0.5f, -1.0f, 0.25f, 5.0f};
    uint16_t last_bf16[4], previous_bf16[4];
    for (size_t index = 0; index < 4; index++) {
        last_bf16[index] = f32_to_bf16(last_values[index]);
        previous_bf16[index] = f32_to_bf16(previous_values[index]);
    }
    h3_gpu_tensor *sample = own(test, h3_gpu_tensor_from_f32(
        test->gpu, sample_values, sizeof(sample_values) / sizeof(*sample_values)));
    h3_gpu_tensor *last = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, last_bf16, sizeof(last_bf16) / sizeof(*last_bf16)));
    h3_gpu_tensor *previous = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, previous_bf16,
        sizeof(previous_bf16) / sizeof(*previous_bf16)));
    require_gpu(test, h3_gpu_begin(test->gpu), "begin Euler update");
    require_gpu(test, h3_gpu_euler_bf16(test->gpu, sample, 2, last, previous,
                                        4, 0.25f, 0.5f),
                "encode Euler update");
    require_gpu(test, h3_gpu_submit(test->gpu), "submit Euler update");
    float got[7];
    require(h3_gpu_tensor_read_f32(sample, got, 7),
            "cannot read Euler-updated sample");
    const float expected[] = {-7.0f, -8.0f, 10.3125f, 19.375f,
                              30.15625f, 40.875f, -9.0f};
    require(memcmp(got, expected, sizeof(expected)) == 0,
            "fused Euler update differs from exact reference");
    float range[4];
    require(h3_gpu_tensor_read_f32_range(sample, 2, range, 4),
            "cannot read Euler sample range");
    require(memcmp(range, expected + 2, sizeof(range)) == 0,
            "F32 range read returned the wrong offset");
}

static double compare(test_context *test, const char *name,
                      h3_gpu_tensor *got_tensor, size_t elements,
                      double *maximum_absolute) {
    uint16_t *got = malloc(elements * sizeof(*got));
    uint16_t *want = load(test, name, H3_DTYPE_BF16, elements);
    require(got != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(got_tensor, got, elements),
            "cannot read completed BF16 tensor");
    double error = 0.0;
    double scale = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double got_value = bf16_to_f32(got[index]);
        double want_value = bf16_to_f32(want[index]);
        double delta = fabs(got_value - want_value);
        double magnitude = fabs(want_value);
        if (delta > error) error = delta;
        if (magnitude > scale) scale = magnitude;
    }
    free(got);
    free(want);
    *maximum_absolute = error;
    return error / (scale > 1e-12 ? scale : 1e-12);
}

static void require_same_bf16(h3_gpu_tensor *left, h3_gpu_tensor *right,
                              size_t elements, const char *message) {
    uint16_t *a = malloc(elements * sizeof(*a));
    uint16_t *b = malloc(elements * sizeof(*b));
    require(a != NULL && b != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(left, a, elements),
            "cannot read first BF16 tensor");
    require(h3_gpu_tensor_read_bf16(right, b, elements),
            "cannot read second BF16 tensor");
    require(memcmp(a, b, elements * sizeof(*a)) == 0, message);
    free(a);
    free(b);
}

static void cleanup(test_context *test) {
    for (size_t index = 0; index < test->owned_count; index++) {
        h3_gpu_tensor_free(test->owned[index]);
    }
    h3_gpu_free(test->gpu);
    h3_st_free_header(&test->fixture);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
                                  "misc/fixtures/h3_dit_bf16.safetensors";
    test_context test;
    memset(&test, 0, sizeof(test));
    char error[512];
    if (!h3_st_read_header(path, &test.fixture, error, sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", error);
        return 1;
    }
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) {
        fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", error);
        h3_st_free_header(&test.fixture);
        return 1;
    }
    test_euler_update(&test);
    test_token_reduction_kernels(&test);

    {
        enum { PATCH_ROWS = 16, PATCH_IN = 32, PATCH_OUT = 5376 };
        size_t input_count = PATCH_ROWS * PATCH_IN;
        size_t weight_count = PATCH_OUT * PATCH_IN;
        size_t output_count = PATCH_ROWS * PATCH_OUT;
        float *input = malloc(input_count * sizeof(*input));
        float *weight = malloc(weight_count * sizeof(*weight));
        float *bias = malloc(PATCH_OUT * sizeof(*bias));
        float *scalar = malloc(output_count * sizeof(*scalar));
        float *tiled = malloc(output_count * sizeof(*tiled));
        require(input && weight && bias && scalar && tiled,
                "patch tile host allocation failed");
        for (size_t index = 0; index < input_count; index++)
            input[index] = (float)((int)(index % 17) - 8) * 0.03125f;
        for (size_t index = 0; index < weight_count; index++)
            weight[index] = (float)((int)(index % 23) - 11) * 0.0078125f;
        for (size_t index = 0; index < PATCH_OUT; index++)
            bias[index] = (float)((int)(index % 7) - 3) * 0.015625f;
        h3_gpu_tensor *gpu_input = own(
            &test, h3_gpu_tensor_from_f32(test.gpu, input, input_count));
        h3_gpu_tensor *gpu_weight = own(
            &test, h3_gpu_tensor_from_f32(test.gpu, weight, weight_count));
        h3_gpu_tensor *gpu_bias = own(
            &test, h3_gpu_tensor_from_f32(test.gpu, bias, PATCH_OUT));
        h3_gpu_tensor *scalar_output = own(
            &test, h3_gpu_tensor_new_f32(test.gpu, output_count));
        h3_gpu_tensor *tiled_output = own(
            &test, h3_gpu_tensor_new_f32(test.gpu, output_count));
        setenv("H3_SCALAR_PATCH", "1", 1);
        require_gpu(&test, h3_gpu_begin(test.gpu),
                    "begin scalar patch stream");
        require_gpu(&test, h3_gpu_linear_f32(
            test.gpu, scalar_output, gpu_input, gpu_weight, gpu_bias,
            PATCH_ROWS, PATCH_IN, PATCH_OUT), "scalar patch linear");
        require_gpu(&test, h3_gpu_submit(test.gpu),
                    "submit scalar patch stream");
        unsetenv("H3_SCALAR_PATCH");
        require_gpu(&test, h3_gpu_begin(test.gpu),
                    "begin tiled patch stream");
        require_gpu(&test, h3_gpu_linear_f32(
            test.gpu, tiled_output, gpu_input, gpu_weight, gpu_bias,
            PATCH_ROWS, PATCH_IN, PATCH_OUT), "tiled patch linear");
        require_gpu(&test, h3_gpu_submit(test.gpu),
                    "submit tiled patch stream");
        require(h3_gpu_tensor_read_f32(scalar_output, scalar, output_count),
                "cannot read scalar patch output");
        require(h3_gpu_tensor_read_f32(tiled_output, tiled, output_count),
                "cannot read tiled patch output");
        require(memcmp(scalar, tiled, output_count * sizeof(*scalar)) == 0,
                "tiled patch output differs from scalar F32");
        free(input); free(weight); free(bias); free(scalar); free(tiled);
    }
    h3_gpu_stats setup_stats;
    require(h3_gpu_get_stats(test.gpu, &setup_stats),
            "cannot read patch-tile counters");

    h3_gpu_tensor *h_in = upload(&test, "x.h_in", SEQUENCE * HIDDEN);
    h3_gpu_tensor *attn_in = upload(&test, "x.attn_in", SEQUENCE * HIDDEN);
    h3_gpu_tensor *t_emb = upload(&test, "x.t_emb", T_ROWS * T_DIM);
    h3_gpu_tensor *rope_cos = upload(&test, "x.rope_cos", SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *rope_sin = upload(&test, "x.rope_sin", SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *norm1 = upload(&test, "norm1.weight", HIDDEN);
    h3_gpu_tensor *norm2 = upload(&test, "norm2.weight", HIDDEN);
    h3_gpu_tensor *adaln_w = upload(&test, "adaln_proj.linear.weight",
                                    MODALITIES * MODULATION_SLOTS * HIDDEN * T_DIM);
    h3_gpu_tensor *adaln_b = upload(&test, "adaln_proj.linear.bias",
                                    MODALITIES * MODULATION_SLOTS * HIDDEN);
    h3_gpu_tensor *qkv_w = upload(&test, "attn.qkv_proj.weight",
                                  INNER * 3 * HIDDEN);
    h3_gpu_tensor *q_norm = upload(&test, "attn.q_norm.weight", HEAD_DIM);
    h3_gpu_tensor *k_norm = upload(&test, "attn.k_norm.weight", HEAD_DIM);
    h3_gpu_tensor *out_w = upload(&test, "attn.out_proj.weight", HIDDEN * INNER);
    h3_gpu_tensor *fc1_w = upload(&test, "mlp.fc1.weight", FFN * 2 * HIDDEN);
    h3_gpu_tensor *fc2_w = upload(&test, "mlp.fc2.weight", HIDDEN * FFN);

    int32_t *runs = load(&test, "x.runs", H3_DTYPE_I32, 12);
    uint32_t row_map[SEQUENCE];
    for (size_t index = 0; index < SEQUENCE; index++) row_map[index] = UINT32_MAX;
    for (size_t run = 0; run < 4; run++) {
        int32_t start = runs[run * 3];
        int32_t stop = runs[run * 3 + 1];
        int32_t row = runs[run * 3 + 2];
        require(start >= 0 && stop >= start && stop <= SEQUENCE,
                "invalid run bounds");
        require(row >= 0 && row < T_ROWS * MODALITIES,
                "invalid modulation row");
        for (int32_t index = start; index < stop; index++) {
            require(row_map[index] == UINT32_MAX, "overlapping runs");
            row_map[index] = (uint32_t)row;
        }
    }
    free(runs);
    for (size_t index = 0; index < SEQUENCE; index++) {
        require(row_map[index] != UINT32_MAX, "runs do not cover sequence");
    }
    h3_gpu_tensor *gpu_rows = own(&test, h3_gpu_tensor_from_u32(test.gpu,
                                                                row_map,
                                                                SEQUENCE));

    h3_gpu_tensor *plain_qkv = fresh(&test, SEQUENCE * INNER * 3);
    h3_gpu_tensor *plain_q = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_k = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_v = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_sdpa = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *plain_fc1 = fresh(&test, SEQUENCE * FFN * 2);
    h3_gpu_tensor *plain_swiglu = fresh(&test, SEQUENCE * FFN);
    h3_gpu_tensor *plain_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *fused_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *t_silu = fresh(&test, T_ROWS * T_DIM);
    h3_gpu_tensor *modulation = fresh(&test, T_ROWS * MODALITIES *
                                             MODULATION_SLOTS * HIDDEN);
    h3_gpu_tensor *mod_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *full_qkv = fresh(&test, SEQUENCE * INNER * 3);
    h3_gpu_tensor *full_q = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_k = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_v = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_sdpa = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *after_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *mod_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *full_fc1 = fresh(&test, SEQUENCE * FFN * 2);
    h3_gpu_tensor *full_swiglu = fresh(&test, SEQUENCE * FFN);
    h3_gpu_tensor *full_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *h_out = fresh(&test, SEQUENCE * HIDDEN);

    require_gpu(&test, h3_gpu_begin(test.gpu), "begin command stream");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_qkv, attn_in, qkv_w,
                                           NULL, SEQUENCE, HIDDEN, INNER * 3),
                "plain QKV");
    require_gpu(&test, h3_gpu_qkv_rope_bf16(test.gpu, plain_q, plain_k, plain_v,
                                             plain_qkv, q_norm, k_norm, rope_cos,
                                             rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                                             ROPE_HALF, 1e-5f), "plain RoPE");
    require_gpu(&test, h3_gpu_sdpa_bf16(test.gpu, plain_sdpa, plain_q, plain_k,
                                        plain_v, SEQUENCE, HEADS, HEAD_DIM,
                                        1.0f / sqrtf((float)HEAD_DIM)),
                "plain attention");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_attn, plain_sdpa,
                                           out_w, NULL, SEQUENCE, INNER, HIDDEN),
                "plain attention output");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_fc1, attn_in, fc1_w,
                                           NULL, SEQUENCE, HIDDEN, FFN * 2),
                "plain MLP input");
    require_gpu(&test, h3_gpu_swiglu_bf16(test.gpu, plain_swiglu, plain_fc1,
                                           SEQUENCE, FFN), "plain SwiGLU");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_mlp, plain_swiglu,
                                           fc2_w, NULL, SEQUENCE, FFN, HIDDEN),
                "plain MLP output");
    require_gpu(&test, h3_gpu_mlp_bf16(test.gpu, fused_mlp, attn_in, fc1_w,
                                        fc2_w, SEQUENCE, HIDDEN, FFN, HIDDEN),
                "fused MLP");

    require_gpu(&test, h3_gpu_silu_bf16(test.gpu, t_silu, t_emb, T_ROWS * T_DIM),
                "AdaLN SiLU");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, modulation, t_silu, adaln_w,
                                           adaln_b, T_ROWS, T_DIM,
                                           MODALITIES * MODULATION_SLOTS * HIDDEN),
                "AdaLN projection");
    require_gpu(&test, h3_gpu_adaln_bf16(test.gpu, mod_attn, h_in, norm1,
                                          modulation, gpu_rows, SEQUENCE, HIDDEN,
                                          MODULATION_SLOTS, 0, 1, 1e-5f),
                "attention AdaLN");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_qkv, mod_attn, qkv_w,
                                           NULL, SEQUENCE, HIDDEN, INNER * 3),
                "full QKV");
    require_gpu(&test, h3_gpu_qkv_rope_bf16(test.gpu, full_q, full_k, full_v,
                                             full_qkv, q_norm, k_norm, rope_cos,
                                             rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                                             ROPE_HALF, 1e-5f), "full RoPE");
    require_gpu(&test, h3_gpu_sdpa_bf16(test.gpu, full_sdpa, full_q, full_k,
                                        full_v, SEQUENCE, HEADS, HEAD_DIM,
                                        1.0f / sqrtf((float)HEAD_DIM)),
                "full attention");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_attn, full_sdpa,
                                           out_w, NULL, SEQUENCE, INNER, HIDDEN),
                "full attention output");
    require_gpu(&test, h3_gpu_continue(test.gpu),
                "continue split command stream");
    require_gpu(&test, h3_gpu_gate_bf16(test.gpu, after_attn, h_in, full_attn,
                                         modulation, gpu_rows, SEQUENCE, HIDDEN,
                                         MODULATION_SLOTS, 2), "attention gate");
    require_gpu(&test, h3_gpu_adaln_bf16(test.gpu, mod_mlp, after_attn, norm2,
                                          modulation, gpu_rows, SEQUENCE, HIDDEN,
                                          MODULATION_SLOTS, 3, 4, 1e-5f),
                "MLP AdaLN");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_fc1, mod_mlp, fc1_w,
                                           NULL, SEQUENCE, HIDDEN, FFN * 2),
                "full MLP input");
    require_gpu(&test, h3_gpu_swiglu_bf16(test.gpu, full_swiglu, full_fc1,
                                           SEQUENCE, FFN), "full SwiGLU");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_mlp, full_swiglu,
                                           fc2_w, NULL, SEQUENCE, FFN, HIDDEN),
                "full MLP output");
    require_gpu(&test, h3_gpu_gate_bf16(test.gpu, h_out, after_attn, full_mlp,
                                         modulation, gpu_rows, SEQUENCE, HIDDEN,
                                         MODULATION_SLOTS, 5), "MLP gate");
    require_gpu(&test, h3_gpu_submit(test.gpu), "submit command stream");

    h3_gpu_stats stats;
    require(h3_gpu_get_stats(test.gpu, &stats), "cannot read Metal counters");

    /* The released H3 checkpoint emits QKV interleaved per head. Verify the
     * production deinterleaver against the conventional fixture layout. */
    size_t qkv_elements = SEQUENCE * INNER * 3;
    uint16_t *plain_qkv_host = malloc(qkv_elements * sizeof(*plain_qkv_host));
    uint16_t *grouped_qkv_host = malloc(qkv_elements * sizeof(*grouped_qkv_host));
    require(plain_qkv_host != NULL && grouped_qkv_host != NULL,
            "grouped QKV allocation failed");
    require(h3_gpu_tensor_read_bf16(plain_qkv, plain_qkv_host, qkv_elements),
            "cannot read conventional QKV");
    for (size_t row = 0; row < SEQUENCE; row++) {
        size_t row_base = row * INNER * 3;
        for (size_t head = 0; head < HEADS; head++) {
            for (size_t kind = 0; kind < 3; kind++) {
                memcpy(grouped_qkv_host + row_base +
                           (head * 3 + kind) * HEAD_DIM,
                       plain_qkv_host + row_base + kind * INNER +
                           head * HEAD_DIM,
                       HEAD_DIM * sizeof(*plain_qkv_host));
            }
        }
    }
    h3_gpu_tensor *grouped_qkv = own(
        &test, h3_gpu_tensor_from_bf16(test.gpu, grouped_qkv_host, qkv_elements));
    h3_gpu_tensor *grouped_q = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *grouped_k = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *grouped_v = fresh(&test, SEQUENCE * INNER);
    free(plain_qkv_host);
    free(grouped_qkv_host);
    require_gpu(&test, h3_gpu_begin(test.gpu), "begin grouped QKV stream");
    require_gpu(&test, h3_gpu_grouped_qkv_rope_bf16(
        test.gpu, grouped_q, grouped_k, grouped_v, grouped_qkv, q_norm, k_norm,
        rope_cos, rope_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
        "grouped QKV RoPE");
    require_gpu(&test, h3_gpu_submit(test.gpu), "submit grouped QKV stream");
    require_same_bf16(plain_q, grouped_q, SEQUENCE * INNER,
                      "grouped query deinterleave mismatch");
    require_same_bf16(plain_k, grouped_k, SEQUENCE * INNER,
                      "grouped key deinterleave mismatch");
    require_same_bf16(plain_v, grouped_v, SEQUENCE * INNER,
                      "grouped value deinterleave mismatch");

    double attn_abs, mlp_abs, block_abs;
    double attn_rel = compare(&test, "x.attn_out", plain_attn,
                              SEQUENCE * HIDDEN, &attn_abs);
    double mlp_rel = compare(&test, "x.mlp_out", plain_mlp,
                             SEQUENCE * HIDDEN, &mlp_abs);
    double fused_mlp_abs;
    double fused_mlp_rel = compare(&test, "x.mlp_out", fused_mlp,
                                   SEQUENCE * HIDDEN, &fused_mlp_abs);
    double block_rel = compare(&test, "x.h_out", h_out,
                               SEQUENCE * HIDDEN, &block_abs);
    printf("BF16 Metal/MLX parity: attention rel %.3g abs %.3g; "
           "MLP rel %.3g abs %.3g; fused MLP rel %.3g abs %.3g; "
           "block rel %.3g abs %.3g\n", attn_rel, attn_abs, mlp_rel,
           mlp_abs, fused_mlp_rel, fused_mlp_abs, block_rel, block_abs);
    require(attn_rel < 1e-2, "BF16 attention exceeds MLX error bound");
    require(mlp_rel < 1e-2, "BF16 MLP exceeds MLX error bound");
    require(fused_mlp_rel < 1e-2,
            "fused BF16 MLP exceeds MLX error bound");
    require(block_rel < 1e-2, "BF16 block exceeds MLX error bound");
    printf("BF16 dispatches: %llu MPS linear, %llu MPS SDPA, %llu direct; "
           "%.2f MiB allocated\n",
           (unsigned long long)stats.mps_linear_dispatches,
           (unsigned long long)stats.mps_sdpa_dispatches,
           (unsigned long long)stats.direct_dispatches,
           (double)stats.allocated_bytes / (1024.0 * 1024.0));
    require(stats.mps_linear_dispatches == 6,
            "wide BF16 linears did not use the cached MPSGraph path");
    require(stats.mps_sdpa_dispatches == 2, "BF16 SDPA counter mismatch");
    require(stats.submissions == setup_stats.submissions + 2,
            "split block did not use two command submissions");
    cleanup(&test);
    puts("ok: portable BF16 Metal block matches the MLX BF16 oracle");
    return 0;
}
