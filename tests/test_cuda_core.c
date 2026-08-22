/* Freestanding correctness test for h3_cuda_core.cu: every one of the 25
 * milestone-A ops against an independent CPU reference. */
#include "h3_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------ bf16 host */
static float host_bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}
static uint16_t host_f32_to_bf16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    b += 0x7fffu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

/* deterministic LCG in [-range, range] */
static uint64_t rng_state = 0x123456789abcdefULL;
static float frand(float range) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = (double)(rng_state >> 11) / (double)(1ULL << 53);
    return (float)((u * 2.0 - 1.0) * range);
}

static int failures = 0;
static void report(const char *name, double max_err, int bad) {
    if (bad) {
        printf("FAIL %-28s max_err=%.3g (%d mismatches)\n", name, max_err,
               bad);
        failures++;
    } else {
        printf("ok   %-28s max_err=%.3g\n", name, max_err);
    }
}

/* f32 compare: abs+rel tolerance */
static void check_f32(const char *name, const float *got, const float *ref,
                      size_t n, double atol, double rtol) {
    double max_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)got[i] - (double)ref[i]);
        double tol = atol + rtol * fabs((double)ref[i]);
        if (err > max_err) max_err = err;
        if (!(err <= tol)) bad++;
    }
    report(name, max_err, bad);
}

/* bf16 compare: exact bits or within 1 bf16 ulp of the reference */
static void check_bf16(const char *name, const uint16_t *got,
                       const uint16_t *ref, size_t n) {
    double max_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        float g = host_bf16_to_f32(got[i]);
        float r = host_bf16_to_f32(ref[i]);
        double err = fabs((double)g - (double)r);
        if (err > max_err) max_err = err;
        if (got[i] == ref[i]) continue;
        double ulp = ldexp((double)fmaxf(fabsf(g), fabsf(r)), -8);
        if (!(err <= ulp + 1e-30)) bad++;
    }
    report(name, max_err, bad);
}

/* bf16 compare for ops routed through cuBLASLt GEMMs: FP32 accumulation is
 * preserved, but the summation order differs from the sequential CPU
 * reference, so bit-exact / 1-ulp agreement cannot be guaranteed. The
 * tolerance (1e-3 + 1e-2 relative) sits just above one bf16 ulp (2^-8
 * relative) of the reference magnitude. With H3_CUDA_NO_CUBLAS=1 the hand
 * kernels stay bit-exact and pass this trivially. */
static void check_bf16_gemm(const char *name, const uint16_t *got,
                            const uint16_t *ref, size_t n) {
    double max_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        float g = host_bf16_to_f32(got[i]);
        float r = host_bf16_to_f32(ref[i]);
        double err = fabs((double)g - (double)r);
        if (err > max_err) max_err = err;
        if (!(err <= 1e-3 + 1e-2 * fabs((double)r))) bad++;
    }
    report(name, max_err, bad);
}

/* f32 compare for ops routed through cuBLASLt GEMMs (the F32 analog of
 * check_bf16_gemm): FP32 accumulation is preserved, but the summation order
 * differs from the sequential CPU reference, so the bit-exact agreement the
 * naive kernel guarantees cannot be. The tolerance (1e-4 abs + 1e-3 rel)
 * sits far above the F32 reordering noise for the K ranges these ops use
 * (hundreds to a few thousand) yet far below any real regression. With
 * H3_CUDA_NO_CUBLAS=1 the hand kernels stay bit-exact and pass this
 * trivially. */
static void check_f32_gemm(const char *name, const float *got,
                           const float *ref, size_t n) {
    double max_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)got[i] - (double)ref[i]);
        if (err > max_err) max_err = err;
        if (!(err <= 1e-4 + 1e-3 * fabs((double)ref[i]))) bad++;
    }
    report(name, max_err, bad);
}

/* ------------------------------------------------------------ helpers */
static h3_gpu *gpu;
static h3_gpu_tensor *mk_f32(const float *v, size_t n) {
    h3_gpu_tensor *t = h3_gpu_tensor_from_f32(gpu, v, n);
    if (!t) { fprintf(stderr, "alloc f32 failed: %s\n", h3_gpu_error(gpu)); exit(2); }
    return t;
}
static h3_gpu_tensor *mk_bf16(const uint16_t *v, size_t n) {
    h3_gpu_tensor *t = h3_gpu_tensor_from_bf16(gpu, v, n);
    if (!t) { fprintf(stderr, "alloc bf16 failed: %s\n", h3_gpu_error(gpu)); exit(2); }
    return t;
}
static h3_gpu_tensor *new_f32(size_t n) {
    h3_gpu_tensor *t = h3_gpu_tensor_new_f32(gpu, n);
    if (!t) { fprintf(stderr, "alloc f32 failed: %s\n", h3_gpu_error(gpu)); exit(2); }
    return t;
}
static h3_gpu_tensor *new_bf16(size_t n) {
    h3_gpu_tensor *t = h3_gpu_tensor_new_bf16(gpu, n);
    if (!t) { fprintf(stderr, "alloc bf16 failed: %s\n", h3_gpu_error(gpu)); exit(2); }
    return t;
}
static void run(void) {
    if (!h3_gpu_submit(gpu)) {
        fprintf(stderr, "submit failed: %s\n", h3_gpu_error(gpu));
        exit(2);
    }
    if (!h3_gpu_begin(gpu)) {
        fprintf(stderr, "begin failed: %s\n", h3_gpu_error(gpu));
        exit(2);
    }
}
#define OP(expr) do { if (!(expr)) { \
    fprintf(stderr, "op failed at %s:%d: %s\n", __FILE__, __LINE__, \
            h3_gpu_error(gpu)); exit(2); } } while (0)

/* ------------------------------------------------------------ CPU refs */
static float ref_silu(float x) { return x / (1.0f + expf(-x)); }

static float ref_erf_approx(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = fabsf(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float p = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
                0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - p * expf(-x * x));
}
static float ref_gelu(float v, int approximate) {
    if (approximate) {
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        return inner <= -10.0f ? 0.0f : inner >= 10.0f ? v :
               0.5f * v * (1.0f + tanhf(inner));
    }
    return v <= -10.0f ? 0.0f : v >= 10.0f ? v :
           0.5f * v * (1.0f + ref_erf_approx(v * 0.7071067811865475f));
}

static void ref_linear_f32(const float *in, const float *w, const float *b,
                           float *out, uint32_t rows, uint32_t in_dim,
                           uint32_t out_dim) {
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < out_dim; c++) {
            float sum = b ? b[c] : 0.0f;
            for (uint32_t k = 0; k < in_dim; k++)
                sum = fmaf(in[r * in_dim + k], w[c * in_dim + k], sum);
            out[r * out_dim + c] = sum;
        }
}

/* ------------------------------------------------------------ tests */
static void test_silu_f32(void) {
    const uint32_t n = 1000;
    float *in = malloc(n * 4), *ref = malloc(n * 4), *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) { in[i] = frand(8.0f); ref[i] = ref_silu(in[i]); }
    h3_gpu_tensor *ti = mk_f32(in, n), *to = new_f32(n);
    OP(h3_gpu_silu_f32(gpu, to, ti, n));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("silu_f32", got, ref, n, 1e-5, 1e-6);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(to);
    free(in); free(ref); free(got);
}

static void test_silu_bf16(void) {
    const uint32_t n = 1000;
    uint16_t *in = malloc(n * 2), *ref = malloc(n * 2), *got = malloc(n * 2);
    for (uint32_t i = 0; i < n; i++) {
        in[i] = host_f32_to_bf16(frand(8.0f));
        ref[i] = host_f32_to_bf16(ref_silu(host_bf16_to_f32(in[i])));
    }
    h3_gpu_tensor *ti = mk_bf16(in, n), *to = new_bf16(n);
    OP(h3_gpu_silu_bf16(gpu, to, ti, n));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("silu_bf16", got, ref, n);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(to);
    free(in); free(ref); free(got);
}

static void test_gelu_bf16(int approximate) {
    const uint32_t n = 512;
    uint16_t *in = malloc(n * 2), *ref = malloc(n * 2), *got = malloc(n * 2);
    for (uint32_t i = 0; i < n; i++) {
        float v = i % 17 == 0 ? (i % 34 == 0 ? 20.0f : -20.0f) : frand(6.0f);
        in[i] = host_f32_to_bf16(v);
        ref[i] = host_f32_to_bf16(ref_gelu(host_bf16_to_f32(in[i]), approximate));
    }
    h3_gpu_tensor *ti = mk_bf16(in, n), *to = new_bf16(n);
    OP(h3_gpu_gelu_bf16(gpu, to, ti, n, approximate));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16(approximate ? "gelu_bf16 tanh" : "gelu_bf16 erf", got, ref, n);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(to);
    free(in); free(ref); free(got);
}

static void test_geglu_f32(void) {
    const uint32_t n = 777;
    float *g = malloc(n * 4), *l = malloc(n * 4), *ref = malloc(n * 4),
          *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) {
        g[i] = frand(4.0f); l[i] = frand(4.0f);
        float x = g[i], cube = x * x * x;
        float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608028654f *
                                              (x + 0.044715f * cube)));
        ref[i] = gelu * l[i];
    }
    h3_gpu_tensor *tg = mk_f32(g, n), *tl = mk_f32(l, n), *to = new_f32(n);
    OP(h3_gpu_geglu_f32(gpu, to, tg, tl, n));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("geglu_f32", got, ref, n, 1e-5, 1e-6);
    h3_gpu_tensor_free(tg); h3_gpu_tensor_free(tl); h3_gpu_tensor_free(to);
    free(g); free(l); free(ref); free(got);
}

static void test_clip_f32(void) {
    const uint32_t n = 500;
    float *in = malloc(n * 4), *ref = malloc(n * 4), *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) {
        in[i] = frand(10.0f);
        ref[i] = fminf(fmaxf(in[i], -1.5f), 2.5f);
    }
    h3_gpu_tensor *ti = mk_f32(in, n), *to = new_f32(n);
    OP(h3_gpu_clip_f32(gpu, to, ti, n, -1.5f, 2.5f));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("clip_f32", got, ref, n, 1e-5, 1e-6);
    /* invalid range must fail */
    if (h3_gpu_clip_f32(gpu, to, ti, n, 3.0f, -3.0f)) {
        printf("FAIL clip_f32 accepted min>max\n"); failures++;
    }
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(to);
    free(in); free(ref); free(got);
}

static void test_add_scaled_f32(void) {
    const uint32_t n = 999;
    float *a = malloc(n * 4), *b = malloc(n * 4), *ref = malloc(n * 4),
          *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) {
        a[i] = frand(3.0f); b[i] = frand(3.0f);
        ref[i] = a[i] * 0.25f + b[i] * -1.75f;
    }
    h3_gpu_tensor *ta = mk_f32(a, n), *tb = mk_f32(b, n), *to = new_f32(n);
    OP(h3_gpu_add_scaled_f32(gpu, to, ta, tb, 0.25f, -1.75f, n));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("add_scaled_f32", got, ref, n, 1e-5, 1e-6);
    h3_gpu_tensor_free(ta); h3_gpu_tensor_free(tb); h3_gpu_tensor_free(to);
    free(a); free(b); free(ref); free(got);
}

static void test_scale_add_f32(void) {
    const uint32_t rows = 3, width = 5, n = rows * width;
    float *r = malloc(n * 4), *b = malloc(n * 4), *s = malloc(width * 4),
          *ref = malloc(n * 4), *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) { r[i] = frand(2.0f); b[i] = frand(2.0f); }
    for (uint32_t i = 0; i < width; i++) s[i] = frand(2.0f);
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t c = 0; c < width; c++)
            ref[row * width + c] = r[row * width + c] +
                                   b[row * width + c] * s[c];
    h3_gpu_tensor *tr = mk_f32(r, n), *tb = mk_f32(b, n),
                  *ts = mk_f32(s, width), *to = new_f32(n);
    OP(h3_gpu_scale_add_f32(gpu, to, tr, tb, ts, rows, width));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("scale_add_f32", got, ref, n, 1e-5, 1e-6);
    h3_gpu_tensor_free(tr); h3_gpu_tensor_free(tb); h3_gpu_tensor_free(ts);
    h3_gpu_tensor_free(to);
    free(r); free(b); free(s); free(ref); free(got);
}

static void test_add_sub_bf16(void) {
    const uint32_t n = 1000;
    uint16_t *a = malloc(n * 2), *b = malloc(n * 2), *refa = malloc(n * 2),
             *refs = malloc(n * 2), *got = malloc(n * 2);
    for (uint32_t i = 0; i < n; i++) {
        a[i] = host_f32_to_bf16(frand(8.0f)); b[i] = host_f32_to_bf16(frand(8.0f));
        refa[i] = host_f32_to_bf16(host_bf16_to_f32(a[i]) + host_bf16_to_f32(b[i]));
        refs[i] = host_f32_to_bf16(host_bf16_to_f32(a[i]) - host_bf16_to_f32(b[i]));
    }
    h3_gpu_tensor *ta = mk_bf16(a, n), *tb = mk_bf16(b, n), *to = new_bf16(n);
    OP(h3_gpu_add_bf16(gpu, to, ta, tb, n));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("add_bf16", got, refa, n);
    OP(h3_gpu_sub_bf16(gpu, to, ta, tb, n));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("sub_bf16", got, refs, n);
    h3_gpu_tensor_free(ta); h3_gpu_tensor_free(tb); h3_gpu_tensor_free(to);
    free(a); free(b); free(refa); free(refs); free(got);
}

static void test_silu_mul_bf16(void) {
    const uint32_t n = 1000;
    uint16_t *g = malloc(n * 2), *u = malloc(n * 2), *ref = malloc(n * 2),
             *got = malloc(n * 2);
    for (uint32_t i = 0; i < n; i++) {
        g[i] = host_f32_to_bf16(frand(6.0f)); u[i] = host_f32_to_bf16(frand(6.0f));
        ref[i] = host_f32_to_bf16(ref_silu(host_bf16_to_f32(g[i])) *
                                  host_bf16_to_f32(u[i]));
    }
    h3_gpu_tensor *tg = mk_bf16(g, n), *tu = mk_bf16(u, n), *to = new_bf16(n);
    OP(h3_gpu_silu_mul_bf16(gpu, to, tg, tu, n));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("silu_mul_bf16", got, ref, n);
    h3_gpu_tensor_free(tg); h3_gpu_tensor_free(tu); h3_gpu_tensor_free(to);
    free(g); free(u); free(ref); free(got);
}

static void ref_swiglu(const float *fused, float *ref, uint32_t rows,
                       uint32_t width) {
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < width; c++) {
            float gate = fused[r * width * 2 + c];
            float up = fused[r * width * 2 + width + c];
            ref[r * width + c] = ref_silu(gate) * up;
        }
}

static void test_swiglu_f32(void) {
    const uint32_t rows = 3, width = 4;
    float *f = malloc(rows * width * 2 * 4), *ref = malloc(rows * width * 4),
          *got = malloc(rows * width * 4);
    for (uint32_t i = 0; i < rows * width * 2; i++) f[i] = frand(4.0f);
    ref_swiglu(f, ref, rows, width);
    h3_gpu_tensor *tf = mk_f32(f, rows * width * 2), *to = new_f32(rows * width);
    OP(h3_gpu_swiglu_f32(gpu, to, tf, rows, width));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, rows * width));
    check_f32("swiglu_f32", got, ref, rows * width, 1e-5, 1e-6);
    h3_gpu_tensor_free(tf); h3_gpu_tensor_free(to);
    free(f); free(ref); free(got);
}

static void test_swiglu_bf16(void) {
    const uint32_t rows = 3, width = 4;
    uint16_t *f = malloc(rows * width * 2 * 2), *ref = malloc(rows * width * 2),
             *got = malloc(rows * width * 2);
    float *ff = malloc(rows * width * 2 * 4), *reff = malloc(rows * width * 4);
    for (uint32_t i = 0; i < rows * width * 2; i++) {
        f[i] = host_f32_to_bf16(frand(4.0f));
        ff[i] = host_bf16_to_f32(f[i]);
    }
    ref_swiglu(ff, reff, rows, width);
    for (uint32_t i = 0; i < rows * width; i++) ref[i] = host_f32_to_bf16(reff[i]);
    h3_gpu_tensor *tf = mk_bf16(f, rows * width * 2),
                  *to = new_bf16(rows * width);
    OP(h3_gpu_swiglu_bf16(gpu, to, tf, rows, width));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, rows * width));
    check_bf16("swiglu_bf16", got, ref, rows * width);
    h3_gpu_tensor_free(tf); h3_gpu_tensor_free(to);
    free(f); free(ref); free(got); free(ff); free(reff);
}

static void test_embedding_bf16(void) {
    const uint32_t vocab = 10, width = 6, tokens = 4;
    uint32_t ids[4] = {0, 7, 3, 15}; /* 15 is out of range -> zeros */
    uint16_t *w = malloc(vocab * width * 2), *got = malloc(tokens * width * 2);
    uint16_t ref[4 * 6];
    for (uint32_t i = 0; i < vocab * width; i++)
        w[i] = host_f32_to_bf16(frand(2.0f));
    for (uint32_t t = 0; t < tokens; t++)
        for (uint32_t c = 0; c < width; c++)
            ref[t * width + c] = ids[t] < vocab ? w[ids[t] * width + c] : 0;
    h3_gpu_tensor *tw = mk_bf16(w, vocab * width),
                  *ti = h3_gpu_tensor_from_u32(gpu, ids, tokens),
                  *to = new_bf16(tokens * width);
    if (!ti) { fprintf(stderr, "u32 alloc failed\n"); exit(2); }
    OP(h3_gpu_embedding_bf16(gpu, to, tw, ti, tokens, vocab, width));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, tokens * width));
    check_bf16("embedding_bf16", got, ref, tokens * width);
    h3_gpu_tensor_free(tw); h3_gpu_tensor_free(ti); h3_gpu_tensor_free(to);
    free(w); free(got);
}

static void ref_rms_norm(const float *in, const float *w, float *ref,
                         uint32_t rows, uint32_t width, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        float ss = 0.0f;
        for (uint32_t c = 0; c < width; c++) ss += in[r * width + c] * in[r * width + c];
        float inv = 1.0f / sqrtf(ss / (float)width + eps);
        for (uint32_t c = 0; c < width; c++)
            ref[r * width + c] = in[r * width + c] * inv * w[c];
    }
}

static void test_rms_norm_f32(void) {
    const uint32_t rows = 3, width = 8, n = rows * width;
    float eps = 1e-5f;
    float *in = malloc(n * 4), *w = malloc(width * 4), *ref = malloc(n * 4),
          *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) in[i] = frand(3.0f);
    for (uint32_t i = 0; i < width; i++) w[i] = frand(2.0f);
    ref_rms_norm(in, w, ref, rows, width, eps);
    h3_gpu_tensor *ti = mk_f32(in, n), *tw = mk_f32(w, width), *to = new_f32(n);
    OP(h3_gpu_rms_norm_f32(gpu, to, ti, tw, rows, width, eps));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("rms_norm_f32", got, ref, n, 1e-5, 1e-5);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(to);
    free(in); free(w); free(ref); free(got);
}

static void test_rms_norm_bf16(void) {
    const uint32_t rows = 3, width = 8, n = rows * width;
    float eps = 1e-5f;
    uint16_t *in = malloc(n * 2), *w = malloc(width * 2), *ref = malloc(n * 2),
             *got = malloc(n * 2);
    float *fin = malloc(n * 4), *fw = malloc(width * 4), *fref = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) { in[i] = host_f32_to_bf16(frand(3.0f)); fin[i] = host_bf16_to_f32(in[i]); }
    for (uint32_t i = 0; i < width; i++) { w[i] = host_f32_to_bf16(frand(2.0f)); fw[i] = host_bf16_to_f32(w[i]); }
    ref_rms_norm(fin, fw, fref, rows, width, eps);
    for (uint32_t i = 0; i < n; i++) ref[i] = host_f32_to_bf16(fref[i]);
    h3_gpu_tensor *ti = mk_bf16(in, n), *tw = mk_bf16(w, width),
                  *to = new_bf16(n);
    OP(h3_gpu_rms_norm_bf16(gpu, to, ti, tw, rows, width, eps));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("rms_norm_bf16", got, ref, n);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(to);
    free(in); free(w); free(ref); free(got); free(fin); free(fw); free(fref);
}

static void ref_layer_norm(const float *in, const float *w, const float *b,
                           float *ref, uint32_t rows, uint32_t width,
                           float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < width; c++) sum += in[r * width + c];
        float mean = sum / (float)width;
        float vs = 0.0f;
        for (uint32_t c = 0; c < width; c++) {
            float d = in[r * width + c] - mean;
            vs += d * d;
        }
        float inv = 1.0f / sqrtf(vs / (float)width + eps);
        for (uint32_t c = 0; c < width; c++)
            ref[r * width + c] = (in[r * width + c] - mean) * inv * w[c] + b[c];
    }
}

static void test_layer_norm_f32(void) {
    const uint32_t rows = 3, width = 8, n = rows * width;
    float eps = 1e-5f;
    float *in = malloc(n * 4), *w = malloc(width * 4), *b = malloc(width * 4),
          *ref = malloc(n * 4), *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) in[i] = frand(3.0f);
    for (uint32_t i = 0; i < width; i++) { w[i] = frand(2.0f); b[i] = frand(1.0f); }
    ref_layer_norm(in, w, b, ref, rows, width, eps);
    h3_gpu_tensor *ti = mk_f32(in, n), *tw = mk_f32(w, width),
                  *tb = mk_f32(b, width), *to = new_f32(n);
    OP(h3_gpu_layer_norm_f32(gpu, to, ti, tw, tb, rows, width, eps));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("layer_norm_f32", got, ref, n, 1e-5, 1e-5);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
}

static void test_layer_norm_bf16(void) {
    const uint32_t rows = 3, width = 8, n = rows * width;
    float eps = 1e-5f;
    uint16_t *in = malloc(n * 2), *w = malloc(width * 2), *b = malloc(width * 2),
             *ref = malloc(n * 2), *got = malloc(n * 2);
    float *fin = malloc(n * 4), *fw = malloc(width * 4), *fb = malloc(width * 4),
          *fref = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) { in[i] = host_f32_to_bf16(frand(3.0f)); fin[i] = host_bf16_to_f32(in[i]); }
    for (uint32_t i = 0; i < width; i++) {
        w[i] = host_f32_to_bf16(frand(2.0f)); fw[i] = host_bf16_to_f32(w[i]);
        b[i] = host_f32_to_bf16(frand(1.0f)); fb[i] = host_bf16_to_f32(b[i]);
    }
    ref_layer_norm(fin, fw, fb, fref, rows, width, eps);
    for (uint32_t i = 0; i < n; i++) ref[i] = host_f32_to_bf16(fref[i]);
    h3_gpu_tensor *ti = mk_bf16(in, n), *tw = mk_bf16(w, width),
                  *tb = mk_bf16(b, width), *to = new_bf16(n);
    OP(h3_gpu_layer_norm_bf16(gpu, to, ti, tw, tb, rows, width, eps));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("layer_norm_bf16", got, ref, n);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
    free(fin); free(fw); free(fb); free(fref);
}

static void test_weight_norm_f32(void) {
    const uint32_t outer = 4, inner = 6, n = outer * inner;
    float *v = malloc(n * 4), *m = malloc(outer * 4), *ref = malloc(n * 4),
          *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) v[i] = frand(3.0f);
    for (uint32_t i = 0; i < outer; i++) m[i] = fabsf(frand(2.0f)) + 0.1f;
    for (uint32_t r = 0; r < outer; r++) {
        float ss = 0.0f;
        for (uint32_t i = 0; i < inner; i++) ss += v[r * inner + i] * v[r * inner + i];
        float scale = m[r] / sqrtf(ss);
        for (uint32_t i = 0; i < inner; i++) ref[r * inner + i] = v[r * inner + i] * scale;
    }
    h3_gpu_tensor *tv = mk_f32(v, n), *tm = mk_f32(m, outer), *to = new_f32(n);
    OP(h3_gpu_weight_norm_f32(gpu, to, tv, tm, outer, inner));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("weight_norm_f32", got, ref, n, 1e-5, 1e-5);
    /* zero dims must fail */
    if (h3_gpu_weight_norm_f32(gpu, to, tv, tm, 0, inner)) {
        printf("FAIL weight_norm_f32 accepted outer=0\n"); failures++;
    }
    h3_gpu_tensor_free(tv); h3_gpu_tensor_free(tm); h3_gpu_tensor_free(to);
    free(v); free(m); free(ref); free(got);
}

static void test_head_rms_norm_bf16(void) {
    const uint32_t seq = 2, heads = 3, dim = 8, n = seq * heads * dim;
    float eps = 1e-6f;
    uint16_t *x = malloc(n * 2), *w = malloc(dim * 2), *ref = malloc(n * 2),
             *got = malloc(n * 2);
    float *fw = malloc(dim * 4);
    for (uint32_t i = 0; i < n; i++) x[i] = host_f32_to_bf16(frand(3.0f));
    for (uint32_t i = 0; i < dim; i++) { w[i] = host_f32_to_bf16(frand(2.0f)); fw[i] = host_bf16_to_f32(w[i]); }
    memcpy(ref, x, n * 2);
    for (uint32_t s = 0; s < seq; s++)
        for (uint32_t h = 0; h < heads; h++) {
            size_t base = ((size_t)s * heads + h) * dim;
            float ss = 0.0f;
            for (uint32_t d = 0; d < dim; d++) {
                float v = host_bf16_to_f32(ref[base + d]);
                ss += v * v;
            }
            float inv = 1.0f / sqrtf(ss / (float)dim + eps);
            for (uint32_t d = 0; d < dim; d++)
                ref[base + d] = host_f32_to_bf16(
                    host_bf16_to_f32(ref[base + d]) * inv * fw[d]);
        }
    h3_gpu_tensor *tx = mk_bf16(x, n), *tw = mk_bf16(w, dim);
    OP(h3_gpu_head_rms_norm_bf16(gpu, tx, tw, seq, heads, dim, eps));
    run();
    OP(h3_gpu_tensor_read_bf16(tx, got, n));
    check_bf16("head_rms_norm_bf16", got, ref, n);
    h3_gpu_tensor_free(tx); h3_gpu_tensor_free(tw);
    free(x); free(w); free(ref); free(got); free(fw);
}

static void test_linear_f32_small(void) {
    const uint32_t rows = 5, in_dim = 7, out_dim = 9;
    float *in = malloc(rows * in_dim * 4), *w = malloc(out_dim * in_dim * 4),
          *b = malloc(out_dim * 4), *ref = malloc(rows * out_dim * 4),
          *got = malloc(rows * out_dim * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) in[i] = frand(1.0f);
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w[i] = frand(1.0f);
    for (uint32_t i = 0; i < out_dim; i++) b[i] = frand(1.0f);
    ref_linear_f32(in, w, b, ref, rows, in_dim, out_dim);
    h3_gpu_tensor *ti = mk_f32(in, rows * in_dim),
                  *tw = mk_f32(w, out_dim * in_dim),
                  *tb = mk_f32(b, out_dim), *to = new_f32(rows * out_dim);
    OP(h3_gpu_linear_f32(gpu, to, ti, tw, tb, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, rows * out_dim));
    check_f32_gemm("linear_f32 small+bias", got, ref, rows * out_dim);
    /* no bias */
    ref_linear_f32(in, w, NULL, ref, rows, in_dim, out_dim);
    OP(h3_gpu_linear_f32(gpu, to, ti, tw, NULL, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, rows * out_dim));
    check_f32_gemm("linear_f32 small nobias", got, ref, rows * out_dim);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
}

static void test_linear_f32_big(void) {
    const uint32_t rows = 64, in_dim = 512, out_dim = 512;
    float *in = malloc(rows * in_dim * 4), *w = malloc(out_dim * in_dim * 4),
          *b = malloc(out_dim * 4), *ref = malloc(rows * out_dim * 4),
          *got = malloc(rows * out_dim * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) in[i] = frand(0.2f);
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w[i] = frand(0.2f);
    for (uint32_t i = 0; i < out_dim; i++) b[i] = frand(0.2f);
    ref_linear_f32(in, w, b, ref, rows, in_dim, out_dim);
    h3_gpu_tensor *ti = mk_f32(in, rows * in_dim),
                  *tw = mk_f32(w, out_dim * in_dim),
                  *tb = mk_f32(b, out_dim), *to = new_f32(rows * out_dim);
    h3_gpu_stats before, after;
    OP(h3_gpu_get_stats(gpu, &before));
    OP(h3_gpu_linear_f32(gpu, to, ti, tw, tb, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_get_stats(gpu, &after));
    OP(h3_gpu_tensor_read_f32(to, got, rows * out_dim));
    check_f32_gemm("linear_f32 64x512x512", got, ref, rows * out_dim);
    if (after.mps_linear_dispatches != before.mps_linear_dispatches + 1) {
        printf("FAIL linear_f32 big did not bump mps_linear_dispatches "
               "(%llu -> %llu)\n",
               (unsigned long long)before.mps_linear_dispatches,
               (unsigned long long)after.mps_linear_dispatches);
        failures++;
    } else {
        printf("ok   linear_f32 big bumped mps_linear_dispatches\n");
    }
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
}

/* Same 64x512x512 projection with TF32 enabled: products see 10-bit
 * mantissas, so the answer is only required to land within 1e-2 relative
 * (+1e-3 absolute) of the F32 reference, versus 1e-3 for exact F32. */
static void test_linear_f32_tf32(void) {
    const uint32_t rows = 64, in_dim = 512, out_dim = 512;
    float *in = malloc(rows * in_dim * 4), *w = malloc(out_dim * in_dim * 4),
          *b = malloc(out_dim * 4), *ref = malloc(rows * out_dim * 4),
          *got = malloc(rows * out_dim * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) in[i] = frand(0.2f);
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w[i] = frand(0.2f);
    for (uint32_t i = 0; i < out_dim; i++) b[i] = frand(0.2f);
    ref_linear_f32(in, w, b, ref, rows, in_dim, out_dim);
    h3_gpu_tensor *ti = mk_f32(in, rows * in_dim),
                  *tw = mk_f32(w, out_dim * in_dim),
                  *tb = mk_f32(b, out_dim), *to = new_f32(rows * out_dim);
    setenv("H3_CUDA_TF32", "1", 1);
    OP(h3_gpu_linear_f32(gpu, to, ti, tw, tb, rows, in_dim, out_dim));
    run();
    setenv("H3_CUDA_TF32", "0", 1);
    OP(h3_gpu_tensor_read_f32(to, got, rows * out_dim));
    double max_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < (size_t)rows * out_dim; i++) {
        double err = fabs((double)got[i] - (double)ref[i]);
        if (err > max_err) max_err = err;
        if (!(err <= 1e-3 + 1e-2 * fabs((double)ref[i]))) bad++;
    }
    report("linear_f32 64x512x512 tf32", max_err, bad);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
}

static void test_linear_bf16_small(void) {
    const uint32_t rows = 5, in_dim = 7, out_dim = 9;
    uint16_t *in = malloc(rows * in_dim * 2), *w = malloc(out_dim * in_dim * 2),
             *b = malloc(out_dim * 2), *ref = malloc(rows * out_dim * 2),
             *got = malloc(rows * out_dim * 2);
    float *fin = malloc(rows * in_dim * 4), *fw = malloc(out_dim * in_dim * 4),
          *fb = malloc(out_dim * 4), *fref = malloc(rows * out_dim * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) { in[i] = host_f32_to_bf16(frand(1.0f)); fin[i] = host_bf16_to_f32(in[i]); }
    for (uint32_t i = 0; i < out_dim * in_dim; i++) { w[i] = host_f32_to_bf16(frand(1.0f)); fw[i] = host_bf16_to_f32(w[i]); }
    for (uint32_t i = 0; i < out_dim; i++) { b[i] = host_f32_to_bf16(frand(1.0f)); fb[i] = host_bf16_to_f32(b[i]); }
    ref_linear_f32(fin, fw, fb, fref, rows, in_dim, out_dim);
    for (uint32_t i = 0; i < rows * out_dim; i++) ref[i] = host_f32_to_bf16(fref[i]);
    h3_gpu_tensor *ti = mk_bf16(in, rows * in_dim),
                  *tw = mk_bf16(w, out_dim * in_dim),
                  *tb = mk_bf16(b, out_dim), *to = new_bf16(rows * out_dim);
    OP(h3_gpu_linear_bf16(gpu, to, ti, tw, tb, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, rows * out_dim));
    check_bf16_gemm("linear_bf16 small+bias", got, ref, rows * out_dim);
    /* no bias */
    ref_linear_f32(fin, fw, NULL, fref, rows, in_dim, out_dim);
    for (uint32_t i = 0; i < rows * out_dim; i++) ref[i] = host_f32_to_bf16(fref[i]);
    OP(h3_gpu_linear_bf16(gpu, to, ti, tw, NULL, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, rows * out_dim));
    check_bf16_gemm("linear_bf16 small nobias", got, ref, rows * out_dim);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
    free(fin); free(fw); free(fb); free(fref);
}

static void test_linear_bf16_big(void) {
    const uint32_t rows = 32, in_dim = 256, out_dim = 256;
    uint16_t *in = malloc(rows * in_dim * 2), *w = malloc(out_dim * in_dim * 2),
             *ref = malloc(rows * out_dim * 2), *got = malloc(rows * out_dim * 2);
    float *fin = malloc(rows * in_dim * 4), *fw = malloc(out_dim * in_dim * 4),
          *fref = malloc(rows * out_dim * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) { in[i] = host_f32_to_bf16(frand(0.2f)); fin[i] = host_bf16_to_f32(in[i]); }
    for (uint32_t i = 0; i < out_dim * in_dim; i++) { w[i] = host_f32_to_bf16(frand(0.2f)); fw[i] = host_bf16_to_f32(w[i]); }
    ref_linear_f32(fin, fw, NULL, fref, rows, in_dim, out_dim);
    for (uint32_t i = 0; i < rows * out_dim; i++) ref[i] = host_f32_to_bf16(fref[i]);
    h3_gpu_tensor *ti = mk_bf16(in, rows * in_dim),
                  *tw = mk_bf16(w, out_dim * in_dim),
                  *to = new_bf16(rows * out_dim);
    h3_gpu_stats before, after;
    OP(h3_gpu_get_stats(gpu, &before));
    OP(h3_gpu_linear_bf16(gpu, to, ti, tw, NULL, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_get_stats(gpu, &after));
    OP(h3_gpu_tensor_read_bf16(to, got, rows * out_dim));
    check_bf16_gemm("linear_bf16 32x256x256", got, ref, rows * out_dim);
    if (after.mps_linear_dispatches != before.mps_linear_dispatches + 1) {
        printf("FAIL linear_bf16 big did not bump mps_linear_dispatches\n");
        failures++;
    } else {
        printf("ok   linear_bf16 big bumped mps_linear_dispatches\n");
    }
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(to);
    free(in); free(w); free(ref); free(got); free(fin); free(fw); free(fref);
}

/* Misaligned GEMM staging: element-offset views into a tensor hand cuBLAS
 * pointers that are element-aligned but not 16-byte aligned, and odd dims
 * make the leading-dimension byte stride misaligned too; without staging
 * cublasLt serves those from its slow align1 SIMT kernels. h3_gpu.h keeps
 * h3_gpu_tensor opaque, so the CUDA backend's layout (h3_cuda_internal.h)
 * is completed here to build 1-element-offset views. The base tensor owns
 * the allocation; the views must not be freed. */
struct h3_gpu_tensor {
    void *data;
    size_t elements;
    size_t bytes;
    h3_gpu_dtype dtype;
    struct h3_gpu *owner;
};
static void offset_view(h3_gpu_tensor *view, h3_gpu_tensor *base,
                        size_t offset) {
    *view = *base;
    view->data = (char *)view->data +
                 offset * (view->dtype == H3_GPU_BF16 ? 2 : 4);
    view->elements -= offset;
}

static void test_linear_f32_misaligned(void) {
    /* odd dims: pointer AND leading-dimension misalignment, bias misaligned
     * too (staging + fallback bias kernel) */
    const uint32_t rows = 6, in_dim = 7, out_dim = 9;
    float *in = malloc(rows * in_dim * 4), *w = malloc(out_dim * in_dim * 4),
          *b = malloc(out_dim * 4), *ref = malloc(rows * out_dim * 4),
          *got = malloc(rows * out_dim * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) in[i] = frand(1.0f);
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w[i] = frand(1.0f);
    for (uint32_t i = 0; i < out_dim; i++) b[i] = frand(1.0f);
    ref_linear_f32(in, w, b, ref, rows, in_dim, out_dim);
    h3_gpu_tensor *ti = new_f32(rows * in_dim + 1),
                  *tw = mk_f32(w, out_dim * in_dim),
                  *tb = new_f32(out_dim + 1),
                  *to = new_f32(rows * out_dim + 1);
    OP(h3_gpu_tensor_write_f32_range(ti, 1, in, rows * in_dim));
    OP(h3_gpu_tensor_write_f32_range(tb, 1, b, out_dim));
    h3_gpu_tensor vi, vb, vo;
    offset_view(&vi, ti, 1);
    offset_view(&vb, tb, 1);
    offset_view(&vo, to, 1);
    OP(h3_gpu_linear_f32(gpu, &vo, &vi, tw, &vb, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_f32_range(to, 1, got, rows * out_dim));
    check_f32_gemm("linear_f32 misaligned+bias", got, ref, rows * out_dim);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);

    /* even dims: pointer-only misalignment, aligned bias (bias epilogue) */
    const uint32_t rows2 = 5, in_dim2 = 8, out_dim2 = 16;
    float *in2 = malloc(rows2 * in_dim2 * 4), *w2 = malloc(out_dim2 * in_dim2 * 4),
          *b2 = malloc(out_dim2 * 4), *ref2 = malloc(rows2 * out_dim2 * 4),
          *got2 = malloc(rows2 * out_dim2 * 4);
    for (uint32_t i = 0; i < rows2 * in_dim2; i++) in2[i] = frand(1.0f);
    for (uint32_t i = 0; i < out_dim2 * in_dim2; i++) w2[i] = frand(1.0f);
    for (uint32_t i = 0; i < out_dim2; i++) b2[i] = frand(1.0f);
    ref_linear_f32(in2, w2, b2, ref2, rows2, in_dim2, out_dim2);
    ti = new_f32(rows2 * in_dim2 + 1);
    tw = mk_f32(w2, out_dim2 * in_dim2);
    tb = mk_f32(b2, out_dim2);
    to = new_f32(rows2 * out_dim2 + 1);
    OP(h3_gpu_tensor_write_f32_range(ti, 1, in2, rows2 * in_dim2));
    offset_view(&vi, ti, 1);
    offset_view(&vo, to, 1);
    OP(h3_gpu_linear_f32(gpu, &vo, &vi, tw, tb, rows2, in_dim2, out_dim2));
    run();
    OP(h3_gpu_tensor_read_f32_range(to, 1, got2, rows2 * out_dim2));
    check_f32_gemm("linear_f32 misaligned epi", got2, ref2, rows2 * out_dim2);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
    free(in2); free(w2); free(b2); free(ref2); free(got2);
}

static void test_linear_bf16_misaligned(void) {
    /* in_dim=12 (24-byte ld) and out_dim=10 (20-byte ld): pointer and ld
     * misalignment for BF16, aligned bias so the epilogue serves the staged
     * GEMM */
    const uint32_t rows = 4, in_dim = 12, out_dim = 10;
    uint16_t *in = malloc(rows * in_dim * 2), *w = malloc(out_dim * in_dim * 2),
             *b = malloc(out_dim * 2), *ref = malloc(rows * out_dim * 2),
             *got = malloc(rows * out_dim * 2);
    float *fin = malloc(rows * in_dim * 4), *fw = malloc(out_dim * in_dim * 4),
          *fb = malloc(out_dim * 4), *fref = malloc(rows * out_dim * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) { in[i] = host_f32_to_bf16(frand(1.0f)); fin[i] = host_bf16_to_f32(in[i]); }
    for (uint32_t i = 0; i < out_dim * in_dim; i++) { w[i] = host_f32_to_bf16(frand(1.0f)); fw[i] = host_bf16_to_f32(w[i]); }
    for (uint32_t i = 0; i < out_dim; i++) { b[i] = host_f32_to_bf16(frand(1.0f)); fb[i] = host_bf16_to_f32(b[i]); }
    ref_linear_f32(fin, fw, fb, fref, rows, in_dim, out_dim);
    for (uint32_t i = 0; i < rows * out_dim; i++) ref[i] = host_f32_to_bf16(fref[i]);
    h3_gpu_tensor *ti = new_bf16(rows * in_dim + 1),
                  *tw = mk_bf16(w, out_dim * in_dim),
                  *tb = mk_bf16(b, out_dim),
                  *to = new_bf16(rows * out_dim + 1);
    OP(h3_gpu_tensor_write_bf16_range(ti, 1, in, rows * in_dim));
    h3_gpu_tensor vi, vo;
    offset_view(&vi, ti, 1);
    offset_view(&vo, to, 1);
    OP(h3_gpu_linear_bf16(gpu, &vo, &vi, tw, tb, rows, in_dim, out_dim));
    run();
    uint16_t *raw = malloc((rows * out_dim + 1) * 2);
    OP(h3_gpu_tensor_read_bf16(to, raw, rows * out_dim + 1));
    memcpy(got, raw + 1, rows * out_dim * 2);
    check_bf16_gemm("linear_bf16 misaligned", got, ref, rows * out_dim);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got);
    free(fin); free(fw); free(fb); free(fref);
    free(raw);
}

/* patch projection: F32 input/weight/bias, BF16 output, out_dim=5376 */
static void test_patch_linear_bf16(void) {
    const uint32_t rows = 16, in_dim = 32, out_dim = 5376;
    const size_t in_off = 5, out_off = 7; /* element offsets for _offset */
    float *in = malloc((rows * in_dim + in_off) * 4),
          *w = malloc(out_dim * in_dim * 4), *b = malloc(out_dim * 4),
          *ref = malloc(rows * out_dim * 4);
    uint16_t *got = malloc((rows * out_dim + out_off) * 2),
             *refb = malloc(rows * out_dim * 2);
    for (size_t i = 0; i < rows * in_dim + in_off; i++) in[i] = frand(0.5f);
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w[i] = frand(0.5f);
    for (uint32_t i = 0; i < out_dim; i++) b[i] = frand(0.5f);
    ref_linear_f32(in + in_off, w, b, ref, rows, in_dim, out_dim);
    for (uint32_t i = 0; i < rows * out_dim; i++) refb[i] = host_f32_to_bf16(ref[i]);

    /* plain patch_linear_bf16 */
    h3_gpu_tensor *ti = mk_f32(in + in_off, rows * in_dim),
                  *tw = mk_f32(w, out_dim * in_dim),
                  *tb = mk_f32(b, out_dim), *to = new_bf16(rows * out_dim);
    OP(h3_gpu_patch_linear_bf16(gpu, to, ti, tw, tb, rows, in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, rows * out_dim));
    check_bf16_gemm("patch_linear_bf16 16x32x5376", got, refb,
                    rows * out_dim);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(to);

    /* offset variant */
    ti = mk_f32(in, rows * in_dim + in_off);
    to = new_bf16(rows * out_dim + out_off);
    OP(h3_gpu_patch_linear_bf16_offset(gpu, to, out_off, ti, in_off, tw, tb,
                                       rows, in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, rows * out_dim + out_off));
    check_bf16_gemm("patch_linear_bf16_offset", got + out_off, refb,
                    rows * out_dim);

    /* invalid shapes must fail */
    if (h3_gpu_patch_linear_bf16(gpu, to, ti, tw, tb, rows, 64, out_dim) ||
        h3_gpu_patch_linear_bf16(gpu, to, ti, tw, tb, rows, in_dim, 1024)) {
        printf("FAIL patch_linear accepted invalid shape\n"); failures++;
    }
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got); free(refb);
}

static void test_patch_linear_bf16_map(void) {
    const uint32_t rows = 8, out_rows = 16, in_dim = 96, out_dim = 5376;
    float *in = malloc(rows * in_dim * 4), *w = malloc(out_dim * in_dim * 4),
          *b = malloc(out_dim * 4), *ref = malloc(rows * out_dim * 4);
    uint16_t *got = malloc(out_rows * out_dim * 2),
             *refb = malloc(rows * out_dim * 2);
    uint32_t map[8] = {3, 0, 15, 7, 5, 2, 11, 9}; /* scattered, unique */
    uint16_t *sentinel = malloc(out_rows * out_dim * 2);
    for (uint32_t i = 0; i < rows * in_dim; i++) in[i] = frand(0.5f);
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w[i] = frand(0.5f);
    for (uint32_t i = 0; i < out_dim; i++) b[i] = frand(0.5f);
    for (uint32_t i = 0; i < out_rows * out_dim; i++)
        sentinel[i] = host_f32_to_bf16(-99.0f);
    ref_linear_f32(in, w, b, ref, rows, in_dim, out_dim);
    for (uint32_t i = 0; i < rows * out_dim; i++) refb[i] = host_f32_to_bf16(ref[i]);

    h3_gpu_tensor *ti = mk_f32(in, rows * in_dim),
                  *tw = mk_f32(w, out_dim * in_dim),
                  *tb = mk_f32(b, out_dim),
                  *tm = h3_gpu_tensor_from_u32(gpu, map, rows),
                  *to = mk_bf16(sentinel, out_rows * out_dim);
    if (!tm) { fprintf(stderr, "u32 alloc failed\n"); exit(2); }
    OP(h3_gpu_patch_linear_bf16_map(gpu, to, ti, tw, tb, tm, out_rows, rows,
                                    in_dim, out_dim));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, out_rows * out_dim));
    /* mapped rows (later writes win for the duplicate); GEMM tolerance as in
     * check_bf16_gemm — cuBLAS reorders the F32 accumulation */
    int written[16] = {0};
    for (uint32_t r = 0; r < rows; r++) written[map[r]] = 1;
    double max_err = 0.0;
    int bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < out_dim; c++) {
            float g = host_bf16_to_f32(got[map[r] * out_dim + c]);
            float e = host_bf16_to_f32(refb[r * out_dim + c]);
            double err = fabs((double)g - (double)e);
            if (err > max_err) max_err = err;
            if (!(err <= 1e-3 + 1e-2 * fabs((double)e))) bad++;
        }
    }
    report("patch_linear_bf16_map rows", max_err, bad);
    /* unmapped rows keep the sentinel */
    bad = 0;
    for (uint32_t orow = 0; orow < out_rows; orow++) {
        if (written[orow]) continue;
        for (uint32_t c = 0; c < out_dim; c++)
            if (got[orow * out_dim + c] != sentinel[orow * out_dim + c]) bad++;
    }
    report("patch_linear_bf16_map untouched", 0.0, bad);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tb);
    h3_gpu_tensor_free(tm); h3_gpu_tensor_free(to);
    free(in); free(w); free(b); free(ref); free(got); free(refb);
    free(sentinel);
}

static void test_mlp_bf16(void) {
    const uint32_t rows = 4, in_dim = 8, hidden = 6, out_dim = 5;
    uint16_t *in = malloc(rows * in_dim * 2),
             *fc1 = malloc(2 * hidden * in_dim * 2),
             *fc2 = malloc(out_dim * hidden * 2),
             *ref = malloc(rows * out_dim * 2), *got = malloc(rows * out_dim * 2);
    float *fin = malloc(rows * in_dim * 4), *ffc1 = malloc(2 * hidden * in_dim * 4),
          *ffc2 = malloc(out_dim * hidden * 4);
    for (uint32_t i = 0; i < rows * in_dim; i++) { in[i] = host_f32_to_bf16(frand(1.0f)); fin[i] = host_bf16_to_f32(in[i]); }
    for (uint32_t i = 0; i < 2 * hidden * in_dim; i++) { fc1[i] = host_f32_to_bf16(frand(0.5f)); ffc1[i] = host_bf16_to_f32(fc1[i]); }
    for (uint32_t i = 0; i < out_dim * hidden; i++) { fc2[i] = host_f32_to_bf16(frand(0.5f)); ffc2[i] = host_bf16_to_f32(fc2[i]); }
    /* reference: fused[r][2h], gate = first half, up = second half */
    float *act = malloc(rows * hidden * 4);
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t hh = 0; hh < hidden; hh++) {
            float gate = 0.0f, up = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++) {
                gate += fin[r * in_dim + k] * ffc1[hh * in_dim + k];
                up += fin[r * in_dim + k] * ffc1[(hidden + hh) * in_dim + k];
            }
            act[r * hidden + hh] = ref_silu(gate) * up;
        }
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < hidden; k++)
                sum += act[r * hidden + k] * ffc2[o * hidden + k];
            ref[r * out_dim + o] = host_f32_to_bf16(sum);
        }
    h3_gpu_tensor *ti = mk_bf16(in, rows * in_dim),
                  *t1 = mk_bf16(fc1, 2 * hidden * in_dim),
                  *t2 = mk_bf16(fc2, out_dim * hidden),
                  *to = new_bf16(rows * out_dim);
    h3_gpu_stats before, after;
    OP(h3_gpu_get_stats(gpu, &before));
    OP(h3_gpu_mlp_bf16(gpu, to, ti, t1, t2, rows, in_dim, hidden, out_dim));
    run();
    OP(h3_gpu_get_stats(gpu, &after));
    OP(h3_gpu_tensor_read_bf16(to, got, rows * out_dim));
    check_bf16_gemm("mlp_bf16 4x8x6x5", got, ref, rows * out_dim);
    if (after.mps_linear_dispatches != before.mps_linear_dispatches + 2) {
        printf("FAIL mlp_bf16 did not bump mps_linear_dispatches by 2 "
               "(%llu -> %llu)\n",
               (unsigned long long)before.mps_linear_dispatches,
               (unsigned long long)after.mps_linear_dispatches);
        failures++;
    } else {
        printf("ok   mlp_bf16 bumped mps_linear_dispatches by 2\n");
    }
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(t1); h3_gpu_tensor_free(t2);
    h3_gpu_tensor_free(to);
    free(in); free(fc1); free(fc2); free(ref); free(got);
    free(fin); free(ffc1); free(ffc2); free(act);
}

/* 200 mixed write_range/read_range cycles with alternating contents:
 * exercises pinned-ring slot reuse (the ring has 8 slots, so the loop wraps
 * it many times and forces event waits), slot growth, the >64 MiB
 * synchronous fallback, and end-to-end data integrity. */
static void test_h2d_ring_stress(void) {
    enum { N = 4096, BIG = 5 * 1024 * 1024 };  /* BIG: 20 MiB f32 */
    float *host = (float *)malloc(N * sizeof(float));
    float *back = (float *)malloc(N * sizeof(float));
    uint16_t *hostb = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *backb = (uint16_t *)malloc(N * sizeof(uint16_t));
    float *big = (float *)malloc(BIG * sizeof(float));
    if (!host || !back || !hostb || !backb || !big) {
        fprintf(stderr, "oom\n"); exit(2);
    }
    h3_gpu_tensor *tf = new_f32(N);
    h3_gpu_tensor *tb = new_bf16(N);
    h3_gpu_tensor *tg = new_f32(BIG);
    int bad = 0;
    for (int i = 0; i < 200; i++) {
        size_t offset = (size_t)((i * 37) % 512);
        size_t count = (size_t)(64 + (i * 911) % (N - 512 - 64));
        float seed = (float)i * 0.5f;
        for (size_t k = 0; k < count; k++) host[k] = seed + (float)k;
        for (size_t k = 0; k < count; k++)
            hostb[k] = host_f32_to_bf16(seed - (float)k);
        OP(h3_gpu_tensor_write_f32_range(tf, offset, host, count));
        OP(h3_gpu_tensor_write_bf16_range(tb, offset, hostb, count));
        /* extra writes to a disjoint tail region keep many async copies in
         * flight, so ring slots get reused while earlier copies may still
         * be pending */
        OP(h3_gpu_tensor_write_f32_range(tf, N - 16, host, 16));
        OP(h3_gpu_tensor_write_bf16_range(tb, N - 16, hostb, 16));
        if ((i % 64) == 0) {
            /* slot-growth path: 20 MiB staged write (> initial 1 MiB slot) */
            for (size_t k = 0; k < BIG; k++) big[k] = (float)(i + (int)k);
            OP(h3_gpu_tensor_write_f32_range(tg, 0, big, BIG));
            float probe[4];
            OP(h3_gpu_tensor_read_f32_range(tg, BIG - 4, probe, 4));
            for (int k = 0; k < 4; k++)
                if (probe[k] != (float)(i + BIG - 4 + k)) bad++;
        }
        OP(h3_gpu_tensor_read_f32_range(tf, offset, back, count));
        OP(h3_gpu_tensor_read_bf16(tb, backb, N));
        for (size_t k = 0; k < count; k++)
            if (back[k] != host[k]) { bad++; break; }
        for (size_t k = 0; k < count; k++)
            if (backb[offset + k] != hostb[k]) { bad++; break; }
    }
    /* oversized write (> 64 MiB slot cap) takes the synchronous path */
    {
        enum { HUGE = 20 * 1024 * 1024 };  /* 80 MiB f32 */
        h3_gpu_tensor *th = new_f32(HUGE);
        float *huge = (float *)malloc(HUGE * sizeof(float));
        if (!huge) { fprintf(stderr, "oom\n"); exit(2); }
        for (size_t k = 0; k < HUGE; k++) huge[k] = (float)(k % 977);
        OP(h3_gpu_tensor_write_f32(th, huge, HUGE));
        float probe[4];
        OP(h3_gpu_tensor_read_f32_range(th, HUGE - 4, probe, 4));
        for (int k = 0; k < 4; k++)
            if (probe[k] != (float)((HUGE - 4 + k) % 977)) bad++;
        free(huge);
        h3_gpu_tensor_free(th);
    }
    report("h2d_ring_stress", 0.0, bad);
    free(host); free(back); free(hostb); free(backb); free(big);
    h3_gpu_tensor_free(tf); h3_gpu_tensor_free(tb); h3_gpu_tensor_free(tg);
}

/* Alloc/free churn across the caching allocator: 100 random-size tensors,
 * free half, realloc (likely cache hits), verify contents, free all, and
 * check the stats accounting balances exactly — cache-held bytes must never
 * count as live. */
static void test_alloc_churn(void) {
    enum { N = 100 };
    h3_gpu_stats before, after;
    OP(h3_gpu_get_stats(gpu, &before));
    h3_gpu_tensor *tensors[N];
    size_t elements[N];
    uint64_t allocated = 0, allocations = 0;
    float *host = (float *)malloc(65536 * sizeof(float));
    float *back = (float *)malloc(65536 * sizeof(float));
    if (!host || !back) { fprintf(stderr, "oom\n"); exit(2); }
    /* round 0: allocate all 100 */
    for (int i = 0; i < N; i++) {
        elements[i] = 1 + (size_t)(fabsf(frand(1.0f)) * 65535.0f);
        int kind = i % 3;
        tensors[i] = kind == 0 ? new_f32(elements[i]) :
                     kind == 1 ? new_bf16(elements[i]) :
                     h3_gpu_tensor_new_i8(gpu, elements[i]);
        if (!tensors[i]) {
            fprintf(stderr, "churn alloc failed: %s\n", h3_gpu_error(gpu));
            exit(2);
        }
        allocated += elements[i] * (size_t)(kind == 0 ? 4 : kind == 1 ? 2 : 1);
        allocations++;
    }
    /* free the even half */
    for (int i = 0; i < N; i += 2) {
        h3_gpu_tensor_free(tensors[i]);
        tensors[i] = NULL;
    }
    /* realloc the freed slots with the same sizes: cache-hit candidates */
    for (int i = 0; i < N; i += 2) {
        int kind = i % 3;
        tensors[i] = kind == 0 ? new_f32(elements[i]) :
                     kind == 1 ? new_bf16(elements[i]) :
                     h3_gpu_tensor_new_i8(gpu, elements[i]);
        if (!tensors[i]) {
            fprintf(stderr, "churn realloc failed: %s\n", h3_gpu_error(gpu));
            exit(2);
        }
        allocated += elements[i] * (size_t)(kind == 0 ? 4 : kind == 1 ? 2 : 1);
        allocations++;
    }
    /* data integrity on the f32 tensors: full write + read-back pattern */
    int bad = 0;
    for (int i = 0; i < N; i += 3) {
        for (size_t k = 0; k < elements[i]; k++)
            host[k] = (float)(i * 7 + (int)k);
        OP(h3_gpu_tensor_write_f32(tensors[i], host, elements[i]));
    }
    for (int i = 0; i < N; i += 3) {
        OP(h3_gpu_tensor_read_f32(tensors[i], back, elements[i]));
        for (size_t k = 0; k < elements[i]; k++)
            if (back[k] != (float)(i * 7 + (int)k)) { bad++; break; }
    }
    for (int i = 0; i < N; i++) h3_gpu_tensor_free(tensors[i]);
    OP(h3_gpu_get_stats(gpu, &after));
    if (after.live_bytes != before.live_bytes) {
        printf("FAIL alloc_churn live_bytes leaked: %llu -> %llu\n",
               (unsigned long long)before.live_bytes,
               (unsigned long long)after.live_bytes);
        bad++;
    }
    if (after.allocated_bytes - before.allocated_bytes != allocated) {
        printf("FAIL alloc_churn allocated_bytes off: got %llu want %llu\n",
               (unsigned long long)(after.allocated_bytes -
                                    before.allocated_bytes),
               (unsigned long long)allocated);
        bad++;
    }
    if (after.tensor_allocations - before.tensor_allocations != allocations) {
        printf("FAIL alloc_churn tensor_allocations off: got %llu want %llu\n",
               (unsigned long long)(after.tensor_allocations -
                                    before.tensor_allocations),
               (unsigned long long)allocations);
        bad++;
    }
    report("alloc_churn", 0.0, bad);
    free(host); free(back);
}

/* File -> device weight loading: 40 MiB of known bf16 values at a nonzero
 * file offset (above the parallel-read threshold), read back bit-exact;
 * h3_gpu_tensor_read_file_bf16 re-filling the same tensor; a small load for
 * the sequential path; and a short read past EOF, which must fail. */
static void test_file_load_bf16(void) {
    const size_t n = (size_t)20 << 20;  /* 40 MiB of bf16 */
    const uint64_t header = 4096;
    uint16_t *host = (uint16_t *)malloc(n * sizeof(uint16_t));
    uint16_t *back = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!host || !back) { fprintf(stderr, "oom\n"); exit(2); }
    for (size_t i = 0; i < n; i++) host[i] = host_f32_to_bf16(frand(4.0f));
    char path[] = "/tmp/h3_cuda_file_load_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { fprintf(stderr, "mkstemp failed: %s\n", strerror(errno)); exit(2); }
    unsigned char junk[4096];
    memset(junk, 0xa5, sizeof(junk));
    if (write(fd, junk, sizeof(junk)) != (ssize_t)sizeof(junk)) {
        fprintf(stderr, "write header failed\n"); exit(2);
    }
    size_t written = 0;
    while (written < n * sizeof(uint16_t)) {
        ssize_t w = write(fd, (const unsigned char *)host + written,
                          n * sizeof(uint16_t) - written);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { fprintf(stderr, "write payload failed\n"); exit(2); }
        written += (size_t)w;
    }
    close(fd);
    int bad = 0;
    /* large load: parallel pread path */
    h3_gpu_tensor *big = h3_gpu_tensor_load_bf16(gpu, path, header, n);
    if (!big) {
        fprintf(stderr, "load_bf16 failed: %s\n", h3_gpu_error(gpu));
        exit(2);
    }
    OP(h3_gpu_tensor_read_bf16(big, back, n));
    if (memcmp(back, host, n * sizeof(uint16_t))) bad++;
    /* re-fill the same tensor from the file at the same offset */
    char error[256];
    if (!h3_gpu_tensor_read_file_bf16(big, path, header, n, error,
                                      sizeof(error))) {
        fprintf(stderr, "read_file_bf16 failed: %s\n", error);
        exit(2);
    }
    memset(back, 0, n * sizeof(uint16_t));
    OP(h3_gpu_tensor_read_bf16(big, back, n));
    if (memcmp(back, host, n * sizeof(uint16_t))) bad++;
    h3_gpu_tensor_free(big);
    /* small load: sequential path */
    h3_gpu_tensor *small = h3_gpu_tensor_load_bf16(gpu, path, header, 1024);
    if (!small) {
        fprintf(stderr, "small load_bf16 failed: %s\n", h3_gpu_error(gpu));
        exit(2);
    }
    OP(h3_gpu_tensor_read_bf16(small, back, 1024));
    if (memcmp(back, host, 1024 * sizeof(uint16_t))) bad++;
    h3_gpu_tensor_free(small);
    /* reading past EOF must fail cleanly */
    if (h3_gpu_tensor_load_bf16(gpu, path, header, n + 4096)) {
        printf("FAIL file_load_bf16 accepted a short read\n");
        bad++;
    }
    report("file_load_bf16", 0.0, bad);
    unlink(path);
    free(host);
    free(back);
}

static void test_validation(void) {
    /* ops without an active command must fail */
    h3_gpu_stats st;
    OP(h3_gpu_get_stats(gpu, &st));
    if (!h3_gpu_submit(gpu)) { fprintf(stderr, "submit failed\n"); exit(2); }
    float v[4] = {0, 0, 0, 0};
    h3_gpu_tensor *t = mk_f32(v, 4);
    if (h3_gpu_silu_f32(gpu, t, t, 4)) {
        printf("FAIL silu_f32 ran without active command\n"); failures++;
    } else {
        printf("ok   silu_f32 requires active command\n");
    }
    OP(h3_gpu_begin(gpu));
    /* wrong dtype must fail */
    uint16_t bv[4] = {0, 0, 0, 0};
    h3_gpu_tensor *tb = mk_bf16(bv, 4);
    if (h3_gpu_silu_f32(gpu, t, tb, 4)) {
        printf("FAIL silu_f32 accepted bf16 input\n"); failures++;
    } else {
        printf("ok   silu_f32 rejects wrong dtype\n");
    }
    h3_gpu_tensor_free(t); h3_gpu_tensor_free(tb);
}

int main(void) {
    char error[512];
    /* F32 GEMMs default to TF32 tensor cores on CUDA (H3_CUDA_TF32); the
     * exact-F32 checks below pin it off, and test_linear_f32_tf32 covers the
     * TF32 path with its own tolerance. */
    setenv("H3_CUDA_TF32", "0", 1);
    gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_gpu_create failed: %s\n", error); return 2; }
    OP(h3_gpu_begin(gpu));

    test_silu_f32();
    test_silu_bf16();
    test_gelu_bf16(0);
    test_gelu_bf16(1);
    test_geglu_f32();
    test_clip_f32();
    test_add_scaled_f32();
    test_scale_add_f32();
    test_add_sub_bf16();
    test_silu_mul_bf16();
    test_swiglu_f32();
    test_swiglu_bf16();
    test_embedding_bf16();
    test_rms_norm_f32();
    test_rms_norm_bf16();
    test_layer_norm_f32();
    test_layer_norm_bf16();
    test_weight_norm_f32();
    test_head_rms_norm_bf16();
    test_linear_f32_small();
    test_linear_f32_big();
    test_linear_f32_tf32();
    test_linear_bf16_small();
    test_linear_bf16_big();
    test_linear_f32_misaligned();
    test_linear_bf16_misaligned();
    test_patch_linear_bf16();
    test_patch_linear_bf16_map();
    test_mlp_bf16();
    test_h2d_ring_stress();
    test_alloc_churn();
    test_file_load_bf16();
    test_validation();

    if (!h3_gpu_submit(gpu)) { fprintf(stderr, "final submit failed\n"); return 2; }
    h3_gpu_free(gpu);
    if (failures) { printf("\n%d FAILURES\n", failures); return 1; }
    printf("\nALL TESTS PASSED\n");
    return 0;
}
