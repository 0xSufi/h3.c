/* Freestanding correctness test for h3_cuda_dit.cu: the 12 DiT fused ops
 * against independent CPU references, plus the 7 Metal-4-only stubs. */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        printf("FAIL %-32s max_err=%.3g (%d mismatches)\n", name, max_err,
               bad);
        failures++;
    } else {
        printf("ok   %-32s max_err=%.3g\n", name, max_err);
    }
}

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
static h3_gpu_tensor *mk_u32(const uint32_t *v, size_t n) {
    h3_gpu_tensor *t = h3_gpu_tensor_from_u32(gpu, v, n);
    if (!t) { fprintf(stderr, "alloc u32 failed: %s\n", h3_gpu_error(gpu)); exit(2); }
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
/* shared shapes */
#define ROWS 4
#define WIDTH 8
#define SLOTS 3
#define EPS 1e-5f
static const uint32_t ROW_MAP[ROWS] = {2, 0, 3, 1};
#define SHIFT_SLOT 0
#define SCALE_SLOT 1
#define GATE_SLOT 2

static void ref_adaln(const float *in, const float *w, const float *mod,
                      const uint32_t *row_map, float *ref, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        float ss = 0.0f;
        for (uint32_t c = 0; c < width; c++)
            ss += in[r * width + c] * in[r * width + c];
        float inv = 1.0f / sqrtf(ss / (float)width + eps);
        size_t base = (size_t)row_map[r] * slots * width;
        for (uint32_t c = 0; c < width; c++) {
            float normalized = in[r * width + c] * inv * w[c];
            float shift = mod[base + (size_t)shift_slot * width + c];
            float scale = mod[base + (size_t)scale_slot * width + c];
            ref[r * width + c] = normalized * (1.0f + scale) + shift;
        }
    }
}

static void ref_gate(const float *res, const float *branch, const float *mod,
                     const uint32_t *row_map, float *ref, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    for (uint32_t r = 0; r < rows; r++) {
        size_t base = (size_t)row_map[r] * slots * width;
        for (uint32_t c = 0; c < width; c++) {
            float gate = mod[base + (size_t)gate_slot * width + c];
            ref[r * width + c] =
                res[r * width + c] + branch[r * width + c] * gate;
        }
    }
}

static void fill_bf16(uint16_t *v, float *f, size_t n, float range) {
    for (size_t i = 0; i < n; i++) {
        v[i] = host_f32_to_bf16(frand(range));
        if (f) f[i] = host_bf16_to_f32(v[i]);
    }
}

/* ------------------------------------------------------------ tests */
static void test_adaln_f32(void) {
    const uint32_t n = ROWS * WIDTH, mod_n = ROWS * SLOTS * WIDTH;
    float *in = malloc(n * 4), *w = malloc(WIDTH * 4), *mod = malloc(mod_n * 4),
          *ref = malloc(n * 4), *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) in[i] = frand(3.0f);
    for (uint32_t i = 0; i < WIDTH; i++) w[i] = frand(2.0f);
    for (uint32_t i = 0; i < mod_n; i++) mod[i] = frand(1.0f);
    ref_adaln(in, w, mod, ROW_MAP, ref, ROWS, WIDTH, SLOTS, SHIFT_SLOT,
              SCALE_SLOT, EPS);
    h3_gpu_tensor *ti = mk_f32(in, n), *tw = mk_f32(w, WIDTH),
                  *tm = mk_f32(mod, mod_n), *tr = mk_u32(ROW_MAP, ROWS),
                  *to = new_f32(n);
    OP(h3_gpu_adaln_f32(gpu, to, ti, tw, tm, tr, ROWS, WIDTH, SLOTS,
                        SHIFT_SLOT, SCALE_SLOT, EPS));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("adaln_f32", got, ref, n, 1e-5, 1e-5);
    /* out-of-range slot must fail */
    if (h3_gpu_adaln_f32(gpu, to, ti, tw, tm, tr, ROWS, WIDTH, SLOTS,
                         SLOTS, SCALE_SLOT, EPS)) {
        printf("FAIL adaln_f32 accepted shift_slot >= slots\n"); failures++;
    } else {
        printf("ok   adaln_f32 rejects bad slot\n");
    }
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tm);
    h3_gpu_tensor_free(tr); h3_gpu_tensor_free(to);
    free(in); free(w); free(mod); free(ref); free(got);
}

static void test_gate_f32(void) {
    const uint32_t n = ROWS * WIDTH, mod_n = ROWS * SLOTS * WIDTH;
    float *res = malloc(n * 4), *br = malloc(n * 4), *mod = malloc(mod_n * 4),
          *ref = malloc(n * 4), *got = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) { res[i] = frand(3.0f); br[i] = frand(3.0f); }
    for (uint32_t i = 0; i < mod_n; i++) mod[i] = frand(1.0f);
    ref_gate(res, br, mod, ROW_MAP, ref, ROWS, WIDTH, SLOTS, GATE_SLOT);
    h3_gpu_tensor *tr = mk_f32(res, n), *tb = mk_f32(br, n),
                  *tm = mk_f32(mod, mod_n), *tmap = mk_u32(ROW_MAP, ROWS),
                  *to = new_f32(n);
    OP(h3_gpu_gate_f32(gpu, to, tr, tb, tm, tmap, ROWS, WIDTH, SLOTS,
                       GATE_SLOT));
    run();
    OP(h3_gpu_tensor_read_f32(to, got, n));
    check_f32("gate_f32", got, ref, n, 1e-5, 1e-5);
    h3_gpu_tensor_free(tr); h3_gpu_tensor_free(tb); h3_gpu_tensor_free(tm);
    h3_gpu_tensor_free(tmap); h3_gpu_tensor_free(to);
    free(res); free(br); free(mod); free(ref); free(got);
}

static void test_adaln_bf16(void) {
    const uint32_t n = ROWS * WIDTH, mod_n = ROWS * SLOTS * WIDTH;
    const size_t in_off = 5;
    uint16_t *in = malloc((n + in_off) * 2), *w = malloc(WIDTH * 2),
             *mod = malloc(mod_n * 2), *ref = malloc(n * 2), *got = malloc(n * 2);
    float *fin = malloc((n + in_off) * 4), *fw = malloc(WIDTH * 4),
          *fmod = malloc(mod_n * 4), *fref = malloc(n * 4);
    fill_bf16(in, fin, n + in_off, 3.0f);
    fill_bf16(w, fw, WIDTH, 2.0f);
    fill_bf16(mod, fmod, mod_n, 1.0f);

    /* plain */
    ref_adaln(fin, fw, fmod, ROW_MAP, fref, ROWS, WIDTH, SLOTS, SHIFT_SLOT,
              SCALE_SLOT, EPS);
    for (uint32_t i = 0; i < n; i++) ref[i] = host_f32_to_bf16(fref[i]);
    h3_gpu_tensor *ti = mk_bf16(in, n), *tw = mk_bf16(w, WIDTH),
                  *tm = mk_bf16(mod, mod_n), *tr = mk_u32(ROW_MAP, ROWS),
                  *to = new_bf16(n);
    OP(h3_gpu_adaln_bf16(gpu, to, ti, tw, tm, tr, ROWS, WIDTH, SLOTS,
                         SHIFT_SLOT, SCALE_SLOT, EPS));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("adaln_bf16", got, ref, n);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(to);

    /* offset variant */
    ref_adaln(fin + in_off, fw, fmod, ROW_MAP, fref, ROWS, WIDTH, SLOTS,
              SHIFT_SLOT, SCALE_SLOT, EPS);
    for (uint32_t i = 0; i < n; i++) ref[i] = host_f32_to_bf16(fref[i]);
    ti = mk_bf16(in, n + in_off);
    to = new_bf16(n);
    OP(h3_gpu_adaln_bf16_offset(gpu, to, ti, in_off, tw, tm, tr, ROWS, WIDTH,
                                SLOTS, SHIFT_SLOT, SCALE_SLOT, EPS));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("adaln_bf16_offset", got, ref, n);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tm);
    h3_gpu_tensor_free(tr); h3_gpu_tensor_free(to);
    free(in); free(w); free(mod); free(ref); free(got);
    free(fin); free(fw); free(fmod); free(fref);
}

static void test_adaln_linear_bf16(void) {
    const uint32_t rows = ROWS, width = WIDTH, out_dim = 5;
    const uint32_t n = rows * width, mod_n = rows * SLOTS * width;
    const size_t in_off = 3;
    uint16_t *in = malloc((n + in_off) * 2), *nw = malloc(width * 2),
             *mod = malloc(mod_n * 2), *w = malloc(out_dim * width * 2),
             *b = malloc(out_dim * 2), *ref = malloc(rows * out_dim * 2),
             *got = malloc(rows * out_dim * 2);
    float *fin = malloc((n + in_off) * 4), *fnw = malloc(width * 4),
          *fmod = malloc(mod_n * 4), *fw = malloc(out_dim * width * 4),
          *fb = malloc(out_dim * 4);
    fill_bf16(in, fin, n + in_off, 3.0f);
    fill_bf16(nw, fnw, width, 2.0f);
    fill_bf16(mod, fmod, mod_n, 1.0f);
    fill_bf16(w, fw, out_dim * width, 0.5f);
    fill_bf16(b, fb, out_dim, 1.0f);

    /* reference, two passes exactly like the kernels */
    float *fref_inv = malloc(rows * 4);
    float *finv_got = malloc(rows * 4);
    for (uint32_t r = 0; r < rows; r++) {
        float ss = 0.0f;
        for (uint32_t c = 0; c < width; c++) {
            float v = fin[in_off + r * width + c];
            ss += v * v;
        }
        fref_inv[r] = 1.0f / sqrtf(ss / (float)width + EPS);
    }
    float *normed = malloc(n * 4);
    for (uint32_t r = 0; r < rows; r++) {
        size_t base = (size_t)ROW_MAP[r] * SLOTS * width;
        for (uint32_t c = 0; c < width; c++) {
            float value = fin[in_off + r * width + c];
            float shift = fmod[base + (size_t)SHIFT_SLOT * width + c];
            float scale = fmod[base + (size_t)SCALE_SLOT * width + c];
            float nr = value * fref_inv[r] * fnw[c];
            normed[r * width + c] =
                host_bf16_to_f32(host_f32_to_bf16(nr * (1.0f + scale) + shift));
        }
    }
    for (int use_bias = 1; use_bias >= 0; use_bias--) {
        float *fref = malloc(rows * out_dim * 4);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t o = 0; o < out_dim; o++) {
                float sum = use_bias ? fb[o] : 0.0f;
                for (uint32_t k = 0; k < width; k++)
                    sum = fmaf(normed[r * width + k], fw[o * width + k], sum);
                fref[r * out_dim + o] = sum;
            }
        for (uint32_t i = 0; i < rows * out_dim; i++)
            ref[i] = host_f32_to_bf16(fref[i]);
        h3_gpu_tensor *ti = mk_bf16(in, n + in_off),
                      *tnw = mk_bf16(nw, width), *tm = mk_bf16(mod, mod_n),
                      *tr = mk_u32(ROW_MAP, rows), *tw = mk_bf16(w, out_dim * width),
                      *tb = use_bias ? mk_bf16(b, out_dim) : NULL,
                      *tinv = new_f32(rows), *to = new_bf16(rows * out_dim);
        OP(h3_gpu_adaln_linear_bf16(gpu, to, tinv, ti, in_off, tnw, tm, tr,
                                    tw, tb, rows, width, out_dim, SLOTS,
                                    SHIFT_SLOT, SCALE_SLOT, EPS));
        run();
        OP(h3_gpu_tensor_read_bf16(to, got, rows * out_dim));
        check_bf16_gemm(use_bias ? "adaln_linear_bf16 bias"
                                 : "adaln_linear_bf16 nobias",
                        got, ref, rows * out_dim);
        if (use_bias) {
            OP(h3_gpu_tensor_read_f32(tinv, finv_got, rows));
            check_f32("adaln_linear_bf16 inverse", finv_got, fref_inv, rows,
                      1e-6, 1e-5);
        }
        h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tnw);
        h3_gpu_tensor_free(tm); h3_gpu_tensor_free(tr); h3_gpu_tensor_free(tw);
        if (tb) h3_gpu_tensor_free(tb);
        h3_gpu_tensor_free(tinv); h3_gpu_tensor_free(to);
        free(fref);
    }
    free(in); free(nw); free(mod); free(w); free(b); free(ref); free(got);
    free(fin); free(fnw); free(fmod); free(fw); free(fb);
    free(fref_inv); free(finv_got); free(normed);
}

static void test_gate_bf16(void) {
    const uint32_t n = ROWS * WIDTH, mod_n = ROWS * SLOTS * WIDTH;
    uint16_t *res = malloc(n * 2), *br = malloc(n * 2), *mod = malloc(mod_n * 2),
             *ref = malloc(n * 2), *got = malloc(n * 2);
    float *fres = malloc(n * 4), *fbr = malloc(n * 4), *fmod = malloc(mod_n * 4),
          *fref = malloc(n * 4);
    fill_bf16(res, fres, n, 3.0f);
    fill_bf16(br, fbr, n, 3.0f);
    fill_bf16(mod, fmod, mod_n, 1.0f);
    ref_gate(fres, fbr, fmod, ROW_MAP, fref, ROWS, WIDTH, SLOTS, GATE_SLOT);
    for (uint32_t i = 0; i < n; i++) ref[i] = host_f32_to_bf16(fref[i]);
    h3_gpu_tensor *tr = mk_bf16(res, n), *tb = mk_bf16(br, n),
                  *tm = mk_bf16(mod, mod_n), *tmap = mk_u32(ROW_MAP, ROWS),
                  *to = new_bf16(n);
    OP(h3_gpu_gate_bf16(gpu, to, tr, tb, tm, tmap, ROWS, WIDTH, SLOTS,
                        GATE_SLOT));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, n));
    check_bf16("gate_bf16", got, ref, n);
    h3_gpu_tensor_free(tr); h3_gpu_tensor_free(tb); h3_gpu_tensor_free(tm);
    h3_gpu_tensor_free(tmap); h3_gpu_tensor_free(to);
    free(res); free(br); free(mod); free(ref); free(got);
    free(fres); free(fbr); free(fmod); free(fref);
}

static void test_gate_adaln_bf16(void) {
    const uint32_t n = ROWS * WIDTH, mod_n = ROWS * SLOTS * WIDTH;
    uint16_t *res = malloc(n * 2), *br = malloc(n * 2),
             *gmod = malloc(mod_n * 2), *nmod = malloc(mod_n * 2),
             *nw = malloc(WIDTH * 2), *ref_gated = malloc(n * 2),
             *ref_out = malloc(n * 2), *got_gated = malloc(n * 2),
             *got_out = malloc(n * 2);
    float *fres = malloc(n * 4), *fbr = malloc(n * 4), *fgmod = malloc(mod_n * 4),
          *fnmod = malloc(mod_n * 4), *fnw = malloc(WIDTH * 4);
    fill_bf16(res, fres, n, 3.0f);
    fill_bf16(br, fbr, n, 3.0f);
    fill_bf16(gmod, fgmod, mod_n, 1.0f);
    fill_bf16(nmod, fnmod, mod_n, 1.0f);
    fill_bf16(nw, fnw, WIDTH, 2.0f);
    /* reference: gate, round to bf16, then adaln over the rounded row */
    for (uint32_t r = 0; r < ROWS; r++) {
        size_t base = (size_t)ROW_MAP[r] * SLOTS * WIDTH;
        float ss = 0.0f;
        for (uint32_t c = 0; c < WIDTH; c++) {
            float gate = fgmod[base + (size_t)GATE_SLOT * WIDTH + c];
            uint16_t g = host_f32_to_bf16(fres[r * WIDTH + c] +
                                          fbr[r * WIDTH + c] * gate);
            ref_gated[r * WIDTH + c] = g;
            float v = host_bf16_to_f32(g);
            ss += v * v;
        }
        float inv = 1.0f / sqrtf(ss / (float)WIDTH + EPS);
        for (uint32_t c = 0; c < WIDTH; c++) {
            float normalized = host_bf16_to_f32(ref_gated[r * WIDTH + c]) *
                inv * fnw[c];
            float shift = fnmod[base + (size_t)SHIFT_SLOT * WIDTH + c];
            float scale = fnmod[base + (size_t)SCALE_SLOT * WIDTH + c];
            ref_out[r * WIDTH + c] =
                host_f32_to_bf16(normalized * (1.0f + scale) + shift);
        }
    }
    h3_gpu_tensor *tr = mk_bf16(res, n), *tb = mk_bf16(br, n),
                  *tg = mk_bf16(gmod, mod_n), *tn = mk_bf16(nmod, mod_n),
                  *tw = mk_bf16(nw, WIDTH), *tmap = mk_u32(ROW_MAP, ROWS),
                  *tgated = new_bf16(n), *to = new_bf16(n);
    OP(h3_gpu_gate_adaln_bf16(gpu, tgated, to, tr, tb, tw, tg, tn, tmap,
                              ROWS, WIDTH, SLOTS, GATE_SLOT, SHIFT_SLOT,
                              SCALE_SLOT, EPS));
    run();
    OP(h3_gpu_tensor_read_bf16(tgated, got_gated, n));
    OP(h3_gpu_tensor_read_bf16(to, got_out, n));
    check_bf16("gate_adaln_bf16 gated", got_gated, ref_gated, n);
    check_bf16("gate_adaln_bf16 output", got_out, ref_out, n);
    /* width > 5376 must fail */
    if (h3_gpu_gate_adaln_bf16(gpu, tgated, to, tr, tb, tw, tg, tn, tmap,
                               ROWS, 5377, SLOTS, GATE_SLOT, SHIFT_SLOT,
                               SCALE_SLOT, EPS)) {
        printf("FAIL gate_adaln_bf16 accepted width=5377\n"); failures++;
    } else {
        printf("ok   gate_adaln_bf16 rejects width>5376\n");
    }
    h3_gpu_tensor_free(tr); h3_gpu_tensor_free(tb); h3_gpu_tensor_free(tg);
    h3_gpu_tensor_free(tn); h3_gpu_tensor_free(tw); h3_gpu_tensor_free(tmap);
    h3_gpu_tensor_free(tgated); h3_gpu_tensor_free(to);
    free(res); free(br); free(gmod); free(nmod); free(nw);
    free(ref_gated); free(ref_out); free(got_gated); free(got_out);
    free(fres); free(fbr); free(fgmod); free(fnmod); free(fnw);
}

/* ---------------- token pool / expand shared fixture ---------------- */
#define TP_IN_ROWS 6
#define TP_ROWS 3
#define TP_BASE_ROWS 2
#define TP_PREFIX 1
#define TP_UPDATE 0.5f
/* pairs: singleton, real pair, singleton */
static const uint32_t TP_PAIRS[TP_ROWS * 2] = {0, 0, 2, 3, 5, 5};
static const uint32_t TP_BASE_IDX[TP_ROWS] = {0, 0xffffffffu, 1};
/* expand: parents per full row, baseline per reduced row */
static const uint32_t TE_PARENTS[TP_IN_ROWS] = {0, 0, 1, 1, 2, 2};
static const uint32_t TE_BASE_IDX[TP_ROWS] = {0, 1, 0xffffffffu};

/* CPU reference for one pooling pass (shared by both pool kernels) */
static void ref_pool_row(const float *fin, size_t in_off,
                         float *pooled_row /*WIDTH floats*/,
                         uint16_t *pooled_bf16_row, uint32_t row) {
    uint32_t px = TP_PAIRS[row * 2], py = TP_PAIRS[row * 2 + 1];
    for (uint32_t c = 0; c < WIDTH; c++) {
        float first = fin[in_off + (size_t)px * WIDTH + c];
        float pooled;
        if (px != py)
            pooled = (first + fin[in_off + (size_t)py * WIDTH + c]) * 0.5f;
        else
            pooled = first;
        pooled_bf16_row[c] = host_f32_to_bf16(pooled);
        pooled_row[c] = host_bf16_to_f32(pooled_bf16_row[c]);
    }
}

static void test_token_pool_bf16(void) {
    const size_t in_off = 4, orig_off = 6, base_off = 3;
    const size_t in_n = TP_IN_ROWS * WIDTH, out_n = TP_ROWS * WIDTH,
                 base_n = TP_BASE_ROWS * WIDTH;
    uint16_t *in = malloc((in_n + in_off) * 2),
             *orig_sentinel = malloc((in_n + orig_off) * 2),
             *base_sentinel = malloc((base_n + base_off) * 2),
             *got_out = malloc(out_n * 2),
             *got_orig = malloc((in_n + orig_off) * 2),
             *got_base = malloc((base_n + base_off) * 2);
    uint16_t *ref_out = malloc(out_n * 2);
    float *fin = malloc((in_n + in_off) * 4);
    fill_bf16(in, fin, in_n + in_off, 3.0f);
    for (size_t i = 0; i < in_n + orig_off; i++)
        orig_sentinel[i] = host_f32_to_bf16(-77.0f);
    for (size_t i = 0; i < base_n + base_off; i++)
        base_sentinel[i] = host_f32_to_bf16(-88.0f);

    /* reference */
    uint16_t *ref_orig = malloc((in_n + orig_off) * 2);
    uint16_t *ref_base = malloc((base_n + base_off) * 2);
    memcpy(ref_orig, orig_sentinel, (in_n + orig_off) * 2);
    memcpy(ref_base, base_sentinel, (base_n + base_off) * 2);
    for (uint32_t r = 0; r < TP_ROWS; r++) {
        float pooled_row[WIDTH];
        ref_pool_row(fin, in_off, pooled_row, ref_out + r * WIDTH, r);
        uint32_t px = TP_PAIRS[r * 2], py = TP_PAIRS[r * 2 + 1];
        for (uint32_t c = 0; c < WIDTH; c++) {
            ref_orig[orig_off + (size_t)px * WIDTH + c] =
                in[in_off + (size_t)px * WIDTH + c];
            if (px != py)
                ref_orig[orig_off + (size_t)py * WIDTH + c] =
                    in[in_off + (size_t)py * WIDTH + c];
        }
        if (TP_BASE_IDX[r] != 0xffffffffu)
            memcpy(ref_base + base_off + (size_t)TP_BASE_IDX[r] * WIDTH,
                   ref_out + r * WIDTH, WIDTH * 2);
    }

    h3_gpu_tensor *ti = mk_bf16(in, in_n + in_off),
                  *tpairs = mk_u32(TP_PAIRS, TP_ROWS * 2),
                  *tbi = mk_u32(TP_BASE_IDX, TP_ROWS),
                  *torig = mk_bf16(orig_sentinel, in_n + orig_off),
                  *tbase = mk_bf16(base_sentinel, base_n + base_off),
                  *to = new_bf16(out_n);
    OP(h3_gpu_token_pool_bf16(gpu, to, ti, in_off, torig, orig_off, tbase,
                              base_off, tbi, tpairs, TP_IN_ROWS, TP_ROWS,
                              TP_BASE_ROWS, WIDTH));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got_out, out_n));
    OP(h3_gpu_tensor_read_bf16(torig, got_orig, in_n + orig_off));
    OP(h3_gpu_tensor_read_bf16(tbase, got_base, base_n + base_off));
    check_bf16("token_pool_bf16 output", got_out, ref_out, out_n);
    check_bf16("token_pool_bf16 original", got_orig, ref_orig, in_n + orig_off);
    check_bf16("token_pool_bf16 baseline", got_base, ref_base,
               base_n + base_off);
    /* rows > input_rows must fail */
    if (h3_gpu_token_pool_bf16(gpu, to, ti, in_off, torig, orig_off, tbase,
                               base_off, tbi, tpairs, TP_IN_ROWS,
                               TP_IN_ROWS + 1, TP_BASE_ROWS, WIDTH)) {
        printf("FAIL token_pool_bf16 accepted rows>input_rows\n"); failures++;
    } else {
        printf("ok   token_pool_bf16 rejects rows>input_rows\n");
    }
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tpairs); h3_gpu_tensor_free(tbi);
    h3_gpu_tensor_free(torig); h3_gpu_tensor_free(tbase); h3_gpu_tensor_free(to);
    free(in); free(orig_sentinel); free(base_sentinel);
    free(got_out); free(got_orig); free(got_base);
    free(ref_out); free(fin); free(ref_orig); free(ref_base);
}

static void test_token_pool_adaln_bf16(void) {
    const size_t in_off = 4, orig_off = 6, base_off = 3;
    const size_t in_n = TP_IN_ROWS * WIDTH, out_n = TP_ROWS * WIDTH,
                 base_n = TP_BASE_ROWS * WIDTH, mod_n = 4 * SLOTS * WIDTH;
    uint16_t *in = malloc((in_n + in_off) * 2),
             *nw = malloc(WIDTH * 2), *mod = malloc(mod_n * 2),
             *ref_res = malloc(out_n * 2), *ref_out = malloc(out_n * 2),
             *got_res = malloc(out_n * 2), *got_out = malloc(out_n * 2);
    float *fin = malloc((in_n + in_off) * 4), *fnw = malloc(WIDTH * 4),
          *fmod = malloc(mod_n * 4);
    fill_bf16(in, fin, in_n + in_off, 3.0f);
    fill_bf16(nw, fnw, WIDTH, 2.0f);
    fill_bf16(mod, fmod, mod_n, 1.0f);

    /* reference: pool -> residual, then adaln over the rounded pooled row */
    for (uint32_t r = 0; r < TP_ROWS; r++) {
        float pooled_row[WIDTH];
        ref_pool_row(fin, in_off, pooled_row, ref_res + r * WIDTH, r);
        float ss = 0.0f;
        for (uint32_t c = 0; c < WIDTH; c++) ss += pooled_row[c] * pooled_row[c];
        float inv = 1.0f / sqrtf(ss / (float)WIDTH + EPS);
        size_t base = (size_t)ROW_MAP[r] * SLOTS * WIDTH;
        for (uint32_t c = 0; c < WIDTH; c++) {
            float normalized = pooled_row[c] * inv * fnw[c];
            float shift = fmod[base + (size_t)SHIFT_SLOT * WIDTH + c];
            float scale = fmod[base + (size_t)SCALE_SLOT * WIDTH + c];
            ref_out[r * WIDTH + c] =
                host_f32_to_bf16(normalized * (1.0f + scale) + shift);
        }
    }
    uint16_t *orig_sentinel = malloc((in_n + orig_off) * 2),
             *base_sentinel = malloc((base_n + base_off) * 2);
    for (size_t i = 0; i < in_n + orig_off; i++)
        orig_sentinel[i] = host_f32_to_bf16(-77.0f);
    for (size_t i = 0; i < base_n + base_off; i++)
        base_sentinel[i] = host_f32_to_bf16(-88.0f);
    h3_gpu_tensor *ti = mk_bf16(in, in_n + in_off),
                  *tpairs = mk_u32(TP_PAIRS, TP_ROWS * 2),
                  *tbi = mk_u32(TP_BASE_IDX, TP_ROWS),
                  *torig = mk_bf16(orig_sentinel, in_n + orig_off),
                  *tbase = mk_bf16(base_sentinel, base_n + base_off),
                  *tnw = mk_bf16(nw, WIDTH), *tm = mk_bf16(mod, mod_n),
                  *tmap = mk_u32(ROW_MAP, TP_ROWS),
                  *tres = new_bf16(out_n), *to = new_bf16(out_n);
    OP(h3_gpu_token_pool_adaln_bf16(gpu, tres, to, ti, in_off, torig,
                                    orig_off, tbase, base_off, tbi, tpairs,
                                    tnw, tm, tmap, TP_IN_ROWS, TP_ROWS,
                                    TP_BASE_ROWS, WIDTH, SLOTS, SHIFT_SLOT,
                                    SCALE_SLOT, EPS));
    run();
    OP(h3_gpu_tensor_read_bf16(tres, got_res, out_n));
    OP(h3_gpu_tensor_read_bf16(to, got_out, out_n));
    check_bf16("token_pool_adaln residual", got_res, ref_res, out_n);
    check_bf16("token_pool_adaln output", got_out, ref_out, out_n);
    h3_gpu_tensor_free(ti); h3_gpu_tensor_free(tpairs); h3_gpu_tensor_free(tbi);
    h3_gpu_tensor_free(torig); h3_gpu_tensor_free(tbase);
    h3_gpu_tensor_free(tnw); h3_gpu_tensor_free(tm); h3_gpu_tensor_free(tmap);
    h3_gpu_tensor_free(tres); h3_gpu_tensor_free(to);
    free(in); free(nw); free(mod); free(ref_res); free(ref_out);
    free(got_res); free(got_out); free(fin); free(fnw); free(fmod);
    free(orig_sentinel); free(base_sentinel);
}

/* CPU reference for one expand row (shared by both expand kernels) */
static void ref_expand_row(const float *forig, size_t orig_off,
                           const float *freduced, const float *fbase,
                           size_t base_off, uint16_t *restored_bf16_row,
                           float *restored_row, uint32_t row) {
    uint32_t parent = TE_PARENTS[row];
    uint32_t baseline_row = TE_BASE_IDX[parent];
    int direct = row < TP_PREFIX || baseline_row == 0xffffffffu;
    for (uint32_t c = 0; c < WIDTH; c++) {
        size_t destination = (size_t)row * WIDTH + c;
        float restored = freduced[(size_t)parent * WIDTH + c];
        if (!direct) {
            float update = restored -
                fbase[base_off + (size_t)baseline_row * WIDTH + c];
            restored = forig[orig_off + destination] + TP_UPDATE * update;
        }
        restored_bf16_row[c] = direct ?
            host_f32_to_bf16(restored) : host_f32_to_bf16(restored);
        restored_row[c] = host_bf16_to_f32(restored_bf16_row[c]);
    }
}

static void test_token_expand_delta_bf16(void) {
    const size_t orig_off = 5, base_off = 2;
    const size_t full_n = TP_IN_ROWS * WIDTH, red_n = TP_ROWS * WIDTH,
                 base_n = TP_BASE_ROWS * WIDTH;
    uint16_t *orig = malloc((full_n + orig_off) * 2), *reduced = malloc(red_n * 2),
             *base = malloc((base_n + base_off) * 2), *ref = malloc(full_n * 2),
             *got = malloc(full_n * 2);
    float *forig = malloc((full_n + orig_off) * 4), *freduced = malloc(red_n * 4),
          *fbase = malloc((base_n + base_off) * 4);
    fill_bf16(orig, forig, full_n + orig_off, 3.0f);
    fill_bf16(reduced, freduced, red_n, 3.0f);
    fill_bf16(base, fbase, base_n + base_off, 3.0f);
    for (uint32_t r = 0; r < TP_IN_ROWS; r++) {
        float row_f[WIDTH];
        ref_expand_row(forig, orig_off, freduced, fbase, base_off,
                       ref + r * WIDTH, row_f, r);
    }
    h3_gpu_tensor *torig = mk_bf16(orig, full_n + orig_off),
                  *tred = mk_bf16(reduced, red_n),
                  *tbase = mk_bf16(base, base_n + base_off),
                  *tbi = mk_u32(TE_BASE_IDX, TP_ROWS),
                  *tpar = mk_u32(TE_PARENTS, TP_IN_ROWS),
                  *to = new_bf16(full_n);
    OP(h3_gpu_token_expand_delta_bf16(gpu, to, torig, orig_off, tred, tbase,
                                      base_off, tbi, tpar, TP_IN_ROWS,
                                      TP_ROWS, TP_BASE_ROWS, WIDTH,
                                      TP_PREFIX, TP_UPDATE));
    run();
    OP(h3_gpu_tensor_read_bf16(to, got, full_n));
    check_bf16("token_expand_delta_bf16", got, ref, full_n);
    h3_gpu_tensor_free(torig); h3_gpu_tensor_free(tred); h3_gpu_tensor_free(tbase);
    h3_gpu_tensor_free(tbi); h3_gpu_tensor_free(tpar); h3_gpu_tensor_free(to);
    free(orig); free(reduced); free(base); free(ref); free(got);
    free(forig); free(freduced); free(fbase);
}

static void test_token_expand_adaln_bf16(void) {
    const size_t orig_off = 5, base_off = 2;
    const size_t full_n = TP_IN_ROWS * WIDTH, red_n = TP_ROWS * WIDTH,
                 base_n = TP_BASE_ROWS * WIDTH, mod_n = TP_IN_ROWS * SLOTS * WIDTH;
    uint16_t *orig = malloc((full_n + orig_off) * 2), *reduced = malloc(red_n * 2),
             *base = malloc((base_n + base_off) * 2), *nw = malloc(WIDTH * 2),
             *mod = malloc(mod_n * 2), *ref_res = malloc(full_n * 2),
             *ref_out = malloc(full_n * 2), *got_res = malloc(full_n * 2),
             *got_out = malloc(full_n * 2);
    float *forig = malloc((full_n + orig_off) * 4), *freduced = malloc(red_n * 4),
          *fbase = malloc((base_n + base_off) * 4), *fnw = malloc(WIDTH * 4),
          *fmod = malloc(mod_n * 4);
    fill_bf16(orig, forig, full_n + orig_off, 3.0f);
    fill_bf16(reduced, freduced, red_n, 3.0f);
    fill_bf16(base, fbase, base_n + base_off, 3.0f);
    fill_bf16(nw, fnw, WIDTH, 2.0f);
    fill_bf16(mod, fmod, mod_n, 1.0f);
    /* row_map for 6 rows (modulation slots dimensioned accordingly) */
    uint32_t expand_map[TP_IN_ROWS] = {0, 3, 1, 5, 2, 4};
    for (uint32_t r = 0; r < TP_IN_ROWS; r++) {
        float restored_row[WIDTH];
        ref_expand_row(forig, orig_off, freduced, fbase, base_off,
                       ref_res + r * WIDTH, restored_row, r);
        float ss = 0.0f;
        for (uint32_t c = 0; c < WIDTH; c++)
            ss += restored_row[c] * restored_row[c];
        float inv = 1.0f / sqrtf(ss / (float)WIDTH + EPS);
        size_t mb = (size_t)expand_map[r] * SLOTS * WIDTH;
        for (uint32_t c = 0; c < WIDTH; c++) {
            float normalized = restored_row[c] * inv * fnw[c];
            float shift = fmod[mb + (size_t)SHIFT_SLOT * WIDTH + c];
            float scale = fmod[mb + (size_t)SCALE_SLOT * WIDTH + c];
            ref_out[r * WIDTH + c] =
                host_f32_to_bf16(normalized * (1.0f + scale) + shift);
        }
    }
    h3_gpu_tensor *torig = mk_bf16(orig, full_n + orig_off),
                  *tred = mk_bf16(reduced, red_n),
                  *tbase = mk_bf16(base, base_n + base_off),
                  *tbi = mk_u32(TE_BASE_IDX, TP_ROWS),
                  *tpar = mk_u32(TE_PARENTS, TP_IN_ROWS),
                  *tnw = mk_bf16(nw, WIDTH), *tm = mk_bf16(mod, mod_n),
                  *tmap = mk_u32(expand_map, TP_IN_ROWS),
                  *tres = new_bf16(full_n), *to = new_bf16(full_n);
    OP(h3_gpu_token_expand_adaln_bf16(gpu, tres, to, torig, orig_off, tred,
                                      tbase, base_off, tbi, tpar, tnw, tm,
                                      tmap, TP_IN_ROWS, TP_ROWS,
                                      TP_BASE_ROWS, WIDTH, TP_PREFIX,
                                      TP_UPDATE, SLOTS, SHIFT_SLOT,
                                      SCALE_SLOT, EPS));
    run();
    OP(h3_gpu_tensor_read_bf16(tres, got_res, full_n));
    OP(h3_gpu_tensor_read_bf16(to, got_out, full_n));
    check_bf16("token_expand_adaln residual", got_res, ref_res, full_n);
    check_bf16("token_expand_adaln output", got_out, ref_out, full_n);
    h3_gpu_tensor_free(torig); h3_gpu_tensor_free(tred); h3_gpu_tensor_free(tbase);
    h3_gpu_tensor_free(tbi); h3_gpu_tensor_free(tpar); h3_gpu_tensor_free(tnw);
    h3_gpu_tensor_free(tm); h3_gpu_tensor_free(tmap); h3_gpu_tensor_free(tres);
    h3_gpu_tensor_free(to);
    free(orig); free(reduced); free(base); free(nw); free(mod);
    free(ref_res); free(ref_out); free(got_res); free(got_out);
    free(forig); free(freduced); free(fbase); free(fnw); free(fmod);
}

static void test_euler_bf16(void) {
    const uint32_t elements = 37;
    const size_t sample_off = 3;
    const float delta = 0.125f, ratio = -0.5f;
    float *sample = malloc((elements + sample_off) * 4),
          *ref = malloc((elements + sample_off) * 4),
          *got = malloc((elements + sample_off) * 4);
    uint16_t *last = malloc(elements * 2), *prev = malloc(elements * 2);
    for (size_t i = 0; i < elements + sample_off; i++) sample[i] = frand(4.0f);
    for (uint32_t i = 0; i < elements; i++) {
        last[i] = host_f32_to_bf16(frand(2.0f));
        prev[i] = host_f32_to_bf16(frand(2.0f));
    }
    memcpy(ref, sample, (elements + sample_off) * 4);
    for (uint32_t i = 0; i < elements; i++) {
        float last_value = host_bf16_to_f32(last[i]);
        float velocity = fmaf(ratio,
                              last_value - host_bf16_to_f32(prev[i]),
                              last_value);
        size_t idx = sample_off + i;
        ref[idx] = fmaf(delta, velocity, ref[idx]);
    }
    h3_gpu_tensor *ts = mk_f32(sample, elements + sample_off),
                  *tl = mk_bf16(last, elements), *tp = mk_bf16(prev, elements);
    OP(h3_gpu_euler_bf16(gpu, ts, sample_off, tl, tp, elements, delta,
                         ratio));
    run();
    OP(h3_gpu_tensor_read_f32(ts, got, elements + sample_off));
    check_f32("euler_bf16", got, ref, elements + sample_off, 1e-6, 1e-6);
    /* out-of-range offset must fail */
    if (h3_gpu_euler_bf16(gpu, ts, elements + sample_off + 1, tl, tp,
                          elements, delta, ratio)) {
        printf("FAIL euler_bf16 accepted bad offset\n"); failures++;
    } else {
        printf("ok   euler_bf16 rejects bad offset\n");
    }
    h3_gpu_tensor_free(ts); h3_gpu_tensor_free(tl); h3_gpu_tensor_free(tp);
    free(sample); free(ref); free(got); free(last); free(prev);
}

static void test_stubs(void) {
    int ok_all = 1;
    if (h3_gpu_quantize_weight_int8(gpu, NULL, NULL, NULL, 4, 8) != 0)
        ok_all = 0;
    else if (!strstr(h3_gpu_error(gpu), "not available in the CUDA backend"))
        ok_all = 0;
    if (h3_gpu_linear_int8_bf16(gpu, NULL, NULL, NULL, NULL, NULL, NULL,
                                1, 1, 1, 0) != 0)
        ok_all = 0;
    else if (!strstr(h3_gpu_error(gpu), "not available in the CUDA backend"))
        ok_all = 0;
    if (h3_gpu_linear_int8_head_major_bf16(gpu, NULL, NULL, NULL, NULL, NULL,
                                           NULL, 1, 1, 1, 1) != 0)
        ok_all = 0;
    else if (!strstr(h3_gpu_error(gpu), "not available in the CUDA backend"))
        ok_all = 0;
    if (h3_gpu_mlp_int8_bf16(gpu, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                             NULL, NULL, NULL, NULL, 1, 1, 1, 1, 0, 0, 0,
                             0) != 0)
        ok_all = 0;
    else if (!strstr(h3_gpu_error(gpu), "not available in the CUDA backend"))
        ok_all = 0;
    if (h3_gpu_mlp_nax_bf16(gpu, NULL, NULL, NULL, NULL, NULL, 1, 1, 1,
                            1) != 0)
        ok_all = 0;
    else if (!strstr(h3_gpu_error(gpu), "not available in the CUDA backend"))
        ok_all = 0;
    if (h3_gpu_grouped_qkv_linear_rope_int8(gpu, NULL, NULL, NULL, NULL,
                                            NULL, NULL, NULL, NULL, NULL,
                                            NULL, NULL, NULL, 1, 1, 1, 1, 1,
                                            1e-5f, 0, 0, 0, 0) != 0)
        ok_all = 0;
    else if (!strstr(h3_gpu_error(gpu), "not available in the CUDA backend"))
        ok_all = 0;
    if (h3_gpu_gate_adaln_quantize_int8(gpu, NULL, NULL, NULL, NULL, NULL,
                                        NULL, NULL, NULL, NULL, 1, 1, 1, 1,
                                        0, 1, 2, 1e-5f) != 0)
        ok_all = 0;
    else if (!strstr(h3_gpu_error(gpu), "not available in the CUDA backend"))
        ok_all = 0;
    if (!ok_all) {
        printf("FAIL int8/nax stubs did not all fail cleanly\n");
        failures++;
    } else {
        printf("ok   7 int8/nax stubs fail cleanly\n");
    }
}

int main(void) {
    /* Exact F32 for the patch-projection GEMM checks (see test_cuda_core.c). */
    setenv("H3_CUDA_TF32", "0", 1);
    char error[512];
    gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_gpu_create failed: %s\n", error); return 2; }
    OP(h3_gpu_begin(gpu));

    test_adaln_f32();
    test_gate_f32();
    test_adaln_bf16();
    test_adaln_linear_bf16();
    test_gate_bf16();
    test_gate_adaln_bf16();
    test_token_pool_bf16();
    test_token_pool_adaln_bf16();
    test_token_expand_delta_bf16();
    test_token_expand_adaln_bf16();
    test_euler_bf16();
    test_stubs();

    if (!h3_gpu_submit(gpu)) { fprintf(stderr, "final submit failed\n"); return 2; }
    h3_gpu_free(gpu);
    if (failures) { printf("\n%d FAILURES\n", failures); return 1; }
    printf("\nALL TESTS PASSED\n");
    return 0;
}
