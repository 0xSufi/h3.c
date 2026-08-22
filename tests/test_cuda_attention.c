/* Freestanding CUDA attention-op tests: links only h3_gpu_cuda.o,
 * h3_cuda_core.o, h3_cuda_attention.o. Every op is checked against an
 * independent CPU reference written below. */
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t c_f32_to_bf16(float v) {
    uint32_t b;
    memcpy(&b, &v, 4);
    b += 0x7fffu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}
static float c_bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static uint32_t rng_state = 0x12345678u;
static float frand(void) { /* [-1, 1] */
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xffffff) / (float)0x800000 - 1.0f;
}

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);           \
            failures++;                                                      \
        }                                                                    \
    } while (0)

/* Compare f32 GPU output vs f32 reference; returns max abs diff. */
static float check_f32(const char *name, const float *got, const float *ref,
                       size_t n, float tol) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - ref[i]);
        if (d > worst) worst = d;
        if (d > tol) {
            printf("FAIL: %s[%zu] got %g want %g\n", name, i, got[i], ref[i]);
            failures++;
            return worst;
        }
    }
    printf("ok: %-36s max|d|=%.3g\n", name, worst);
    return worst;
}

/* Compare bf16 GPU output vs float reference (rounded to bf16); counts
 * elements off by more than one ulp. */
static int check_bf16(const char *name, const uint16_t *got, const float *ref,
                      size_t n, int max_ulp) {
    int worst_ulp = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        uint16_t want = c_f32_to_bf16(ref[i]);
        int ulp = abs((int)got[i] - (int)want);
        float d = fabsf(c_bf16_to_f32(got[i]) - c_bf16_to_f32(want));
        if (ulp > worst_ulp) worst_ulp = ulp;
        if (d > worst) worst = d;
        if (ulp > max_ulp) {
            printf("FAIL: %s[%zu] got %g (0x%04x) want %g (0x%04x)\n", name,
                   i, c_bf16_to_f32(got[i]), got[i], c_bf16_to_f32(want), want);
            failures++;
            return worst_ulp;
        }
    }
    printf("ok: %-36s max_ulp=%d max|d|=%.3g\n", name, worst_ulp, worst);
    return worst_ulp;
}

/* Compare bf16 GPU output vs float reference with a combined
 * absolute+relative tolerance (values widened to f32 first). Used for the
 * flash SDPA/GQA kernels: online softmax plus the tiled P*V reduction
 * reorder the summation relative to the CPU reference, so bit-exactness is
 * not expected; FP32 accumulation and RNE bf16 output rounding are
 * preserved, so a tight 1e-3 abs + 1e-2 rel bound (about one bf16 ulp)
 * still applies. */
static int check_bf16_tol(const char *name, const uint16_t *got,
                          const float *ref, size_t n, float abs_tol,
                          float rel_tol) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float g = c_bf16_to_f32(got[i]);
        float d = fabsf(g - ref[i]);
        if (d > worst) worst = d;
        if (d > abs_tol + rel_tol * fabsf(ref[i])) {
            printf("FAIL: %s[%zu] got %g (0x%04x) want %g\n", name, i, g,
                   got[i], ref[i]);
            failures++;
            return 1;
        }
    }
    printf("ok: %-36s max|d|=%.3g\n", name, worst);
    return 0;
}

/* Set while the suite runs with H3_CUDA_SDPA_NAIVE=1: the naive SDPA/GQA
 * kernels keep the original tight (near bit-exact) tolerances; the flash
 * kernels use check_bf16_tol above. */
static int sdpa_naive_mode = 0;

/* Flash f32 SDPA tolerance: same summation reorder as bf16, but f32 outputs
 * stay much tighter than the bf16 rounding grid. */
#define SDPA_FLASH_F32_TOL 1e-4f
#define SDPA_FLASH_BF16_ABS 1e-3f
#define SDPA_FLASH_BF16_REL 1e-2f

/* ------------------------------------------------------------------ refs */

/* ref for h3_qkv_rope_f32 / h3_qkv_rope_bf16 (weighted norm, optional
 * grouped layout) and h3_video_qkv_rope_f32 (grouped layout, unweighted). */
static void ref_qkv_rope(const float *qkv, const float *qw, const float *kw,
                         const float *rc, const float *rs, float *q_out,
                         float *k_out, float *v_out, uint32_t seq,
                         uint32_t heads, uint32_t hd, uint32_t rope_half,
                         int grouped, int weighted, float eps) {
    size_t inner = (size_t)heads * hd;
    for (uint32_t row = 0; row < seq; row++)
        for (uint32_t head = 0; head < heads; head++) {
            size_t row_base = (size_t)row * inner * 3;
            size_t qb, kb, vb;
            if (grouped) {
                qb = row_base + (size_t)head * hd * 3;
                kb = qb + hd;
                vb = kb + hd;
            } else {
                qb = row_base + (size_t)head * hd;
                kb = qb + inner;
                vb = qb + inner * 2;
            }
            float qs = 0.0f, ks = 0.0f;
            for (uint32_t d = 0; d < hd; d++) {
                qs = fmaf(qkv[qb + d], qkv[qb + d], qs);
                ks = fmaf(qkv[kb + d], qkv[kb + d], ks);
            }
            float qi = 1.0f / sqrtf(qs / (float)hd + eps);
            float ki = 1.0f / sqrtf(ks / (float)hd + eps);
            for (uint32_t dim = 0; dim < hd; dim++) {
                float w0q = weighted ? qw[dim] : 1.0f;
                float w0k = weighted ? kw[dim] : 1.0f;
                float q0 = qkv[qb + dim] * qi * w0q;
                float k0 = qkv[kb + dim] * ki * w0k;
                if (dim < rope_half) {
                    uint32_t pair = dim + rope_half;
                    float q1 = qkv[qb + pair] * qi * (weighted ? qw[pair] : 1.0f);
                    float k1 = qkv[kb + pair] * ki * (weighted ? kw[pair] : 1.0f);
                    float c = rc[row * rope_half + dim];
                    float s = rs[row * rope_half + dim];
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (dim < rope_half * 2) {
                    uint32_t pair = dim - rope_half;
                    float q1 = qkv[qb + pair] * qi * (weighted ? qw[pair] : 1.0f);
                    float k1 = qkv[kb + pair] * ki * (weighted ? kw[pair] : 1.0f);
                    float c = rc[row * rope_half + pair];
                    float s = rs[row * rope_half + pair];
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                size_t o = ((size_t)row * heads + head) * hd + dim;
                q_out[o] = q0;
                k_out[o] = k0;
                v_out[o] = qkv[vb + dim];
            }
        }
}

static void ref_vision_qkv_rope(const float *qkv, const float *rc,
                                const float *rs, float *q_out, float *k_out,
                                float *v_out, uint32_t seq, uint32_t heads,
                                uint32_t hd, uint32_t rope_half) {
    size_t inner = (size_t)heads * hd;
    for (uint32_t row = 0; row < seq; row++)
        for (uint32_t head = 0; head < heads; head++) {
            size_t row_base = (size_t)row * inner * 3;
            size_t qb = row_base + (size_t)head * hd;
            size_t kb = row_base + inner + (size_t)head * hd;
            size_t vb = row_base + inner * 2 + (size_t)head * hd;
            for (uint32_t dim = 0; dim < hd; dim++) {
                uint32_t ri = row * rope_half + dim % rope_half;
                float c = rc[ri], s = rs[ri];
                uint32_t pair =
                    dim < rope_half ? dim + rope_half : dim - rope_half;
                float q0 = qkv[qb + dim], q1 = qkv[qb + pair];
                float k0 = qkv[kb + dim], k1 = qkv[kb + pair];
                size_t o = ((size_t)row * heads + head) * hd + dim;
                q_out[o] = dim < rope_half ? q0 * c - q1 * s : q0 * c + q1 * s;
                k_out[o] = dim < rope_half ? k0 * c - k1 * s : k0 * c + k1 * s;
                v_out[o] = qkv[vb + dim];
            }
        }
}

/* Brute-force SDPA. query/key/value are [batch, seq, heads, hd]; output is
 * [batch, seq, heads, hd] or [batch, heads, seq, hd] when head_major. */
static void ref_sdpa(const float *q, const float *k, const float *v,
                     float *out, uint32_t batch, uint32_t seq, uint32_t heads,
                     uint32_t hd, float scale, int causal, int head_major) {
    float *scores = malloc(seq * sizeof(float));
    for (uint32_t b = 0; b < batch; b++)
        for (uint32_t h = 0; h < heads; h++)
            for (uint32_t i = 0; i < seq; i++) {
                size_t bb = (size_t)b * seq * heads * hd;
                const float *qp = q + bb + ((size_t)i * heads + h) * hd;
                uint32_t kc = causal ? i + 1 : seq;
                float mx = -INFINITY;
                for (uint32_t j = 0; j < kc; j++) {
                    const float *kp = k + bb + ((size_t)j * heads + h) * hd;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < hd; d++)
                        dot = fmaf(qp[d], kp[d], dot);
                    scores[j] = dot * scale;
                    if (scores[j] > mx) mx = scores[j];
                }
                float sum = 0.0f;
                for (uint32_t j = 0; j < kc; j++) {
                    scores[j] = expf(scores[j] - mx);
                    sum += scores[j];
                }
                float inv = 1.0f / sum;
                for (uint32_t d = 0; d < hd; d++) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < kc; j++) {
                        float vv = v[bb + ((size_t)j * heads + h) * hd + d];
                        acc = fmaf(scores[j] * inv, vv, acc);
                    }
                    size_t o = head_major
                                   ? bb + ((size_t)h * seq + i) * hd + d
                                   : bb + ((size_t)i * heads + h) * hd + d;
                    out[o] = acc;
                }
            }
    free(scores);
}

static void ref_gqa(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                    float *out, uint32_t seq, uint32_t qh, uint32_t kvh,
                    uint32_t hd, float scale) {
    float *scores = malloc(seq * sizeof(float));
    float *sq = malloc(hd * sizeof(float));
    for (uint32_t i = 0; i < seq; i++)
        for (uint32_t h = 0; h < qh; h++) {
            uint32_t kh = h / (qh / kvh);
            size_t qb = ((size_t)i * qh + h) * hd;
            for (uint32_t d = 0; d < hd; d++)
                sq[d] = c_bf16_to_f32(
                    c_f32_to_bf16(c_bf16_to_f32(q[qb + d]) * scale));
            uint32_t kc = i + 1;
            float mx = -INFINITY;
            for (uint32_t j = 0; j < kc; j++) {
                size_t kb = ((size_t)j * kvh + kh) * hd;
                float dot = 0.0f;
                for (uint32_t d = 0; d < hd; d++)
                    dot = fmaf(sq[d], c_bf16_to_f32(k[kb + d]), dot);
                scores[j] = dot;
                if (dot > mx) mx = dot;
            }
            float sum = 0.0f;
            for (uint32_t j = 0; j < kc; j++) {
                scores[j] = expf(scores[j] - mx);
                sum += scores[j];
            }
            float inv = 1.0f / sum;
            for (uint32_t d = 0; d < hd; d++) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < kc; j++)
                    acc = fmaf(scores[j] * inv,
                               c_bf16_to_f32(
                                   v[((size_t)j * kvh + kh) * hd + d]),
                               acc);
                out[qb + d] = acc;
            }
        }
    free(scores);
    free(sq);
}

static void ref_text_qk_rope(const float *qi, const float *ki, const float *qw,
                             const float *kw, const float *rc, const float *rs,
                             float *qo, float *ko, uint32_t seq, uint32_t qh,
                             uint32_t kvh, uint32_t hd, float eps) {
    uint32_t half = hd / 2;
    for (uint32_t row = 0; row < seq; row++) {
        for (uint32_t h = 0; h < qh; h++) {
            size_t qb = ((size_t)row * qh + h) * hd;
            float qs = 0.0f;
            for (uint32_t d = 0; d < hd; d++)
                qs = fmaf(qi[qb + d], qi[qb + d], qs);
            float inv = 1.0f / sqrtf(qs / (float)hd + eps);
            for (uint32_t dim = 0; dim < hd; dim++) {
                uint32_t pair = dim < half ? dim + half : dim - half;
                float c = rc[row * half + dim % half];
                float s = rs[row * half + dim % half];
                float q0 = qi[qb + dim] * inv * qw[dim];
                float q1 = qi[qb + pair] * inv * qw[pair];
                qo[qb + dim] =
                    dim < half ? q0 * c - q1 * s : q0 * c + q1 * s;
            }
        }
        for (uint32_t h = 0; h < kvh; h++) {
            size_t kb = ((size_t)row * kvh + h) * hd;
            float ks = 0.0f;
            for (uint32_t d = 0; d < hd; d++)
                ks = fmaf(ki[kb + d], ki[kb + d], ks);
            float inv = 1.0f / sqrtf(ks / (float)hd + eps);
            for (uint32_t dim = 0; dim < hd; dim++) {
                uint32_t pair = dim < half ? dim + half : dim - half;
                float c = rc[row * half + dim % half];
                float s = rs[row * half + dim % half];
                float k0 = ki[kb + dim] * inv * kw[dim];
                float k1 = ki[kb + pair] * inv * kw[pair];
                ko[kb + dim] =
                    dim < half ? k0 * c - k1 * s : k0 * c + k1 * s;
            }
        }
    }
}

static void ref_rope_text(float *q, float *k, const float *rc, const float *rs,
                          uint32_t seq, uint32_t qh, uint32_t kvh,
                          uint32_t hd) {
    uint32_t half = hd / 2;
    for (uint32_t row = 0; row < seq; row++) {
        for (uint32_t h = 0; h < qh; h++) {
            size_t b = ((size_t)row * qh + h) * hd;
            for (uint32_t d = 0; d < half; d++) {
                float f0 = q[b + d], f1 = q[b + half + d];
                float c = rc[row * half + d], s = rs[row * half + d];
                q[b + d] = f0 * c - f1 * s;
                q[b + half + d] = f1 * c + f0 * s;
            }
        }
        for (uint32_t h = 0; h < kvh; h++) {
            size_t b = ((size_t)row * kvh + h) * hd;
            for (uint32_t d = 0; d < half; d++) {
                float f0 = k[b + d], f1 = k[b + half + d];
                float c = rc[row * half + d], s = rs[row * half + d];
                k[b + d] = f0 * c - f1 * s;
                k[b + half + d] = f1 * c + f0 * s;
            }
        }
    }
}

static void ref_head_rms(float *t, const float *w, uint32_t seq,
                         uint32_t heads, uint32_t hd, float eps) {
    for (uint32_t row = 0; row < seq; row++)
        for (uint32_t h = 0; h < heads; h++) {
            size_t b = ((size_t)row * heads + h) * hd;
            float sum = 0.0f;
            for (uint32_t d = 0; d < hd; d++)
                sum = fmaf(t[b + d], t[b + d], sum);
            float inv = 1.0f / sqrtf(sum / (float)hd + eps);
            for (uint32_t d = 0; d < hd; d++) t[b + d] *= inv * w[d];
        }
}

/* ------------------------------------------------------------ test bodies */

static float *f32_buf(size_t n) {
    float *p = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) p[i] = frand();
    return p;
}
static uint16_t *bf16_buf(size_t n) {
    uint16_t *p = malloc(n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i++) p[i] = c_f32_to_bf16(frand());
    return p;
}
static float *bf16_as_f32(const uint16_t *p, size_t n) {
    float *o = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) o[i] = c_bf16_to_f32(p[i]);
    return o;
}
/* unit-norm-ish rope tables */
static float *rope_table(uint32_t seq, uint32_t half, int cosine) {
    float *p = malloc((size_t)seq * half * sizeof(float));
    for (uint32_t r = 0; r < seq; r++)
        for (uint32_t d = 0; d < half; d++) {
            float angle = (r + 1) * (d + 1) * 0.05f;
            p[r * half + d] = cosine ? cosf(angle) : sinf(angle);
        }
    return p;
}
static uint16_t *rope_table_bf16(uint32_t seq, uint32_t half, int cosine) {
    float *f = rope_table(seq, half, cosine);
    uint16_t *p = malloc((size_t)seq * half * sizeof(uint16_t));
    for (size_t i = 0; i < (size_t)seq * half; i++) p[i] = c_f32_to_bf16(f[i]);
    free(f);
    return p;
}

static void run_suite(h3_gpu *gpu) {
    const uint32_t seq = 5, heads = 2, hd = 8, half = 3;
    const size_t inner = (size_t)heads * hd, count = (size_t)seq * inner;
    float eps = 1e-5f;

    /* ---- error path: no active command buffer ---- */
    {
        h3_gpu_tensor *dummy = h3_gpu_tensor_new_f32(gpu, 16);
        CHECK(!h3_gpu_sdpa_f32(gpu, dummy, dummy, dummy, dummy, 2, 2, 2, 1.0f),
              "sdpa without begin must fail");
        h3_gpu_tensor_free(dummy);
    }

    /* ---- 1. h3_gpu_qkv_rope_f32 ---- */
    {
        float *qkv = f32_buf(count * 3), *qn = f32_buf(hd), *kn = f32_buf(hd);
        float *rc = rope_table(seq, half, 1), *rs = rope_table(seq, half, 0);
        h3_gpu_tensor *tqkv = h3_gpu_tensor_from_f32(gpu, qkv, count * 3);
        h3_gpu_tensor *tqn = h3_gpu_tensor_from_f32(gpu, qn, hd);
        h3_gpu_tensor *tkn = h3_gpu_tensor_from_f32(gpu, kn, hd);
        h3_gpu_tensor *trc = h3_gpu_tensor_from_f32(gpu, rc, seq * half);
        h3_gpu_tensor *trs = h3_gpu_tensor_from_f32(gpu, rs, seq * half);
        h3_gpu_tensor *tq = h3_gpu_tensor_new_f32(gpu, count);
        h3_gpu_tensor *tk = h3_gpu_tensor_new_f32(gpu, count);
        h3_gpu_tensor *tv = h3_gpu_tensor_new_f32(gpu, count);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_qkv_rope_f32(gpu, tq, tk, tv, tqkv, tqn, tkn, trc, trs,
                                  seq, heads, hd, half, eps),
              "qkv_rope_f32 call");
        /* error path: rope_half too large */
        CHECK(!h3_gpu_qkv_rope_f32(gpu, tq, tk, tv, tqkv, tqn, tkn, trc, trs,
                                   seq, heads, hd, hd / 2 + 1, eps),
              "qkv_rope_f32 oversized rope_half must fail");
        CHECK(h3_gpu_submit(gpu), "submit");
        float *rq = malloc(count * 4), *rk = malloc(count * 4),
              *rv = malloc(count * 4);
        ref_qkv_rope(qkv, qn, kn, rc, rs, rq, rk, rv, seq, heads, hd, half, 0,
                     1, eps);
        float *gq = malloc(count * 4), *gk = malloc(count * 4),
              *gv = malloc(count * 4);
        h3_gpu_tensor_read_f32(tq, gq, count);
        h3_gpu_tensor_read_f32(tk, gk, count);
        h3_gpu_tensor_read_f32(tv, gv, count);
        check_f32("qkv_rope_f32 q", gq, rq, count, 1e-5f);
        check_f32("qkv_rope_f32 k", gk, rk, count, 1e-5f);
        check_f32("qkv_rope_f32 v", gv, rv, count, 1e-5f);
        h3_gpu_tensor *all[] = {tqkv, tqn, tkn, trc, trs, tq, tk, tv};
        for (int i = 0; i < 8; i++) h3_gpu_tensor_free(all[i]);
        free(qkv); free(qn); free(kn); free(rc); free(rs);
        free(rq); free(rk); free(rv); free(gq); free(gk); free(gv);
    }

    /* ---- 2. h3_gpu_video_qkv_rope_f32 (grouped layout, unweighted) ---- */
    {
        float *qkv = f32_buf(count * 3);
        float *rc = rope_table(seq, half, 1), *rs = rope_table(seq, half, 0);
        h3_gpu_tensor *tqkv = h3_gpu_tensor_from_f32(gpu, qkv, count * 3);
        h3_gpu_tensor *trc = h3_gpu_tensor_from_f32(gpu, rc, seq * half);
        h3_gpu_tensor *trs = h3_gpu_tensor_from_f32(gpu, rs, seq * half);
        h3_gpu_tensor *tq = h3_gpu_tensor_new_f32(gpu, count);
        h3_gpu_tensor *tk = h3_gpu_tensor_new_f32(gpu, count);
        h3_gpu_tensor *tv = h3_gpu_tensor_new_f32(gpu, count);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_video_qkv_rope_f32(gpu, tq, tk, tv, tqkv, trc, trs, seq,
                                        heads, hd, half, eps),
              "video_qkv_rope_f32 call");
        /* error path: wrong dtype */
        h3_gpu_tensor *bad = h3_gpu_tensor_new_bf16(gpu, count);
        CHECK(!h3_gpu_video_qkv_rope_f32(gpu, tq, tk, tv, bad, trc, trs, seq,
                                         heads, hd, half, eps),
              "video_qkv_rope_f32 bf16 input must fail");
        h3_gpu_tensor_free(bad);
        CHECK(h3_gpu_submit(gpu), "submit");
        float *rq = malloc(count * 4), *rk = malloc(count * 4),
              *rv = malloc(count * 4);
        ref_qkv_rope(qkv, NULL, NULL, rc, rs, rq, rk, rv, seq, heads, hd,
                     half, 1, 0, eps);
        float *gq = malloc(count * 4), *gk = malloc(count * 4),
              *gv = malloc(count * 4);
        h3_gpu_tensor_read_f32(tq, gq, count);
        h3_gpu_tensor_read_f32(tk, gk, count);
        h3_gpu_tensor_read_f32(tv, gv, count);
        check_f32("video_qkv_rope_f32 q", gq, rq, count, 1e-5f);
        check_f32("video_qkv_rope_f32 k", gk, rk, count, 1e-5f);
        check_f32("video_qkv_rope_f32 v", gv, rv, count, 1e-5f);
        h3_gpu_tensor *all[] = {tqkv, trc, trs, tq, tk, tv};
        for (int i = 0; i < 6; i++) h3_gpu_tensor_free(all[i]);
        free(qkv); free(rc); free(rs);
        free(rq); free(rk); free(rv); free(gq); free(gk); free(gv);
    }

    /* ---- 3/4. SDPA f32 non-causal and causal (batch=2) ---- */
    {
        const uint32_t sseq = 5, shd = 4;
        const size_t sc = (size_t)sseq * heads * shd;
        float scale = 0.7f;
        float *q = f32_buf(sc), *k = f32_buf(sc), *v = f32_buf(sc);
        h3_gpu_tensor *tq = h3_gpu_tensor_from_f32(gpu, q, sc);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_f32(gpu, k, sc);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_f32(gpu, v, sc);
        h3_gpu_tensor *to = h3_gpu_tensor_new_f32(gpu, sc);
        h3_gpu_stats before, after;
        h3_gpu_get_stats(gpu, &before);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_sdpa_f32(gpu, to, tq, tk, tv, sseq, heads, shd, scale),
              "sdpa_f32 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        h3_gpu_get_stats(gpu, &after);
        CHECK(after.mps_sdpa_dispatches == before.mps_sdpa_dispatches + 1,
              "sdpa_f32 must bump mps_sdpa_dispatches");
        float *ref = malloc(sc * 4), *got = malloc(sc * 4);
        ref_sdpa(q, k, v, ref, 1, sseq, heads, shd, scale, 0, 0);
        h3_gpu_tensor_read_f32(to, got, sc);
        check_f32("sdpa_f32", got, ref, sc,
                  sdpa_naive_mode ? 1e-5f : SDPA_FLASH_F32_TOL);

        /* causal with batch=2 */
        const uint32_t batch = 2;
        const size_t bc = batch * sc;
        float *q2 = f32_buf(bc), *k2 = f32_buf(bc), *v2 = f32_buf(bc);
        h3_gpu_tensor *tq2 = h3_gpu_tensor_from_f32(gpu, q2, bc);
        h3_gpu_tensor *tk2 = h3_gpu_tensor_from_f32(gpu, k2, bc);
        h3_gpu_tensor *tv2 = h3_gpu_tensor_from_f32(gpu, v2, bc);
        h3_gpu_tensor *to2 = h3_gpu_tensor_new_f32(gpu, bc);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_sdpa_causal_f32(gpu, to2, tq2, tk2, tv2, batch, sseq,
                                     heads, shd, scale),
              "sdpa_causal_f32 call");
        /* error path: dtype mismatch */
        h3_gpu_tensor *bad = h3_gpu_tensor_new_bf16(gpu, bc);
        CHECK(!h3_gpu_sdpa_causal_f32(gpu, to2, bad, tk2, tv2, batch, sseq,
                                      heads, shd, scale),
              "sdpa_causal_f32 dtype mismatch must fail");
        h3_gpu_tensor_free(bad);
        CHECK(h3_gpu_submit(gpu), "submit");
        float *ref2 = malloc(bc * 4), *got2 = malloc(bc * 4);
        ref_sdpa(q2, k2, v2, ref2, batch, sseq, heads, shd, scale, 1, 0);
        h3_gpu_tensor_read_f32(to2, got2, bc);
        check_f32("sdpa_causal_f32", got2, ref2, bc,
                  sdpa_naive_mode ? 1e-5f : SDPA_FLASH_F32_TOL);
        h3_gpu_tensor *all[] = {tq, tk, tv, to, tq2, tk2, tv2, to2};
        for (int i = 0; i < 8; i++) h3_gpu_tensor_free(all[i]);
        free(q); free(k); free(v); free(ref); free(got);
        free(q2); free(k2); free(v2); free(ref2); free(got2);
    }

    /* ---- 5/6. SDPA bf16 + head-major output ---- */
    {
        const uint32_t sseq = 5, shd = 4;
        const size_t sc = (size_t)sseq * heads * shd;
        float scale = 0.5f;
        uint16_t *q = bf16_buf(sc), *k = bf16_buf(sc), *v = bf16_buf(sc);
        float *qf = bf16_as_f32(q, sc), *kf = bf16_as_f32(k, sc),
              *vf = bf16_as_f32(v, sc);
        h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, sc);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, sc);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, sc);
        h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, sc);
        h3_gpu_tensor *toh = h3_gpu_tensor_new_bf16(gpu, sc);
        h3_gpu_stats before, after;
        h3_gpu_get_stats(gpu, &before);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_sdpa_bf16(gpu, to, tq, tk, tv, sseq, heads, shd, scale),
              "sdpa_bf16 call");
        CHECK(h3_gpu_sdpa_bf16_head_major_output(gpu, toh, tq, tk, tv, sseq,
                                                 heads, shd, scale),
              "sdpa_bf16_head_major call");
        CHECK(h3_gpu_submit(gpu), "submit");
        h3_gpu_get_stats(gpu, &after);
        CHECK(after.mps_sdpa_dispatches == before.mps_sdpa_dispatches + 2,
              "bf16 sdpas must bump mps_sdpa_dispatches");
        float *ref = malloc(sc * 4), *refh = malloc(sc * 4);
        ref_sdpa(qf, kf, vf, ref, 1, sseq, heads, shd, scale, 0, 0);
        ref_sdpa(qf, kf, vf, refh, 1, sseq, heads, shd, scale, 0, 1);
        uint16_t *got = malloc(sc * 2), *goth = malloc(sc * 2);
        h3_gpu_tensor_read_bf16(to, got, sc);
        h3_gpu_tensor_read_bf16(toh, goth, sc);
        if (sdpa_naive_mode) {
            check_bf16("sdpa_bf16", got, ref, sc, 1);
            check_bf16("sdpa_bf16_head_major", goth, refh, sc, 1);
        } else {
            check_bf16_tol("sdpa_bf16", got, ref, sc, SDPA_FLASH_BF16_ABS,
                           SDPA_FLASH_BF16_REL);
            check_bf16_tol("sdpa_bf16_head_major", goth, refh, sc,
                           SDPA_FLASH_BF16_ABS, SDPA_FLASH_BF16_REL);
        }
        h3_gpu_tensor *all[] = {tq, tk, tv, to, toh};
        for (int i = 0; i < 5; i++) h3_gpu_tensor_free(all[i]);
        free(q); free(k); free(v); free(qf); free(kf); free(vf);
        free(ref); free(refh); free(got); free(goth);
    }

    /* ---- 6b. SDPA bf16 at head_dim 128 (tensor-core dispatch shape) ----
     * On sm_80+ this exercises h3k_sdpa_mma_bf16; elsewhere it stays on the
     * scalar flash kernel. seq 100 crosses both the 16-row Q tiles and the
     * 64-row KV tiles. The mma path rounds the f32 softmax probabilities to
     * bf16 before P*V (one rounding boundary beyond the scalar flash), which
     * stays inside the standard flash tolerances below. */
    {
        const uint32_t sseq = 100, sh = 3, shd = 128;
        const size_t sc = (size_t)sseq * sh * shd;
        float scale = 0.088f;
        uint16_t *q = bf16_buf(sc), *k = bf16_buf(sc), *v = bf16_buf(sc);
        float *qf = bf16_as_f32(q, sc), *kf = bf16_as_f32(k, sc),
              *vf = bf16_as_f32(v, sc);
        h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, sc);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, sc);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, sc);
        h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, sc);
        h3_gpu_tensor *toh = h3_gpu_tensor_new_bf16(gpu, sc);
        h3_gpu_stats before, after;
        h3_gpu_get_stats(gpu, &before);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_sdpa_bf16(gpu, to, tq, tk, tv, sseq, sh, shd, scale),
              "sdpa_bf16 hd128 call");
        CHECK(h3_gpu_sdpa_bf16_head_major_output(gpu, toh, tq, tk, tv, sseq,
                                                 sh, shd, scale),
              "sdpa_bf16_head_major hd128 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        h3_gpu_get_stats(gpu, &after);
        CHECK(after.mps_sdpa_dispatches == before.mps_sdpa_dispatches + 2,
              "hd128 bf16 sdpas must bump mps_sdpa_dispatches");
        float *ref = malloc(sc * 4), *refh = malloc(sc * 4);
        ref_sdpa(qf, kf, vf, ref, 1, sseq, sh, shd, scale, 0, 0);
        ref_sdpa(qf, kf, vf, refh, 1, sseq, sh, shd, scale, 0, 1);
        uint16_t *got = malloc(sc * 2), *goth = malloc(sc * 2);
        h3_gpu_tensor_read_bf16(to, got, sc);
        h3_gpu_tensor_read_bf16(toh, goth, sc);
        /* Both modes use the tolerance check here: at seq=100/hd=128 even
         * the naive kernel exceeds the 1-ulp bound the small-shape tests
         * hold, because GPU expf differs from the CPU reference by ~1 f32
         * ulp per key and cancellation near zero amplifies that to a few
         * bf16 ulps (observed: 5 ulps at |out| ~ 7e-7). */
        check_bf16_tol("sdpa_bf16_hd128", got, ref, sc, SDPA_FLASH_BF16_ABS,
                       SDPA_FLASH_BF16_REL);
        check_bf16_tol("sdpa_bf16_hd128_head_major", goth, refh, sc,
                       SDPA_FLASH_BF16_ABS, SDPA_FLASH_BF16_REL);
        h3_gpu_tensor *all[] = {tq, tk, tv, to, toh};
        for (int i = 0; i < 5; i++) h3_gpu_tensor_free(all[i]);
        free(q); free(k); free(v); free(qf); free(kf); free(vf);
        free(ref); free(refh); free(got); free(goth);
    }

    /* ---- 6c. SDPA f32 at head_dim 64 (FA2 route through bf16) ----
     * On sm_80+ h3_cuda_sdpa routes non-causal f32 SDPA at head_dim 64/128
     * through the bf16 FA2 kernel (the video VAE decoder's attention shape),
     * rounding Q/K/V to bf16 in the workspace and widening the result;
     * H3_CUDA_SDPA_F32_EXACT=1 keeps the f32 kernels. The inputs here are
     * bf16-representable, so the f32 reference sees the same values and the
     * check holds the route to the bf16 output rounding (abs 1e-2); in the
     * naive pass the exact f32 kernel runs and 1e-5 applies. seq 130 crosses
     * the 128-row Q tile and the 64-row KV tile. */
    {
        const uint32_t sseq = 130, sh = 3, shd = 64;
        const size_t sc = (size_t)sseq * sh * shd;
        float scale = 0.125f;
        uint16_t *q = bf16_buf(sc), *k = bf16_buf(sc), *v = bf16_buf(sc);
        float *qf = bf16_as_f32(q, sc), *kf = bf16_as_f32(k, sc),
              *vf = bf16_as_f32(v, sc);
        h3_gpu_tensor *tq = h3_gpu_tensor_from_f32(gpu, qf, sc);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_f32(gpu, kf, sc);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_f32(gpu, vf, sc);
        h3_gpu_tensor *to = h3_gpu_tensor_new_f32(gpu, sc);
        h3_gpu_stats before, after;
        h3_gpu_get_stats(gpu, &before);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_sdpa_f32(gpu, to, tq, tk, tv, sseq, sh, shd, scale),
              "sdpa_f32 hd64 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        h3_gpu_get_stats(gpu, &after);
        CHECK(after.mps_sdpa_dispatches == before.mps_sdpa_dispatches + 1,
              "hd64 f32 sdpa must bump mps_sdpa_dispatches once");
        float *ref = malloc(sc * 4), *got = malloc(sc * 4);
        ref_sdpa(qf, kf, vf, ref, 1, sseq, sh, shd, scale, 0, 0);
        h3_gpu_tensor_read_f32(to, got, sc);
        check_f32("sdpa_f32_hd64", got, ref, sc,
                  sdpa_naive_mode ? 1e-5f : 1e-2f);
        h3_gpu_tensor *all[] = {tq, tk, tv, to};
        for (int i = 0; i < 4; i++) h3_gpu_tensor_free(all[i]);
        free(q); free(k); free(v); free(qf); free(kf); free(vf);
        free(ref); free(got);
    }

    /* ---- 7. h3_gpu_gqa_causal_bf16 ---- */
    {
        const uint32_t gseq = 6, qh = 4, kvh = 2, ghd = 8;
        const size_t qc = (size_t)gseq * qh * ghd, kc = (size_t)gseq * kvh * ghd;
        float scale = 0.4f;
        uint16_t *q = bf16_buf(qc), *k = bf16_buf(kc), *v = bf16_buf(kc);
        h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, qc);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, kc);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, kc);
        h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, qc);
        h3_gpu_stats before, after;
        h3_gpu_get_stats(gpu, &before);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_gqa_causal_bf16(gpu, to, tq, tk, tv, gseq, qh, kvh, ghd,
                                     scale),
              "gqa_causal_bf16 call");
        /* error path: head_dim > 128 */
        CHECK(!h3_gpu_gqa_causal_bf16(gpu, to, tq, tk, tv, gseq, qh, kvh,
                                      132, scale),
              "gqa head_dim>128 must fail");
        CHECK(h3_gpu_submit(gpu), "submit");
        h3_gpu_get_stats(gpu, &after);
        CHECK(after.direct_dispatches == before.direct_dispatches + 1,
              "gqa must bump direct_dispatches");
        CHECK(after.mps_sdpa_dispatches == before.mps_sdpa_dispatches,
              "gqa must not bump mps_sdpa_dispatches");
        float *ref = malloc(qc * 4);
        ref_gqa(q, k, v, ref, gseq, qh, kvh, ghd, scale);
        uint16_t *got = malloc(qc * 2);
        h3_gpu_tensor_read_bf16(to, got, qc);
        if (sdpa_naive_mode)
            check_bf16("gqa_causal_bf16", got, ref, qc, 2);
        else
            check_bf16_tol("gqa_causal_bf16", got, ref, qc,
                           SDPA_FLASH_BF16_ABS, SDPA_FLASH_BF16_REL);
        h3_gpu_tensor *all[] = {tq, tk, tv, to};
        for (int i = 0; i < 4; i++) h3_gpu_tensor_free(all[i]);
        free(q); free(k); free(v); free(ref); free(got);
    }

    /* ---- 8/9. qkv_rope_bf16 ungrouped + grouped ---- */
    for (int grouped = 0; grouped <= 1; grouped++) {
        uint16_t *qkv = bf16_buf(count * 3), *qn = bf16_buf(hd),
                 *kn = bf16_buf(hd);
        uint16_t *rc = rope_table_bf16(seq, half, 1),
                 *rs = rope_table_bf16(seq, half, 0);
        float *qkvf = bf16_as_f32(qkv, count * 3), *qnf = bf16_as_f32(qn, hd),
              *knf = bf16_as_f32(kn, hd);
        float *rcf = bf16_as_f32(rc, seq * half),
              *rsf = bf16_as_f32(rs, seq * half);
        h3_gpu_tensor *tqkv = h3_gpu_tensor_from_bf16(gpu, qkv, count * 3);
        h3_gpu_tensor *tqn = h3_gpu_tensor_from_bf16(gpu, qn, hd);
        h3_gpu_tensor *tkn = h3_gpu_tensor_from_bf16(gpu, kn, hd);
        h3_gpu_tensor *trc = h3_gpu_tensor_from_bf16(gpu, rc, seq * half);
        h3_gpu_tensor *trs = h3_gpu_tensor_from_bf16(gpu, rs, seq * half);
        h3_gpu_tensor *tq = h3_gpu_tensor_new_bf16(gpu, count);
        h3_gpu_tensor *tk = h3_gpu_tensor_new_bf16(gpu, count);
        h3_gpu_tensor *tv = h3_gpu_tensor_new_bf16(gpu, count);
        CHECK(h3_gpu_begin(gpu), "begin");
        int ok = grouped ? h3_gpu_grouped_qkv_rope_bf16(
                               gpu, tq, tk, tv, tqkv, tqn, tkn, trc, trs, seq,
                               heads, hd, half, eps)
                         : h3_gpu_qkv_rope_bf16(gpu, tq, tk, tv, tqkv, tqn,
                                                tkn, trc, trs, seq, heads, hd,
                                                half, eps);
        CHECK(ok, "qkv_rope_bf16 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        float *rq = malloc(count * 4), *rk = malloc(count * 4),
              *rv = malloc(count * 4);
        ref_qkv_rope(qkvf, qnf, knf, rcf, rsf, rq, rk, rv, seq, heads, hd,
                     half, grouped, 1, eps);
        uint16_t *gq = malloc(count * 2), *gk = malloc(count * 2),
                 *gv = malloc(count * 2);
        h3_gpu_tensor_read_bf16(tq, gq, count);
        h3_gpu_tensor_read_bf16(tk, gk, count);
        h3_gpu_tensor_read_bf16(tv, gv, count);
        const char *tag = grouped ? "grouped_qkv_rope_bf16" : "qkv_rope_bf16";
        char name[64];
        snprintf(name, sizeof(name), "%s q", tag);
        check_bf16(name, gq, rq, count, 1);
        snprintf(name, sizeof(name), "%s k", tag);
        check_bf16(name, gk, rk, count, 1);
        snprintf(name, sizeof(name), "%s v", tag);
        check_bf16(name, gv, rv, count, 1);
        h3_gpu_tensor *all[] = {tqkv, tqn, tkn, trc, trs, tq, tk, tv};
        for (int i = 0; i < 8; i++) h3_gpu_tensor_free(all[i]);
        free(qkv); free(qn); free(kn); free(rc); free(rs);
        free(qkvf); free(qnf); free(knf); free(rcf); free(rsf);
        free(rq); free(rk); free(rv); free(gq); free(gk); free(gv);
    }

    /* ---- 10. grouped_qkv_linear_rope_bf16 (composition) ---- */
    {
        const uint32_t rows = 5, in_dim = 6, lh = 2, lhd = 4, lhalf = 2;
        const uint32_t linner = lh * lhd;
        const size_t qkvc = (size_t)rows * linner * 3;
        const size_t outc = (size_t)rows * linner;
        uint16_t *in = bf16_buf((size_t)rows * in_dim);
        uint16_t *w = bf16_buf((size_t)linner * 3 * in_dim);
        uint16_t *qn = bf16_buf(lhd), *kn = bf16_buf(lhd);
        uint16_t *rc = rope_table_bf16(rows, lhalf, 1),
                 *rs = rope_table_bf16(rows, lhalf, 0);
        h3_gpu_tensor *tin = h3_gpu_tensor_from_bf16(gpu, in, rows * in_dim);
        h3_gpu_tensor *tw =
            h3_gpu_tensor_from_bf16(gpu, w, (size_t)linner * 3 * in_dim);
        h3_gpu_tensor *tqn = h3_gpu_tensor_from_bf16(gpu, qn, lhd);
        h3_gpu_tensor *tkn = h3_gpu_tensor_from_bf16(gpu, kn, lhd);
        h3_gpu_tensor *trc = h3_gpu_tensor_from_bf16(gpu, rc, rows * lhalf);
        h3_gpu_tensor *trs = h3_gpu_tensor_from_bf16(gpu, rs, rows * lhalf);
        h3_gpu_tensor *tqkv = h3_gpu_tensor_new_bf16(gpu, qkvc);
        h3_gpu_tensor *tq = h3_gpu_tensor_new_bf16(gpu, outc);
        h3_gpu_tensor *tk = h3_gpu_tensor_new_bf16(gpu, outc);
        h3_gpu_tensor *tv = h3_gpu_tensor_new_bf16(gpu, outc);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_grouped_qkv_linear_rope_bf16(
                  gpu, tq, tk, tv, tqkv, tin, tw, tqn, tkn, trc, trs, rows,
                  in_dim, lh, lhd, lhalf, eps),
              "grouped_qkv_linear_rope_bf16 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        /* CPU reference: linear (float fma, RNE to bf16) then grouped rope. */
        float *qkvb = malloc(qkvc * 4);
        float *inf = bf16_as_f32(in, (size_t)rows * in_dim);
        float *wf = bf16_as_f32(w, (size_t)linner * 3 * in_dim);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t o = 0; o < linner * 3; o++) {
                float sum = 0.0f;
                for (uint32_t i = 0; i < in_dim; i++)
                    sum = fmaf(inf[(size_t)r * in_dim + i],
                               wf[(size_t)o * in_dim + i], sum);
                qkvb[(size_t)r * linner * 3 + o] =
                    c_bf16_to_f32(c_f32_to_bf16(sum));
            }
        float *qnf = bf16_as_f32(qn, lhd), *knf = bf16_as_f32(kn, lhd);
        float *rcf = bf16_as_f32(rc, (size_t)rows * lhalf),
              *rsf = bf16_as_f32(rs, (size_t)rows * lhalf);
        float *rq = malloc(outc * 4), *rk = malloc(outc * 4),
              *rv = malloc(outc * 4);
        ref_qkv_rope(qkvb, qnf, knf, rcf, rsf, rq, rk, rv, rows, lh, lhd,
                     lhalf, 1, 1, eps);
        uint16_t *gq = malloc(outc * 2), *gk = malloc(outc * 2),
                 *gv = malloc(outc * 2);
        h3_gpu_tensor_read_bf16(tq, gq, outc);
        h3_gpu_tensor_read_bf16(tk, gk, outc);
        h3_gpu_tensor_read_bf16(tv, gv, outc);
        check_bf16("grouped_linear_rope q", gq, rq, outc, 2);
        check_bf16("grouped_linear_rope k", gk, rk, outc, 2);
        check_bf16("grouped_linear_rope v", gv, rv, outc, 2);
        h3_gpu_tensor *all[] = {tin, tw, tqn, tkn, trc, trs, tqkv, tq, tk, tv};
        for (int i = 0; i < 10; i++) h3_gpu_tensor_free(all[i]);
        free(in); free(w); free(qn); free(kn); free(rc); free(rs);
        free(inf); free(wf); free(qkvb); free(qnf); free(knf);
        free(rcf); free(rsf); free(rq); free(rk); free(rv);
        free(gq); free(gk); free(gv);
    }

    /* ---- 11. vision_qkv_rope_bf16 (rope_half*2 == head_dim) ---- */
    {
        const uint32_t vhd = 8, vhalf = 4;
        const size_t vc = (size_t)seq * heads * vhd;
        uint16_t *qkv = bf16_buf(vc * 3);
        uint16_t *rc = rope_table_bf16(seq, vhalf, 1),
                 *rs = rope_table_bf16(seq, vhalf, 0);
        float *qkvf = bf16_as_f32(qkv, vc * 3);
        float *rcf = bf16_as_f32(rc, seq * vhalf),
              *rsf = bf16_as_f32(rs, seq * vhalf);
        h3_gpu_tensor *tqkv = h3_gpu_tensor_from_bf16(gpu, qkv, vc * 3);
        h3_gpu_tensor *trc = h3_gpu_tensor_from_bf16(gpu, rc, seq * vhalf);
        h3_gpu_tensor *trs = h3_gpu_tensor_from_bf16(gpu, rs, seq * vhalf);
        h3_gpu_tensor *tq = h3_gpu_tensor_new_bf16(gpu, vc);
        h3_gpu_tensor *tk = h3_gpu_tensor_new_bf16(gpu, vc);
        h3_gpu_tensor *tv = h3_gpu_tensor_new_bf16(gpu, vc);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_vision_qkv_rope_bf16(gpu, tq, tk, tv, tqkv, trc, trs,
                                          seq, heads, vhd, vhalf),
              "vision_qkv_rope_bf16 call");
        /* error path: rope_half*2 != head_dim */
        CHECK(!h3_gpu_vision_qkv_rope_bf16(gpu, tq, tk, tv, tqkv, trc, trs,
                                           seq, heads, vhd, vhalf - 1),
              "vision rope_half mismatch must fail");
        CHECK(h3_gpu_submit(gpu), "submit");
        float *rq = malloc(vc * 4), *rk = malloc(vc * 4), *rv = malloc(vc * 4);
        ref_vision_qkv_rope(qkvf, rcf, rsf, rq, rk, rv, seq, heads, vhd,
                            vhalf);
        uint16_t *gq = malloc(vc * 2), *gk = malloc(vc * 2), *gv = malloc(vc * 2);
        h3_gpu_tensor_read_bf16(tq, gq, vc);
        h3_gpu_tensor_read_bf16(tk, gk, vc);
        h3_gpu_tensor_read_bf16(tv, gv, vc);
        check_bf16("vision_qkv_rope q", gq, rq, vc, 1);
        check_bf16("vision_qkv_rope k", gk, rk, vc, 1);
        check_bf16("vision_qkv_rope v", gv, rv, vc, 1);
        h3_gpu_tensor *all[] = {tqkv, trc, trs, tq, tk, tv};
        for (int i = 0; i < 6; i++) h3_gpu_tensor_free(all[i]);
        free(qkv); free(rc); free(rs); free(qkvf); free(rcf); free(rsf);
        free(rq); free(rk); free(rv); free(gq); free(gk); free(gv);
    }

    /* ---- 12. text_qk_rope_bf16 ---- */
    {
        const uint32_t qh = 4, kvh = 2, thd = 8;
        const size_t qc = (size_t)seq * qh * thd, kc = (size_t)seq * kvh * thd;
        uint16_t *qi = bf16_buf(qc), *ki = bf16_buf(kc), *qn = bf16_buf(thd),
                 *kn = bf16_buf(thd);
        uint16_t *rc = rope_table_bf16(seq, thd / 2, 1),
                 *rs = rope_table_bf16(seq, thd / 2, 0);
        float *qif = bf16_as_f32(qi, qc), *kif = bf16_as_f32(ki, kc),
              *qnf = bf16_as_f32(qn, thd), *knf = bf16_as_f32(kn, thd);
        float *rcf = bf16_as_f32(rc, seq * thd / 2),
              *rsf = bf16_as_f32(rs, seq * thd / 2);
        h3_gpu_tensor *tqi = h3_gpu_tensor_from_bf16(gpu, qi, qc);
        h3_gpu_tensor *tki = h3_gpu_tensor_from_bf16(gpu, ki, kc);
        h3_gpu_tensor *tqn = h3_gpu_tensor_from_bf16(gpu, qn, thd);
        h3_gpu_tensor *tkn = h3_gpu_tensor_from_bf16(gpu, kn, thd);
        h3_gpu_tensor *trc = h3_gpu_tensor_from_bf16(gpu, rc, seq * thd / 2);
        h3_gpu_tensor *trs = h3_gpu_tensor_from_bf16(gpu, rs, seq * thd / 2);
        h3_gpu_tensor *tqo = h3_gpu_tensor_new_bf16(gpu, qc);
        h3_gpu_tensor *tko = h3_gpu_tensor_new_bf16(gpu, kc);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_text_qk_rope_bf16(gpu, tqo, tko, tqi, tki, tqn, tkn, trc,
                                       trs, seq, qh, kvh, thd, eps),
              "text_qk_rope_bf16 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        float *rq = malloc(qc * 4), *rk = malloc(kc * 4);
        ref_text_qk_rope(qif, kif, qnf, knf, rcf, rsf, rq, rk, seq, qh, kvh,
                         thd, eps);
        uint16_t *gq = malloc(qc * 2), *gk = malloc(kc * 2);
        h3_gpu_tensor_read_bf16(tqo, gq, qc);
        h3_gpu_tensor_read_bf16(tko, gk, kc);
        check_bf16("text_qk_rope q", gq, rq, qc, 1);
        check_bf16("text_qk_rope k", gk, rk, kc, 1);
        h3_gpu_tensor *all[] = {tqi, tki, tqn, tkn, trc, trs, tqo, tko};
        for (int i = 0; i < 8; i++) h3_gpu_tensor_free(all[i]);
        free(qi); free(ki); free(qn); free(kn); free(rc); free(rs);
        free(qif); free(kif); free(qnf); free(knf); free(rcf); free(rsf);
        free(rq); free(rk); free(gq); free(gk);
    }

    /* ---- 13. head_rms_norm_bf16 (lives in h3_cuda_core.cu) ---- */
    {
        const uint32_t nhd = 8;
        const size_t nc = (size_t)seq * heads * nhd;
        uint16_t *t = bf16_buf(nc), *w = bf16_buf(nhd);
        float *tf = bf16_as_f32(t, nc), *wf = bf16_as_f32(w, nhd);
        h3_gpu_tensor *tt = h3_gpu_tensor_from_bf16(gpu, t, nc);
        h3_gpu_tensor *tw = h3_gpu_tensor_from_bf16(gpu, w, nhd);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_head_rms_norm_bf16(gpu, tt, tw, seq, heads, nhd, eps),
              "head_rms_norm_bf16 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        ref_head_rms(tf, wf, seq, heads, nhd, eps);
        uint16_t *got = malloc(nc * 2);
        h3_gpu_tensor_read_bf16(tt, got, nc);
        check_bf16("head_rms_norm_bf16", got, tf, nc, 1);
        h3_gpu_tensor_free(tt);
        h3_gpu_tensor_free(tw);
        free(t); free(w); free(tf); free(wf); free(got);
    }

    /* ---- 14. rope_text_bf16 (f32 tables, in-place) ---- */
    {
        const uint32_t qh = 4, kvh = 2, thd = 8;
        const size_t qc = (size_t)seq * qh * thd, kc = (size_t)seq * kvh * thd;
        uint16_t *q = bf16_buf(qc), *k = bf16_buf(kc);
        float *rc = rope_table(seq, thd / 2, 1), *rs = rope_table(seq, thd / 2, 0);
        float *qf = bf16_as_f32(q, qc), *kf = bf16_as_f32(k, kc);
        h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, qc);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, kc);
        h3_gpu_tensor *trc = h3_gpu_tensor_from_f32(gpu, rc, seq * thd / 2);
        h3_gpu_tensor *trs = h3_gpu_tensor_from_f32(gpu, rs, seq * thd / 2);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_rope_text_bf16(gpu, tq, tk, trc, trs, seq, qh, kvh, thd),
              "rope_text_bf16 call");
        CHECK(h3_gpu_submit(gpu), "submit");
        ref_rope_text(qf, kf, rc, rs, seq, qh, kvh, thd);
        uint16_t *gq = malloc(qc * 2), *gk = malloc(kc * 2);
        h3_gpu_tensor_read_bf16(tq, gq, qc);
        h3_gpu_tensor_read_bf16(tk, gk, kc);
        check_bf16("rope_text q", gq, qf, qc, 1);
        check_bf16("rope_text k", gk, kf, kc, 1);
        h3_gpu_tensor *all[] = {tq, tk, trc, trs};
        for (int i = 0; i < 4; i++) h3_gpu_tensor_free(all[i]);
        free(q); free(k); free(rc); free(rs); free(qf); free(kf);
        free(gq); free(gk);
    }

    /* ---- 15/16. audio_qkv_split_f32 + audio_attention_pool_f32 ---- */
    {
        const uint32_t batch = 2, len = 3, ah = 2, ahd = 8, odim = 2;
        const size_t width = (size_t)ah * ahd, ac = (size_t)batch * len * width;
        float *qkv = f32_buf(ac * 3), *qb = f32_buf(width),
              *kb = f32_buf(width), *vb = f32_buf(width);
        h3_gpu_tensor *tqkv = h3_gpu_tensor_from_f32(gpu, qkv, ac * 3);
        h3_gpu_tensor *tqb = h3_gpu_tensor_from_f32(gpu, qb, width);
        h3_gpu_tensor *tkb = h3_gpu_tensor_from_f32(gpu, kb, width);
        h3_gpu_tensor *tvb = h3_gpu_tensor_from_f32(gpu, vb, width);
        h3_gpu_tensor *tq = h3_gpu_tensor_new_f32(gpu, ac);
        h3_gpu_tensor *tk = h3_gpu_tensor_new_f32(gpu, ac);
        h3_gpu_tensor *tv = h3_gpu_tensor_new_f32(gpu, ac);
        h3_gpu_tensor *tp = h3_gpu_tensor_new_f32(gpu, (size_t)batch * len * odim);
        CHECK(h3_gpu_begin(gpu), "begin");
        CHECK(h3_gpu_audio_qkv_split_f32(gpu, tq, tk, tv, tqkv, tqb, tkb, tvb,
                                         batch, len, ah, ahd),
              "audio_qkv_split_f32 call");
        CHECK(h3_gpu_audio_attention_pool_f32(gpu, tp, tq, batch, len, ah, ahd,
                                              odim),
              "audio_attention_pool_f32 call");
        /* error path: head_dim % output_dim != 0 */
        CHECK(!h3_gpu_audio_attention_pool_f32(gpu, tp, tq, batch, len, ah,
                                               ahd, 3),
              "attention_pool bad output_dim must fail");
        CHECK(h3_gpu_submit(gpu), "submit");
        float *rq = malloc(ac * 4), *rk = malloc(ac * 4), *rv = malloc(ac * 4);
        for (size_t g = 0; g < ac; g++) {
            size_t col = g % width, base = (g / width) * width * 3;
            rq[g] = qkv[base + col] + qb[col];
            rk[g] = qkv[base + width + col] + kb[col];
            rv[g] = qkv[base + width * 2 + col] + vb[col];
        }
        float *rp = malloc((size_t)batch * len * odim * 4);
        uint32_t pool = ahd / odim;
        for (size_t g = 0; g < (size_t)batch * len * odim; g++) {
            size_t col = g % odim, row = g / odim;
            float sum = 0.0f;
            for (uint32_t h = 0; h < ah; h++) {
                size_t base = (row * ah + h) * ahd + col * pool;
                for (uint32_t it = 0; it < pool; it++) sum += rq[base + it];
            }
            rp[g] = sum / (float)(ah * pool);
        }
        float *gq = malloc(ac * 4), *gk = malloc(ac * 4), *gv = malloc(ac * 4),
              *gp = malloc((size_t)batch * len * odim * 4);
        h3_gpu_tensor_read_f32(tq, gq, ac);
        h3_gpu_tensor_read_f32(tk, gk, ac);
        h3_gpu_tensor_read_f32(tv, gv, ac);
        h3_gpu_tensor_read_f32(tp, gp, (size_t)batch * len * odim);
        check_f32("audio_qkv_split q", gq, rq, ac, 1e-5f);
        check_f32("audio_qkv_split k", gk, rk, ac, 1e-5f);
        check_f32("audio_qkv_split v", gv, rv, ac, 1e-5f);
        check_f32("audio_attention_pool", gp, rp, (size_t)batch * len * odim,
                  1e-5f);
        h3_gpu_tensor *all[] = {tqkv, tqb, tkb, tvb, tq, tk, tv, tp};
        for (int i = 0; i < 8; i++) h3_gpu_tensor_free(all[i]);
        free(qkv); free(qb); free(kb); free(vb);
        free(rq); free(rk); free(rv); free(rp);
        free(gq); free(gk); free(gv); free(gp);
    }

}

int main(void) {
    char err[512];
    h3_gpu *gpu = h3_gpu_create(NULL, err, sizeof(err));
    if (!gpu) {
        printf("no GPU: %s\n", err);
        return 77;
    }
    /* Run the whole suite twice: pass 1 forces the default flash SDPA/GQA
     * kernels (env var cleared so an exported H3_CUDA_SDPA_NAIVE cannot
     * change what is checked); pass 2 sets H3_CUDA_SDPA_NAIVE=1 so the naive
     * fallback kernels run and are verified against the same CPU references
     * with the original near-bit-exact tolerances. */
    unsetenv("H3_CUDA_SDPA_NAIVE");
    sdpa_naive_mode = 0;
    printf("== pass 1: flash SDPA kernels ==\n");
    run_suite(gpu);
    setenv("H3_CUDA_SDPA_NAIVE", "1", 1);
    sdpa_naive_mode = 1;
    printf("== pass 2: naive SDPA kernels (H3_CUDA_SDPA_NAIVE=1) ==\n");
    run_suite(gpu);
    h3_gpu_free(gpu);
    if (failures) {
        printf("FAILURES: %d\n", failures);
        return 1;
    }
    printf("all attention tests passed\n");
    return 0;
}
