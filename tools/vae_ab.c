/* Decode one dumped video latent (H3_DUMP_LATENT from h3.c) with the CUDA
 * video VAE decoder under different backend precision settings and report
 * how far each variant lands from the exact F32 decode. Build (Linux):
 *
 *   gcc -std=c11 -D_DEFAULT_SOURCE -O2 -I. tools/vae_ab.c libh3.a \
 *       -L$CUDA_HOME/lib64 -lcudart -lcublasLt -lcublas -lstdc++ -lm \
 *       -lpthread -o vae_ab
 *   ./vae_ab MiniMax-H3/FL2VA/video_vae/source latent.bin
 *
 * Variants: exact (H3_CUDA_TF32=0, H3_CUDA_SDPA_F32_EXACT=1), tf32 (TF32
 * GEMMs, exact attention), attn (bf16 FA2 attention, exact GEMMs), fast
 * (both), bf16gemm (H3_CUDA_F32_GEMM=bf16 + bf16 attention), fp8gemm
 * (H3_CUDA_F32_GEMM=fp8 + bf16 attention). Reports rel-L2, max abs error,
 * and PSNR over RGB in [0,1]. */
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int decode(const char *weights, const float *latent, int t, int h,
                  int w, const char *tf32, const char *exact,
                  const char *f32_gemm, h3_video_frames *out,
                  double *seconds) {
    setenv("H3_CUDA_TF32", tf32, 1);
    setenv("H3_CUDA_SDPA_F32_EXACT", exact, 1);
    setenv("H3_CUDA_F32_GEMM", f32_gemm, 1);
    char error[512];
    memset(out, 0, sizeof(*out));
    double start = now();
    int ok = h3_video_vae_decode(weights, "h3_shaders.metal", latent, t, h, w,
                                 NULL, NULL, out, error, sizeof(error));
    *seconds = now() - start;
    if (!ok) fprintf(stderr, "decode failed: %s\n", error);
    return ok;
}

static void compare(const char *name, const h3_video_frames *ref,
                    const h3_video_frames *got, double seconds) {
    size_t n = (size_t)ref->frames * ref->height * ref->width * 3;
    if ((size_t)got->frames * got->height * got->width * 3 != n) {
        printf("%-6s shape mismatch\n", name);
        return;
    }
    double num = 0, den = 0, mse = 0, max_abs = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got->rgb[i] - (double)ref->rgb[i];
        num += d * d;
        den += (double)ref->rgb[i] * (double)ref->rgb[i];
        mse += d * d;
        if (fabs(d) > max_abs) max_abs = fabs(d);
    }
    mse /= (double)n;
    double psnr = mse > 0 ? 10.0 * log10(1.0 / mse) : INFINITY;
    printf("%-6s %7.2f s  rel-L2 %.3e  max|d| %.3e  PSNR %.2f dB\n", name,
           seconds, den > 0 ? sqrt(num / den) : 0.0, max_abs, psnr);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <video_vae_weight_dir> <latent.bin>\n",
                argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 2; }
    int32_t header[5];
    if (fread(header, sizeof(header), 1, f) != 1 || header[0] != 0x544c3348 ||
        header[1] != 24) {
        fprintf(stderr, "bad latent header\n");
        return 2;
    }
    int t = header[2], h = header[3], w = header[4];
    size_t count = (size_t)24 * t * h * w;
    float *latent = malloc(count * sizeof(float));
    if (!latent || fread(latent, sizeof(float), count, f) != count) {
        fprintf(stderr, "short latent\n");
        return 2;
    }
    fclose(f);
    printf("latent: t=%d h=%d w=%d\n", t, h, w);
    h3_video_frames exact, variant;
    double s_exact, s_var;
    if (!decode(argv[1], latent, t, h, w, "0", "1", "", &exact, &s_exact))
        return 1;
    printf("%-6s %7.2f s  (reference: F32 GEMMs, f32 attention)\n", "exact",
           s_exact);
    const char *names[] = {"tf32", "attn", "fast", "bf16gemm", "fp8gemm",
                           "exact2"};
    const char *tf32s[] = {"1", "0", "1", "1", "1", "0"};
    const char *exacts[] = {"1", "0", "0", "0", "0", "1"};
    const char *gemms[] = {"", "", "", "bf16", "fp8", ""};
    for (int i = 0; i < 6; i++) {
        if (!decode(argv[1], latent, t, h, w, tf32s[i], exacts[i], gemms[i],
                    &variant, &s_var))
            return 1;
        compare(names[i], &exact, &variant, s_var);
        h3_video_frames_free(&variant);
    }
    h3_video_frames_free(&exact);
    free(latent);
    return 0;
}
