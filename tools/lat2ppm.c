/* Decode a dumped video latent (H3_DUMP_LATENT) with the CUDA video VAE
 * and write three PPM frames (first/middle/last) for visual inspection.
 *
 *   gcc -std=c11 -D_DEFAULT_SOURCE -O2 -I. tools/lat2ppm.c libh3.a \
 *       -L$CUDA_HOME/lib64 -lcudart -lcublasLt -lcublas -lstdc++ -lm \
 *       -lpthread -o lat2ppm
 *   ./lat2ppm <video_vae_weight_dir> <latent.bin> <prefix>
 */
#include "h3_video_vae.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int write_ppm(const char *path, const h3_video_frames *frames,
                     int index) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", frames->width, frames->height);
    size_t pixels = (size_t)frames->height * frames->width * 3;
    const float *rgb = frames->rgb + (size_t)index * pixels;
    for (size_t i = 0; i < pixels; i++) {
        float v = rgb[i] * 255.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        fputc((int)(v + 0.5f), f);
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <weights> <latent.bin> <prefix>\n",
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
    h3_video_frames frames;
    char error[512];
    if (!h3_video_vae_decode(argv[1], "h3_shaders.metal", latent, t, h, w,
                             NULL, NULL, &frames, error, sizeof(error))) {
        fprintf(stderr, "decode failed: %s\n", error);
        return 1;
    }
    char path[512];
    int picks[3] = {0, frames.frames / 2, frames.frames - 1};
    const char *names[3] = {"first", "mid", "last"};
    for (int i = 0; i < 3; i++) {
        snprintf(path, sizeof(path), "%s-%s.ppm", argv[3], names[i]);
        if (!write_ppm(path, &frames, picks[i])) { perror(path); return 1; }
    }
    printf("%s: %d frames %dx%d written\n", argv[3], frames.frames,
           frames.width, frames.height);
    h3_video_frames_free(&frames);
    free(latent);
    return 0;
}
