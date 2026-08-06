#include "h3_ffmpeg.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_av_mux.c: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    enum { WIDTH = 32, HEIGHT = 32, FRAMES = 8, SAMPLES = 3200 };
    const char *path = argc > 1 ? argv[1] : "/tmp/h3-av-mux-test.mp4";
    uint8_t *rgb = malloc(WIDTH * HEIGHT * 3 * FRAMES);
    float *pcm = malloc(2 * SAMPLES * sizeof(*pcm));
    if (!rgb || !pcm) die("out of memory creating mux fixture");
    for (int frame = 0; frame < FRAMES; frame++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++) {
                size_t pixel = ((size_t)frame * HEIGHT * WIDTH +
                                (size_t)y * WIDTH + (size_t)x) * 3;
                rgb[pixel] = (uint8_t)(x * 8);
                rgb[pixel + 1] = (uint8_t)(y * 8);
                rgb[pixel + 2] = (uint8_t)(frame * 32);
            }
    for (int channel = 0; channel < 2; channel++)
        for (int sample = 0; sample < SAMPLES; sample++)
            pcm[(size_t)channel * SAMPLES + (size_t)sample] =
                0.05f * sinf(2.0f * 3.14159265358979323846f *
                             (float)(220 + channel * 110) * (float)sample /
                             32000.0f);
    char error[512];
    if (!h3_ffmpeg_write_av_rgb24_f32(path, rgb, FRAMES, WIDTH, HEIGHT, 24,
                                      pcm, SAMPLES, 2, 32000,
                                      error, sizeof(error))) die(error);
    struct stat status;
    if (stat(path, &status) != 0 || status.st_size < 1000)
        die("FFmpeg did not create a nonempty A/V container");
    printf("ok: concurrent FFmpeg video/PCM pipes created %s (%lld bytes)\n",
           path, (long long)status.st_size);
    free(rgb);
    free(pcm);
    return 0;
}
