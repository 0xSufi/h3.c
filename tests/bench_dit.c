#include "h3_dit.h"
#include "h3_safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    TEXT_ROWS = 6,
    TEXT_WIDTH = 5120,
    LATENT_T = 7,
    LATENT_H = 32,
    LATENT_W = 32,
    AUDIO_T = 37,
    VIDEO_ELEMENTS = 24 * LATENT_T * LATENT_H * LATENT_W,
    AUDIO_ELEMENTS = 32 * 2 * AUDIO_T
};

static void die(const char *message) {
    fprintf(stderr, "h3_dit_bench: %s\n", message);
    exit(1);
}

static double seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint16_t *load_text(const char *path) {
    char error[512];
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, sizeof(error))) die(error);
    const h3_st_tensor *tensor = h3_st_find(&header, "x.output");
    size_t elements = TEXT_ROWS * TEXT_WIDTH;
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(tensor) != elements)
        die("prompt fixture has the wrong schema");
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values || !h3_st_read_data(&header, tensor, values,
                                     elements * sizeof(*values),
                                     error, sizeof(error))) die(error);
    h3_st_free_header(&header);
    return values;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *prompt_fixture = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_prompt_bf16.safetensors";
    uint16_t *text_values = load_text(prompt_fixture);
    float *video = calloc(VIDEO_ELEMENTS, sizeof(*video));
    float *audio = calloc(AUDIO_ELEMENTS, sizeof(*audio));
    float *video_velocity = malloc(VIDEO_ELEMENTS * sizeof(*video_velocity));
    float *audio_velocity = malloc(AUDIO_ELEMENTS * sizeof(*audio_velocity));
    if (!video || !audio || !video_velocity || !audio_velocity)
        die("out of memory allocating benchmark latents");
    for (size_t index = 0; index < VIDEO_ELEMENTS; index++)
        video[index] = (float)((int)(index % 257) - 128) / 128.0f;
    for (size_t index = 0; index < AUDIO_ELEMENTS; index++)
        audio[index] = (float)((int)(index % 127) - 63) / 64.0f;

    h3_text_embedding text = {
        .tokens = TEXT_ROWS,
        .width = TEXT_WIDTH,
        .values = text_values
    };
    h3_layout_spec spec = {TEXT_ROWS, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                           22, NULL, 0, NULL, 0};
    h3_layout layout;
    h3_sigma_schedule sigmas;
    char error[512];
    if (!h3_layout_build(&spec, &layout, error, sizeof(error)) ||
        !h3_schedule_build(20, &sigmas)) die("cannot build benchmark layout");
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/transformer", model_root);
    double load_start = seconds();
    h3_dit *dit = h3_dit_load_t2va(
        weights, "h3_shaders.metal", &text, &layout, &sigmas, 50, 1,
        NULL, NULL, error, sizeof(error));
    if (!dit) die(error);
    double load_seconds = seconds() - load_start;

    static const int steps[] = {0, 3, 6, 9, 12, 15, 18};
    double forward_start = seconds();
    for (size_t index = 0; index < sizeof(steps) / sizeof(*steps); index++) {
        double step_start = seconds();
        if (!h3_dit_forward(dit, steps[index], video, audio,
                            video_velocity, audio_velocity,
                            error, sizeof(error))) die(error);
        for (size_t element = 0; element < VIDEO_ELEMENTS; element++)
            video[element] += video_velocity[element] * 0.001f;
        for (size_t element = 0; element < AUDIO_ELEMENTS; element++)
            audio[element] += audio_velocity[element] * 0.001f;
        printf("  core %zu step %d %.3fs\n", index + 1, steps[index],
               seconds() - step_start);
    }
    double forward_seconds = seconds() - forward_start;
    printf("DiT 512 load %.3fs, seven forwards %.3fs, combined %.3fs\n",
           load_seconds, forward_seconds, load_seconds + forward_seconds);

    h3_dit_free(dit);
    h3_layout_free(&layout);
    free(text_values); free(video); free(audio);
    free(video_velocity); free(audio_velocity);
    return 0;
}
