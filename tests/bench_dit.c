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

static void run_command_ab(h3_dit *dit, int interval, float *video,
                           float *audio, float *video_velocity,
                           float *audio_velocity) {
    char error[512];
    setenv("H3_DIT_COMMAND_BLOCKS", "0", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating AB reference outputs");
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));

    static const int split_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double split_seconds = 0.0;
    int baseline_count = 0;
    int split_count = 0;
    char interval_text[16];
    snprintf(interval_text, sizeof(interval_text), "%d", interval);
    for (size_t index = 0;
         index < sizeof(split_pattern) / sizeof(*split_pattern); index++) {
        if (split_pattern[index])
            setenv("H3_DIT_COMMAND_BLOCKS", interval_text, 1);
        else
            setenv("H3_DIT_COMMAND_BLOCKS", "0", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_reference, video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_reference, audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("command split changed DiT output bytes");
        if (split_pattern[index]) {
            split_seconds += elapsed;
            split_count++;
            printf("  AB split %d %.3fs\n", interval, elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  AB baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DIT_COMMAND_BLOCKS");
    printf("DiT command AB baseline %.4fs, split-%d %.4fs, ratio %.4f, "
           "outputs byte-identical\n", baseline_seconds / baseline_count,
           interval, split_seconds / split_count,
           split_seconds * baseline_count /
               (baseline_seconds * split_count));
    free(video_reference);
    free(audio_reference);
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
    unsigned active_blocks = 50;
    const char *layers = getenv("H3_BENCH_LAYERS");
    if (layers && *layers) {
        char *end = NULL;
        long parsed = strtol(layers, &end, 10);
        if (end == layers || *end || parsed < 25 || parsed > 50)
            die("H3_BENCH_LAYERS must be in [25, 50]");
        active_blocks = (unsigned)parsed;
    }
    double load_start = seconds();
    h3_dit *dit = h3_dit_load_t2va(
        weights, "h3_shaders.metal", &text, &layout, &sigmas, active_blocks, 1,
        NULL, NULL, error, sizeof(error));
    if (!dit) die(error);
    double load_seconds = seconds() - load_start;

    const char *command_ab = getenv("H3_BENCH_COMMAND_AB");
    if (command_ab && *command_ab) {
        char *end = NULL;
        long interval = strtol(command_ab, &end, 10);
        if (end == command_ab || *end || interval < 1 || interval > 50)
            die("H3_BENCH_COMMAND_AB must be in [1, 50]");
        run_command_ab(dit, (int)interval, video, audio, video_velocity,
                       audio_velocity);
        printf("DiT 512/%u-layer load %.3fs before command AB\n",
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }

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
