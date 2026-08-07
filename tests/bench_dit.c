#include "h3_dit.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef H3_BENCH_LATENT_H
#define H3_BENCH_LATENT_H 32
#endif
#ifndef H3_BENCH_LATENT_W
#define H3_BENCH_LATENT_W 32
#endif

enum {
    TEXT_ROWS = 6,
    TEXT_WIDTH = 5120,
    LATENT_T = 7,
    LATENT_H = H3_BENCH_LATENT_H,
    LATENT_W = H3_BENCH_LATENT_W,
    CANVAS_H = LATENT_H * 16,
    CANVAS_W = LATENT_W * 16,
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

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const uint8_t *octets = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; index++) {
        hash ^= octets[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
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

static void run_graph_data_ab(h3_dit *dit, float *video, float *audio,
                              float *video_velocity, float *audio_velocity) {
    char error[512];
    setenv("H3_DISABLE_GRAPH_DATA_CACHE", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating graph-data AB references");
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_GRAPH_DATA_CACHE");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int cache_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double cached_seconds = 0.0;
    int baseline_count = 0;
    int cached_count = 0;
    for (size_t index = 0;
         index < sizeof(cache_pattern) / sizeof(*cache_pattern); index++) {
        if (cache_pattern[index]) unsetenv("H3_DISABLE_GRAPH_DATA_CACHE");
        else setenv("H3_DISABLE_GRAPH_DATA_CACHE", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_reference, video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_reference, audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("graph-data cache changed DiT output bytes");
        if (cache_pattern[index]) {
            cached_seconds += elapsed;
            cached_count++;
            printf("  AB graph-data cached %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  AB graph-data baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_GRAPH_DATA_CACHE");
    printf("DiT graph-data AB baseline %.4fs, cached-weights %.4fs, ratio %.4f, "
           "outputs byte-identical\n", baseline_seconds / baseline_count,
           cached_seconds / cached_count,
           cached_seconds * baseline_count /
               (baseline_seconds * cached_count));
    free(video_reference);
    free(audio_reference);
}

static void run_mps_command_ab(h3_dit *dit, float *video, float *audio,
                               float *video_velocity,
                               float *audio_velocity) {
    char error[512];
    setenv("H3_REUSE_MPS_COMMAND", "0", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating MPS-command AB references");
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    setenv("H3_REUSE_MPS_COMMAND", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int reuse_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double reused_seconds = 0.0;
    int baseline_count = 0;
    int reused_count = 0;
    for (size_t index = 0;
         index < sizeof(reuse_pattern) / sizeof(*reuse_pattern); index++) {
        if (reuse_pattern[index]) setenv("H3_REUSE_MPS_COMMAND", "1", 1);
        else setenv("H3_REUSE_MPS_COMMAND", "0", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_reference, video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_reference, audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("MPS-command reuse changed DiT output bytes");
        if (reuse_pattern[index]) {
            reused_seconds += elapsed;
            reused_count++;
            printf("  AB MPS command reused %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  AB MPS command baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_REUSE_MPS_COMMAND");
    printf("DiT MPS-command AB baseline %.4fs, reused %.4fs, ratio %.4f, "
           "outputs byte-identical\n", baseline_seconds / baseline_count,
           reused_seconds / reused_count,
           reused_seconds * baseline_count /
               (baseline_seconds * reused_count));
    free(video_reference);
    free(audio_reference);
}

static double relative_l2(const float *got, const float *want, size_t count,
                          double *maximum_absolute) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - want[index];
        double magnitude = fabs(delta);
        squared_error += delta * delta;
        squared_reference += (double)want[index] * want[index];
        if (magnitude > maximum) maximum = magnitude;
    }
    *maximum_absolute = maximum;
    return sqrt(squared_error / (squared_reference > 1e-30
                                 ? squared_reference : 1e-30));
}

static void run_nax_mlp_ab(h3_dit *dit, float *video, float *audio,
                           float *video_velocity, float *audio_velocity) {
    char error[512];
    char candidate_blocks[16] = {0};
    const char *blocks = getenv("H3_BENCH_NAX_COMMAND_BLOCKS");
    if (blocks) snprintf(candidate_blocks, sizeof(candidate_blocks), "%s",
                         blocks);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating NAX MLP AB references");
    setenv("H3_DISABLE_NAX_MLP", "1", 1);
    unsetenv("H3_DIT_COMMAND_BLOCKS");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_NAX_MLP");
    if (candidate_blocks[0])
        setenv("H3_DIT_COMMAND_BLOCKS", candidate_blocks, 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int nax_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double nax_seconds = 0.0;
    int baseline_count = 0;
    int nax_count = 0;
    double video_rel = 0.0, audio_rel = 0.0;
    double video_abs = 0.0, audio_abs = 0.0;
    for (size_t index = 0;
         index < sizeof(nax_pattern) / sizeof(*nax_pattern); index++) {
        if (nax_pattern[index]) {
            unsetenv("H3_DISABLE_NAX_MLP");
            if (candidate_blocks[0])
                setenv("H3_DIT_COMMAND_BLOCKS", candidate_blocks, 1);
            else unsetenv("H3_DIT_COMMAND_BLOCKS");
        } else {
            setenv("H3_DISABLE_NAX_MLP", "1", 1);
            unsetenv("H3_DIT_COMMAND_BLOCKS");
        }
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (nax_pattern[index]) {
            video_rel = relative_l2(video_velocity, video_reference,
                                    VIDEO_ELEMENTS, &video_abs);
            audio_rel = relative_l2(audio_velocity, audio_reference,
                                    AUDIO_ELEMENTS, &audio_abs);
            nax_seconds += elapsed;
            nax_count++;
            printf("  NAX MLP AB candidate %.3fs video relL2 %.6g; "
                   "audio relL2 %.6g\n", elapsed, video_rel, audio_rel);
        } else {
            if (memcmp(video_velocity, video_reference,
                       VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                memcmp(audio_velocity, audio_reference,
                       AUDIO_ELEMENTS * sizeof(*audio_reference)))
                die("repeated MPSGraph MLP changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  NAX MLP AB baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_NAX_MLP");
    unsetenv("H3_DIT_COMMAND_BLOCKS");
    printf("DiT NAX MLP AB baseline %.4fs, NAX %.4fs, ratio %.4f; "
           "video relL2 %.6g max %.6g; audio relL2 %.6g max %.6g\n",
           baseline_seconds / baseline_count, nax_seconds / nax_count,
           nax_seconds * baseline_count / (baseline_seconds * nax_count),
           video_rel, video_abs, audio_rel, audio_abs);
    free(video_reference);
    free(audio_reference);
}

static void run_sampler_ab(h3_dit *dit, float *video, float *audio,
                           float *video_velocity, float *audio_velocity) {
    char error[512];
    float *initial_video = malloc(VIDEO_ELEMENTS * sizeof(*initial_video));
    float *initial_audio = malloc(AUDIO_ELEMENTS * sizeof(*initial_audio));
    float *reference_video = malloc(VIDEO_ELEMENTS * sizeof(*reference_video));
    float *reference_audio = malloc(AUDIO_ELEMENTS * sizeof(*reference_audio));
    if (!initial_video || !initial_audio || !reference_video || !reference_audio)
        die("out of memory allocating sampler AB state");
    memcpy(initial_video, video, VIDEO_ELEMENTS * sizeof(*initial_video));
    memcpy(initial_audio, audio, AUDIO_ELEMENTS * sizeof(*initial_audio));

    /* Populate the shape-keyed MPSGraph caches outside the measured runs. */
    if (!h3_dit_forward(dit, 0, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    static const int gpu_pattern[] = {0, 1, 1, 0, 1, 0, 0, 1};
    double cpu_seconds = 0.0;
    double gpu_seconds = 0.0;
    int cpu_count = 0;
    int gpu_count = 0;
    int have_reference = 0;
    const char *gpu_command_blocks = getenv("H3_BENCH_GPU_COMMAND_BLOCKS");
    double video_rel = 0.0, audio_rel = 0.0;
    double video_abs = 0.0, audio_abs = 0.0;
    for (size_t run = 0; run < sizeof(gpu_pattern) / sizeof(*gpu_pattern);
         run++) {
        memcpy(video, initial_video, VIDEO_ELEMENTS * sizeof(*video));
        memcpy(audio, initial_audio, AUDIO_ELEMENTS * sizeof(*audio));
        if (gpu_pattern[run]) {
            unsetenv("H3_CPU_SAMPLER");
            setenv("H3_GPU_SAMPLER", "1", 1);
            if (gpu_command_blocks)
                setenv("H3_DIT_COMMAND_BLOCKS", gpu_command_blocks, 1);
        } else {
            setenv("H3_CPU_SAMPLER", "1", 1);
            unsetenv("H3_GPU_SAMPLER");
            if (gpu_command_blocks) unsetenv("H3_DIT_COMMAND_BLOCKS");
        }
        double start = seconds();
        if (!h3_dit_denoise_euler(dit, video, audio, 3, NULL, NULL,
                                  error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (!have_reference && !gpu_pattern[run]) {
            memcpy(reference_video, video,
                   VIDEO_ELEMENTS * sizeof(*reference_video));
            memcpy(reference_audio, audio,
                   AUDIO_ELEMENTS * sizeof(*reference_audio));
            have_reference = 1;
        }
        if (gpu_pattern[run]) {
            video_rel = relative_l2(video, reference_video, VIDEO_ELEMENTS,
                                    &video_abs);
            audio_rel = relative_l2(audio, reference_audio, AUDIO_ELEMENTS,
                                    &audio_abs);
            gpu_seconds += elapsed;
            gpu_count++;
            printf("  sampler AB GPU %.3fs video relL2 %.6g max %.6g; "
                   "audio relL2 %.6g max %.6g\n", elapsed, video_rel,
                   video_abs, audio_rel, audio_abs);
        } else {
            if (have_reference &&
                (memcmp(video, reference_video, sizeof(*video) * VIDEO_ELEMENTS) ||
                 memcmp(audio, reference_audio, sizeof(*audio) * AUDIO_ELEMENTS)))
                die("repeated CPU sampler changed output bytes");
            cpu_seconds += elapsed;
            cpu_count++;
            printf("  sampler AB CPU %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_GPU_SAMPLER");
    unsetenv("H3_CPU_SAMPLER");
    if (gpu_command_blocks) unsetenv("H3_DIT_COMMAND_BLOCKS");
    printf("DiT sampler AB CPU %.4fs, GPU-chain %.4fs, ratio %.4f; "
           "video relL2 %.6g max %.6g; audio relL2 %.6g max %.6g\n",
           cpu_seconds / cpu_count, gpu_seconds / gpu_count,
           gpu_seconds * cpu_count / (cpu_seconds * gpu_count),
           video_rel, video_abs, audio_rel, audio_abs);
    free(initial_video);
    free(initial_audio);
    free(reference_video);
    free(reference_audio);
}

static void run_token_reduction_ab(h3_dit *dit, float *video, float *audio,
                                   float *video_velocity,
                                   float *audio_velocity) {
    char error[512];
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating token-reduction AB references");
    setenv("H3_DISABLE_TOKEN_REDUCTION", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_TOKEN_REDUCTION");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int candidate_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double candidate_seconds = 0.0;
    int baseline_count = 0;
    int candidate_count = 0;
    uint64_t candidate_video_hash = 0, candidate_audio_hash = 0;
    double video_relative = 0.0, audio_relative = 0.0;
    double video_absolute = 0.0, audio_absolute = 0.0;
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        if (candidate_pattern[run]) unsetenv("H3_DISABLE_TOKEN_REDUCTION");
        else setenv("H3_DISABLE_TOKEN_REDUCTION", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (candidate_pattern[run]) {
            uint64_t video_hash = hash_bytes(
                video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
            uint64_t audio_hash = hash_bytes(
                audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
            if (candidate_count &&
                (video_hash != candidate_video_hash ||
                 audio_hash != candidate_audio_hash))
                die("token-reduction candidate changed output bytes");
            candidate_video_hash = video_hash;
            candidate_audio_hash = audio_hash;
            video_relative = relative_l2(
                video_velocity, video_reference, VIDEO_ELEMENTS,
                &video_absolute);
            audio_relative = relative_l2(
                audio_velocity, audio_reference, AUDIO_ELEMENTS,
                &audio_absolute);
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  token reduction candidate %.3fs video relL2 %.6g; "
                   "audio relL2 %.6g\n", elapsed, video_relative,
                   audio_relative);
        } else {
            if (memcmp(video_reference, video_velocity,
                       VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                memcmp(audio_reference, audio_velocity,
                       AUDIO_ELEMENTS * sizeof(*audio_reference)))
                die("token-reduction baseline changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  token reduction baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_TOKEN_REDUCTION");
    printf("DiT token reduction AB baseline %.4fs, candidate %.4fs, "
           "ratio %.4f; video relL2 %.6g max %.6g; audio relL2 %.6g "
           "max %.6g\n", baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           video_relative, video_absolute, audio_relative, audio_absolute);
    printf("token reduction candidate hashes video %016llx audio %016llx\n",
           (unsigned long long)candidate_video_hash,
           (unsigned long long)candidate_audio_hash);
    h3_gpu_stats stats;
    if (!h3_dit_get_gpu_stats(dit, &stats))
        die("cannot read token-reduction GPU allocation stats");
    printf("token reduction cumulative GPU allocations %.3f GiB\n",
           (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0));
    free(video_reference);
    free(audio_reference);
}

static void run_token_reduction_denoise_ab(h3_dit *dit, float *video,
                                   float *audio, int reuse_interval) {
    char error[512];
    float *initial_video = malloc(VIDEO_ELEMENTS * sizeof(*initial_video));
    float *initial_audio = malloc(AUDIO_ELEMENTS * sizeof(*initial_audio));
    float *reference_video = malloc(VIDEO_ELEMENTS * sizeof(*reference_video));
    float *reference_audio = malloc(AUDIO_ELEMENTS * sizeof(*reference_audio));
    if (!initial_video || !initial_audio || !reference_video || !reference_audio)
        die("out of memory allocating token-reduction denoise AB state");
    memcpy(initial_video, video, VIDEO_ELEMENTS * sizeof(*initial_video));
    memcpy(initial_audio, audio, AUDIO_ELEMENTS * sizeof(*initial_audio));
    static const int candidate_pattern[] = {0, 1, 1, 0};
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0, have_reference = 0;
    uint64_t candidate_video_hash = 0, candidate_audio_hash = 0;
    double video_relative = 0.0, audio_relative = 0.0;
    double video_absolute = 0.0, audio_absolute = 0.0;
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        memcpy(video, initial_video, VIDEO_ELEMENTS * sizeof(*video));
        memcpy(audio, initial_audio, AUDIO_ELEMENTS * sizeof(*audio));
        if (candidate_pattern[run]) unsetenv("H3_DISABLE_TOKEN_REDUCTION");
        else setenv("H3_DISABLE_TOKEN_REDUCTION", "1", 1);
        double start = seconds();
        if (!h3_dit_denoise_euler(dit, video, audio, reuse_interval, NULL, NULL,
                                  error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (!have_reference && !candidate_pattern[run]) {
            memcpy(reference_video, video,
                   VIDEO_ELEMENTS * sizeof(*reference_video));
            memcpy(reference_audio, audio,
                   AUDIO_ELEMENTS * sizeof(*reference_audio));
            have_reference = 1;
        }
        if (candidate_pattern[run]) {
            uint64_t video_hash = hash_bytes(
                video, VIDEO_ELEMENTS * sizeof(*video));
            uint64_t audio_hash = hash_bytes(
                audio, AUDIO_ELEMENTS * sizeof(*audio));
            if (candidate_count &&
                (video_hash != candidate_video_hash ||
                 audio_hash != candidate_audio_hash))
                die("token-reduction denoise candidate changed output bytes");
            candidate_video_hash = video_hash;
            candidate_audio_hash = audio_hash;
            video_relative = relative_l2(video, reference_video,
                                         VIDEO_ELEMENTS, &video_absolute);
            audio_relative = relative_l2(audio, reference_audio,
                                         AUDIO_ELEMENTS, &audio_absolute);
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  token reduction denoise candidate %.3fs video relL2 "
                   "%.6g; audio relL2 %.6g\n", elapsed, video_relative,
                   audio_relative);
        } else {
            if (have_reference &&
                (memcmp(video, reference_video,
                        VIDEO_ELEMENTS * sizeof(*reference_video)) ||
                 memcmp(audio, reference_audio,
                        AUDIO_ELEMENTS * sizeof(*reference_audio))))
                die("token-reduction denoise baseline changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  token reduction denoise baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_TOKEN_REDUCTION");
    printf("DiT token reduction denoise AB baseline %.4fs, candidate %.4fs, "
           "ratio %.4f; video relL2 %.6g max %.6g; audio relL2 %.6g max "
           "%.6g\n", baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           video_relative, video_absolute, audio_relative, audio_absolute);
    printf("token reduction denoise reuse-%d hashes video %016llx audio "
           "%016llx\n", reuse_interval,
           (unsigned long long)candidate_video_hash,
           (unsigned long long)candidate_audio_hash);
    free(initial_video); free(initial_audio);
    free(reference_video); free(reference_audio);
}

static void run_cross_adaln_ab(h3_dit *dit, float *video, float *audio,
                               float *video_velocity,
                               float *audio_velocity) {
    char error[512];
    static const int candidate_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0;
    uint64_t video_hash = 0, audio_hash = 0;
    setenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    video_hash = hash_bytes(
        video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
    audio_hash = hash_bytes(
        audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
    unsetenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    if (hash_bytes(video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_velocity)) != video_hash ||
        hash_bytes(audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_velocity)) != audio_hash)
        die("cross-block AdaLN fusion warmup changed output bytes");
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        if (candidate_pattern[run])
            unsetenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN");
        else
            setenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        uint64_t current_video = hash_bytes(
            video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
        uint64_t current_audio = hash_bytes(
            audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
        if (run && (current_video != video_hash || current_audio != audio_hash))
            die("cross-block AdaLN fusion changed output bytes");
        video_hash = current_video;
        audio_hash = current_audio;
        if (candidate_pattern[run]) {
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  cross-block AdaLN fused %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  cross-block AdaLN separate %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN");
    double baseline = baseline_seconds / baseline_count;
    double candidate = candidate_seconds / candidate_count;
    printf("DiT cross-block AdaLN AB separate %.4fs, fused %.4fs, "
           "ratio %.4f; hashes video %016llx audio %016llx\n",
           baseline, candidate, candidate / baseline,
           (unsigned long long)video_hash, (unsigned long long)audio_hash);
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
    int sampler_ab = getenv("H3_BENCH_SAMPLER_AB") != NULL;
    int token_reduction_ab =
        getenv("H3_BENCH_TOKEN_REDUCTION_AB") != NULL;
    int cross_adaln_ab = getenv("H3_BENCH_CROSS_ADALN_AB") != NULL;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error)) ||
        !((sampler_ab || token_reduction_ab || cross_adaln_ab)
              ? h3_serving_schedule_build(20, &sigmas)
              : h3_schedule_build(20, &sigmas)))
        die("cannot build benchmark layout");
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/transformer", model_root);
    unsigned active_blocks = 50;
    int reuse_interval = 1;
    const char *layers = getenv("H3_BENCH_LAYERS");
    if (layers && *layers) {
        char *end = NULL;
        long parsed = strtol(layers, &end, 10);
        if (end == layers || *end || parsed < 25 || parsed > 50)
            die("H3_BENCH_LAYERS must be in [25, 50]");
        active_blocks = (unsigned)parsed;
    }
    const char *reuse = getenv("H3_BENCH_REUSE");
    if (reuse && *reuse) {
        char *end = NULL;
        long parsed = strtol(reuse, &end, 10);
        if (end == reuse || *end || parsed < 1 || parsed > 3)
            die("H3_BENCH_REUSE must be in [1, 3]");
        reuse_interval = (int)parsed;
    }
    double load_start = seconds();
    h3_dit *dit = h3_dit_load_t2va(
        weights, "h3_shaders.metal", &text, &layout, &sigmas, active_blocks, 1,
        token_reduction_ab || cross_adaln_ab,
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
        printf("DiT %ux%u/%u-layer load %.3fs before command AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_GRAPH_DATA_AB")) {
        run_graph_data_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before graph-data AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_MPS_COMMAND_AB")) {
        run_mps_command_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before MPS-command AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_NAX_MLP_AB")) {
        run_nax_mlp_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before NAX MLP AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (sampler_ab) {
        run_sampler_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before sampler AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (cross_adaln_ab) {
        run_cross_adaln_ab(dit, video, audio, video_velocity,
                           audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before cross-block AdaLN AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (token_reduction_ab) {
        if (!strcmp(getenv("H3_BENCH_TOKEN_REDUCTION_AB"), "denoise"))
            run_token_reduction_denoise_ab(dit, video, audio,
                                           reuse_interval);
        else
            run_token_reduction_ab(dit, video, audio, video_velocity,
                                   audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before token reduction AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
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
    printf("DiT %ux%u load %.3fs, seven forwards %.3fs, combined %.3fs\n",
           (unsigned)CANVAS_W, (unsigned)CANVAS_H,
           load_seconds, forward_seconds,
           load_seconds + forward_seconds);

    h3_dit_free(dit);
    h3_layout_free(&layout);
    free(text_values); free(video); free(audio);
    free(video_velocity); free(audio_velocity);
    return 0;
}
