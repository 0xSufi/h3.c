#include "h3_audio_vae.h"

#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 32,
    LATENT_DIM = 2048,
    DECODER_DIM = 1024,
    STEREO = 2,
    STAGES = 7,
    RESBLOCKS = 3,
    RESIDUAL_PAIRS = 3,
    FILTER_SIZE = 12,
    SAMPLE_RATE = 32000,
    HOP_LENGTH = 800
};

static const uint32_t upsample_rates[STAGES] = {5, 5, 2, 2, 2, 2, 2};
static const uint32_t upsample_kernels[STAGES] = {9, 9, 4, 4, 4, 4, 4};
static const uint32_t residual_kernels[RESBLOCKS] = {3, 7, 11};
static const uint32_t residual_dilations[RESIDUAL_PAIRS] = {1, 3, 5};

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *vector;
    h3_gpu_tensor *magnitude;
    h3_gpu_tensor *bias;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t padding;
    uint32_t dilation;
    uint32_t stride;
    int transpose;
} audio_conv;

typedef struct {
    h3_gpu_tensor *alpha;
    h3_gpu_tensor *beta;
} audio_activation;

typedef struct {
    audio_activation activations[RESIDUAL_PAIRS * 2];
    audio_conv convs1[RESIDUAL_PAIRS];
    audio_conv convs2[RESIDUAL_PAIRS];
} audio_resblock;

typedef struct {
    audio_conv upsample;
    audio_resblock blocks[RESBLOCKS];
} audio_stage;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_gpu_tensor *upsample_filter;
    h3_gpu_tensor *downsample_filter;
    h3_gpu_tensor *hidden;
    uint32_t length;
} audio_context;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(audio_context *audio, int ok, char *error,
                  size_t error_size, const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(audio->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void free_conv(audio_conv *conv) {
    free_tensor(&conv->weight);
    free_tensor(&conv->vector);
    free_tensor(&conv->magnitude);
    free_tensor(&conv->bias);
    memset(conv, 0, sizeof(*conv));
}

static void free_activation(audio_activation *activation) {
    free_tensor(&activation->alpha);
    free_tensor(&activation->beta);
}

static void free_resblock(audio_resblock *block) {
    for (int index = 0; index < RESIDUAL_PAIRS * 2; index++)
        free_activation(&block->activations[index]);
    for (int index = 0; index < RESIDUAL_PAIRS; index++) {
        free_conv(&block->convs1[index]);
        free_conv(&block->convs2[index]);
    }
}

static void free_stage(audio_stage *stage) {
    free_conv(&stage->upsample);
    for (int index = 0; index < RESBLOCKS; index++)
        free_resblock(&stage->blocks[index]);
}

static void cleanup(audio_context *audio) {
    if (!audio) return;
    free_tensor(&audio->upsample_filter);
    free_tensor(&audio->downsample_filter);
    free_tensor(&audio->hidden);
    h3_weight_store_free(audio->weights);
    h3_gpu_free(audio->gpu);
    memset(audio, 0, sizeof(*audio));
}

static h3_gpu_tensor *load_f32(audio_context *audio, const char *name,
                               int ndim, const uint64_t *shape, char *error,
                               size_t error_size) {
    return h3_weight_load_f32(audio->weights, audio->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(audio_context *audio, const char *name,
                         uint64_t width, char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(audio, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f3(audio_context *audio, const char *name,
                         uint64_t first, uint64_t second, uint64_t third,
                         char *error, size_t error_size) {
    uint64_t shape[] = {first, second, third};
    return load_f32(audio, name, 3, shape, error, error_size);
}

static int load_plain_conv(audio_context *audio, audio_conv *conv,
                           const char *prefix, uint32_t input_channels,
                           uint32_t output_channels, uint32_t kernel,
                           uint32_t padding, uint32_t dilation, int has_bias,
                           char *error, size_t error_size) {
    char name[192];
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->padding = padding;
    conv->dilation = dilation;
    conv->stride = 1;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = f3(audio, name, output_channels, input_channels, kernel,
                      error, error_size);
    if (!conv->weight) return 0;
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        conv->bias = f1(audio, name, output_channels, error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int load_normalized_conv(audio_context *audio, audio_conv *conv,
                                const char *prefix,
                                uint32_t input_channels,
                                uint32_t output_channels, uint32_t kernel,
                                uint32_t padding, uint32_t dilation,
                                uint32_t stride, int transpose, int has_bias,
                                char *error, size_t error_size) {
    char name[192];
    uint32_t outer = transpose ? input_channels : output_channels;
    uint32_t inner_channels = transpose ? output_channels : input_channels;
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->padding = padding;
    conv->dilation = dilation;
    conv->stride = stride;
    conv->transpose = transpose;
    snprintf(name, sizeof(name), "%s.weight_v", prefix);
    conv->vector = f3(audio, name, outer, inner_channels, kernel,
                      error, error_size);
    if (!conv->vector) return 0;
    snprintf(name, sizeof(name), "%s.weight_g", prefix);
    conv->magnitude = f3(audio, name, outer, 1, 1, error, error_size);
    if (!conv->magnitude) return 0;
    size_t elements = (size_t)outer * inner_channels * kernel;
    conv->weight = h3_gpu_tensor_new_f32(audio->gpu, elements);
    if (!conv->weight) {
        fail(error, error_size, "cannot allocate normalized %s weight: %s",
             prefix, h3_gpu_error(audio->gpu));
        return 0;
    }
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        conv->bias = f1(audio, name, output_channels, error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int normalize_conv(audio_context *audio, audio_conv *conv,
                          char *error, size_t error_size) {
    uint32_t outer = conv->transpose ? conv->input_channels :
                                      conv->output_channels;
    uint32_t inner_channels = conv->transpose ? conv->output_channels :
                                               conv->input_channels;
    uint64_t inner = (uint64_t)inner_channels * conv->kernel;
    if (inner > UINT32_MAX) {
        fail(error, error_size, "audio convolution weight is too large");
        return 0;
    }
    return gpu_op(audio, h3_gpu_weight_norm_f32(
        audio->gpu, conv->weight, conv->vector, conv->magnitude, outer,
        (uint32_t)inner), error, error_size, "AudioVAE weight normalization");
}

static void retire_normalization_inputs(audio_conv *conv) {
    free_tensor(&conv->vector);
    free_tensor(&conv->magnitude);
}

static int load_activation(audio_context *audio,
                           audio_activation *activation, const char *prefix,
                           uint32_t channels, char *error, size_t error_size) {
    char name[224];
    snprintf(name, sizeof(name), "%s.act.alpha", prefix);
    activation->alpha = f1(audio, name, channels, error, error_size);
    if (!activation->alpha) return 0;
    snprintf(name, sizeof(name), "%s.act.beta", prefix);
    activation->beta = f1(audio, name, channels, error, error_size);
    return activation->beta != NULL;
}

static int load_filters(audio_context *audio, char *error,
                        size_t error_size) {
    audio->upsample_filter = f3(
        audio, "decoder.activation_post.upsample.filter", 1, 1, FILTER_SIZE,
        error, error_size);
    audio->downsample_filter = f3(
        audio, "decoder.activation_post.downsample.lowpass.filter", 1, 1,
        FILTER_SIZE, error, error_size);
    return audio->upsample_filter && audio->downsample_filter;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "audio VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "audio VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "audio VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "audio VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving audio VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open audio VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid audio VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read audio VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size,
                 "audio VAE latent standard deviation is invalid");
            return 0;
        }
    }
    return ok;
}

static int prepare_input(audio_context *audio, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t elements = (size_t)STEREO * audio->length * LATENT_CHANNELS;
    float *rows = malloc(elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory transposing audio latent");
        return 0;
    }
    size_t destination = 0;
    for (int stereo = 0; stereo < STEREO; stereo++)
        for (uint32_t time = 0; time < audio->length; time++)
            for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                size_t source = ((size_t)channel * STEREO + (size_t)stereo) *
                                audio->length + time;
                rows[destination++] = input[source] * deviation[channel] +
                                      mean[channel];
            }
    h3_gpu_tensor *latent = h3_gpu_tensor_from_f32(audio->gpu, rows, elements);
    free(rows);
    if (!latent) {
        fail(error, error_size, "cannot allocate audio latent: %s",
             h3_gpu_error(audio->gpu));
        return 0;
    }

    audio_conv projection = {0}, pre = {0};
    h3_gpu_tensor *projected = NULL, *hidden = NULL;
    int ok = load_plain_conv(audio, &projection, "dec_in_proj",
                             LATENT_CHANNELS, LATENT_DIM, 1, 0, 1, 1,
                             error, error_size) &&
             load_normalized_conv(audio, &pre, "decoder.conv_pre",
                                  LATENT_DIM, DECODER_DIM, 7, 3, 1, 1, 0, 1,
                                  error, error_size);
    if (!ok) goto done;
    projected = h3_gpu_tensor_new_f32(
        audio->gpu, (size_t)STEREO * audio->length * LATENT_DIM);
    hidden = h3_gpu_tensor_new_f32(
        audio->gpu, (size_t)STEREO * audio->length * DECODER_DIM);
    if (!projected || !hidden) {
        fail(error, error_size, "cannot allocate AudioVAE input activations: %s",
             h3_gpu_error(audio->gpu));
        ok = 0;
        goto done;
    }
    ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE input") &&
         normalize_conv(audio, &pre, error, error_size) &&
         gpu_op(audio, h3_gpu_conv1d_f32(
             audio->gpu, projected, latent, projection.weight, projection.bias,
             STEREO, audio->length, LATENT_CHANNELS, LATENT_DIM, 1, 0, 1),
             error, error_size, "AudioVAE input projection") &&
         gpu_op(audio, h3_gpu_conv1d_f32(
             audio->gpu, hidden, projected, pre.weight, pre.bias, STEREO,
             audio->length, LATENT_DIM, DECODER_DIM, 7, 3, 1), error,
             error_size, "AudioVAE pre convolution") &&
         gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                "submit AudioVAE input");
    if (ok) {
        audio->hidden = hidden;
        hidden = NULL;
    }

done:
    h3_gpu_tensor_free(latent);
    h3_gpu_tensor_free(projected);
    h3_gpu_tensor_free(hidden);
    free_conv(&projection);
    free_conv(&pre);
    return ok;
}

static int load_stage(audio_context *audio, audio_stage *stage, int index,
                      char *error, size_t error_size) {
    uint32_t input_channels = DECODER_DIM >> index;
    uint32_t output_channels = DECODER_DIM >> (index + 1);
    uint32_t rate = upsample_rates[index];
    uint32_t kernel = upsample_kernels[index];
    uint32_t padding = (kernel - rate) / 2;
    char prefix[192];
    snprintf(prefix, sizeof(prefix), "decoder.ups.%d.0", index);
    if (!load_normalized_conv(audio, &stage->upsample, prefix,
                              input_channels, output_channels, kernel, padding,
                              1, rate, 1, 1, error, error_size)) return 0;

    for (int block_index = 0; block_index < RESBLOCKS; block_index++) {
        int global = index * RESBLOCKS + block_index;
        audio_resblock *block = &stage->blocks[block_index];
        uint32_t residual_kernel = residual_kernels[block_index];
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            for (int which = 0; which < 2; which++) {
                int activation = pair * 2 + which;
                snprintf(prefix, sizeof(prefix),
                         "decoder.resblocks.%d.activations.%d", global,
                         activation);
                if (!load_activation(audio, &block->activations[activation],
                                     prefix, output_channels, error,
                                     error_size)) return 0;
            }
            uint32_t dilation = residual_dilations[pair];
            uint32_t dilated_padding =
                dilation * (residual_kernel - 1) / 2;
            snprintf(prefix, sizeof(prefix),
                     "decoder.resblocks.%d.convs1.%d", global, pair);
            if (!load_normalized_conv(
                    audio, &block->convs1[pair], prefix, output_channels,
                    output_channels, residual_kernel, dilated_padding,
                    dilation, 1, 0, 1, error, error_size)) return 0;
            snprintf(prefix, sizeof(prefix),
                     "decoder.resblocks.%d.convs2.%d", global, pair);
            if (!load_normalized_conv(
                    audio, &block->convs2[pair], prefix, output_channels,
                    output_channels, residual_kernel,
                    (residual_kernel - 1) / 2, 1, 1, 0, 1, error,
                    error_size)) return 0;
        }
    }
    return 1;
}

static int normalize_stage(audio_context *audio, audio_stage *stage,
                           char *error, size_t error_size) {
    if (!gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE stage normalization") ||
        !normalize_conv(audio, &stage->upsample, error, error_size)) return 0;
    for (int block = 0; block < RESBLOCKS; block++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++)
            if (!normalize_conv(audio, &stage->blocks[block].convs1[pair],
                                error, error_size) ||
                !normalize_conv(audio, &stage->blocks[block].convs2[pair],
                                error, error_size)) return 0;
    if (!gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                "submit AudioVAE stage normalization")) return 0;
    retire_normalization_inputs(&stage->upsample);
    for (int block = 0; block < RESBLOCKS; block++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            retire_normalization_inputs(&stage->blocks[block].convs1[pair]);
            retire_normalization_inputs(&stage->blocks[block].convs2[pair]);
        }
    return 1;
}

static int run_activation(audio_context *audio, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const audio_activation *activation,
                          uint32_t length, uint32_t channels, char *error,
                          size_t error_size) {
    return gpu_op(audio, h3_gpu_alias_free_snake_f32(
        audio->gpu, output, input, activation->alpha, activation->beta,
        audio->upsample_filter, audio->downsample_filter, STEREO, length,
        channels), error, error_size, "AudioVAE alias-free SnakeBeta");
}

static int run_conv(audio_context *audio, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, const audio_conv *conv,
                    uint32_t length, char *error, size_t error_size) {
    int ok = conv->transpose ? h3_gpu_conv_transpose1d_f32(
        audio->gpu, output, input, conv->weight, conv->bias, STEREO, length,
        conv->input_channels, conv->output_channels, conv->kernel, conv->stride,
        conv->padding) : h3_gpu_conv1d_f32(
        audio->gpu, output, input, conv->weight, conv->bias, STEREO, length,
        conv->input_channels, conv->output_channels, conv->kernel, conv->padding,
        conv->dilation);
    return gpu_op(audio, ok, error, error_size,
                  conv->transpose ? "AudioVAE transposed convolution" :
                                    "AudioVAE residual convolution");
}

static int run_stage(audio_context *audio, audio_stage *stage, int index,
                     char *error, size_t error_size) {
    uint32_t input_length = audio->length;
    uint32_t channels = DECODER_DIM >> (index + 1);
    uint32_t kernel = upsample_kernels[index];
    uint32_t stride = upsample_rates[index];
    uint32_t padding = (kernel - stride) / 2;
    uint64_t expanded = (uint64_t)(input_length - 1) * stride + kernel -
                        2 * padding;
    if (expanded > UINT32_MAX) {
        fail(error, error_size, "AudioVAE stage length overflows");
        return 0;
    }
    uint32_t output_length = (uint32_t)expanded;
    uint64_t elements64 = (uint64_t)STEREO * output_length * channels;
    if (elements64 > UINT32_MAX || elements64 > SIZE_MAX) {
        fail(error, error_size, "AudioVAE stage activation is too large");
        return 0;
    }
    size_t elements = (size_t)elements64;
    h3_gpu_tensor *upsampled = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *sum = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *work = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *branch = h3_gpu_tensor_new_f32(audio->gpu, elements);
    int ok = upsampled && sum && work && activated && branch;
    if (!ok) {
        fail(error, error_size, "cannot allocate AudioVAE stage activations: %s",
             h3_gpu_error(audio->gpu));
        goto done;
    }
    ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE stage") &&
         run_conv(audio, upsampled, audio->hidden, &stage->upsample,
                  input_length, error, error_size);
    for (int block_index = 0; ok && block_index < RESBLOCKS; block_index++) {
        audio_resblock *block = &stage->blocks[block_index];
        h3_gpu_tensor *target = block_index == 0 ? sum : work;
        ok = gpu_op(audio, h3_gpu_copy_f32(audio->gpu, target, 0, upsampled, 0,
                                           elements), error, error_size,
                    "copy AudioVAE residual input");
        for (int pair = 0; ok && pair < RESIDUAL_PAIRS; pair++) {
            ok = run_activation(audio, activated, target,
                                &block->activations[pair * 2], output_length,
                                channels, error, error_size) &&
                 run_conv(audio, branch, activated, &block->convs1[pair],
                          output_length, error, error_size) &&
                 run_activation(audio, activated, branch,
                                &block->activations[pair * 2 + 1],
                                output_length, channels, error, error_size) &&
                 run_conv(audio, branch, activated, &block->convs2[pair],
                          output_length, error, error_size) &&
                 gpu_op(audio, h3_gpu_add_scaled_f32(
                     audio->gpu, target, target, branch, 1.0f, 1.0f,
                     (uint32_t)elements), error, error_size,
                     "AudioVAE residual add");
        }
        if (ok && block_index == 1)
            ok = gpu_op(audio, h3_gpu_add_scaled_f32(
                audio->gpu, sum, sum, work, 1.0f, 1.0f,
                (uint32_t)elements), error, error_size,
                "AudioVAE resblock accumulation");
        if (ok && block_index == 2)
            ok = gpu_op(audio, h3_gpu_add_scaled_f32(
                audio->gpu, sum, sum, work, 1.0f / 3.0f, 1.0f / 3.0f,
                (uint32_t)elements), error, error_size,
                "AudioVAE resblock average");
    }
    if (ok) ok = gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                        "submit AudioVAE stage");
    if (ok) {
        free_tensor(&audio->hidden);
        audio->hidden = sum;
        sum = NULL;
        audio->length = output_length;
    }

done:
    h3_gpu_tensor_free(upsampled);
    h3_gpu_tensor_free(sum);
    h3_gpu_tensor_free(work);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(branch);
    return ok;
}

static int decode_output(audio_context *audio, h3_audio_waveform *output,
                         char *error, size_t error_size) {
    audio_activation activation = {0};
    audio_conv post = {0};
    h3_gpu_tensor *activated = NULL, *waveform = NULL;
    size_t hidden_elements = (size_t)STEREO * audio->length * 8;
    size_t waveform_elements = (size_t)STEREO * audio->length;
    int ok = load_activation(audio, &activation, "decoder.activation_post", 8,
                             error, error_size) &&
             load_normalized_conv(audio, &post, "decoder.conv_post", 8, 1, 7,
                                  3, 1, 1, 0, 0, error, error_size);
    if (!ok) goto done;
    activated = h3_gpu_tensor_new_f32(audio->gpu, hidden_elements);
    waveform = h3_gpu_tensor_new_f32(audio->gpu, waveform_elements);
    if (!activated || !waveform || waveform_elements > UINT32_MAX) {
        fail(error, error_size, "cannot allocate AudioVAE output: %s",
             h3_gpu_error(audio->gpu));
        ok = 0;
        goto done;
    }
    ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE output") &&
         normalize_conv(audio, &post, error, error_size) &&
         run_activation(audio, activated, audio->hidden, &activation,
                        audio->length, 8, error, error_size) &&
         run_conv(audio, waveform, activated, &post, audio->length,
                  error, error_size) &&
         gpu_op(audio, h3_gpu_clip_f32(audio->gpu, waveform, waveform,
                                       (uint32_t)waveform_elements,
                                       -1.0f, 1.0f), error, error_size,
                "AudioVAE output clip") &&
         gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                "submit AudioVAE output");
    if (!ok) goto done;
    output->pcm = malloc(waveform_elements * sizeof(*output->pcm));
    if (!output->pcm || !h3_gpu_tensor_read_f32(
            waveform, output->pcm, waveform_elements)) {
        fail(error, error_size, "cannot read AudioVAE waveform");
        free(output->pcm);
        output->pcm = NULL;
        ok = 0;
        goto done;
    }
    output->channels = STEREO;
    output->samples = (int)audio->length;
    output->sample_rate = SAMPLE_RATE;

done:
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(waveform);
    free_activation(&activation);
    free_conv(&post);
    return ok;
}

int h3_audio_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_length,
                        h3_audio_vae_progress progress, void *progress_opaque,
                        h3_audio_waveform *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !*weight_directory || !shader_source_path ||
        !*shader_source_path || !normalized_latent || !output ||
        latent_length < 1 || latent_length > INT32_MAX / HOP_LENGTH) {
        fail(error, error_size, "invalid AudioVAE decode arguments");
        return 0;
    }
    audio_context audio = {0};
    audio.length = (uint32_t)latent_length;
    float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, mean, deviation,
                                   error, error_size)) return 0;
    audio.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!audio.gpu) return 0;
    audio.weights = h3_weight_store_open(weight_directory, error, error_size);
    int ok = audio.weights && load_filters(&audio, error, error_size) &&
             prepare_input(&audio, normalized_latent, mean, deviation,
                           error, error_size);
    for (int index = 0; ok && index < STAGES; index++) {
        audio_stage stage = {0};
        ok = load_stage(&audio, &stage, index, error, error_size) &&
             normalize_stage(&audio, &stage, error, error_size) &&
             run_stage(&audio, &stage, index, error, error_size);
        free_stage(&stage);
        if (ok && progress) progress(index + 1, STAGES, progress_opaque);
    }
    if (ok) ok = audio.length == (uint32_t)latent_length * HOP_LENGTH;
    if (!ok && error && error_size && !error[0])
        fail(error, error_size, "AudioVAE produced the wrong sample count");
    if (ok) ok = decode_output(&audio, output, error, error_size);
    if (ok && !h3_gpu_get_stats(audio.gpu, &output->gpu_stats)) {
        fail(error, error_size, "cannot read AudioVAE GPU statistics");
        ok = 0;
    }
    cleanup(&audio);
    if (!ok) h3_audio_waveform_free(output);
    return ok;
}

void h3_audio_waveform_free(h3_audio_waveform *waveform) {
    if (!waveform) return;
    free(waveform->pcm);
    memset(waveform, 0, sizeof(*waveform));
}
