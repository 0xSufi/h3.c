#include "h3_internal.h"
#include "h3_audio_vae.h"
#include "h3_host.h"
#include "h3_dit.h"
#include "h3_ffmpeg.h"
#include "h3_metal.h"
#include "h3_multimodal.h"
#include "h3_safetensors.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_video_encoder.h"
#include "h3_video_vae.h"
#include "h3_vision_encoder.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char h3_global_error[512];

void h3_set_error(h3_ctx *ctx, const char *format, ...) {
    char *destination = ctx ? ctx->error : h3_global_error;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(destination, 512, format, arguments);
    va_end(arguments);
}

static int h3_is_file(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

static char *h3_path(const char *root, const char *relative) {
    size_t size = strlen(root) + strlen(relative) + 2;
    char *result = malloc(size);
    if (result) snprintf(result, size, "%s/%s", root, relative);
    return result;
}

static int h3_require_file(h3_ctx *ctx, const char *relative) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving model path");
        return 0;
    }
    int exists = h3_is_file(path);
    if (!exists) h3_set_error(ctx, "missing required model file: %s", path);
    free(path);
    return exists;
}

static int h3_inventory(h3_ctx *ctx, const char *relative,
                        h3_component_info *info) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving component path");
        return 0;
    }
    char detail[384];
    int ok = h3_st_inventory_dir(path, info, detail, sizeof(detail));
    if (!ok) h3_set_error(ctx, "%s", detail);
    free(path);
    return ok;
}

h3_ctx *h3_load_dir(const char *model_dir) {
    h3_global_error[0] = '\0';
    if (!model_dir || !*model_dir) {
        h3_set_error(NULL, "model directory is required");
        return NULL;
    }
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        h3_set_error(NULL, "out of memory creating H3 context");
        return NULL;
    }
    ctx->model_dir = strdup(model_dir);
    if (!ctx->model_dir) {
        h3_set_error(NULL, "out of memory copying model path");
        free(ctx);
        return NULL;
    }
    if (!h3_require_file(ctx, "FL2VA/transformer/config.json") ||
        !h3_require_file(ctx, "FL2VA/tokenizer/tokenizer.json") ||
        !h3_inventory(ctx, "FL2VA/text_encoder", &ctx->model.text_encoder) ||
        !h3_inventory(ctx, "FL2VA/transformer", &ctx->model.fl2va_transformer) ||
        !h3_inventory(ctx, "FL2VA/video_vae/source", &ctx->model.video_vae) ||
        !h3_inventory(ctx, "FL2VA/audio_vae", &ctx->model.audio_vae)) {
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    /* Ref2VA is selected only by ordered-reference requests. Keep prompt-only
     * FL2VA usable while that optional 62 GiB checkpoint is not installed. */
    char *ref_index = h3_path(
        ctx->model_dir, "Ref2VA/transformer/model.safetensors.index.json");
    if (!ref_index) {
        h3_set_error(ctx, "out of memory resolving optional Ref2VA path");
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    int has_ref2va = h3_is_file(ref_index);
    free(ref_index);
    if (has_ref2va && !h3_inventory(
            ctx, "Ref2VA/transformer", &ctx->model.ref2va_transformer)) {
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    char metal_error[256];
    if (!h3_metal_probe(&ctx->device, metal_error, sizeof(metal_error))) {
        h3_set_error(ctx, "%s", metal_error);
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    return ctx;
}

void h3_free(h3_ctx *ctx) {
    if (!ctx) return;
    free(ctx->model_dir);
    free(ctx);
}

const char *h3_last_error(const h3_ctx *ctx) {
    return ctx ? ctx->error : h3_global_error;
}

const h3_device_info *h3_device(const h3_ctx *ctx) {
    return ctx ? &ctx->device : NULL;
}

const h3_model_info *h3_model(const h3_ctx *ctx) {
    return ctx ? &ctx->model : NULL;
}

static int h3_valid_params(h3_ctx *ctx, const h3_params *params) {
    if (!params) {
        h3_set_error(ctx, "generation parameters are required");
        return 0;
    }
    if (params->width < 32 || params->height < 32 ||
        params->width % H3_CANVAS_MULTIPLE ||
        params->height % H3_CANVAS_MULTIPLE) {
        h3_set_error(ctx, "width and height must be multiples of 32 and at least 32");
        return 0;
    }
    if ((int64_t)params->width * params->height > H3_MAX_PIXELS) {
        h3_set_error(ctx, "canvas exceeds the released 768*1344 pixel limit");
        return 0;
    }
    if ((params->render_width == 0) != (params->render_height == 0)) {
        h3_set_error(ctx, "render width and height must be set together");
        return 0;
    }
    if (params->render_width) {
        if (params->render_width < 32 || params->render_height < 32 ||
            params->render_width % H3_CANVAS_MULTIPLE ||
            params->render_height % H3_CANVAS_MULTIPLE ||
            params->render_width > params->width ||
            params->render_height > params->height ||
            (int64_t)params->render_width * params->height !=
                (int64_t)params->render_height * params->width) {
            h3_set_error(ctx,
                "internal render canvas must be same-aspect multiples of 32 "
                "no larger than the output canvas");
            return 0;
        }
    }
    if (params->frames < 5 || h3_align_frame_count(params->frames) > 362) {
        h3_set_error(ctx, "frames must align within the released 5..362 range");
        return 0;
    }
    if (params->steps < 2 || params->steps > H3_MAX_STEPS) {
        h3_set_error(ctx, "sigma points must be in [2, 1000]");
        return 0;
    }
    if (params->denoise_reuse < 1 || params->denoise_reuse > 3) {
        h3_set_error(ctx, "denoise reuse must be in [1, 3]");
        return 0;
    }
    if (params->dit_layers < H3_MIN_DIT_LAYERS ||
        params->dit_layers > H3_DEFAULT_DIT_LAYERS) {
        h3_set_error(ctx, "DiT layers must be in [35, 50]");
        return 0;
    }
    if (params->mlp_layers &&
        (params->mlp_layers < H3_MIN_DIT_LAYERS ||
         params->mlp_layers > params->dit_layers)) {
        h3_set_error(ctx, "MLP layers must be in [35, DiT layers]");
        return 0;
    }
    if (params->core_reuse < 1 || params->core_reuse > 6) {
        h3_set_error(ctx, "core reuse must be in [1, 6]");
        return 0;
    }
    if (params->token_reduction != 0 && params->token_reduction != 1) {
        h3_set_error(ctx, "token reduction must be zero or one");
        return 0;
    }
    if (params->core_reuse > 1 && params->denoise_reuse > 1) {
        h3_set_error(ctx, "core reuse and denoiser reuse cannot be combined");
        return 0;
    }
    if (params->reference_count && !params->references) {
        h3_set_error(ctx, "reference_count is nonzero but references is NULL");
        return 0;
    }
    if (params->reference_count > 12) {
        h3_set_error(ctx, "Ref2VA supports at most 12 references");
        return 0;
    }
    if (params->reference_image_size != H3_REFERENCE_IMAGE_MATCH &&
        params->reference_image_size != H3_REFERENCE_IMAGE_MAX) {
        h3_set_error(ctx, "unknown reference image sizing policy");
        return 0;
    }
    if (params->reference_count && (params->first_frame || params->last_frame)) {
        h3_set_error(ctx, "full references cannot be combined with frame anchors");
        return 0;
    }
    size_t images = 0, videos = 0, audio_inputs = 0, visual = 0;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (!reference->path || !*reference->path) {
            h3_set_error(ctx, "reference %zu has no input path", index + 1);
            return 0;
        }
        switch (reference->kind) {
        case H3_REFERENCE_IMAGE:
            images++; visual++;
            break;
        case H3_REFERENCE_VIDEO:
            videos++; visual++;
            if (reference->include_embedded_audio) audio_inputs++;
            break;
        case H3_REFERENCE_AUDIO:
            audio_inputs++;
            break;
        case H3_REFERENCE_VIDEO_AUDIO:
            videos++; visual++; audio_inputs++;
            if (!reference->audio_path || !*reference->audio_path) {
                h3_set_error(ctx,
                    "video+audio reference %zu has no soundtrack path",
                    index + 1);
                return 0;
            }
            break;
        default:
            h3_set_error(ctx, "reference %zu has an unknown type", index + 1);
            return 0;
        }
    }
    if (images > 9 || videos > 3 || audio_inputs > 3) {
        h3_set_error(ctx,
            "Ref2VA limits are 9 images, 3 videos, and 3 audio inputs");
        return 0;
    }
    if (params->reference_count && !visual) {
        h3_set_error(ctx, "reference audio requires an image or video reference");
        return 0;
    }
    return 1;
}

typedef struct {
    h3_ctx *ctx;
    const h3_params *params;
    int cancelled;
} h3_generation_progress;

static void h3_progress_emit(h3_generation_progress *state, const char *phase,
                             int completed, int total) {
    if (!state || state->cancelled || !state->params->on_progress) return;
    if (state->params->on_progress(phase, completed, total,
                                   state->params->callback_opaque)) {
        state->cancelled = 1;
        h3_set_error(state->ctx, "generation cancelled during %s", phase);
    }
}

static void h3_text_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "text encoder", completed, total);
}

static void h3_dit_progress_bridge(const char *phase, int completed, int total,
                                   void *opaque) {
    h3_progress_emit(opaque, phase, completed, total);
}

static void h3_vae_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "video VAE load", completed, total);
}

static void h3_audio_vae_progress_bridge(int completed, int total,
                                         void *opaque) {
    h3_progress_emit(opaque, "audio VAE", completed, total);
}

static void h3_audio_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    h3_progress_emit(opaque, "audio VAE encoder", completed, total);
}

static void h3_video_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    h3_progress_emit(opaque, "video VAE encoder", completed, total);
}

static void h3_vision_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "Qwen vision", completed, total);
}

/* Qwen consumes reference video as time-major two-frame blocks, while the
 * visual VAE and media boundary retain channel-major [3,T,H,W]. */
static float *h3_extract_vision_pair(const float *pixels, int frames,
                                     int height, int width,
                                     int first, int second) {
    if (!pixels || frames < 1 || height < 1 || width < 1 || first < 0 ||
        second < 0 || first >= frames || second >= frames) return NULL;
    size_t area = (size_t)height * (size_t)width;
    if (area > SIZE_MAX / 6 || 6 * area > SIZE_MAX / sizeof(float)) return NULL;
    float *pair = malloc(6 * area * sizeof(*pair));
    if (!pair) return NULL;
    const int times[2] = {first, second};
    for (int time = 0; time < 2; time++)
        for (int channel = 0; channel < 3; channel++) {
            size_t source = ((size_t)channel * (size_t)frames +
                             (size_t)times[time]) * area;
            size_t destination = ((size_t)time * 3 +
                                  (size_t)channel) * area;
            memcpy(pair + destination, pixels + source,
                   area * sizeof(*pair));
        }
    return pair;
}

h3_result *h3_generate(h3_ctx *ctx, const char *prompt,
                       const h3_params *params) {
    if (!ctx) return NULL;
    ctx->error[0] = '\0';
    if (!prompt || !*prompt) {
        h3_set_error(ctx, "prompt must not be empty");
        return NULL;
    }
    if (!h3_valid_params(ctx, params)) return NULL;
    int render_width = params->render_width ? params->render_width :
                                               params->width;
    int render_height = params->render_height ? params->render_height :
                                                 params->height;
    if (h3_align_frame_count(params->frames) < 22) {
        h3_set_error(ctx,
            "generation requires at least one trained 22-frame decoder chunk");
        return NULL;
    }
    int ref2va = params->reference_count != 0;
    if (ref2va && !ctx->model.ref2va_transformer.files) {
        h3_set_error(ctx, "ordered references require the Ref2VA checkpoint");
        return NULL;
    }
    h3_generation_progress progress = {ctx, params, 0};
    h3_tokenizer *tokenizer = NULL;
    uint32_t *ids = NULL;
    size_t token_count = 0;
    size_t visual_capacity = ref2va ? params->reference_count :
        (size_t)(params->first_frame != NULL) +
        (size_t)(params->last_frame != NULL);
    size_t visual_count = 0;
    float **condition_pixels = NULL;
    int *condition_widths = NULL;
    int *condition_heights = NULL;
    int *condition_frames = NULL;
    size_t *visual_reference_indices = NULL;
    size_t *reference_visual_indices = NULL;
    h3_vision_output *vision_outputs = NULL;
    size_t vision_output_count = 0;
    h3_reference_presentation *presentations = NULL;
    double **presentation_timestamps = NULL;
    h3_layout_ref *layout_references = NULL;
    int keyframes[2] = {0, 0};
    size_t keyframe_count = 0;
    float *condition_video_rows = NULL;
    size_t condition_video_elements = 0;
    float *condition_audio_rows = NULL;
    size_t condition_audio_elements = 0;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3_layout layout;
    memset(&layout, 0, sizeof(layout));
    h3_dit *dit = NULL;
    float *video = NULL, *audio = NULL;
    h3_video_frames frames;
    memset(&frames, 0, sizeof(frames));
    h3_audio_waveform waveform;
    memset(&waveform, 0, sizeof(waveform));
    uint8_t *rgb8 = NULL;
    h3_result *result = NULL;
    char *tokenizer_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/tokenizer/tokenizer.json" : "FL2VA/tokenizer/tokenizer.json");
    char *text_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/text_encoder" : "FL2VA/text_encoder");
    char *dit_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/transformer" : "FL2VA/transformer");
    char *vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/video_vae/source" : "FL2VA/video_vae/source");
    char *audio_vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/audio_vae" : "FL2VA/audio_vae");
    if (!tokenizer_path || !text_path || !dit_path || !vae_path ||
        !audio_vae_path) {
        h3_set_error(ctx, "out of memory resolving generation model paths");
        goto cleanup;
    }
    if (visual_capacity) {
        condition_pixels = calloc(visual_capacity, sizeof(*condition_pixels));
        condition_widths = calloc(visual_capacity, sizeof(*condition_widths));
        condition_heights = calloc(visual_capacity, sizeof(*condition_heights));
        condition_frames = calloc(visual_capacity, sizeof(*condition_frames));
        visual_reference_indices = calloc(
            visual_capacity, sizeof(*visual_reference_indices));
        reference_visual_indices = malloc(
            params->reference_count * sizeof(*reference_visual_indices));
        if (reference_visual_indices)
            for (size_t index = 0; index < params->reference_count; index++)
                reference_visual_indices[index] = SIZE_MAX;
        if (ref2va) {
            layout_references = calloc(visual_capacity,
                                       sizeof(*layout_references));
            presentations = calloc(params->reference_count,
                                   sizeof(*presentations));
            presentation_timestamps = calloc(
                params->reference_count, sizeof(*presentation_timestamps));
        }
        if (!condition_pixels || !condition_widths || !condition_heights ||
            !condition_frames || !visual_reference_indices ||
            !reference_visual_indices ||
            (ref2va && (!layout_references || !presentations ||
                        !presentation_timestamps))) {
            h3_set_error(ctx, "out of memory preparing visual references");
            goto cleanup;
        }
    }
    char detail[512];
    h3_progress_emit(&progress, "tokenizer", 0, 1);
    tokenizer = h3_tokenizer_load(tokenizer_path, detail, sizeof(detail));
    if (!tokenizer) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_progress_emit(&progress, "tokenizer", 1, 1);
    if (progress.cancelled) goto cleanup;
    h3_temporal_shape temporal = h3_temporal(params->frames);
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);

    if (ref2va) {
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            if (reference->kind == H3_REFERENCE_AUDIO) {
                presentations[index].kind = H3_PRESENTATION_AUDIO;
                layout_references[index] = (h3_layout_ref){
                    H3_LAYOUT_REF_AUDIO, 0, 0, 0, 0};
                continue;
            }
            int source_width, source_height, media_width, media_height;
            if (!h3_ffprobe_visual_size(reference->path,
                                        &source_width, &source_height,
                                        detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (reference->kind == H3_REFERENCE_IMAGE) {
                if (!h3_reference_image_canvas(
                        source_width, source_height,
                        render_width, render_height,
                        params->reference_image_size == H3_REFERENCE_IMAGE_MAX ?
                        2048 : 0, &media_width, &media_height)) {
                    h3_set_error(ctx,
                        "cannot resolve reference image %zu canvas", index + 1);
                    goto cleanup;
                }
                if (!h3_ffmpeg_read_image_f32(
                        reference->path, media_width, media_height,
                        H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                        detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                condition_frames[visual_count] = 1;
                presentations[index].kind = H3_PRESENTATION_IMAGE;
                presentations[index].vision_count = 1;
                vision_output_count++;
            } else {
                if (!h3_reference_video_canvas(
                        source_width, source_height,
                        &media_width, &media_height)) {
                    h3_set_error(ctx,
                        "cannot resolve reference video %zu canvas", index + 1);
                    goto cleanup;
                }
                if (!h3_ffmpeg_read_video_f32(
                        reference->path, media_width, media_height,
                        temporal.frame_count, &condition_pixels[visual_count],
                        &condition_frames[visual_count],
                        detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                size_t samples = ((size_t)condition_frames[visual_count] + 11) / 12;
                size_t blocks = (samples + 1) / 2;
                if (vision_output_count > SIZE_MAX - blocks) {
                    h3_set_error(ctx, "reference video vision count overflows");
                    goto cleanup;
                }
                presentation_timestamps[index] = malloc(
                    blocks * sizeof(*presentation_timestamps[index]));
                if (!presentation_timestamps[index]) {
                    h3_set_error(ctx,
                        "out of memory allocating reference video timestamps");
                    goto cleanup;
                }
                for (size_t block = 0; block < blocks; block++) {
                    size_t first = 2 * block;
                    size_t second = first + 1 < samples ? first + 1 : first;
                    presentation_timestamps[index][block] =
                        ((double)first + (double)second) / 4.0;
                }
                presentations[index].kind = H3_PRESENTATION_VIDEO;
                presentations[index].vision_count = blocks;
                presentations[index].timestamps = presentation_timestamps[index];
                vision_output_count += blocks;
            }
            condition_widths[visual_count] = media_width;
            condition_heights[visual_count] = media_height;
            visual_reference_indices[visual_count] = index;
            reference_visual_indices[index] = visual_count;
            int ref_latent_w, ref_latent_h;
            h3_latent_canvas(media_width, media_height,
                             &ref_latent_w, &ref_latent_h);
            layout_references[index] = (h3_layout_ref){
                reference->kind == H3_REFERENCE_IMAGE ?
                    H3_LAYOUT_REF_IMAGE : H3_LAYOUT_REF_VIDEO,
                h3_video_encoder_latent_t(condition_frames[visual_count]),
                ref_latent_h, ref_latent_w, 0};
            visual_count++;
        }

        size_t total_audio_samples = 0;
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            const char *audio_path = NULL;
            int truncate = 0;
            int max_samples = 32000 * 15;
            if (reference->kind == H3_REFERENCE_AUDIO) {
                audio_path = reference->path;
            } else if (reference->kind == H3_REFERENCE_VIDEO_AUDIO) {
                audio_path = reference->audio_path;
                truncate = 1;
            } else if (reference->kind == H3_REFERENCE_VIDEO &&
                       reference->include_embedded_audio) {
                audio_path = reference->path;
                truncate = 1;
            }
            if (!audio_path) continue;
            if (truncate) {
                size_t visual = reference_visual_indices[index];
                if (visual == SIZE_MAX) {
                    h3_set_error(ctx,
                        "video soundtrack %zu has no decoded video", index + 1);
                    goto cleanup;
                }
                max_samples = (int)llround(
                    (double)condition_frames[visual] * 32000.0 / H3_FPS);
                if (max_samples < 64000) {
                    h3_set_error(ctx,
                        "video soundtrack %zu requires at least 2 seconds; "
                        "request at least 56 output frames", index + 1);
                    goto cleanup;
                }
            }
            float *pcm = NULL;
            int samples = 0;
            if (!h3_ffmpeg_read_audio_f32(
                    audio_path, max_samples, truncate, &pcm, &samples,
                    detail, sizeof(detail))) {
                free(pcm);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if ((size_t)samples > (size_t)32000 * 15 - total_audio_samples) {
                free(pcm);
                h3_set_error(ctx,
                    "ordered reference audio exceeds 15 seconds in total");
                goto cleanup;
            }
            h3_audio_latent latent;
            memset(&latent, 0, sizeof(latent));
            if (!h3_audio_vae_encode(
                    audio_vae_path, "h3_shaders.metal", pcm, samples,
                    h3_audio_encoder_progress_bridge, &progress, &latent,
                    detail, sizeof(detail))) {
                free(pcm);
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            free(pcm);
            if (latent.channels != 32 || latent.stereo != 2 ||
                latent.length < 1 ||
                (size_t)latent.length > SIZE_MAX / 2 / 32) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "reference audio encoder produced invalid geometry");
                goto cleanup;
            }
            size_t elements = (size_t)latent.length * 2 * 32;
            if (condition_audio_elements > SIZE_MAX - elements ||
                condition_audio_elements + elements >
                    SIZE_MAX / sizeof(*condition_audio_rows)) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "reference audio row count overflows");
                goto cleanup;
            }
            float *grown = realloc(
                condition_audio_rows,
                (condition_audio_elements + elements) * sizeof(*grown));
            if (!grown) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "out of memory packing reference audio conditions");
                goto cleanup;
            }
            condition_audio_rows = grown;
            float *rows = condition_audio_rows + condition_audio_elements;
            for (int stereo = 0; stereo < 2; stereo++)
                for (int time = 0; time < latent.length; time++)
                    for (int channel = 0; channel < 32; channel++) {
                        size_t source = ((size_t)channel * 2 +
                                         (size_t)stereo) * latent.length +
                                        (size_t)time;
                        size_t destination = ((size_t)stereo * latent.length +
                                              (size_t)time) * 32 +
                                             (size_t)channel;
                        rows[destination] = latent.values[source];
                    }
            h3_rng condition_rng;
            h3_rng_seed(&condition_rng, params->seed + 1);
            for (size_t element = 0; element < elements; element++)
                rows[element] = 0.999f * rows[element] +
                                0.001f * h3_rng_normal(&condition_rng);
            condition_audio_elements += elements;
            total_audio_samples += (size_t)samples;
            layout_references[index].audio_t = latent.length;
            presentations[index].has_audio = 1;
            h3_audio_latent_free(&latent);
            if (progress.cancelled) goto cleanup;
        }
    } else {
        if (params->first_frame) {
            keyframes[keyframe_count++] = 0;
            if (!h3_ffmpeg_read_image_f32(
                    params->first_frame, render_width, render_height,
                    H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = render_width;
            condition_heights[visual_count] = render_height;
            condition_frames[visual_count] = 1;
            visual_count++;
        }
        if (params->last_frame) {
            keyframes[keyframe_count++] = temporal.frame_count - 1;
            if (!h3_ffmpeg_read_image_f32(
                    params->last_frame, render_width, render_height,
                    H3_IMAGE_FIT_COVER, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = render_width;
            condition_heights[visual_count] = render_height;
            condition_frames[visual_count] = 1;
            visual_count++;
        }
        vision_output_count = visual_count;
    }
    if (vision_output_count) {
        vision_outputs = calloc(vision_output_count, sizeof(*vision_outputs));
        if (!vision_outputs) {
            h3_set_error(ctx, "out of memory allocating Qwen vision outputs");
            goto cleanup;
        }
    }

    if (visual_count) {
        for (size_t image = 0; image < visual_count; image++) {
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            int image_latent_t = h3_video_encoder_latent_t(
                condition_frames[image]);
            size_t rows = (size_t)image_latent_t *
                          (size_t)image_latent_h * (size_t)image_latent_w / 4;
            if (rows > SIZE_MAX / 96 ||
                condition_video_elements > SIZE_MAX - rows * 96) {
                h3_set_error(ctx, "condition row count overflows");
                goto cleanup;
            }
            condition_video_elements += rows * 96;
        }
        if (condition_video_elements > SIZE_MAX /
                                       sizeof(*condition_video_rows)) {
            h3_set_error(ctx, "condition storage size overflows");
            goto cleanup;
        }
        condition_video_rows = malloc(condition_video_elements *
                                      sizeof(*condition_video_rows));
        if (!condition_video_rows) {
            h3_set_error(ctx, "out of memory allocating visual condition rows");
            goto cleanup;
        }
        size_t condition_offset = 0;
        for (size_t image = 0; image < visual_count; image++) {
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            int image_latent_t = h3_video_encoder_latent_t(
                condition_frames[image]);
            size_t row_elements = (size_t)image_latent_t *
                                  (size_t)image_latent_h *
                                  (size_t)image_latent_w / 4 * 96;
            h3_video_latent latent;
            memset(&latent, 0, sizeof(latent));
            if (!h3_video_vae_encode(
                    vae_path, "h3_shaders.metal", condition_pixels[image],
                    condition_frames[image], condition_heights[image],
                    condition_widths[image],
                    h3_video_encoder_progress_bridge, &progress,
                    &latent, detail, sizeof(detail))) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (latent.time != image_latent_t ||
                latent.height != image_latent_h ||
                latent.width != image_latent_w) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx,
                    "visual condition VAE produced unexpected latent geometry");
                goto cleanup;
            }
            float *rows = condition_video_rows + condition_offset;
            if (!h3_dit_patchify_video(
                    latent.values, 24, image_latent_t,
                    image_latent_h, image_latent_w,
                    rows, row_elements)) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "cannot patchify visual condition latent");
                goto cleanup;
            }
            h3_video_latent_free(&latent);
            h3_rng condition_rng;
            h3_rng_seed(&condition_rng, params->seed);
            float *noise = malloc(row_elements * sizeof(*noise));
            if (!noise) {
                h3_set_error(ctx, "out of memory augmenting visual condition");
                goto cleanup;
            }
            h3_rng_fill_normal(&condition_rng, noise, row_elements);
            for (size_t element = 0; element < row_elements; element++)
                rows[element] = 0.999f * rows[element] + 0.001f * noise[element];
            free(noise);
            condition_offset += row_elements;
            if (progress.cancelled) goto cleanup;
        }
        if (condition_offset != condition_video_elements) {
            h3_set_error(ctx, "visual condition packing size mismatch");
            goto cleanup;
        }
        size_t vision_cursor = 0;
        for (size_t image = 0; image < visual_count; image++) {
            size_t reference_index = visual_reference_indices[image];
            if (!ref2va ||
                params->references[reference_index].kind == H3_REFERENCE_IMAGE) {
                if (!h3_vision_encode_bf16(
                        text_path, "h3_shaders.metal", condition_pixels[image],
                        1, condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                if (ref2va)
                    presentations[reference_index].vision =
                        &vision_outputs[vision_cursor];
                vision_cursor++;
            } else {
                size_t blocks = presentations[reference_index].vision_count;
                presentations[reference_index].vision =
                    &vision_outputs[vision_cursor];
                size_t samples = ((size_t)condition_frames[image] + 11) / 12;
                for (size_t block = 0; block < blocks; block++) {
                    size_t first_sample = 2 * block;
                    size_t second_sample = first_sample + 1 < samples ?
                                           first_sample + 1 : first_sample;
                    int first = (int)(first_sample * 12);
                    int second = (int)(second_sample * 12);
                    float *pair = h3_extract_vision_pair(
                        condition_pixels[image], condition_frames[image],
                        condition_heights[image], condition_widths[image],
                        first, second);
                    if (!pair) {
                        h3_set_error(ctx,
                            "out of memory extracting Qwen video pair");
                        goto cleanup;
                    }
                    int ok = h3_vision_encode_bf16(
                        text_path, "h3_shaders.metal", pair, 2,
                        condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail));
                    free(pair);
                    if (!ok) {
                        h3_set_error(ctx, "%s", detail);
                        goto cleanup;
                    }
                    vision_cursor++;
                    if (progress.cancelled) goto cleanup;
                }
            }
            free(condition_pixels[image]);
            condition_pixels[image] = NULL;
            if (progress.cancelled) goto cleanup;
        }
        if (vision_cursor != vision_output_count) {
            h3_set_error(ctx, "Qwen reference vision count mismatch");
            goto cleanup;
        }
        h3_progress_emit(&progress, "text encoder", 0, 50);
        int text_ok = ref2va ? h3_multimodal_encode_ref2va_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt,
                presentations, params->reference_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail)) :
            h3_multimodal_encode_fl2va_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt,
                vision_outputs, visual_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail));
        if (!text_ok) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        for (size_t image = 0; image < vision_output_count; image++)
            h3_vision_output_free(&vision_outputs[image]);
    } else {
        if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,
                                 detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_progress_emit(&progress, "text encoder", 0, 50);
        if (!h3_text_encode_bf16(
                text_path, "h3_shaders.metal", ids, token_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    if (progress.cancelled) goto cleanup;

    h3_layout_spec spec = {(int)text.tokens, temporal.video_t, latent_h,
                           latent_w, temporal.audio_t, temporal.frame_count,
                           keyframes, keyframe_count,
                           layout_references,
                           ref2va ? params->reference_count : 0};
    if (!h3_layout_build(&spec, &layout, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(params->steps, &sigmas)) {
        h3_set_error(ctx, "cannot construct the requested sigma schedule");
        goto cleanup;
    }
    if (visual_count) {
        dit = h3_dit_load_conditioned(
            dit_path, "h3_shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers,
            (unsigned)(params->mlp_layers ? params->mlp_layers :
                       params->dit_layers),
            (unsigned)params->core_reuse,
            params->token_reduction,
            condition_video_rows, condition_video_elements,
            condition_audio_rows, condition_audio_elements,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    } else {
        dit = h3_dit_load_t2va(
            dit_path, "h3_shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers,
            (unsigned)(params->mlp_layers ? params->mlp_layers :
                       params->dit_layers),
            (unsigned)params->core_reuse,
            params->token_reduction,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    }
    if (!dit) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_text_embedding_free(&text);
    free(condition_video_rows);
    condition_video_rows = NULL;
    free(condition_audio_rows);
    condition_audio_rows = NULL;
    if (progress.cancelled) goto cleanup;
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    video = malloc(video_count * sizeof(*video));
    audio = malloc(audio_count * sizeof(*audio));
    if (!video || !audio) {
        h3_set_error(ctx, "out of memory allocating joint H3 noise");
        goto cleanup;
    }
    /* The released server initializes each modality from a separate generator
     * carrying the same requested seed. */
    h3_rng video_rng, audio_rng;
    h3_rng_seed(&video_rng, params->seed);
    h3_rng_seed(&audio_rng, params->seed);
    h3_rng_fill_normal(&video_rng, video, video_count);
    h3_rng_fill_normal(&audio_rng, audio, audio_count);
    if (!h3_dit_denoise_euler(
            dit, video, audio, params->denoise_reuse,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_dit_free(dit);
    dit = NULL;
    if (progress.cancelled) goto cleanup;
    h3_progress_emit(&progress, "audio VAE", 0, 7);
    if (!h3_audio_vae_decode(audio_vae_path, "h3_shaders.metal", audio,
                             temporal.audio_t, h3_audio_vae_progress_bridge,
                             &progress, &waveform, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    free(audio);
    audio = NULL;
    if (progress.cancelled) goto cleanup;
    if (!h3_video_vae_decode(vae_path, "h3_shaders.metal", video,
                             temporal.video_t, latent_h, latent_w,
                             h3_vae_progress_bridge, &progress, &frames,
                             detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    free(video);
    video = NULL;
    if (progress.cancelled) goto cleanup;
    size_t rgb_count = (size_t)frames.frames * (size_t)frames.height *
                       (size_t)frames.width * 3;
    rgb8 = malloc(rgb_count);
    if (!rgb8) {
        h3_set_error(ctx, "out of memory converting generated RGB frames");
        goto cleanup;
    }
    for (size_t index = 0; index < rgb_count; index++) {
        float scaled = frames.rgb[index] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        rgb8[index] = (uint8_t)lrintf(scaled);
    }
    int output_width = frames.width;
    int output_height = frames.height;
    if (output_width != params->width || output_height != params->height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(
                rgb8, frames.frames, output_width, output_height,
                params->width, params->height, &resized)) {
            h3_set_error(ctx, "cannot resize generated RGB frames");
            goto cleanup;
        }
        free(rgb8);
        rgb8 = resized;
        output_width = params->width;
        output_height = params->height;
    }
    if (params->on_frame) {
        size_t frame_bytes = (size_t)output_width * (size_t)output_height * 3;
        for (int index = 0; index < frames.frames; index++) {
            h3_frame frame = {output_width, output_height, output_width * 3,
                              rgb8 + (size_t)index * frame_bytes,
                              index, frames.frames};
            if (params->on_frame(&frame, params->callback_opaque)) {
                h3_set_error(ctx, "generation cancelled while delivering frame %d",
                             index);
                goto cleanup;
            }
        }
    }
    if (params->output_path && *params->output_path) {
        h3_progress_emit(&progress, "FFmpeg", 0, frames.frames);
        if (!h3_ffmpeg_write_av_rgb24_f32(
                params->output_path, rgb8, frames.frames, output_width,
                output_height, H3_FPS, waveform.pcm, waveform.samples,
                waveform.channels, waveform.sample_rate,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_progress_emit(&progress, "FFmpeg", frames.frames, frames.frames);
    }
    result = calloc(1, sizeof(*result));
    if (!result) {
        h3_set_error(ctx, "out of memory creating generation result");
        goto cleanup;
    }
    result->width = output_width;
    result->height = output_height;
    result->frames = frames.frames;
    result->fps = H3_FPS;
    result->sample_rate = waveform.sample_rate;
    result->seed = params->seed;

cleanup:
    free(tokenizer_path); free(text_path); free(dit_path); free(vae_path);
    free(audio_vae_path);
    h3_tokenizer_free(tokenizer);
    h3_tokenizer_ids_free(ids);
    for (size_t image = 0; image < visual_capacity; image++) {
        if (condition_pixels) free(condition_pixels[image]);
    }
    for (size_t image = 0; image < vision_output_count; image++)
        if (vision_outputs) h3_vision_output_free(&vision_outputs[image]);
    for (size_t index = 0; index < params->reference_count; index++)
        if (presentation_timestamps) free(presentation_timestamps[index]);
    free(condition_pixels);
    free(condition_widths);
    free(condition_heights);
    free(condition_frames);
    free(visual_reference_indices);
    free(reference_visual_indices);
    free(vision_outputs);
    free(presentations);
    free(presentation_timestamps);
    free(layout_references);
    free(condition_video_rows);
    free(condition_audio_rows);
    h3_text_embedding_free(&text);
    h3_layout_free(&layout);
    h3_dit_free(dit);
    free(video); free(audio); free(rgb8);
    h3_video_frames_free(&frames);
    h3_audio_waveform_free(&waveform);
    return result;
}

void h3_result_free(h3_result *result) {
    free(result);
}
