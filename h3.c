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
    if (params->frames < 5 || h3_align_frame_count(params->frames) > 362) {
        h3_set_error(ctx, "frames must align within the released 5..362 range");
        return 0;
    }
    if (params->steps < 2 || params->steps > H3_MAX_STEPS) {
        h3_set_error(ctx, "sigma points must be in [2, 1000]");
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
    size_t images = 0, videos = 0, explicit_audio = 0, visual = 0;
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
            break;
        case H3_REFERENCE_AUDIO:
            explicit_audio++;
            break;
        case H3_REFERENCE_VIDEO_AUDIO:
            videos++; visual++; explicit_audio++;
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
    if (images > 9 || videos > 3 || explicit_audio > 3) {
        h3_set_error(ctx,
            "Ref2VA limits are 9 images, 3 videos, and 3 explicit audio inputs");
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

static void h3_video_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    h3_progress_emit(opaque, "video VAE encoder", completed, total);
}

static void h3_vision_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "Qwen vision", completed, total);
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
    for (size_t index = 0; index < params->reference_count; index++) {
        if (params->references[index].kind != H3_REFERENCE_IMAGE) {
            h3_set_error(ctx,
                "video/audio references enter in the next incremental Ref2VA slice");
            return NULL;
        }
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
    h3_vision_output *vision_outputs = NULL;
    h3_layout_ref *layout_references = NULL;
    int keyframes[2] = {0, 0};
    size_t keyframe_count = 0;
    float *condition_video_rows = NULL;
    size_t condition_video_elements = 0;
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
        vision_outputs = calloc(visual_capacity, sizeof(*vision_outputs));
        if (ref2va)
            layout_references = calloc(visual_capacity,
                                       sizeof(*layout_references));
        if (!condition_pixels || !condition_widths || !condition_heights ||
            !vision_outputs || (ref2va && !layout_references)) {
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
    h3_latent_canvas(params->width, params->height, &latent_w, &latent_h);

    if (ref2va) {
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            int source_width, source_height, image_width, image_height;
            if (!h3_ffprobe_visual_size(reference->path,
                                        &source_width, &source_height,
                                        detail, sizeof(detail)) ||
                !h3_reference_image_canvas(
                    source_width, source_height, params->width, params->height,
                    params->reference_image_size == H3_REFERENCE_IMAGE_MAX ?
                    2048 : 0, &image_width, &image_height)) {
                if (!detail[0]) snprintf(detail, sizeof(detail),
                    "cannot resolve reference image %zu canvas", index + 1);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (!h3_ffmpeg_read_image_f32(
                    reference->path, image_width, image_height,
                    H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = image_width;
            condition_heights[visual_count] = image_height;
            int ref_latent_w, ref_latent_h;
            h3_latent_canvas(image_width, image_height,
                             &ref_latent_w, &ref_latent_h);
            layout_references[visual_count] = (h3_layout_ref){
                H3_LAYOUT_REF_IMAGE, 1, ref_latent_h, ref_latent_w, 0};
            visual_count++;
        }
    } else {
        if (params->first_frame) {
            keyframes[keyframe_count++] = 0;
            if (!h3_ffmpeg_read_image_f32(
                    params->first_frame, params->width, params->height,
                    H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = params->width;
            condition_heights[visual_count] = params->height;
            visual_count++;
        }
        if (params->last_frame) {
            keyframes[keyframe_count++] = temporal.frame_count - 1;
            if (!h3_ffmpeg_read_image_f32(
                    params->last_frame, params->width, params->height,
                    H3_IMAGE_FIT_COVER, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = params->width;
            condition_heights[visual_count] = params->height;
            visual_count++;
        }
    }

    if (visual_count) {
        for (size_t image = 0; image < visual_count; image++) {
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            size_t rows = (size_t)image_latent_h * (size_t)image_latent_w / 4;
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
            size_t row_elements = (size_t)image_latent_h *
                                  (size_t)image_latent_w / 4 * 96;
            h3_video_latent latent;
            memset(&latent, 0, sizeof(latent));
            if (!h3_video_vae_encode(
                    vae_path, "h3_shaders.metal", condition_pixels[image],
                    1, condition_heights[image], condition_widths[image],
                    h3_video_encoder_progress_bridge, &progress,
                    &latent, detail, sizeof(detail))) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (latent.time != 1 || latent.height != image_latent_h ||
                latent.width != image_latent_w) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx,
                    "visual condition VAE produced unexpected latent geometry");
                goto cleanup;
            }
            float *rows = condition_video_rows + condition_offset;
            if (!h3_dit_patchify_video(
                    latent.values, 24, 1, image_latent_h, image_latent_w,
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
        for (size_t image = 0; image < visual_count; image++) {
            if (!h3_vision_encode_bf16(
                    text_path, "h3_shaders.metal", condition_pixels[image],
                    1, condition_heights[image], condition_widths[image],
                    h3_vision_progress_bridge, &progress,
                    &vision_outputs[image], detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (progress.cancelled) goto cleanup;
        }
        h3_progress_emit(&progress, "text encoder", 0, 50);
        if (!h3_multimodal_encode_fl2va_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt,
                vision_outputs, visual_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        for (size_t image = 0; image < visual_count; image++) {
            h3_vision_output_free(&vision_outputs[image]);
            free(condition_pixels[image]);
            condition_pixels[image] = NULL;
        }
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
            condition_video_rows, condition_video_elements, NULL, 0,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    } else {
        dit = h3_dit_load_t2va(
            dit_path, "h3_shaders.metal", &text, &layout, &sigmas,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    }
    if (!dit) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_text_embedding_free(&text);
    free(condition_video_rows);
    condition_video_rows = NULL;
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
            dit, video, audio, h3_dit_progress_bridge, &progress,
            detail, sizeof(detail))) {
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
    if (params->on_frame) {
        size_t frame_bytes = (size_t)frames.width * (size_t)frames.height * 3;
        for (int index = 0; index < frames.frames; index++) {
            h3_frame frame = {frames.width, frames.height, frames.width * 3,
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
                params->output_path, rgb8, frames.frames, frames.width,
                frames.height, H3_FPS, waveform.pcm, waveform.samples,
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
    result->width = frames.width;
    result->height = frames.height;
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
        if (vision_outputs) h3_vision_output_free(&vision_outputs[image]);
    }
    free(condition_pixels);
    free(condition_widths);
    free(condition_heights);
    free(vision_outputs);
    free(layout_references);
    free(condition_video_rows);
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
