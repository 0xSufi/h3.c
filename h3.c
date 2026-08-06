#include "h3_internal.h"
#include "h3_host.h"
#include "h3_metal.h"
#include "h3_safetensors.h"

#include <errno.h>
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
        !h3_require_file(ctx, "Ref2VA/transformer/config.json") ||
        !h3_require_file(ctx, "FL2VA/tokenizer/tokenizer.json") ||
        !h3_inventory(ctx, "FL2VA/text_encoder", &ctx->model.text_encoder) ||
        !h3_inventory(ctx, "FL2VA/transformer", &ctx->model.fl2va_transformer) ||
        !h3_inventory(ctx, "Ref2VA/transformer", &ctx->model.ref2va_transformer) ||
        !h3_inventory(ctx, "FL2VA/video_vae/source", &ctx->model.video_vae) ||
        !h3_inventory(ctx, "FL2VA/audio_vae", &ctx->model.audio_vae)) {
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
    if (params->steps < 1 || params->steps > H3_MAX_STEPS) {
        h3_set_error(ctx, "steps must be in [1, 1000]");
        return 0;
    }
    if (params->reference_count && !params->references) {
        h3_set_error(ctx, "reference_count is nonzero but references is NULL");
        return 0;
    }
    if (params->reference_count && (params->first_frame || params->last_frame)) {
        h3_set_error(ctx, "full references cannot be combined with frame anchors");
        return 0;
    }
    return 1;
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
    h3_set_error(ctx, "generation reaches M5; this build currently provides M1 inspection");
    return NULL;
}

void h3_result_free(h3_result *result) {
    free(result);
}
