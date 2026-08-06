#include "h3_dit.h"

#include "h3_dit_schedule.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    AUDIO_STREAMS = 2,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2
};

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
} h3_dit_block;

struct h3_dit {
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_dit_schedule *schedule;
    int fused_mlp;
    h3_layout layout;
    h3_sigma_schedule sigmas;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    uint32_t text_rows;
    uint32_t video_condition_rows;
    uint32_t audio_condition_rows;
    uint32_t audio_rows;
    uint32_t video_rows;
    uint32_t video_total_rows;
    uint32_t audio_total_rows;
    uint32_t audio_target_start;
    uint32_t video_target_start;
    uint32_t sequence;
    h3_gpu_tensor *refined_text;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor **row_maps;
    h3_gpu_tensor **final_audio_maps;
    h3_gpu_tensor **final_video_maps;
    h3_gpu_tensor *video_patch_w;
    h3_gpu_tensor *video_patch_b;
    h3_gpu_tensor *audio_patch_w;
    h3_gpu_tensor *audio_patch_b;
    h3_dit_block blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *final_video_w;
    h3_gpu_tensor *final_video_b;
    h3_gpu_tensor *final_audio_w;
    h3_gpu_tensor *final_audio_b;
    h3_gpu_tensor *video_input;
    h3_gpu_tensor *audio_input;
    h3_gpu_tensor *video_projected_f32;
    h3_gpu_tensor *audio_projected_f32;
    h3_gpu_tensor *video_projected;
    h3_gpu_tensor *audio_projected;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *final_audio_input;
    h3_gpu_tensor *final_video_input;
    h3_gpu_tensor *final_audio_norm;
    h3_gpu_tensor *final_video_norm;
    h3_gpu_tensor *final_audio_f32;
    h3_gpu_tensor *final_video_f32;
    h3_gpu_tensor *audio_output;
    h3_gpu_tensor *video_output;
    h3_gpu_tensor *audio_output_bf16;
    h3_gpu_tensor *video_output_bf16;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(dit->gpu));
    return 0;
}

static void report(h3_dit_progress progress, void *opaque, const char *phase,
                   int completed, int total) {
    if (progress) progress(phase, completed, total, opaque);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *bf1(h3_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 1, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf2(h3_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 2, shape,
                               error, error_size);
}

static h3_gpu_tensor *f1(h3_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 1, shape,
                              error, error_size);
}

static h3_gpu_tensor *f2(h3_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 2, shape,
                              error, error_size);
}

static int copy_layout(h3_dit *dit, const h3_layout *layout,
                       char *error, size_t error_size) {
    dit->layout = *layout;
    dit->layout.segments = NULL;
    dit->layout.positions = NULL;
    if (layout->segment_count) {
        dit->layout.segments = malloc(layout->segment_count *
                                      sizeof(*layout->segments));
        if (!dit->layout.segments) goto oom;
        memcpy(dit->layout.segments, layout->segments,
               layout->segment_count * sizeof(*layout->segments));
    }
    if (layout->seq_len) {
        dit->layout.positions = malloc(layout->seq_len *
                                       sizeof(*layout->positions));
        if (!dit->layout.positions) goto oom;
        memcpy(dit->layout.positions, layout->positions,
               layout->seq_len * sizeof(*layout->positions));
    }
    return 1;
oom:
    fail(error, error_size, "out of memory copying packed H3 layout");
    h3_layout_free(&dit->layout);
    return 0;
}

static int validate_layout(h3_dit *dit, const h3_text_embedding *text,
                           char *error, size_t error_size) {
    const h3_layout *layout = &dit->layout;
    if (!text || !text->values || text->width != TEXT_DIM || !text->tokens ||
        layout->signature[0] != (int)text->tokens ||
        !layout->segments || layout->segment_count < 3 ||
        layout->segments[0].kind != H3_SEG_TEXT ||
        layout->segments[layout->segment_count - 1].kind != H3_SEG_VIDEO ||
        layout->signature[1] < 1 || layout->signature[2] < 2 ||
        layout->signature[3] < 2 || layout->signature[4] < 1 ||
        layout->signature[2] % 2 || layout->signature[3] % 2 ||
        layout->seq_len > UINT32_MAX || text->tokens > UINT32_MAX ||
        layout->img_cond_rows > UINT32_MAX ||
        layout->audio_cond_rows > UINT32_MAX ||
        layout->audio_target_rows > UINT32_MAX ||
        layout->img_target_rows > UINT32_MAX) {
        fail(error, error_size,
             "DiT requires a valid contiguous H3 packed layout");
        return 0;
    }
    size_t cursor = 0, text_rows = 0, video_condition = 0;
    size_t audio_condition = 0, video_target = 0, audio_target = 0;
    unsigned target_video_segments = 0, target_audio_segments = 0;
    for (size_t index = 0; index < layout->segment_count; index++) {
        const h3_segment *segment = &layout->segments[index];
        if (segment->start != cursor || segment->stop < segment->start ||
            segment->stop > layout->seq_len) {
            fail(error, error_size, "DiT layout segments are not contiguous");
            return 0;
        }
        size_t rows = segment->stop - segment->start;
        switch (segment->kind) {
        case H3_SEG_TEXT: text_rows += rows; break;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE: video_condition += rows; break;
        case H3_SEG_REF_AUDIO: audio_condition += rows; break;
        case H3_SEG_AUDIO:
            audio_target += rows;
            target_audio_segments++;
            dit->audio_target_start = (uint32_t)segment->start;
            break;
        case H3_SEG_VIDEO:
            video_target += rows;
            target_video_segments++;
            dit->video_target_start = (uint32_t)segment->start;
            break;
        default:
            fail(error, error_size, "DiT layout contains an unknown segment");
            return 0;
        }
        cursor = segment->stop;
    }
    if (cursor != layout->seq_len || text_rows != text->tokens ||
        video_condition != layout->img_cond_rows ||
        audio_condition != layout->audio_cond_rows ||
        video_target != layout->img_target_rows ||
        audio_target != layout->audio_target_rows ||
        target_video_segments != 1 || target_audio_segments != 1 ||
        video_condition > UINT32_MAX - video_target ||
        audio_condition > UINT32_MAX - audio_target) {
        fail(error, error_size, "DiT layout row-source counts are inconsistent");
        return 0;
    }
    if (text->tags) {
        for (size_t index = 0; index < text->tokens; index++) {
            if (text->tags[index] >= H3_DIT_MODALITIES) {
                fail(error, error_size, "DiT text presentation has an invalid tag");
                return 0;
            }
        }
    }
    dit->latent_t = layout->signature[1];
    dit->latent_h = layout->signature[2];
    dit->latent_w = layout->signature[3];
    dit->audio_t = layout->signature[4];
    dit->text_rows = (uint32_t)text->tokens;
    dit->video_condition_rows = (uint32_t)video_condition;
    dit->audio_condition_rows = (uint32_t)audio_condition;
    dit->audio_rows = (uint32_t)layout->audio_target_rows;
    dit->video_rows = (uint32_t)layout->img_target_rows;
    dit->video_total_rows = (uint32_t)(video_condition + video_target);
    dit->audio_total_rows = (uint32_t)(audio_condition + audio_target);
    dit->sequence = (uint32_t)layout->seq_len;
    return 1;
}

static int load_block(h3_dit *dit, h3_dit_block *block, const char *prefix,
                      char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
#define LOAD2(field, suffix, rows, columns) do {                               \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf2(dit, name, rows, columns, error, error_size);            \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    LOAD2(out, "attn.out_proj.weight", HIDDEN, INNER);
    LOAD2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD1
#undef LOAD2
    return 1;
}

static void free_block(h3_dit_block *block) {
    free_tensor(&block->norm1);
    free_tensor(&block->norm2);
    free_tensor(&block->qkv);
    free_tensor(&block->q_norm);
    free_tensor(&block->k_norm);
    free_tensor(&block->out);
    free_tensor(&block->fc1);
    free_tensor(&block->fc2);
}

static int run_refiner_block(h3_dit *dit, const h3_dit_block *weight,
                             h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             h3_gpu_tensor *heads, h3_gpu_tensor *branch,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *activated,
                             char *error, size_t error_size) {
    uint32_t rows = dit->text_rows;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm1, rows,
                             HIDDEN, 1e-5f), "refiner attention norm");
    OP(h3_gpu_linear_bf16(dit->gpu, qkv, norm, weight->qkv, NULL, rows,
                           HIDDEN, INNER * 3), "refiner QKV");
    OP(h3_gpu_grouped_qkv_rope_bf16(
                             dit->gpu, query, key, value, qkv, weight->q_norm,
                             weight->k_norm, weight->q_norm, weight->q_norm,
                             rows, HEADS, HEAD_DIM, 0, 1e-5f),
       "refiner QK norm");
    OP(h3_gpu_sdpa_bf16(dit->gpu, heads, query, key, value, rows, HEADS,
                         HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "refiner attention");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, heads, weight->out, NULL, rows,
                           INNER, HIDDEN), "refiner attention output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner attention residual");
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm2, rows,
                             HIDDEN, 1e-5f), "refiner MLP norm");
    OP(h3_gpu_linear_bf16(dit->gpu, fc1, norm, weight->fc1, NULL, rows,
                           HIDDEN, FFN * 2), "refiner MLP input");
    OP(h3_gpu_swiglu_bf16(dit->gpu, activated, fc1, rows, FFN),
       "refiner SwiGLU");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, activated, weight->fc2, NULL,
                           rows, FFN, HIDDEN), "refiner MLP output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner MLP residual");
#undef OP
    return 1;
}

static int refine_text(h3_dit *dit, const h3_text_embedding *text,
                       char *error, size_t error_size) {
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        dit->gpu, text->values, text->tokens * TEXT_DIM);
    h3_gpu_tensor *condition_w = bf2(dit, "condition_proj.weight", HIDDEN,
                                     TEXT_DIM, error, error_size);
    h3_gpu_tensor *condition_b = bf1(dit, "condition_proj.bias", HIDDEN,
                                     error, error_size);
    h3_dit_block refiner[2];
    memset(refiner, 0, sizeof(refiner));
    h3_gpu_tensor *final_norm = NULL;
    h3_gpu_tensor *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *heads = NULL, *branch = NULL, *fc1 = NULL;
    h3_gpu_tensor *activated = NULL;
    int ok = source && condition_w && condition_b &&
        load_block(dit, &refiner[0], "token_refiner.blocks.0.",
                   error, error_size) &&
        load_block(dit, &refiner[1], "token_refiner.blocks.1.",
                   error, error_size);
    if (ok) final_norm = bf1(dit, "token_refiner.final_norm.weight", HIDDEN,
                             error, error_size);
    size_t rows = dit->text_rows;
    if (ok && final_norm) {
        dit->refined_text = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        norm = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        qkv = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER * 3);
        query = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        key = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        value = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        heads = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        branch = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        fc1 = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN * 2);
        activated = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN);
        ok = dit->refined_text && norm && qkv && query && key && value &&
             heads && branch && fc1 && activated;
    }
    if (!ok) {
        if (!error || !*error)
            fail(error, error_size, "cannot allocate token-refiner tensors: %s",
                 h3_gpu_error(dit->gpu));
        goto cleanup;
    }
    ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                "begin token refinement") &&
         gpu_op(dit, h3_gpu_linear_bf16(
             dit->gpu, dit->refined_text, source, condition_w, condition_b,
             dit->text_rows, TEXT_DIM, HIDDEN), error, error_size,
             "condition projection") &&
         run_refiner_block(dit, &refiner[0], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         run_refiner_block(dit, &refiner[1], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         gpu_op(dit, h3_gpu_rms_norm_bf16(
             dit->gpu, dit->refined_text, dit->refined_text, final_norm,
             dit->text_rows, HIDDEN, 1e-5f), error, error_size,
             "refiner final norm") &&
         gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit token refinement");
cleanup:
    free_tensor(&source);
    free_tensor(&condition_w);
    free_tensor(&condition_b);
    free_block(&refiner[0]);
    free_block(&refiner[1]);
    free_tensor(&final_norm);
    free_tensor(&norm);
    free_tensor(&qkv);
    free_tensor(&query);
    free_tensor(&key);
    free_tensor(&value);
    free_tensor(&heads);
    free_tensor(&branch);
    free_tensor(&fc1);
    free_tensor(&activated);
    return ok;
}

static int prepare_rope(h3_dit *dit, char *error, size_t error_size) {
    h3_gpu_tensor *inverse_tensor = f1(dit, "rope.inv_freq", ROPE_FREQS,
                                       error, error_size);
    float inverse[ROPE_FREQS];
    if (!inverse_tensor ||
        !h3_gpu_tensor_read_f32(inverse_tensor, inverse, ROPE_FREQS)) {
        free_tensor(&inverse_tensor);
        if (!error || !*error) fail(error, error_size, "cannot read RoPE frequencies");
        return 0;
    }
    free_tensor(&inverse_tensor);
    size_t count = (size_t)dit->sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines);
        free(sines);
        fail(error, error_size, "out of memory allocating DiT RoPE tables");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++) {
        float axes[] = {(float)dit->layout.positions[row].t,
                        (float)dit->layout.positions[row].h,
                        (float)dit->layout.positions[row].w};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
        }
    }
    h3_gpu_tensor *cos_f32 = h3_gpu_tensor_from_f32(dit->gpu, cosines, count);
    h3_gpu_tensor *sin_f32 = h3_gpu_tensor_from_f32(dit->gpu, sines, count);
    free(cosines);
    free(sines);
    dit->rope_cos = h3_gpu_tensor_new_bf16(dit->gpu, count);
    dit->rope_sin = h3_gpu_tensor_new_bf16(dit->gpu, count);
    int ok = cos_f32 && sin_f32 && dit->rope_cos && dit->rope_sin;
    if (ok) {
        ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                    "begin RoPE setup") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_cos, cos_f32, (uint32_t)count),
                 error, error_size, "RoPE cosine cast") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_sin, sin_f32, (uint32_t)count),
                 error, error_size, "RoPE sine cast") &&
             gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit RoPE setup");
    } else if (!error || !*error) {
        fail(error, error_size, "cannot allocate DiT RoPE buffers: %s",
             h3_gpu_error(dit->gpu));
    }
    free_tensor(&cos_f32);
    free_tensor(&sin_f32);
    return ok;
}

static int prepare_maps(h3_dit *dit, const h3_text_embedding *text,
                        char *error, size_t error_size) {
    int steps = h3_dit_schedule_steps(dit->schedule);
    dit->row_maps = calloc((size_t)steps, sizeof(*dit->row_maps));
    dit->final_audio_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_audio_maps));
    dit->final_video_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_video_maps));
    uint32_t *rows = malloc((size_t)dit->sequence * sizeof(*rows));
    uint32_t *audio = malloc((size_t)dit->audio_rows * sizeof(*audio));
    uint32_t *video = malloc((size_t)dit->video_rows * sizeof(*video));
    if (!dit->row_maps || !dit->final_audio_maps || !dit->final_video_maps ||
        !rows || !audio || !video) {
        fail(error, error_size, "out of memory allocating modulation row maps");
        free(rows); free(audio); free(video);
        return 0;
    }
    for (int step = 0; step < steps; step++) {
        if (!h3_dit_schedule_row_map(dit->schedule, step, &dit->layout,
                                     text->tags, text->tokens, rows,
                                     dit->sequence)) {
            fail(error, error_size, "cannot construct modulation row map");
            free(rows); free(audio); free(video);
            return 0;
        }
        uint32_t audio_row = h3_dit_schedule_audio_row(dit->schedule, step);
        uint32_t video_row = h3_dit_schedule_video_row(dit->schedule, step);
        for (uint32_t index = 0; index < dit->audio_rows; index++)
            audio[index] = audio_row;
        for (uint32_t index = 0; index < dit->video_rows; index++)
            video[index] = video_row;
        dit->row_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, rows, dit->sequence);
        dit->final_audio_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_rows);
        dit->final_video_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_rows);
        if (!dit->row_maps[step] || !dit->final_audio_maps[step] ||
            !dit->final_video_maps[step]) {
            fail(error, error_size, "cannot allocate modulation row maps: %s",
                 h3_gpu_error(dit->gpu));
            free(rows); free(audio); free(video);
            return 0;
        }
    }
    free(rows); free(audio); free(video);
    return 1;
}

static int load_core(h3_dit *dit, h3_dit_progress progress, void *opaque,
                     char *error, size_t error_size) {
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
        if (!load_block(dit, &dit->blocks[index], prefix, error, error_size))
            return 0;
        report(progress, opaque, "load transformer core", (int)index + 1,
               H3_DIT_BLOCKS);
    }
    dit->video_patch_w = f2(dit, "video_patch_proj.weight", HIDDEN,
                            VIDEO_PATCH, error, error_size);
    dit->video_patch_b = f1(dit, "video_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->audio_patch_w = f2(dit, "audio_patch_proj.weight", HIDDEN,
                            AUDIO_CHANNELS, error, error_size);
    dit->audio_patch_b = f1(dit, "audio_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->final_norm = bf1(dit, "final_layer.norm.weight", HIDDEN,
                          error, error_size);
    dit->final_video_w = f2(dit, "final_layer.video_out.weight", VIDEO_PATCH,
                            HIDDEN, error, error_size);
    dit->final_video_b = f1(dit, "final_layer.video_out.bias", VIDEO_PATCH,
                            error, error_size);
    dit->final_audio_w = f2(dit, "final_layer.audio_out.weight", AUDIO_CHANNELS,
                            HIDDEN, error, error_size);
    dit->final_audio_b = f1(dit, "final_layer.audio_out.bias", AUDIO_CHANNELS,
                            error, error_size);
    return dit->video_patch_w && dit->video_patch_b && dit->audio_patch_w &&
           dit->audio_patch_b && dit->final_norm && dit->final_video_w &&
           dit->final_video_b && dit->final_audio_w && dit->final_audio_b;
}

static int allocate_activations(h3_dit *dit, char *error, size_t error_size) {
    size_t sequence = dit->sequence;
    size_t audio = dit->audio_rows;
    size_t video = dit->video_rows;
    size_t audio_total = dit->audio_total_rows;
    size_t video_total = dit->video_total_rows;
#define BF(field, elements) (dit->field = h3_gpu_tensor_new_bf16(dit->gpu, (elements)))
#define F32(field, elements) (dit->field = h3_gpu_tensor_new_f32(dit->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(video_input, video_total * VIDEO_PATCH),
        F32(audio_input, audio_total * AUDIO_CHANNELS),
        F32(video_projected_f32, video_total * HIDDEN),
        F32(audio_projected_f32, audio_total * HIDDEN),
        BF(video_projected, video_total * HIDDEN),
        BF(audio_projected, audio_total * HIDDEN),
        BF(hidden, sequence * HIDDEN),
        BF(mod_attention, sequence * HIDDEN),
        BF(qkv, sequence * INNER * 3),
        BF(query, sequence * INNER),
        BF(key, sequence * INNER),
        BF(value, sequence * INNER),
        BF(attention_heads, sequence * INNER),
        BF(attention_output, sequence * HIDDEN),
        BF(mod_mlp, sequence * HIDDEN),
        BF(mlp_output, sequence * HIDDEN),
        BF(final_audio_input, audio * HIDDEN),
        BF(final_video_input, video * HIDDEN),
        BF(final_audio_norm, audio * HIDDEN),
        BF(final_video_norm, video * HIDDEN),
        F32(final_audio_f32, audio * HIDDEN),
        F32(final_video_f32, video * HIDDEN),
        F32(audio_output, audio * AUDIO_CHANNELS),
        F32(video_output, video * VIDEO_PATCH),
        BF(audio_output_bf16, audio * AUDIO_CHANNELS),
        BF(video_output_bf16, video * VIDEO_PATCH)
    };
#undef BF
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            fail(error, error_size, "cannot allocate DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_mlp) {
        dit->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN * 2);
        dit->activated = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN);
        if (!dit->fc1 || !dit->activated) {
            fail(error, error_size,
                 "cannot allocate diagnostic DiT MLP tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return 1;
}

typedef struct {
    h3_dit_progress callback;
    void *opaque;
} schedule_progress;

static void schedule_report(int completed, int total, void *opaque) {
    schedule_progress *state = opaque;
    report(state->callback, state->opaque, "precompute AdaLN", completed, total);
}

static h3_dit *load_dit(const char *weight_directory,
                        const char *shader_source_path,
                        const h3_text_embedding *text,
                        const h3_layout *layout,
                        const h3_sigma_schedule *sigmas,
                        const float *condition_video_rows,
                        size_t condition_video_elements,
                        const float *condition_audio_rows,
                        size_t condition_audio_elements,
                        h3_dit_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !layout || !sigmas) {
        fail(error, error_size, "invalid DiT load arguments");
        return NULL;
    }
    h3_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        fail(error, error_size, "out of memory creating DiT model");
        return NULL;
    }
    dit->fused_mlp = getenv("H3_DISABLE_FUSED_MLP") == NULL;
    if (!copy_layout(dit, layout, error, error_size) ||
        !validate_layout(dit, text, error, error_size)) goto failed;
    size_t wanted_video_condition =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio_condition =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video_condition ||
        condition_audio_elements != wanted_audio_condition ||
        (wanted_video_condition && !condition_video_rows) ||
        (wanted_audio_condition && !condition_audio_rows)) {
        fail(error, error_size,
             "condition row elements do not match the packed DiT layout");
        goto failed;
    }
    dit->sigmas = *sigmas;
    dit->weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!dit->weights) goto failed;
    dit->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!dit->gpu) goto failed;
    h3_gpu_profile_set_label(dit->gpu, "H3 DiT");
    report(progress, progress_opaque, "refine text", 0, 1);
    if (!refine_text(dit, text, error, error_size)) goto failed;
    report(progress, progress_opaque, "refine text", 1, 1);
    schedule_progress schedule_state = {progress, progress_opaque};
    dit->schedule = h3_dit_schedule_precompute(
        dit->weights, dit->gpu, sigmas, dit->video_condition_rows != 0,
        dit->audio_condition_rows != 0, schedule_report, &schedule_state,
        error, error_size);
    if (!dit->schedule || !prepare_rope(dit, error, error_size) ||
        !prepare_maps(dit, text, error, error_size) ||
        !load_core(dit, progress, progress_opaque, error, error_size) ||
        !allocate_activations(dit, error, error_size)) goto failed;
    if ((wanted_video_condition && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows,
             wanted_video_condition)) ||
        (wanted_audio_condition && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows,
             wanted_audio_condition))) {
        fail(error, error_size, "cannot write persistent DiT condition rows");
        goto failed;
    }
    h3_gpu_profile_mark(dit->gpu, "load");
    return dit;
failed:
    h3_dit_free(dit);
    return NULL;
}

h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    NULL, 0, NULL, 0, progress, progress_opaque,
                    error, error_size);
}

h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    condition_video_rows, condition_video_elements,
                    condition_audio_rows, condition_audio_elements,
                    progress, progress_opaque, error, error_size);
}

static int run_block(h3_dit *dit, unsigned index, int step,
                     char *error, size_t error_size) {
    h3_dit_block *weight = &dit->blocks[index];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(dit->schedule,
                                                            index);
    h3_gpu_tensor *row_map = dit->row_maps[step];
    uint32_t rows = dit->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_attention, dit->hidden,
        weight->norm1, modulation, row_map, rows, HIDDEN, SLOTS, 0, 1, 1e-5f),
       "DiT attention AdaLN");
    OP(h3_gpu_linear_bf16(dit->gpu, dit->qkv, dit->mod_attention,
        weight->qkv, NULL, rows, HIDDEN, INNER * 3), "DiT QKV");
    OP(h3_gpu_grouped_qkv_rope_bf16(
        dit->gpu, dit->query, dit->key, dit->value,
        dit->qkv, weight->q_norm, weight->k_norm, dit->rope_cos, dit->rope_sin,
        rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f), "DiT QK norm/RoPE");
    OP(h3_gpu_sdpa_bf16(dit->gpu, dit->attention_heads, dit->query, dit->key,
        dit->value, rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "DiT full attention");
    OP(h3_gpu_linear_bf16(dit->gpu, dit->attention_output,
        dit->attention_heads, weight->out, NULL, rows, INNER, HIDDEN),
       "DiT attention output");
    OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden,
        dit->attention_output, modulation, row_map, rows, HIDDEN, SLOTS, 2),
       "DiT attention gate");
    OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_mlp, dit->hidden, weight->norm2,
        modulation, row_map, rows, HIDDEN, SLOTS, 3, 4, 1e-5f),
       "DiT MLP AdaLN");
    if (dit->fused_mlp) {
        OP(h3_gpu_mlp_bf16(dit->gpu, dit->mlp_output, dit->mod_mlp,
            weight->fc1, weight->fc2, rows, HIDDEN, FFN, HIDDEN),
           "DiT fused MLP");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->fc1, dit->mod_mlp, weight->fc1,
            NULL, rows, HIDDEN, FFN * 2), "DiT MLP input");
        OP(h3_gpu_swiglu_bf16(dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "DiT SwiGLU");
        OP(h3_gpu_linear_bf16(dit->gpu, dit->mlp_output, dit->activated,
            weight->fc2, NULL, rows, FFN, HIDDEN), "DiT MLP output");
    }
    OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden, dit->mlp_output,
        modulation, row_map, rows, HIDDEN, SLOTS, 5), "DiT MLP gate");
#undef OP
    return 1;
}

static int encode_forward(h3_dit *dit, int step, char *error,
                          size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_begin(dit->gpu), "begin DiT forward");
    OP(h3_gpu_linear_f32(dit->gpu, dit->video_projected_f32, dit->video_input,
        dit->video_patch_w, dit->video_patch_b, dit->video_total_rows,
        VIDEO_PATCH,
        HIDDEN), "video patch projection");
    OP(h3_gpu_linear_f32(dit->gpu, dit->audio_projected_f32, dit->audio_input,
        dit->audio_patch_w, dit->audio_patch_b, dit->audio_total_rows,
        AUDIO_CHANNELS, HIDDEN), "audio patch projection");
    OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->video_projected,
        dit->video_projected_f32, dit->video_total_rows * HIDDEN),
       "video BF16 cast");
    OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->audio_projected,
        dit->audio_projected_f32, dit->audio_total_rows * HIDDEN),
       "audio BF16 cast");
    size_t video_offset = 0;
    size_t audio_offset = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        const h3_segment *segment = &dit->layout.segments[index];
        size_t segment_rows = segment->stop - segment->start;
        size_t destination = segment->start * HIDDEN;
        if (segment->kind == H3_SEG_TEXT) {
            OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                dit->refined_text, 0, segment_rows * HIDDEN),
               "pack refined text");
        } else if (segment->kind == H3_SEG_COND ||
                   segment->kind == H3_SEG_REF_IMAGE ||
                   segment->kind == H3_SEG_VIDEO) {
            OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                dit->video_projected, video_offset * HIDDEN,
                segment_rows * HIDDEN), "pack video source");
            video_offset += segment_rows;
        } else {
            OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                dit->audio_projected, audio_offset * HIDDEN,
                segment_rows * HIDDEN), "pack audio source");
            audio_offset += segment_rows;
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        fail(error, error_size, "DiT segment packing did not consume row sources");
        return 0;
    }
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (!run_block(dit, block, step, error, error_size)) return 0;
    }
    OP(h3_gpu_copy_bf16(dit->gpu, dit->final_audio_input, 0, dit->hidden,
        (size_t)dit->audio_target_start * HIDDEN,
        (size_t)dit->audio_rows * HIDDEN),
       "slice final audio");
    OP(h3_gpu_copy_bf16(dit->gpu, dit->final_video_input, 0, dit->hidden,
        (size_t)dit->video_target_start * HIDDEN,
        (size_t)dit->video_rows * HIDDEN), "slice final video");
    const h3_gpu_tensor *final = h3_dit_schedule_final(dit->schedule);
    OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_audio_norm,
        dit->final_audio_input, dit->final_norm, final,
        dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
        0, 1, 1e-5f), "final audio AdaLN");
    OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_video_norm,
        dit->final_video_input, dit->final_norm, final,
        dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
        0, 1, 1e-5f), "final video AdaLN");
    OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_audio_f32,
        dit->final_audio_norm, dit->audio_rows * HIDDEN), "final audio F32 cast");
    OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_video_f32,
        dit->final_video_norm, dit->video_rows * HIDDEN), "final video F32 cast");
    OP(h3_gpu_linear_f32(dit->gpu, dit->audio_output, dit->final_audio_f32,
        dit->final_audio_w, dit->final_audio_b, dit->audio_rows, HIDDEN,
        AUDIO_CHANNELS), "final audio head");
    OP(h3_gpu_linear_f32(dit->gpu, dit->video_output, dit->final_video_f32,
        dit->final_video_w, dit->final_video_b, dit->video_rows, HIDDEN,
        VIDEO_PATCH), "final video head");
    OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->audio_output_bf16,
        dit->audio_output, dit->audio_rows * AUDIO_CHANNELS),
       "final audio output cast");
    OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->video_output_bf16,
        dit->video_output, dit->video_rows * VIDEO_PATCH),
       "final video output cast");
    OP(h3_gpu_submit(dit->gpu), "submit DiT forward");
#undef OP
    return 1;
}

size_t h3_dit_video_elements(const h3_dit *dit) {
    return dit ? (size_t)VIDEO_CHANNELS * (size_t)dit->latent_t *
        (size_t)dit->latent_h * (size_t)dit->latent_w : 0;
}

size_t h3_dit_audio_elements(const h3_dit *dit) {
    return dit ? (size_t)AUDIO_CHANNELS * AUDIO_STREAMS *
        (size_t)dit->audio_t : 0;
}

int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || step < 0 || step >= h3_dit_schedule_steps(dit->schedule) ||
        !video_latent || !audio_latent || !video_velocity || !audio_velocity) {
        fail(error, error_size, "invalid DiT forward arguments");
        return 0;
    }
    size_t video_row_elements = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_row_elements = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    float *video_rows = malloc(video_row_elements * sizeof(*video_rows));
    float *audio_rows = malloc(audio_row_elements * sizeof(*audio_rows));
    uint16_t *video_out = malloc(video_row_elements * sizeof(*video_out));
    uint16_t *audio_out = malloc(audio_row_elements * sizeof(*audio_out));
    float *video_f32 = malloc(video_row_elements * sizeof(*video_f32));
    float *audio_f32 = malloc(audio_row_elements * sizeof(*audio_f32));
    if (!video_rows || !audio_rows || !video_out || !audio_out ||
        !video_f32 || !audio_f32) {
        fail(error, error_size, "out of memory packing DiT latents");
        free(video_rows); free(audio_rows); free(video_out); free(audio_out);
        free(video_f32); free(audio_f32);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows,
        video_row_elements) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->video_input,
            (size_t)dit->video_condition_rows * VIDEO_PATCH,
            video_rows, video_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->audio_input,
            (size_t)dit->audio_condition_rows * AUDIO_CHANNELS,
            audio_rows, audio_row_elements);
    if (!ok) fail(error, error_size, "cannot pack/write DiT input latents");
    if (ok) ok = encode_forward(dit, step, error, error_size);
    if (ok) ok = h3_gpu_tensor_read_bf16(dit->video_output_bf16, video_out,
                                         video_row_elements) &&
                 h3_gpu_tensor_read_bf16(dit->audio_output_bf16, audio_out,
                                         audio_row_elements);
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot read DiT output");
    if (ok) {
        for (size_t index = 0; index < video_row_elements; index++) {
            uint32_t bits = (uint32_t)video_out[index] << 16;
            memcpy(&video_f32[index], &bits, sizeof(bits));
        }
        for (size_t index = 0; index < audio_row_elements; index++) {
            uint32_t bits = (uint32_t)audio_out[index] << 16;
            memcpy(&audio_f32[index], &bits, sizeof(bits));
        }
    }
    if (ok) ok = h3_dit_unpatchify_video(video_f32, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_velocity,
        h3_dit_video_elements(dit)) &&
        h3_dit_unpack_audio(audio_f32, AUDIO_CHANNELS, dit->audio_t,
                            audio_velocity, h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot unpack DiT output");
    free(video_rows); free(audio_rows); free(video_out); free(audio_out);
    free(video_f32); free(audio_f32);
    return ok;
}

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats) {
    return dit && h3_gpu_get_stats(dit->gpu, stats);
}

static void extrapolate_velocity(float *output, const float *last,
                                 const float *previous, size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) {
        memcpy(output, last, count * sizeof(*output));
        return;
    }
    float denominator = last_sigma - previous_sigma;
    float ratio = denominator != 0.0f
        ? (current_sigma - last_sigma) / denominator : 0.0f;
    /* Reuse intervals are deliberately small. This guard prevents malformed
     * custom schedules from turning one cached evaluation into an explosion. */
    if (ratio < -2.0f) ratio = -2.0f;
    if (ratio > 2.0f) ratio = 2.0f;
    for (size_t index = 0; index < count; index++)
        output[index] = last[index] +
                        ratio * (last[index] - previous[index]);
}

int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid DiT denoising arguments");
        return 0;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *video_denoised = malloc(video_count * sizeof(*video_denoised));
    float *audio_denoised = malloc(audio_count * sizeof(*audio_denoised));
    float *old_video = malloc(video_count * sizeof(*old_video));
    float *old_audio = malloc(audio_count * sizeof(*old_audio));
    float *video_next = malloc(video_count * sizeof(*video_next));
    float *audio_next = malloc(audio_count * sizeof(*audio_next));
    if (!video_velocity || !audio_velocity || !video_denoised ||
        !audio_denoised || !old_video || !old_audio || !video_next ||
        !audio_next) {
        fail(error, error_size, "out of memory allocating RES solver state");
        free(video_velocity); free(audio_velocity); free(video_denoised);
        free(audio_denoised); free(old_video); free(old_audio);
        free(video_next); free(audio_next);
        return 0;
    }
    int ok = 1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                            video_velocity, audio_velocity,
                            error, error_size);
        float sigma = dit->sigmas.video[step];
        float timestep = 1.0f - sigma;
        float sigma_from_timestep = 1.0f - timestep;
        float audio_slope = (float)h3_time_shift_slope(
            sigma, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT);
        if (ok) {
            for (size_t index = 0; index < video_count; index++)
                video_denoised[index] = video_latent[index] +
                    sigma_from_timestep * video_velocity[index];
            for (size_t index = 0; index < audio_count; index++)
                audio_denoised[index] = audio_latent[index] +
                    sigma_from_timestep * audio_velocity[index] * audio_slope;
            ok = h3_res_step(video_next, video_latent, video_denoised,
                             step ? old_video : NULL, video_count,
                             dit->sigmas.video, step, dit->sigmas.steps) &&
                 h3_res_step(audio_next, audio_latent, audio_denoised,
                             step ? old_audio : NULL, audio_count,
                             dit->sigmas.video, step, dit->sigmas.steps);
            if (!ok) fail(error, error_size, "RES solver rejected step %d", step);
        }
        if (ok) {
            memcpy(video_latent, video_next,
                   video_count * sizeof(*video_latent));
            memcpy(audio_latent, audio_next,
                   audio_count * sizeof(*audio_latent));
            memcpy(old_video, video_denoised,
                   video_count * sizeof(*old_video));
            memcpy(old_audio, audio_denoised,
                   audio_count * sizeof(*old_audio));
            report(progress, progress_opaque, "denoise", step + 1,
                   dit->sigmas.steps);
        }
    }
    free(video_velocity); free(audio_velocity); free(video_denoised);
    free(audio_denoised); free(old_video); free(old_audio);
    free(video_next); free(audio_next);
    h3_gpu_profile_mark(dit->gpu, "RES denoise");
    return ok;
}

int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent || reuse_interval < 1 ||
        reuse_interval > 32 ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid Euler denoising arguments");
        return 0;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *last_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*last_video)) : NULL;
    float *previous_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*previous_video)) : NULL;
    float *last_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*last_audio)) : NULL;
    float *previous_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*previous_audio)) : NULL;
    if (!video_velocity || !audio_velocity ||
        (reuse_interval > 1 &&
         (!last_video || !previous_video || !last_audio || !previous_audio))) {
        fail(error, error_size, "out of memory allocating Euler velocities");
        free(video_velocity);
        free(audio_velocity);
        free(last_video);
        free(previous_video);
        free(last_audio);
        free(previous_audio);
        return 0;
    }
    int ok = 1;
    int last_evaluated = -1;
    int previous_evaluated = -1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        int evaluate = reuse_interval == 1 || step == 0 ||
                       step == dit->sigmas.steps - 1 ||
                       step % reuse_interval == 0;
        if (evaluate) {
            ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                                video_velocity, audio_velocity,
                                error, error_size);
            if (ok && reuse_interval > 1) {
                if (last_evaluated >= 0) {
                    memcpy(previous_video, last_video,
                           video_count * sizeof(*previous_video));
                    memcpy(previous_audio, last_audio,
                           audio_count * sizeof(*previous_audio));
                    previous_evaluated = last_evaluated;
                }
                memcpy(last_video, video_velocity,
                       video_count * sizeof(*last_video));
                memcpy(last_audio, audio_velocity,
                       audio_count * sizeof(*last_audio));
                last_evaluated = step;
            }
        } else {
            extrapolate_velocity(
                video_velocity, last_video, previous_video, video_count,
                dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.video[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
            extrapolate_velocity(
                audio_velocity, last_audio, previous_audio, audio_count,
                dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.audio[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
        }
        if (ok) {
            ok = h3_euler_velocity_step(
                     video_latent, video_velocity, video_count,
                     dit->sigmas.video[step], dit->sigmas.video[step + 1]) &&
                 h3_euler_velocity_step(
                     audio_latent, audio_velocity, audio_count,
                     dit->sigmas.audio[step], dit->sigmas.audio[step + 1]);
            if (!ok) fail(error, error_size,
                          "Euler solver rejected step %d", step);
        }
        if (ok) report(progress, progress_opaque, "denoise", step + 1,
                       dit->sigmas.steps);
    }
    free(video_velocity);
    free(audio_velocity);
    free(last_video);
    free(previous_video);
    free(last_audio);
    free(previous_audio);
    h3_gpu_profile_mark(dit->gpu, "Euler denoise");
    return ok;
}

void h3_dit_free(h3_dit *dit) {
    if (!dit) return;
    int steps = h3_dit_schedule_steps(dit->schedule);
    if (dit->row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->row_maps[step]);
    if (dit->final_audio_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_audio_maps[step]);
    if (dit->final_video_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_video_maps[step]);
    free(dit->row_maps);
    free(dit->final_audio_maps);
    free(dit->final_video_maps);
    free_tensor(&dit->refined_text);
    free_tensor(&dit->rope_cos);
    free_tensor(&dit->rope_sin);
    free_tensor(&dit->video_patch_w); free_tensor(&dit->video_patch_b);
    free_tensor(&dit->audio_patch_w); free_tensor(&dit->audio_patch_b);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        free_block(&dit->blocks[block]);
    free_tensor(&dit->final_norm);
    free_tensor(&dit->final_video_w); free_tensor(&dit->final_video_b);
    free_tensor(&dit->final_audio_w); free_tensor(&dit->final_audio_b);
#define FREE(field) free_tensor(&dit->field)
    FREE(video_input); FREE(audio_input);
    FREE(video_projected_f32); FREE(audio_projected_f32);
    FREE(video_projected); FREE(audio_projected); FREE(hidden);
    FREE(mod_attention); FREE(qkv); FREE(query); FREE(key); FREE(value);
    FREE(attention_heads); FREE(attention_output); FREE(mod_mlp); FREE(fc1);
    FREE(activated); FREE(mlp_output); FREE(final_audio_input);
    FREE(final_video_input); FREE(final_audio_norm); FREE(final_video_norm);
    FREE(final_audio_f32); FREE(final_video_f32); FREE(audio_output);
    FREE(video_output);
    FREE(audio_output_bf16); FREE(video_output_bf16);
#undef FREE
    h3_dit_schedule_free(dit->schedule);
    h3_gpu_free(dit->gpu);
    h3_weight_store_free(dit->weights);
    h3_layout_free(&dit->layout);
    free(dit);
}

static int video_shape(int channels, int time, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels < 1 || time < 1 || height < 2 || width < 2 ||
        height % 2 || width % 2) return 0;
    size_t c = (size_t)channels, t = (size_t)time;
    size_t h = (size_t)height, w = (size_t)width;
    if (c > SIZE_MAX / t || c * t > SIZE_MAX / h ||
        c * t * h > SIZE_MAX / w) return 0;
    *latent_count = c * t * h * w;
    *row_count = t * (h / 2) * (w / 2) * c * 4;
    return 1;
}

int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements) {
    size_t latent_count, expected;
    if (!latent || !rows ||
        !video_shape(channels, time, height, width, &latent_count, &expected) ||
        row_elements != expected || latent_count != expected) return 0;
    size_t output = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t input = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            rows[output++] = latent[input];
                        }
    return output == row_elements;
}

int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements) {
    size_t expected, row_count;
    if (!rows || !latent ||
        !video_shape(channels, time, height, width, &expected, &row_count) ||
        latent_elements != expected || row_count != expected) return 0;
    size_t input = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t output = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            latent[output] = rows[input++];
                        }
    return input == row_count;
}

int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements) {
    if (!latent || !rows || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        row_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t output = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t input = ((size_t)channel * 2 + (size_t)stream) *
                               (size_t)time + (size_t)t;
                rows[output++] = latent[input];
            }
    return output == row_elements;
}

int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements) {
    if (!rows || !latent || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        latent_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t input = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t output = ((size_t)channel * 2 + (size_t)stream) *
                                (size_t)time + (size_t)t;
                latent[output] = rows[input++];
            }
    return input == latent_elements;
}
