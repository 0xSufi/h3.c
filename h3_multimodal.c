#include "h3_multimodal.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_VISION_START UINT32_C(151652)
#define H3_VISION_END UINT32_C(151653)
#define H3_IMAGE_PAD UINT32_C(151655)

typedef struct {
    uint32_t *values;
    size_t count;
    size_t capacity;
} h3_ids;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int ids_reserve(h3_ids *ids, size_t extra) {
    if (extra > SIZE_MAX - ids->count) return 0;
    size_t needed = ids->count + extra;
    if (needed <= ids->capacity) return 1;
    size_t capacity = ids->capacity ? ids->capacity : 32;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*ids->values)) return 0;
    uint32_t *values = realloc(ids->values, capacity * sizeof(*values));
    if (!values) return 0;
    ids->values = values;
    ids->capacity = capacity;
    return 1;
}

static int ids_append(h3_ids *ids, const uint32_t *values, size_t count) {
    if (!ids_reserve(ids, count)) return 0;
    if (count) memcpy(ids->values + ids->count, values,
                      count * sizeof(*values));
    ids->count += count;
    return 1;
}

static int ids_push(h3_ids *ids, uint32_t value) {
    return ids_append(ids, &value, 1);
}

static int tokenize_append(const h3_tokenizer *tokenizer, const char *text,
                           h3_ids *ids, char *error, size_t error_size) {
    uint32_t *values = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, text, 0, &values, &count,
                             error, error_size)) return 0;
    int ok = ids_append(ids, values, count);
    h3_tokenizer_ids_free(values);
    if (!ok) fail(error, error_size,
                  "out of memory constructing multimodal token sequence");
    return ok;
}

static int build_positions(const h3_vision_output *images,
                           const h3_text_vision_span *spans,
                           size_t image_count, size_t sequence,
                           uint32_t *positions,
                           char *error, size_t error_size) {
    int64_t offset = 0;
    for (size_t image = 0; image < image_count; image++) {
        size_t start = spans[image].start;
        size_t end = start + spans[image].tokens;
        if (start > sequence || end > sequence ||
            images[image].grid_h < 2 || images[image].grid_w < 2 ||
            images[image].grid_h % 2 || images[image].grid_w % 2) {
            fail(error, error_size, "invalid Qwen vision span geometry");
            return 0;
        }
        if (image == 0) {
            for (size_t axis = 0; axis < 3; axis++)
                for (size_t index = 0; index < start; index++)
                    positions[axis * sequence + index] = (uint32_t)index;
        }
        size_t merged_h = (size_t)images[image].grid_h / 2;
        size_t merged_w = (size_t)images[image].grid_w / 2;
        size_t length_max = merged_h > merged_w ? merged_h : merged_w;
        int64_t base = (int64_t)start + offset;
        int64_t next = (int64_t)start + (int64_t)length_max + offset;
        if (base < 0 || next < 0 ||
            (uint64_t)next + (uint64_t)(sequence - end) > UINT32_MAX) {
            fail(error, error_size, "multimodal mRoPE position overflow");
            return 0;
        }
        for (size_t index = start; index < end; index++)
            positions[index] = (uint32_t)base;
        size_t cursor = start;
        for (size_t row = 0; row < merged_h; row++) {
            for (size_t column = 0; column < merged_w; column++) {
                positions[sequence + cursor] = (uint32_t)(base + (int64_t)row);
                positions[2 * sequence + cursor] =
                    (uint32_t)(base + (int64_t)column);
                cursor++;
            }
        }
        if (cursor != end) {
            fail(error, error_size, "Qwen merged grid does not match token span");
            return 0;
        }
        for (size_t axis = 0; axis < 3; axis++)
            for (size_t index = end; index < sequence; index++)
                positions[axis * sequence + index] =
                    (uint32_t)(next + (int64_t)(index - end));
        offset += (int64_t)length_max - (int64_t)spans[image].tokens;
    }
    return 1;
}

int h3_multimodal_encode_fl2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_vision_output *images, size_t image_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!tokenizer || !weight_directory || !shader_source_path || !prompt ||
        !*prompt || !images || !image_count || !output) {
        fail(error, error_size, "invalid FL2VA multimodal presentation arguments");
        return 0;
    }
    h3_ids ids = {0};
    h3_text_vision_span *spans = calloc(image_count, sizeof(*spans));
    uint32_t *positions = NULL;
    uint8_t *tags = NULL;
    int ok = 0;
    if (!spans) goto oom;
    for (size_t image = 0; image < image_count; image++) {
        const h3_vision_output *vision = &images[image];
        if (!vision->merged || !vision->deepstack[0] ||
            !vision->deepstack[1] || !vision->deepstack[2] ||
            !vision->tokens || vision->tokens > UINT32_MAX) {
            fail(error, error_size, "invalid Qwen vision output");
            goto cleanup;
        }
        char prefix[64];
        int length = snprintf(prefix, sizeof(prefix), "<Picture %zu>: ", image + 1);
        if (length < 0 || (size_t)length >= sizeof(prefix) ||
            !tokenize_append(tokenizer, prefix, &ids, error, error_size) ||
            !ids_push(&ids, H3_VISION_START)) goto cleanup;
        spans[image].start = ids.count;
        spans[image].tokens = vision->tokens;
        spans[image].embeddings = vision->merged;
        for (size_t layer = 0; layer < H3_VISION_DEEPSTACKS; layer++)
            spans[image].deepstack[layer] = vision->deepstack[layer];
        if (!ids_reserve(&ids, vision->tokens + 1)) goto oom;
        for (size_t token = 0; token < vision->tokens; token++)
            ids.values[ids.count++] = H3_IMAGE_PAD;
        ids.values[ids.count++] = H3_VISION_END;
    }
    if (!tokenize_append(tokenizer, prompt, &ids, error, error_size)) goto cleanup;
    if (!ids.count || ids.count > INT_MAX || ids.count > SIZE_MAX / 3 ||
        ids.count > SIZE_MAX / sizeof(*positions)) goto oom;
    positions = calloc(3 * ids.count, sizeof(*positions));
    tags = malloc(ids.count);
    if (!positions || !tags) goto oom;
    memset(tags, 1, ids.count);
    for (size_t image = 0; image < image_count; image++) {
        size_t first = spans[image].start - 1;
        size_t count = spans[image].tokens + 2;
        if (first > ids.count || count > ids.count - first) {
            fail(error, error_size, "multimodal tag span overflow");
            goto cleanup;
        }
        memset(tags + first, 0, count);
    }
    if (!build_positions(images, spans, image_count, ids.count, positions,
                         error, error_size)) goto cleanup;
    ok = h3_text_encode_multimodal_bf16(
        weight_directory, shader_source_path, ids.values, ids.count,
        spans, image_count, positions, tags, progress, progress_opaque,
        output, error, error_size);
    goto cleanup;

oom:
    fail(error, error_size, "out of memory constructing FL2VA presentation");
cleanup:
    free(ids.values);
    free(spans);
    free(positions);
    free(tags);
    return ok;
}
