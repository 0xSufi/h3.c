#ifndef H3_TEXT_ENCODER_H
#define H3_TEXT_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

#define H3_TEXT_HIDDEN_SIZE 5120u

typedef struct {
    size_t tokens;
    size_t width;
    uint16_t *values;
    h3_gpu_stats gpu_stats;
} h3_text_embedding;

typedef void (*h3_text_progress)(int completed_layers, int total_layers,
                                 void *opaque);

/* Run the released first 50 Qwen3-VL language layers. The caller owns the
 * returned BF16 values and releases them with h3_text_embedding_free(). */
int h3_text_encode_bf16(const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);
void h3_text_embedding_free(h3_text_embedding *embedding);

#endif
