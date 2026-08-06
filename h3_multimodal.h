#ifndef H3_MULTIMODAL_H
#define H3_MULTIMODAL_H

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <stddef.h>

/* Construct the released FL2VA `<Picture n>` presentation around already
 * encoded Qwen vision outputs, then run the language decoder. Keeping the
 * presentation builder separate lets the vision and text phases release their
 * large temporary weights independently. */
int h3_multimodal_encode_fl2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_vision_output *images, size_t image_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

#endif
