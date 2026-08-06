#ifndef H3_MULTIMODAL_H
#define H3_MULTIMODAL_H

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <stddef.h>

/* Construct the released `<Picture n>` presentation around already encoded
 * Qwen vision outputs, then run the language decoder. FL2VA anchors and
 * image-only Ref2VA entries intentionally share this exact presentation.
 * Keeping the builder separate lets vision and text release their large
 * temporary weights independently. */
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
