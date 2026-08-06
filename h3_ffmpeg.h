#ifndef H3_FFMPEG_H
#define H3_FFMPEG_H

#include <stddef.h>
#include <stdint.h>

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height, int fps,
                          char *error, size_t error_size);

#endif
