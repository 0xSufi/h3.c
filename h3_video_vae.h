#ifndef H3_VIDEO_VAE_H
#define H3_VIDEO_VAE_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int frames;
    int height;
    int width;
    /* Frame-major, row-major interleaved RGB F32 in [0,1]. */
    float *rgb;
    h3_gpu_stats gpu_stats;
} h3_video_frames;

typedef void (*h3_video_vae_progress)(int completed_blocks, int total_blocks,
                                      void *opaque);

/* Decode aligned H3 temporal chunks, using the released overlap/blend rules in
 * time and 256-pixel overlapping tiles in space. The two-token mode remains
 * available only for the diagnostic MLX fixture. */
int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size);
void h3_video_frames_free(h3_video_frames *frames);

#endif
