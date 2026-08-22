#ifndef H3_CUDA_H
#define H3_CUDA_H

#include "h3.h"

/* Linux counterpart of h3_metal_probe: fills h3_device_info from the CUDA
 * driver. apple_gpu_family and metal4 are always 0 on this backend. */
int h3_cuda_probe(h3_device_info *info, char *error, size_t error_size);

#endif
