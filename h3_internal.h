#ifndef H3_INTERNAL_H
#define H3_INTERNAL_H

#include "h3.h"

#include <stdarg.h>

struct h3_ctx {
    char *model_dir;
    char error[512];
    h3_device_info device;
    h3_model_info model;
};

void h3_set_error(h3_ctx *ctx, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

#endif
