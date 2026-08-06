#include "h3.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s -d MODEL_DIR [--info]\n"
        "       %s -d MODEL_DIR -p PROMPT [-o OUTPUT] [options]\n\n"
        "Options:\n"
        "  -d, --model-dir PATH   MiniMax-H3 local directory\n"
        "  -p, --prompt TEXT      Raw H3 prompt\n"
        "  -o, --output PATH      Output MP4 (default: outputs/h3.mp4)\n"
        "      --width N          Output width (default: 864)\n"
        "      --height N         Output height (default: 480)\n"
        "      --frames N         Requested frames (default: 56)\n"
        "      --steps N          Sampling steps (default: 20)\n"
        "      --seed N           Random seed (default: 42)\n"
        "      --first-frame PATH First-frame conditioning image\n"
        "      --last-frame PATH  Last-frame conditioning image\n"
        "      --show             Graphical-terminal frame display (M5)\n"
        "      --info             Inspect model/device without mapping weights\n"
        "  -h, --help             Show this help\n",
        program, program);
}

static int parse_int(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0 || parsed > INT32_MAX) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (int)parsed;
}

static uint64_t parse_u64(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (uint64_t)parsed;
}

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_component(const char *label, const h3_component_info *item) {
    printf("  %-18s %2zu files  %4zu tensors  %7.3f GiB\n",
           label, item->files, item->tensors, gib(item->tensor_bytes));
}

static void print_info(const h3_ctx *ctx) {
    const h3_device_info *device = h3_device(ctx);
    const h3_model_info *model = h3_model(ctx);
    printf("h3-metal %s\n", H3_VERSION);
    printf("Device: %s (%s)\n", device->name, device->architecture);
    printf("  physical memory       %.1f GiB\n", gib(device->physical_memory));
    printf("  recommended GPU set   %.1f GiB\n", gib(device->recommended_working_set));
    printf("  max Metal buffer      %.1f GiB\n", gib(device->max_buffer_length));
    printf("  Apple GPU family      %d\n", device->apple_gpu_family);
    printf("  Metal 4               %s\n", device->metal4 ? "yes" : "no");
    printf("  unified memory        %s\n", device->unified_memory ? "yes" : "no");
    printf("Native checkpoint inventory (header-only):\n");
    print_component("Qwen3-VL encoder", &model->text_encoder);
    print_component("FL2VA DiT", &model->fl2va_transformer);
    print_component("Ref2VA DiT", &model->ref2va_transformer);
    print_component("video VAE", &model->video_vae);
    print_component("audio VAE", &model->audio_vae);
}

int main(int argc, char **argv) {
    enum { OPT_WIDTH = 1000, OPT_HEIGHT, OPT_FRAMES, OPT_STEPS, OPT_SEED,
           OPT_FIRST, OPT_LAST, OPT_SHOW, OPT_INFO };
    static const struct option options[] = {
        {"model-dir", required_argument, NULL, 'd'},
        {"prompt", required_argument, NULL, 'p'},
        {"output", required_argument, NULL, 'o'},
        {"width", required_argument, NULL, OPT_WIDTH},
        {"height", required_argument, NULL, OPT_HEIGHT},
        {"frames", required_argument, NULL, OPT_FRAMES},
        {"steps", required_argument, NULL, OPT_STEPS},
        {"seed", required_argument, NULL, OPT_SEED},
        {"first-frame", required_argument, NULL, OPT_FIRST},
        {"last-frame", required_argument, NULL, OPT_LAST},
        {"show", no_argument, NULL, OPT_SHOW},
        {"info", no_argument, NULL, OPT_INFO},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };
    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = "outputs/h3.mp4";
    h3_params params = H3_PARAMS_DEFAULT;
    int show = 0;
    int info = 0;
    int option;
    while ((option = getopt_long(argc, argv, "d:p:o:h", options, NULL)) != -1) {
        switch (option) {
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; break;
            case 'h': usage(argv[0]); return 0;
            case OPT_WIDTH: params.width = parse_int(optarg, "width"); break;
            case OPT_HEIGHT: params.height = parse_int(optarg, "height"); break;
            case OPT_FRAMES: params.frames = parse_int(optarg, "frames"); break;
            case OPT_STEPS: params.steps = parse_int(optarg, "steps"); break;
            case OPT_SEED: params.seed = parse_u64(optarg, "seed"); break;
            case OPT_FIRST: params.first_frame = optarg; break;
            case OPT_LAST: params.last_frame = optarg; break;
            case OPT_SHOW: show = 1; break;
            case OPT_INFO: info = 1; break;
            default: usage(argv[0]); return 2;
        }
    }
    if (!model_dir || (!info && !prompt)) {
        usage(argv[0]);
        return 2;
    }
    h3_ctx *ctx = h3_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "h3: %s\n", h3_last_error(NULL));
        return 1;
    }
    if (info) print_info(ctx);
    if (prompt) {
        (void)output;
        (void)show;
        h3_result *result = h3_generate(ctx, prompt, &params);
        if (!result) {
            fprintf(stderr, "h3: %s\n", h3_last_error(ctx));
            h3_free(ctx);
            return 1;
        }
        h3_result_free(result);
    }
    h3_free(ctx);
    return 0;
}
