#include "h3.h"
#include "h3_terminal.h"

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
        "      --steps N          Sigma points (default: 50; 49 forwards)\n"
        "      --seed N           Random seed (default: 42)\n"
        "      --first-frame PATH First-frame conditioning image\n"
        "      --last-frame PATH  Last-frame conditioning image\n"
        "      --ref-image PATH    Append an ordered Ref2VA image\n"
        "      --ref-image-size S  Image sizing: match (default) or max\n"
        "      --ref-video PATH    Append video, including embedded audio\n"
        "      --ref-silent-video PATH  Append video without its audio\n"
        "      --ref-video-audio VIDEO AUDIO  Append video + soundtrack\n"
        "      --ref-audio PATH    Append an ordered standalone audio clip\n"
        "      --show             Graphical-terminal frame display (M5)\n"
        "      --profile          Print per-phase Metal timing and allocation data\n"
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

static h3_reference *append_reference(h3_reference references[12],
                                      size_t *count) {
    if (*count >= 12) {
        fprintf(stderr, "h3: Ref2VA supports at most 12 references\n");
        exit(2);
    }
    h3_reference *reference = &references[(*count)++];
    memset(reference, 0, sizeof(*reference));
    return reference;
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

typedef struct {
    char phase[64];
    int active;
    int completed;
    int total;
    h3_terminal_protocol terminal;
    int display_failed;
} cli_state;

static int cli_progress(const char *phase, int completed, int total,
                        void *opaque) {
    cli_state *state = opaque;
    if (!strcmp(state->phase, phase) && state->completed == completed &&
        state->total == total) return 0;
    if (strcmp(state->phase, phase)) {
        if (state->active) fputc('\n', stderr);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
    }
    state->completed = completed;
    state->total = total;
    state->active = completed < total;
    fprintf(stderr, "\r%-25s %4d/%-4d", phase, completed, total);
    if (!state->active) fputc('\n', stderr);
    fflush(stderr);
    return 0;
}

static int cli_frame(const h3_frame *frame, void *opaque) {
    cli_state *state = opaque;
    if (state->display_failed || state->terminal == H3_TERM_NONE) return 0;
    fprintf(stderr, "h3: frame %d/%d via %s\n", frame->frame_index + 1,
            frame->frame_count, h3_terminal_protocol_name(state->terminal));
    char error[256];
    if (!h3_terminal_display_rgb24(state->terminal, frame->rgb,
                                   frame->width, frame->height, frame->stride,
                                   error, sizeof(error))) {
        fprintf(stderr, "h3: terminal display disabled: %s\n", error);
        state->display_failed = 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    enum { OPT_WIDTH = 1000, OPT_HEIGHT, OPT_FRAMES, OPT_STEPS, OPT_SEED,
           OPT_FIRST, OPT_LAST, OPT_REF_IMAGE, OPT_REF_IMAGE_SIZE,
           OPT_REF_VIDEO, OPT_REF_SILENT_VIDEO, OPT_REF_VIDEO_AUDIO,
           OPT_REF_AUDIO, OPT_SHOW, OPT_PROFILE, OPT_INFO };
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
        {"ref-image", required_argument, NULL, OPT_REF_IMAGE},
        {"ref-image-size", required_argument, NULL, OPT_REF_IMAGE_SIZE},
        {"ref-video", required_argument, NULL, OPT_REF_VIDEO},
        {"ref-silent-video", required_argument, NULL, OPT_REF_SILENT_VIDEO},
        {"ref-video-audio", required_argument, NULL, OPT_REF_VIDEO_AUDIO},
        {"ref-audio", required_argument, NULL, OPT_REF_AUDIO},
        {"show", no_argument, NULL, OPT_SHOW},
        {"profile", no_argument, NULL, OPT_PROFILE},
        {"info", no_argument, NULL, OPT_INFO},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };
    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = "outputs/h3.mp4";
    h3_params params = H3_PARAMS_DEFAULT;
    h3_reference references[12];
    size_t reference_count = 0;
    cli_state cli = {{0}, 0, -1, -1, H3_TERM_NONE, 0};
    int show = 0;
    int profile = 0;
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
            case OPT_REF_IMAGE: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_IMAGE;
                reference->path = optarg;
                break;
            }
            case OPT_REF_IMAGE_SIZE:
                if (!strcmp(optarg, "match"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MATCH;
                else if (!strcmp(optarg, "max"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MAX;
                else {
                    fprintf(stderr,
                        "h3: --ref-image-size must be match or max\n");
                    return 2;
                }
                break;
            case OPT_REF_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 1;
                break;
            }
            case OPT_REF_SILENT_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 0;
                break;
            }
            case OPT_REF_VIDEO_AUDIO: {
                if (optind >= argc) {
                    fprintf(stderr,
                        "h3: --ref-video-audio requires VIDEO and AUDIO\n");
                    return 2;
                }
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO_AUDIO;
                reference->path = optarg;
                reference->audio_path = argv[optind++];
                break;
            }
            case OPT_REF_AUDIO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_AUDIO;
                reference->path = optarg;
                break;
            }
            case OPT_SHOW: show = 1; break;
            case OPT_PROFILE: profile = 1; break;
            case OPT_INFO: info = 1; break;
            default: usage(argv[0]); return 2;
        }
    }
    if (!model_dir || (!info && !prompt)) {
        usage(argv[0]);
        return 2;
    }
    params.references = references;
    params.reference_count = reference_count;
    if (profile) setenv("H3_PROFILE", "1", 1);
    h3_ctx *ctx = h3_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "h3: %s\n", h3_last_error(NULL));
        return 1;
    }
    if (info) print_info(ctx);
    if (prompt) {
        params.output_path = output;
        params.on_progress = cli_progress;
        params.callback_opaque = &cli;
        if (show) {
            cli.terminal = h3_terminal_detect();
            if (cli.terminal == H3_TERM_NONE) {
                fprintf(stderr, "h3: warning: --show needs Kitty, Ghostty, "
                        "iTerm2, WezTerm, or Konsole\n");
            } else {
                fprintf(stderr, "h3: graphical output uses %s\n",
                        h3_terminal_protocol_name(cli.terminal));
                params.on_frame = cli_frame;
            }
        }
        h3_result *result = h3_generate(ctx, prompt, &params);
        if (!result) {
            if (cli.active) fputc('\n', stderr);
            fprintf(stderr, "h3: %s\n", h3_last_error(ctx));
            h3_free(ctx);
            return 1;
        }
        h3_result_free(result);
        fprintf(stderr, "h3: wrote %s\n", output);
    }
    h3_free(ctx);
    return 0;
}
