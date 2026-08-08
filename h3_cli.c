#include "h3_cli.h"

#include "h3_ffmpeg.h"
#include "h3_host.h"
#include "h3_terminal.h"
#include "linenoise.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define H3_CLI_PATH 4096

typedef struct {
    h3_ctx *ctx;
    const char *model_dir;
    h3_params params;
    char *first_frame;
    char *last_frame;
    char output_dir[H3_CLI_PATH];
    char last_output[H3_CLI_PATH];
    char *last_prompt;
    unsigned output_count;
    int random_seed;
    int show;
    int open_output;
    h3_terminal_protocol terminal;
    char phase[64];
    int progress_active;
    int completed;
    int total;
    int display_failed;
} h3_cli_state;

static char *skip_spaces(char *text) {
    while (*text && isspace((unsigned char)*text)) text++;
    return text;
}

static void trim_trailing(char *text) {
    size_t length = strlen(text);
    while (length && isspace((unsigned char)text[length - 1]))
        text[--length] = '\0';
}

static int parse_i32(const char *text, int minimum, int maximum, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    while (end && isspace((unsigned char)*end)) end++;
    if (errno || end == text || !end || *end || parsed < minimum ||
        parsed > maximum) return 0;
    *value = (int)parsed;
    return 1;
}

static int parse_u64_value(const char *text, uint64_t *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    while (end && isspace((unsigned char)*end)) end++;
    if (errno || end == text || !end || *end) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int parse_size(const char *text, int *width, int *height) {
    char *middle = NULL;
    char *end = NULL;
    errno = 0;
    long w = strtol(text, &middle, 10);
    if (errno || middle == text || (*middle != 'x' && *middle != 'X')) return 0;
    errno = 0;
    long h = strtol(middle + 1, &end, 10);
    while (end && isspace((unsigned char)*end)) end++;
    if (errno || end == middle + 1 || !end || *end || w < 1 || h < 1 ||
        w > INT32_MAX || h > INT32_MAX) return 0;
    *width = (int)w;
    *height = (int)h;
    return 1;
}

static int set_directory(char destination[H3_CLI_PATH], const char *path) {
    if (!path || !*path || strlen(path) >= H3_CLI_PATH) return 0;
    if (mkdir(path, 0755) != 0 && errno != EEXIST) return 0;
    struct stat status;
    if (stat(path, &status) != 0 || !S_ISDIR(status.st_mode)) return 0;
    snprintf(destination, H3_CLI_PATH, "%s", path);
    return 1;
}

static uint64_t random_seed(void) {
    uint64_t value;
    arc4random_buf(&value, sizeof(value));
    return value;
}

static int cli_progress(const char *phase, int completed, int total,
                        void *opaque) {
    h3_cli_state *state = opaque;
    if (!strcmp(state->phase, phase) && state->completed == completed &&
        state->total == total) return 0;
    if (strcmp(state->phase, phase)) {
        if (state->progress_active) fputc('\n', stderr);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
    }
    state->completed = completed;
    state->total = total;
    state->progress_active = completed < total;
    fprintf(stderr, "\r%-25s %4d/%-4d", phase, completed, total);
    if (!state->progress_active) fputc('\n', stderr);
    fflush(stderr);
    return 0;
}

static int cli_frame(const h3_frame *frame, void *opaque) {
    h3_cli_state *state = opaque;
    if (!state->show || state->display_failed ||
        state->terminal == H3_TERM_NONE) return 0;
    if (state->progress_active) {
        fputc('\n', stderr);
        state->progress_active = 0;
    }
    if (frame->denoise_step >= 0)
        fprintf(stderr, "h3: preview %d/%d\n", frame->denoise_step + 1,
                frame->denoise_steps);
    char error[256];
    if (!h3_terminal_display_rgb24(state->terminal, frame->rgb,
                                   frame->width, frame->height, frame->stride,
                                   error, sizeof(error))) {
        fprintf(stderr, "h3: display disabled: %s\n", error);
        state->display_failed = 1;
    }
    return 0;
}

static void print_help(void) {
    puts("Commands:");
    puts("  !help                    Show this help");
    puts("  !status                  Show current settings");
    puts("  !seed [N|random]         Set or show the seed");
    puts("  !size [WxH]              Set or show output size");
    puts("  !render-size [WxH|native]  Set internal render size");
    puts("  !frames [N]              Set or show requested frames");
    puts("  !seconds [N]             Set duration at 24 fps");
    puts("  !steps [N]               Set or show denoising steps");
    puts("  !reuse [N]               Set or show denoiser reuse");
    puts("  !layers [N]              Set or show active DiT blocks");
    puts("  !core-reuse [N]          Set or show core reuse");
    puts("  !token-reduction [on|off]  Toggle token reduction");
    puts("  !first [PATH|clear]      Set, show, or clear first frame");
    puts("  !last [PATH|clear]       Set, show, or clear last frame");
    puts("  !show [on|off]           Toggle denoising previews");
    puts("  !zoom N                  Set terminal image zoom");
    puts("  !open [on|off]           Toggle opening completed videos");
    puts("  !output [DIR]            Set or show the output directory");
    puts("  !save [PATH]             Copy the last generated video");
    puts("  !again                   Repeat the last prompt");
    puts("  !cache                   Show reusable-session cache state");
    puts("  !cache clear             Clear reusable-session caches");
    puts("  !quit                    Exit");
    puts("\nType a prompt on a line by itself to generate a video.");
}

static void print_status(const h3_cli_state *state) {
    int aligned = h3_align_frame_count(state->params.frames);
    printf("Size: %dx%d", state->params.width, state->params.height);
    if (state->params.render_width)
        printf(" (render %dx%d)", state->params.render_width,
               state->params.render_height);
    printf("\nFrames: %d requested, %d generated (%.3g seconds)\n",
           state->params.frames, aligned, (double)aligned / H3_FPS);
    printf("Steps: %d | reuse: %d | layers: %d | core reuse: %d | tokens: %s\n",
           state->params.steps, state->params.denoise_reuse,
           state->params.dit_layers, state->params.core_reuse,
           state->params.token_reduction ? "reduced" : "full");
    if (state->random_seed) puts("Seed: random");
    else printf("Seed: %" PRIu64 "\n", state->params.seed);
    printf("First: %s\n", state->first_frame ? state->first_frame : "none");
    printf("Last: %s\n", state->last_frame ? state->last_frame : "none");
    printf("Show: %s | open: %s | output: %s\n",
           state->show ? "on" : "off", state->open_output ? "on" : "off",
           state->output_dir);
}

static int parse_toggle(char *argument, int current, int *value) {
    argument = skip_spaces(argument);
    if (!*argument) {
        *value = !current;
        return 1;
    }
    if (!strcasecmp(argument, "on") || !strcmp(argument, "1")) {
        *value = 1;
        return 1;
    }
    if (!strcasecmp(argument, "off") || !strcmp(argument, "0")) {
        *value = 0;
        return 1;
    }
    return 0;
}

static int validate_anchor(const char *path) {
    int width, height;
    char error[512];
    if (h3_ffprobe_visual_size(path, &width, &height, error, sizeof(error)))
        return 1;
    fprintf(stderr, "h3: %s\n", error);
    return 0;
}

static void set_anchor(h3_cli_state *state, int first, char *argument) {
    argument = skip_spaces(argument);
    char **slot = first ? &state->first_frame : &state->last_frame;
    const char *name = first ? "First" : "Last";
    if (!*argument) {
        printf("%s: %s\n", name, *slot ? *slot : "none");
        return;
    }
    if (!strcasecmp(argument, "clear")) {
        free(*slot);
        *slot = NULL;
        printf("%s: none\n", name);
        return;
    }
    if (state->params.reference_count) {
        fprintf(stderr, "h3: frame anchors cannot be combined with Ref2VA references\n");
        return;
    }
    if (!validate_anchor(argument)) return;
    char *copy = strdup(argument);
    if (!copy) {
        fprintf(stderr, "h3: out of memory copying anchor path\n");
        return;
    }
    free(*slot);
    *slot = copy;
    printf("%s: %s\n", name, *slot);
}

static int copy_file(const char *source, const char *destination) {
    FILE *input = fopen(source, "rb");
    if (!input) return 0;
    FILE *output = fopen(destination, "wb");
    if (!output) {
        fclose(input);
        return 0;
    }
    unsigned char buffer[1024 * 1024];
    int ok = 1;
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), input)) != 0) {
        if (fwrite(buffer, 1, count, output) != count) {
            ok = 0;
            break;
        }
    }
    if (ferror(input)) ok = 0;
    if (fclose(input) != 0) ok = 0;
    if (fclose(output) != 0) ok = 0;
    return ok;
}

static void open_video(const char *path) {
    pid_t process;
    char *arguments[] = {"open", (char *)path, NULL};
    int status = posix_spawnp(&process, "open", NULL, NULL, arguments, environ);
    if (status != 0)
        fprintf(stderr, "h3: cannot open %s: %s\n", path, strerror(status));
}

static int generate(h3_cli_state *state, const char *prompt) {
    char output[H3_CLI_PATH];
    unsigned number = ++state->output_count;
    int length = snprintf(output, sizeof(output), "%s/video-%04u.mp4",
                          state->output_dir, number);
    if (length < 0 || (size_t)length >= sizeof(output)) {
        fprintf(stderr, "h3: output path is too long\n");
        return 0;
    }
    h3_params params = state->params;
    params.seed = state->random_seed ? random_seed() : state->params.seed;
    params.first_frame = state->first_frame;
    params.last_frame = state->last_frame;
    params.output_path = output;
    params.preview_denoise = state->show && state->terminal != H3_TERM_NONE;
    params.on_frame = params.preview_denoise ? cli_frame : NULL;
    params.on_progress = cli_progress;
    params.callback_opaque = state;
    state->phase[0] = '\0';
    state->progress_active = 0;
    state->display_failed = 0;
    printf("Seed: %" PRIu64 "\n", params.seed);
    fflush(stdout);
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    h3_result *result = h3_generate(state->ctx, prompt, &params);
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (!result) {
        if (state->progress_active) fputc('\n', stderr);
        fprintf(stderr, "h3: %s\n", h3_last_error(state->ctx));
        return 0;
    }
    h3_result_free(result);
    double elapsed = (double)(end.tv_sec - begin.tv_sec) +
        (double)(end.tv_nsec - begin.tv_nsec) / 1e9;
    snprintf(state->last_output, sizeof(state->last_output), "%s", output);
    free(state->last_prompt);
    state->last_prompt = strdup(prompt);
    printf("Done -> %s [%.2fs]\n", output, elapsed);
    if (state->open_output) open_video(output);
    return 1;
}

static int set_integer(char *argument, const char *name, int minimum,
                       int maximum, int *slot) {
    argument = skip_spaces(argument);
    if (!*argument) {
        printf("%s: %d\n", name, *slot);
        return 1;
    }
    int value;
    if (!parse_i32(argument, minimum, maximum, &value)) {
        fprintf(stderr, "h3: invalid %s\n", name);
        return 0;
    }
    *slot = value;
    printf("%s: %d\n", name, *slot);
    return 1;
}

static int process_command(h3_cli_state *state, char *line, int *repeat) {
    char *command = skip_spaces(line + 1);
    char *argument = command;
    while (*argument && !isspace((unsigned char)*argument)) argument++;
    if (*argument) *argument++ = '\0';
    argument = skip_spaces(argument);

    if (!strcasecmp(command, "help")) print_help();
    else if (!strcasecmp(command, "status")) print_status(state);
    else if (!strcasecmp(command, "quit") || !strcasecmp(command, "exit"))
        return 1;
    else if (!strcasecmp(command, "again")) {
        if (!state->last_prompt) fprintf(stderr, "h3: no previous prompt\n");
        else *repeat = 1;
    } else if (!strcasecmp(command, "seed")) {
        if (!*argument) {
            if (state->random_seed) puts("Seed: random");
            else printf("Seed: %" PRIu64 "\n", state->params.seed);
        } else if (!strcasecmp(argument, "random") || !strcmp(argument, "-1")) {
            state->random_seed = 1;
            puts("Seed: random");
        } else {
            uint64_t value;
            if (!parse_u64_value(argument, &value))
                fprintf(stderr, "h3: invalid seed\n");
            else {
                state->params.seed = value;
                state->random_seed = 0;
                printf("Seed: %" PRIu64 "\n", value);
            }
        }
    } else if (!strcasecmp(command, "size")) {
        if (!*argument) printf("Size: %dx%d\n", state->params.width,
                               state->params.height);
        else {
            int width, height;
            if (!parse_size(argument, &width, &height))
                fprintf(stderr, "h3: size must be WxH\n");
            else {
                state->params.width = width;
                state->params.height = height;
                printf("Size: %dx%d\n", width, height);
            }
        }
    } else if (!strcasecmp(command, "render-size")) {
        if (!*argument) {
            if (state->params.render_width)
                printf("Render size: %dx%d\n", state->params.render_width,
                       state->params.render_height);
            else puts("Render size: native");
        } else if (!strcasecmp(argument, "native")) {
            state->params.render_width = state->params.render_height = 0;
            puts("Render size: native");
        } else {
            int width, height;
            if (!parse_size(argument, &width, &height))
                fprintf(stderr, "h3: render size must be WxH or native\n");
            else {
                state->params.render_width = width;
                state->params.render_height = height;
                printf("Render size: %dx%d\n", width, height);
            }
        }
    } else if (!strcasecmp(command, "frames")) {
        set_integer(argument, "Frames", 1, INT32_MAX, &state->params.frames);
    } else if (!strcasecmp(command, "seconds")) {
        if (!*argument) {
            printf("Seconds: %.3g (%d aligned frames)\n",
                   (double)h3_align_frame_count(state->params.frames) / H3_FPS,
                   h3_align_frame_count(state->params.frames));
        } else {
            char *end = NULL;
            errno = 0;
            double seconds = strtod(argument, &end);
            while (end && isspace((unsigned char)*end)) end++;
            double frames = seconds * H3_FPS;
            if (errno || end == argument || !end || *end ||
                !isfinite(seconds) || seconds <= 0 ||
                frames > INT32_MAX) {
                fprintf(stderr, "h3: invalid duration\n");
            } else {
                state->params.frames = (int)llround(frames);
                printf("Seconds: %.3g (%d aligned frames)\n", seconds,
                       h3_align_frame_count(state->params.frames));
            }
        }
    } else if (!strcasecmp(command, "steps")) {
        set_integer(argument, "Steps", 1, H3_MAX_STEPS, &state->params.steps);
    } else if (!strcasecmp(command, "reuse")) {
        set_integer(argument, "Reuse", 1, 32, &state->params.denoise_reuse);
    } else if (!strcasecmp(command, "layers")) {
        set_integer(argument, "Layers", H3_MIN_DIT_LAYERS,
                    H3_DEFAULT_DIT_LAYERS, &state->params.dit_layers);
    } else if (!strcasecmp(command, "core-reuse")) {
        set_integer(argument, "Core reuse", 1, 6, &state->params.core_reuse);
    } else if (!strcasecmp(command, "token-reduction")) {
        int value;
        if (!parse_toggle(argument, state->params.token_reduction, &value))
            fprintf(stderr, "h3: use on or off\n");
        else {
            state->params.token_reduction = value;
            printf("Token reduction: %s\n", value ? "on" : "off");
        }
    } else if (!strcasecmp(command, "first")) set_anchor(state, 1, argument);
    else if (!strcasecmp(command, "last")) set_anchor(state, 0, argument);
    else if (!strcasecmp(command, "show")) {
        int value;
        if (!parse_toggle(argument, state->show, &value))
            fprintf(stderr, "h3: use on or off\n");
        else if (value && state->terminal == H3_TERM_NONE)
            fprintf(stderr, "h3: no supported graphical terminal detected\n");
        else {
            state->show = value;
            printf("Show: %s\n", value ? "on" : "off");
        }
    } else if (!strcasecmp(command, "zoom")) {
        int value;
        if (!parse_i32(argument, 1, INT32_MAX, &value) ||
            !h3_terminal_set_zoom(value)) fprintf(stderr, "h3: invalid zoom\n");
        else printf("Zoom: %dx\n", value);
    } else if (!strcasecmp(command, "open")) {
        int value;
        if (!parse_toggle(argument, state->open_output, &value))
            fprintf(stderr, "h3: use on or off\n");
        else {
            state->open_output = value;
            printf("Open: %s\n", value ? "on" : "off");
        }
    } else if (!strcasecmp(command, "output")) {
        if (!*argument) printf("Output: %s\n", state->output_dir);
        else if (!set_directory(state->output_dir, argument))
            fprintf(stderr, "h3: cannot use output directory %s: %s\n",
                    argument, strerror(errno));
        else printf("Output: %s\n", state->output_dir);
    } else if (!strcasecmp(command, "save")) {
        if (!state->last_output[0]) fprintf(stderr, "h3: no video to save\n");
        else {
            char destination[H3_CLI_PATH];
            if (*argument) snprintf(destination, sizeof(destination), "%s", argument);
            else snprintf(destination, sizeof(destination), "h3-%ld.mp4", (long)time(NULL));
            if (!copy_file(state->last_output, destination))
                fprintf(stderr, "h3: cannot save %s: %s\n", destination,
                        strerror(errno));
            else printf("Saved: %s\n", destination);
        }
    } else if (!strcasecmp(command, "cache")) {
        if (!strcasecmp(argument, "clear")) {
            h3_cache_clear(state->ctx);
            puts("Cache cleared.");
        } else if (*argument) {
            fprintf(stderr, "h3: use !cache or !cache clear\n");
        } else {
            h3_cache_info info;
            h3_cache_get_info(state->ctx, &info);
            printf("Cache: embeddings %zu (%.1f MiB), DiT %s, video VAE %s\n",
                   info.embedding_entries,
                   (double)info.embedding_bytes / (1024.0 * 1024.0),
                   info.prepared_dit ? "resident" : "empty",
                   info.video_decoder ? "resident" : "empty");
        }
    } else {
        fprintf(stderr, "h3: unknown command; type !help\n");
    }
    return 0;
}

int h3_cli_run(h3_ctx *ctx, const char *model_dir,
               const h3_params *initial, int show,
               int seed_was_given) {
    h3_cli_state state;
    memset(&state, 0, sizeof(state));
    state.ctx = ctx;
    h3_cache_set_enabled(ctx, 1);
    state.model_dir = model_dir;
    state.params = *initial;
    state.params.output_path = NULL;
    state.params.first_frame = NULL;
    state.params.last_frame = NULL;
    state.params.on_frame = NULL;
    state.params.on_progress = NULL;
    state.params.callback_opaque = NULL;
    state.random_seed = !seed_was_given;
    state.terminal = h3_terminal_detect();
    state.show = show && state.terminal != H3_TERM_NONE;
    if (initial->first_frame) state.first_frame = strdup(initial->first_frame);
    if (initial->last_frame) state.last_frame = strdup(initial->last_frame);
    snprintf(state.output_dir, sizeof(state.output_dir), "/tmp/h3-XXXXXX");
    if (!mkdtemp(state.output_dir)) {
        fprintf(stderr, "h3: cannot create session directory: %s\n",
                strerror(errno));
        free(state.first_frame);
        free(state.last_frame);
        return 1;
    }

    char history[H3_CLI_PATH] = {0};
    const char *user_home = getenv("HOME");
    if (user_home && strlen(user_home) + sizeof("/.h3_history") < sizeof(history)) {
        snprintf(history, sizeof(history), "%s/.h3_history", user_home);
        linenoiseHistoryLoad(history);
    }
    linenoiseHistorySetMaxLen(500);
    linenoiseSetMultiLine(1);

    printf("H3 Interactive Mode\nModel: %s\nOutputs: %s\n", model_dir,
           state.output_dir);
    if (show && state.terminal == H3_TERM_NONE)
        puts("Display unavailable in this terminal.");
    puts("Type a prompt to generate video; !help lists commands.");
    print_status(&state);

    char *line;
    while ((line = linenoise("h3> ")) != NULL) {
        trim_trailing(line);
        char *input = skip_spaces(line);
        if (*input) {
            linenoiseHistoryAdd(input);
            int repeat = 0;
            int quit = *input == '!' ? process_command(&state, input, &repeat) : 0;
            if (quit) {
                free(line);
                break;
            }
            if (repeat) {
                char *prompt = strdup(state.last_prompt);
                if (!prompt) fprintf(stderr, "h3: out of memory copying prompt\n");
                else {
                    generate(&state, prompt);
                    free(prompt);
                }
            }
            else if (*input != '!') generate(&state, input);
        }
        free(line);
    }
    if (*history) linenoiseHistorySave(history);
    free(state.first_frame);
    free(state.last_frame);
    free(state.last_prompt);
    puts("Goodbye.");
    return 0;
}
