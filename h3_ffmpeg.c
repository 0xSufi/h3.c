#include "h3_ffmpeg.h"

#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int make_parents(const char *path, char *error, size_t error_size) {
    char *copy = strdup(path);
    if (!copy) {
        fail(error, error_size, "out of memory resolving output directory");
        return 0;
    }
    for (char *cursor = copy + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
            fail(error, error_size, "cannot create output directory %s: %s",
                 copy, strerror(errno));
            free(copy);
            return 0;
        }
        *cursor = '/';
    }
    free(copy);
    return 1;
}

static int write_all(int descriptor, const uint8_t *data, size_t bytes,
                     char *error, size_t error_size) {
    while (bytes) {
        ssize_t written = write(descriptor, data,
                                bytes > (size_t)SSIZE_MAX ?
                                (size_t)SSIZE_MAX : bytes);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            fail(error, error_size, "cannot stream RGB frames to FFmpeg: %s",
                 written < 0 ? strerror(errno) : "short write");
            return 0;
        }
        data += (size_t)written;
        bytes -= (size_t)written;
    }
    return 1;
}

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height, int fps,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !frames || frame_count < 1 || width < 2 ||
        height < 2 || fps < 1 || width % 2 || height % 2) {
        fail(error, error_size, "invalid FFmpeg RGB output arguments");
        return 0;
    }
    size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / 3 ||
        (size_t)frame_count > SIZE_MAX / (pixels * 3)) {
        fail(error, error_size, "RGB video size overflows");
        return 0;
    }
    if (!make_parents(path, error, error_size)) return 0;
    int stream[2];
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFmpeg pipe: %s", strerror(errno));
        return 0;
    }
    char size[64], rate[32];
    snprintf(size, sizeof(size), "%dx%d", width, height);
    snprintf(rate, sizeof(rate), "%d", fps);
    char *arguments[] = {
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", size, "-framerate", rate,
        "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(&actions, stream[0], STDIN_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, "ffmpeg", &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[0]);
    if (code) {
        close(stream[1]);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    struct sigaction ignore, previous;
    memset(&ignore, 0, sizeof(ignore));
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    sigaction(SIGPIPE, &ignore, &previous);
    int ok = write_all(stream[1], frames,
                       (size_t)frame_count * pixels * 3,
                       error, error_size);
    close(stream[1]);
    sigaction(SIGPIPE, &previous, NULL);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        if (ok) fail(error, error_size, "cannot wait for FFmpeg: %s",
                     strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (ok) fail(error, error_size, "FFmpeg exited with status %d",
                     WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 0;
    }
    return ok;
}
