"""ComfyUI nodes for the h3 engine (MiniMax-H3 text/image-to-video).

Runs the `h3` binary as a subprocess (the engine does all GPU work itself —
on Linux through its CUDA backend), reports denoising progress to the
ComfyUI progress bar, saves the MP4 (with the generated audio) into the
ComfyUI output directory, previews it in the graph, and returns the decoded
frames as an IMAGE batch plus the file path.

Configuration (environment variables, read when the node runs):
  H3_DIR        repository / build directory holding the `h3` binary
                (default ~/h3.c)
  H3_MODEL_DIR  MiniMax-H3 snapshot directory (default $H3_DIR/MiniMax-H3)
  H3_FFMPEG     ffmpeg binary used to decode frames (default ffmpeg)
"""
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management
import comfy.utils

H3_DIR = os.path.expanduser(os.environ.get("H3_DIR", "~/h3.c"))
H3_BIN = os.path.join(H3_DIR, "h3")
H3_MODEL_DIR = os.environ.get("H3_MODEL_DIR", os.path.join(H3_DIR, "MiniMax-H3"))
H3_FFMPEG = os.environ.get("H3_FFMPEG", "ffmpeg")

# Speed/precision presets. The engine's defaults are the exact BF16 path;
# the FP8 paths and the DiT schedule controls are the validated knobs from
# the README (DGX Spark, 15 s clip: 21.5 min / 13.7 min / 7.7 min / 5.6 min).
PRESETS = {
    "exact bf16 (reference quality, slowest)": {"env": {}, "args": []},
    "fp8 (e4m3 GEMMs + attention)": {
        "env": {"H3_CUDA_FP8": "1", "H3_CUDA_SDPA_FP8": "1"}, "args": []},
    "fast (fp8 + --layers 45 --reuse 2)": {
        "env": {"H3_CUDA_FP8": "1", "H3_CUDA_SDPA_FP8": "1"},
        "args": ["--layers", "45", "--reuse", "2"]},
    "fastest (fast + --token-reduction)": {
        "env": {"H3_CUDA_FP8": "1", "H3_CUDA_SDPA_FP8": "1"},
        "args": ["--layers", "45", "--reuse", "2", "--token-reduction"]},
}

_PROGRESS = re.compile(rb"(denoise|text encoder|load transformer core|video VAE load|FFmpeg)\s+(\d+)/(\d+)")


def _save_png(image, path):
    """IMAGE tensor [B,H,W,3] in 0..1 -> first frame as PNG."""
    frame = (image[0].detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(frame).save(path)


def _decode_frames(path):
    """MP4 -> float32 tensor [N,H,W,3] in 0..1 via ffmpeg raw RGB."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    width, height = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        [H3_FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    frame_bytes = width * height * 3
    count = len(raw) // frame_bytes
    if count == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    array = np.frombuffer(raw[:count * frame_bytes], dtype=np.uint8)
    array = array.reshape(count, height, width, 3).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def _run_h3(command, env, steps):
    """Run h3, stream its progress, honor ComfyUI interruption."""
    process = subprocess.Popen(command, cwd=H3_DIR, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pbar = comfy.utils.ProgressBar(steps)
    tail = []
    phase = {"name": ""}

    def pump():
        buffer = b""
        while True:
            chunk = process.stdout.read(256)
            if not chunk:
                break
            buffer += chunk
            while True:
                cut = min([i for i in (buffer.find(b"\r"), buffer.find(b"\n")) if i >= 0], default=-1)
                if cut < 0:
                    break
                line, buffer = buffer[:cut], buffer[cut + 1:]
                if not line.strip():
                    continue
                match = _PROGRESS.search(line)
                if match:
                    name = match.group(1).decode()
                    done, total = int(match.group(2)), int(match.group(3))
                    if name == "denoise" and total:
                        pbar.update_absolute(min(done, steps), steps)
                    if name != phase["name"]:
                        phase["name"] = name
                        print(f"[h3] {name} ...", flush=True)
                else:
                    text = line.decode(errors="replace").rstrip()
                    tail.append(text)
                    del tail[:-40]
                    print(f"[h3] {text}", flush=True)
        if buffer.strip():
            tail.append(buffer.decode(errors="replace"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        while process.poll() is None:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
            reader.join(timeout=0.25)
    finally:
        reader.join(timeout=5)
    if process.returncode != 0:
        raise RuntimeError("h3 failed (exit %d):\n%s" % (process.returncode, "\n".join(tail[-15:])))


class H3TextToVideo:
    """Generate a video (with audio) from a prompt with the h3 engine."""

    CATEGORY = "H3"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    DESCRIPTION = ("Runs the h3 MiniMax-H3 engine. Saves the MP4 (video + "
                   "generated audio) to the output directory, previews it, "
                   "and returns the frames as an IMAGE batch.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default":
                    "A cat wearing sunglasses rides a skateboard through a neon-lit city at night, cinematic"}),
                "preset": (list(PRESETS.keys()), {"default": "fast (fp8 + --layers 45 --reuse 2)"}),
                "width": ("INT", {"default": 864, "min": 256, "max": 1280, "step": 16}),
                "height": ("INT", {"default": 480, "min": 256, "max": 1280, "step": 16}),
                "seconds": ("INT", {"default": 2, "min": 1, "max": 15, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 2, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "control_after_generate": "randomize"}),
                "filename_prefix": ("STRING", {"default": "h3/video"}),
                "return_frames": ("BOOLEAN", {"default": True, "tooltip":
                    "Decode the MP4 into an IMAGE batch (a 15 s 864x480 clip is ~1.8 GB of float frames)."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "First-frame conditioning image (FL2VA)."}),
                "last_frame": ("IMAGE", {"tooltip": "Last-frame conditioning image (FL2VA)."}),
                "extra_args": ("STRING", {"default": "", "tooltip":
                    "Extra h3 command-line flags, e.g. --core-reuse 4 --ref-image /path.png"}),
                "model_dir": ("STRING", {"default": H3_MODEL_DIR}),
            },
        }

    def generate(self, prompt, preset, width, height, seconds, steps, seed,
                 filename_prefix, return_frames, first_frame=None,
                 last_frame=None, extra_args="", model_dir=None):
        if not os.path.isfile(H3_BIN):
            raise RuntimeError(f"h3 binary not found at {H3_BIN} (set H3_DIR)")
        model_dir = os.path.expanduser(model_dir or H3_MODEL_DIR)
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"MiniMax-H3 model directory not found: {model_dir}")
        output_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, width, height)
        file = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(full_folder, file)

        command = [H3_BIN, "-d", model_dir, "-p", prompt, "-o", path,
                   "--width", str(width), "--height", str(height),
                   "--seconds", str(seconds), "--steps", str(steps),
                   "--seed", str(seed)]
        spec = PRESETS[preset]
        command += spec["args"]
        temp_dir = tempfile.mkdtemp(prefix="h3_", dir=folder_paths.get_temp_directory())
        try:
            if first_frame is not None:
                first_path = os.path.join(temp_dir, "first.png")
                _save_png(first_frame, first_path)
                command += ["--first-frame", first_path]
            if last_frame is not None:
                last_path = os.path.join(temp_dir, "last.png")
                _save_png(last_frame, last_path)
                command += ["--last-frame", last_path]
            if extra_args.strip():
                command += shlex.split(extra_args)
            env = os.environ.copy()
            env.update(spec["env"])
            env.setdefault("H3_PROFILE", "1")
            print("[h3] " + " ".join(shlex.quote(c) for c in command), flush=True)
            _run_h3(command, env, steps)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if not os.path.isfile(path):
            raise RuntimeError(f"h3 finished but {path} was not written")
        if return_frames:
            frames = _decode_frames(path)
        else:
            frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return {
            "ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}],
                   "animated": (True,)},
            "result": (frames, path),
        }


NODE_CLASS_MAPPINGS = {"H3TextToVideo": H3TextToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"H3TextToVideo": "H3 Text to Video (MiniMax-H3)"}
