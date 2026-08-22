#!/usr/bin/env python3
"""Minimal web UI for the h3 engine (MiniMax-H3 multimodal video).

One page, one queue: prompts (plus optional first/last-frame images and,
when the Ref2VA checkpoint is installed, ordered reference images / video /
audio) go in, MP4s with the generated soundtrack come out. The server runs
the `h3` binary one job at a time, streams its progress, and keeps every
result with its settings in a gallery.

Environment:
  H3_DIR          directory with the built `h3` binary   (default ~/h3.c)
  H3_MODEL_DIR    MiniMax-H3 snapshot                    (default $H3_DIR/MiniMax-H3)
  H3_WEBUI_DATA   uploads/outputs directory              (default ~/h3-webui)
  H3_WEBUI_PORT   listen port                            (default 7860)

Only aiohttp beyond the standard library. Trusted-network tool: it executes
the local h3 binary with caller-supplied flags and has no authentication.
"""
import asyncio
import json
import os
import random
import re
import shlex
import time
import uuid
from pathlib import Path

from aiohttp import web

H3_DIR = Path(os.environ.get("H3_DIR", "~/h3.c")).expanduser()
H3_BIN = H3_DIR / "h3"
MODEL_DIR = Path(os.environ.get("H3_MODEL_DIR", str(H3_DIR / "MiniMax-H3"))).expanduser()
DATA_DIR = Path(os.environ.get("H3_WEBUI_DATA", "~/h3-webui")).expanduser()
OUTPUTS = DATA_DIR / "outputs"
UPLOADS = DATA_DIR / "uploads"
PORT = int(os.environ.get("H3_WEBUI_PORT", "7860"))
STATIC = Path(__file__).resolve().parent

PRESETS = {
    "exact": {"label": "exact bf16 (reference quality)", "env": {}, "args": []},
    "fp8": {"label": "fp8 (e4m3 GEMMs + attention)",
            "env": {"H3_CUDA_FP8": "1", "H3_CUDA_SDPA_FP8": "1"}, "args": []},
    "fast": {"label": "fast (fp8 + layers 45, reuse 2)",
             "env": {"H3_CUDA_FP8": "1", "H3_CUDA_SDPA_FP8": "1"},
             "args": ["--layers", "45", "--reuse", "2"]},
    "fastest": {"label": "fastest (fast + token reduction)",
                "env": {"H3_CUDA_FP8": "1", "H3_CUDA_SDPA_FP8": "1"},
                "args": ["--layers", "45", "--reuse", "2", "--token-reduction"]},
}

# Progress phases in engine order: (h3 progress label, ui label, start%, end%).
PHASES = [
    ("tokenizer", "tokenizing", 0, 1),
    ("text encoder", "encoding prompt", 1, 6),
    ("refine text", "refining text", 6, 7),
    ("video encoder", "encoding references", 7, 9),
    ("audio encoder", "encoding reference audio", 9, 10),
    ("precompute AdaLN", "preparing schedule", 10, 11),
    ("load transformer core", "loading DiT weights", 11, 16),
    ("denoise", "denoising", 16, 80),
    ("audio VAE", "decoding audio", 80, 83),
    ("video VAE load", "loading video decoder", 83, 87),
    ("FFmpeg", "encoding MP4", 95, 100),
]
_PROGRESS = re.compile(rb"^(" + b"|".join(re.escape(name.encode()) for name, *_ in PHASES) +
                       rb")\s+(\d+)/(\d+)\s*$")

jobs = {}          # id -> job dict (also mirrors finished jobs from disk)
queue = asyncio.Queue()


def now():
    return time.time()


def public(job):
    keys = ("id", "created", "status", "phase", "percent", "step", "steps",
            "prompt", "settings", "error", "file", "elapsed", "eta", "refs")
    return {k: job.get(k) for k in keys}


def load_history():
    for meta in sorted(OUTPUTS.glob("*.json")):
        try:
            job = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        if job.get("id") and (OUTPUTS / (job["id"] + ".mp4")).exists():
            job["status"] = "done"
            jobs[job["id"]] = job


_ref2va_cache = {"at": 0.0, "ready": False}


def ref2va_ready():
    """Ref2VA is usable once its directory exists and huggingface_hub has no
    in-flight partials for it (files land complete; the *.incomplete markers
    under the snapshot's .cache flag a download still in progress)."""
    if now() - _ref2va_cache["at"] > 10:
        ready = (MODEL_DIR / "Ref2VA" / "model_index.json").is_file()
        if ready:
            download = MODEL_DIR / ".cache" / "huggingface" / "download" / "Ref2VA"
            ready = not download.is_dir() or                 next(download.rglob("*.incomplete"), None) is None
        _ref2va_cache.update(at=now(), ready=ready)
    return _ref2va_cache["ready"]


def caps():
    return {
        "model_dir": str(MODEL_DIR),
        "engine": H3_BIN.exists(),
        "ref2va": ref2va_ready(),
        "presets": {k: v["label"] for k, v in PRESETS.items()},
    }


async def save_upload(field, job_id, name):
    path = UPLOADS / f"{job_id}_{name}{Path(field.filename or name).suffix or ''}"
    with open(path, "wb") as sink:
        field.file.seek(0)
        while True:
            chunk = field.file.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)
    return path


async def handle_generate(request):
    form = await request.post()
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    prompt = (form.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "prompt is empty"}, status=400)
    preset = form.get("preset") if form.get("preset") in PRESETS else "fast"
    try:
        width = max(256, min(1280, int(form.get("width", 864)) // 16 * 16))
        height = max(256, min(1280, int(form.get("height", 480)) // 16 * 16))
        seconds = max(1, min(15, int(form.get("seconds", 5))))
        steps = max(2, min(100, int(form.get("steps", 20))))
    except ValueError:
        return web.json_response({"error": "bad numeric setting"}, status=400)
    if form.get("random_seed") == "on" or not str(form.get("seed", "")).strip():
        seed = random.randrange(2 ** 32)
    else:
        try:
            seed = int(form.get("seed")) & 0xffffffff
        except ValueError:
            return web.json_response({"error": "bad seed"}, status=400)

    command = [str(H3_BIN), "-d", str(MODEL_DIR), "-p", prompt,
               "-o", str(OUTPUTS / (job_id + ".mp4")),
               "--width", str(width), "--height", str(height),
               "--seconds", str(seconds), "--steps", str(steps),
               "--seed", str(seed)] + PRESETS[preset]["args"]
    refs = []
    single = {"first_frame": "--first-frame", "last_frame": "--last-frame",
              "ref_audio": "--ref-audio"}
    for name, flag in single.items():
        field = form.get(name)
        if getattr(field, "file", None):
            path = await save_upload(field, job_id, name)
            command += [flag, str(path)]
            refs.append(name.replace("_", " "))
    video_field = form.get("ref_video")
    if getattr(video_field, "file", None):
        path = await save_upload(video_field, job_id, "ref_video")
        flag = "--ref-silent-video" if form.get("ref_video_mute") == "on" else "--ref-video"
        command += [flag, str(path)]
        refs.append("ref video" + (" (muted)" if flag == "--ref-silent-video" else ""))
    for index, field in enumerate(form.getall("ref_images", [])):
        if getattr(field, "file", None):
            path = await save_upload(field, job_id, f"ref_image{index}")
            command += ["--ref-image", str(path)]
            refs.append(f"ref image {index + 1}")
    extra = (form.get("extra_args") or "").strip()
    if extra:
        try:
            command += shlex.split(extra)
        except ValueError as error:
            return web.json_response({"error": f"extra args: {error}"}, status=400)

    job = {
        "id": job_id, "created": now(), "status": "queued", "phase": "queued",
        "percent": 0, "step": 0, "steps": steps, "prompt": prompt,
        "settings": {"preset": preset, "width": width, "height": height,
                     "seconds": seconds, "steps": steps, "seed": seed,
                     "extra_args": extra},
        "refs": refs, "error": None, "file": None, "elapsed": 0, "eta": None,
        "command": command, "env": dict(PRESETS[preset]["env"]), "cancel": False,
    }
    jobs[job_id] = job
    await queue.put(job_id)
    return web.json_response({"id": job_id})


async def run_job(job):
    job["status"] = "running"
    job["phase"] = "starting"
    started = now()
    env = os.environ.copy()
    env.update(job["env"])
    env.setdefault("H3_PROFILE", "1")
    process = await asyncio.create_subprocess_exec(
        *job["command"], cwd=str(H3_DIR), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    tail = []
    buffer = b""
    denoise_started = None
    try:
        while True:
            if job["cancel"]:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except asyncio.TimeoutError:
                    process.kill()
                job["status"] = "cancelled"
                job["phase"] = "cancelled"
                return
            try:
                chunk = await asyncio.wait_for(process.stdout.read(256), 0.5)
            except asyncio.TimeoutError:
                job["elapsed"] = round(now() - started, 1)
                continue
            if not chunk:
                break
            buffer += chunk
            while True:
                cut = [i for i in (buffer.find(b"\r"), buffer.find(b"\n")) if i >= 0]
                if not cut:
                    break
                line, buffer = buffer[:min(cut)], buffer[min(cut) + 1:]
                stripped = line.strip()
                if not stripped:
                    continue
                match = _PROGRESS.match(stripped)
                if match:
                    name = match.group(1).decode()
                    done, total = int(match.group(2)), int(match.group(3))
                    for phase_name, label, start, end in PHASES:
                        if phase_name == name:
                            frac = done / total if total else 0
                            job["phase"] = label
                            job["percent"] = round(start + (end - start) * frac, 1)
                            break
                    if name == "denoise":
                        job["step"], job["steps"] = done, total
                        if done and denoise_started is None:
                            denoise_started = now()
                        if done and denoise_started and done < total:
                            pace = (now() - denoise_started) / done
                            tail_estimate = 15 + 5.5 * job["settings"]["seconds"]
                            job["eta"] = round(pace * (total - done) + tail_estimate)
                        elif done >= total:
                            job["eta"] = round(15 + 5.5 * job["settings"]["seconds"])
                else:
                    text = stripped.decode(errors="replace")
                    tail.append(text)
                    del tail[:-30]
                job["elapsed"] = round(now() - started, 1)
        await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
    job["elapsed"] = round(now() - started, 1)
    output = OUTPUTS / (job["id"] + ".mp4")
    if process.returncode == 0 and output.exists():
        job["status"] = "done"
        job["phase"] = "done"
        job["percent"] = 100
        job["file"] = output.name
        job["eta"] = None
        meta = {k: job[k] for k in ("id", "created", "status", "prompt",
                                    "settings", "refs", "file", "elapsed")}
        (OUTPUTS / (job["id"] + ".json")).write_text(json.dumps(meta, indent=1))
    else:
        job["status"] = "error"
        job["phase"] = "error"
        job["error"] = "\n".join(tail[-8:]) or f"h3 exited with {process.returncode}"


async def worker(app):
    while True:
        job_id = await queue.get()
        job = jobs.get(job_id)
        if not job or job["cancel"]:
            if job:
                job["status"] = "cancelled"
                job["phase"] = "cancelled"
            continue
        try:
            await run_job(job)
        except Exception as error:  # noqa: BLE001 - surface anything to the UI
            job["status"] = "error"
            job["error"] = str(error)


async def handle_jobs(request):
    ordered = sorted(jobs.values(), key=lambda j: j.get("created", 0), reverse=True)
    return web.json_response({"caps": caps(), "jobs": [public(j) for j in ordered[:100]]})


async def handle_cancel(request):
    job = jobs.get(request.match_info["job_id"])
    if not job:
        return web.json_response({"error": "unknown job"}, status=404)
    job["cancel"] = True
    return web.json_response({"ok": True})


async def handle_index(request):
    return web.FileResponse(STATIC / "index.html")


async def on_startup(app):
    app["worker"] = asyncio.create_task(worker(app))


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    load_history()
    app = web.Application(client_max_size=2 * 1024 ** 3)
    app.router.add_get("/", handle_index)
    app.router.add_post("/api/generate", handle_generate)
    app.router.add_get("/api/jobs", handle_jobs)
    app.router.add_post("/api/jobs/{job_id}/cancel", handle_cancel)
    app.router.add_static("/media/", OUTPUTS, show_index=False)
    app.on_startup.append(on_startup)
    print(f"h3 web UI on http://0.0.0.0:{PORT}  (engine {H3_BIN}, model {MODEL_DIR})")
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)


if __name__ == "__main__":
    main()
