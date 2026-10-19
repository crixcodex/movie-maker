"""Single-file ComfyUI movie studio.

Install: pip install gradio requests
Also install ffmpeg and ffprobe on PATH. Export a working five-second,
image-conditioned video workflow from ComfyUI using Export (API).

Required environment configuration (node IDs are workflow-specific):
  COMFY_WORKFLOW=/workspace/workflow_api.json
  COMFY_PROMPT_NODE=6
  COMFY_IMAGE_NODE=10
  COMFY_VIDEO_NODE=42  # output node exposing videos/gifs/images in history
Optional: COMFY_SEED_NODE, COMFY_SEED_INPUT (default seed),
  COMFY_PROMPT_INPUT (text), COMFY_IMAGE_INPUT (image),
  COMFY_OUTPUT_DIR (/workspace/ComfyUI/output), STUDIO_DIR (/workspace),
  COMFY_URL (http://127.0.0.1:8188), SHOT_TIMEOUT (3600 seconds).

The workflow itself must produce five-second clips. Storyboard expansion is a
deterministic placeholder, not an LLM, and does not guarantee story continuity.
Shutdown is opt-in; install/configure the vastai CLI and set CONTAINER_ID first.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque

import gradio as gr
import requests


BASE_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
ROOT = Path(os.getenv("STUDIO_DIR", "/workspace")).resolve()
COMFY_OUTPUT = Path(os.getenv("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output")).resolve()
RUN_LOCK = threading.Lock()
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def command(args: list[str], timeout: int = 300) -> str:
    """Never interpolate user prompts or file paths into a shell command."""
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-4000:]}")
    return result.stdout


def duration(path: Path) -> float:
    value = float(command(["ffprobe", "-v", "error", "-show_entries",
                           "format=duration", "-of", "default=nw=1:nk=1", str(path)]).strip())
    if not 0 < value < float("inf"):
        raise RuntimeError(f"Invalid video duration: {path}")
    return value


def load_workflow() -> dict:
    path = Path(os.getenv("COMFY_WORKFLOW", "/workspace/workflow_api.json"))
    with path.open(encoding="utf-8") as handle:
        workflow = json.load(handle)
    if not isinstance(workflow, dict) or "nodes" in workflow:
        raise ValueError("Export the workflow in API format, not the UI graph format.")
    for variable, input_variable, default in [
        ("COMFY_PROMPT_NODE", "COMFY_PROMPT_INPUT", "text"),
        ("COMFY_IMAGE_NODE", "COMFY_IMAGE_INPUT", "image"),
    ]:
        node = os.getenv(variable, "")
        field = os.getenv(input_variable, default)
        if node not in workflow or field not in workflow[node].get("inputs", {}):
            raise ValueError(f"Configure {variable} and {input_variable} for your workflow.")
    if os.getenv("COMFY_VIDEO_NODE", "") not in workflow:
        raise ValueError("Set COMFY_VIDEO_NODE to the video output node ID.")
    seed_node = os.getenv("COMFY_SEED_NODE")
    if seed_node and os.getenv("COMFY_SEED_INPUT", "seed") not in workflow.get(seed_node, {}).get("inputs", {}):
        raise ValueError("COMFY_SEED_NODE / COMFY_SEED_INPUT does not exist.")
    return workflow


def storyboard(plot: str, count: int) -> list[str]:
    """One nonempty line per shot; a single paragraph uses test expansion."""
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
             for line in plot.splitlines() if line.strip()]
    if len(lines) > 1:
        if len(lines) != count:
            raise ValueError(f"Storyboard has {len(lines)} lines; set shot count to match.")
        return lines
    beats = ["establish the setting", "introduce the character's objective",
             "follow the character into the scene", "reveal a new obstacle",
             "show the character reacting", "advance the action",
             "reveal a consequence", "build toward a turning point",
             "resolve the immediate conflict", "close on a cinematic final image"]
    return [f"{plot.strip()}\nShot {i + 1}/{count}, five seconds: "
            f"{beats[min(9, i * 10 // count)]}. "
            "Maintain the reference character's identity, costume, and visual style. "
            "Continuous cinematic motion, coherent scene geography."
            for i in range(count)]


def api(session: requests.Session, method: str, route: str, **kwargs):
    response = session.request(method, BASE_URL + route, timeout=(10, 60), **kwargs)
    if not response.ok:
        raise RuntimeError(f"ComfyUI {route}: HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def output_video(history: dict) -> Path:
    """Resolve only the configured node's saved video, never the newest disk file."""
    outputs = history.get("outputs", {}).get(os.environ["COMFY_VIDEO_NODE"], {})
    candidates = []
    for key in ("videos", "gifs", "images"):
        for item in outputs.get(key, []):
            if not isinstance(item, dict) or item.get("type", "output") != "output":
                continue
            filename = item.get("filename", "")
            if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            path = (COMFY_OUTPUT / item.get("subfolder", "") / filename).resolve()
            if not path.is_relative_to(COMFY_OUTPUT):
                raise ValueError("ComfyUI returned an output outside COMFY_OUTPUT_DIR.")
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one saved video from COMFY_VIDEO_NODE; "
                           f"found {len(candidates)}. Check the output node/history schema.")
    path = candidates[0]
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Completed output is not available locally: {path}")
    return path


def normalize(source: Path, target: Path):
    """Uniform silent H.264 clips make concat reliable across changing outputs.

    This studio intentionally generates silent video; add a soundtrack in post.
    Normalization is a one-time encode; cumulative assembly uses stream copy.
    """
    seconds = duration(source)
    if abs(seconds - 5) > 0.25:
        raise RuntimeError(f"Workflow produced {seconds:.2f}s; configure five-second output.")
    command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-map", "0:v:0", "-an", "-vf",
             "scale=1280:720:force_original_aspect_ratio=decrease,"
             "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24",
             "-t", "5", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-video_track_timescale", "12288",
             "-movflags", "+faststart", str(target)], timeout=600)


def stitch(folder: Path, expected: float) -> Path:
    temporary = folder / "master.pending.mp4"
    master = folder / "current_movie_stream.mp4"
    command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
             "-safe", "1", "-i", str(folder / "filelist.txt"), "-c", "copy",
             "-movflags", "+faststart", str(temporary)], timeout=600)
    actual = duration(temporary)
    if abs(actual - expected) > max(0.5, expected * 0.001):
        raise RuntimeError(f"Master duration mismatch: {actual:.2f}s vs {expected:.2f}s.")
    os.replace(temporary, master)  # Readers never see a half-written master.
    return master


def schedule_shutdown(folder: Path):
    """Run only after verified final assembly, in a detached POSIX shell."""
    instance = os.getenv("CONTAINER_ID", "")
    if os.name != "posix" or not re.fullmatch(r"[0-9]+", instance):
        raise RuntimeError("Shutdown requires Linux and a numeric CONTAINER_ID.")
    if not shutil.which("vastai"):
        raise RuntimeError("vastai CLI is missing; shutdown was not scheduled.")
    # Equivalent to sleep 10 && vastai stop instance $CONTAINER_ID. Validate the
    # ID and quote the log path; background the group so Gradio can finish.
    log = shlex.quote(str(folder / "shutdown.log"))
    result = os.system(f"nohup sh -c 'sleep 10 && vastai stop instance {instance}' "
                       f"> {log} 2>&1 < /dev/null &")
    if result:
        raise RuntimeError(f"Could not schedule shutdown (shell status {result}).")


def worker(plot: str, image: str, count: int, shutdown: bool, events: queue.Queue):
    """Independent worker continues even if the browser disconnects."""
    def emit(message: str, clip=None, final=None):
        events.put((message, clip, final))

    try:
        for binary in ("ffmpeg", "ffprobe"):
            if not shutil.which(binary):
                raise RuntimeError(f"Install {binary} and add it to PATH.")
        workflow = load_workflow()
        shots = storyboard(plot, count)
        if shutdown and (os.name != "posix" or not re.fullmatch(r"[0-9]+", os.getenv("CONTAINER_ID", ""))
                         or not shutil.which("vastai")):
            raise ValueError("Auto-shutdown needs Linux, vastai CLI, and numeric CONTAINER_ID.")
        folder = ROOT / "movie_runs" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        (folder / "storyboard.json").write_text(json.dumps(shots, indent=2), encoding="utf-8")
        emit(f"Run directory: {folder}")
        with requests.Session() as session:
            with open(image, "rb") as handle:
                uploaded = api(session, "POST", "/upload/image", files={
                    "image": (f"studio_{uuid.uuid4().hex}{Path(image).suffix}", handle)},
                    data={"type": "input", "overwrite": "false"})
            reference = "/".join(filter(None, [uploaded.get("subfolder"), uploaded["name"]]))
            total_duration = 0.0
            for index, prompt in enumerate(shots, 1):
                graph = copy.deepcopy(workflow)
                graph[os.environ["COMFY_PROMPT_NODE"]]["inputs"][os.getenv("COMFY_PROMPT_INPUT", "text")] = prompt
                graph[os.environ["COMFY_IMAGE_NODE"]]["inputs"][os.getenv("COMFY_IMAGE_INPUT", "image")] = reference
                if os.getenv("COMFY_SEED_NODE"):
                    graph[os.environ["COMFY_SEED_NODE"]]["inputs"][os.getenv("COMFY_SEED_INPUT", "seed")] = uuid.uuid4().int % (2**63)
                submitted = api(session, "POST", "/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
                if submitted.get("node_errors") or not submitted.get("prompt_id"):
                    raise RuntimeError(f"Workflow validation failed: {submitted}")
                prompt_id = submitted["prompt_id"]
                emit(f"Rendering Shot {index}/{count}... (prompt {prompt_id})")
                deadline = time.monotonic() + float(os.getenv("SHOT_TIMEOUT", "3600"))
                next_notice = time.monotonic() + 15
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Shot {index} timed out; ComfyUI may still be rendering {prompt_id}.")
                    try:
                        history = api(session, "GET", f"/history/{prompt_id}").get(prompt_id)
                    except requests.RequestException as exc:
                        emit(f"Polling connection interrupted; retrying: {exc}")
                        time.sleep(3)
                        continue
                    if history:
                        status = history.get("status", {})
                        if status.get("status_str") == "error" or any(
                            msg[0] in {"execution_error", "execution_interrupted"}
                            for msg in status.get("messages", []) if isinstance(msg, (list, tuple)) and msg
                        ):
                            raise RuntimeError(f"ComfyUI execution failed: {status}")
                        if status.get("completed"):
                            break
                    if time.monotonic() >= next_notice:
                        emit(f"Rendering Shot {index}/{count}... waiting for ComfyUI")
                        next_notice = time.monotonic() + 15
                    time.sleep(2)
                source = output_video(history)
                emit(f"Shot {index}/{count} saved: {source.name}; preparing preview...")
                clip = folder / f"shot_{index:04d}.mp4"
                normalize(source, clip)
                # Controlled relative filenames avoid concat quoting/injection issues.
                with (folder / "filelist.txt").open("a", encoding="utf-8") as handle:
                    handle.write(f"file '{clip.name}'\n")
                total_duration += duration(clip)
                master = stitch(folder, total_duration)
                transport = folder / f"shot_{index:04d}.ts"
                command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip),
                         "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(transport)])
                emit(f"Completed Shot {index}/{count}. Master: {total_duration:.1f}s", str(transport))
            final = folder / "final_movie.mp4"
            shutil.copy2(master, final)
            duration(final)
            emit(f"Movie complete: {final}", final=str(final))
            if shutdown:
                schedule_shutdown(folder)
                emit("GPU stop scheduled in 10 seconds. Download promptly; see shutdown.log for CLI results.")
    except Exception as exc:
        emit(f"ERROR: {type(exc).__name__}: {exc}")
        emit("Run stopped. Completed clips are preserved. No further shutdown action will be attempted.")
    finally:
        RUN_LOCK.release()
        events.put(None)


def generate(plot, image, count, shutdown):
    """Gradio runs queued generators off the request thread; work has its own thread."""
    if not plot or not plot.strip() or not image:
        raise gr.Error("Provide a plot/storyboard and character reference image.")
    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("A movie is already running on this studio instance.")
    events = queue.Queue()
    thread = threading.Thread(target=worker, args=(plot, image, int(count), shutdown, events), daemon=True)
    try:
        thread.start()
    except Exception:
        RUN_LOCK.release()
        raise
    lines = deque(maxlen=250)
    yield gr.skip(), "Starting Autopilot...", None
    while True:
        try:
            event = events.get(timeout=1)
        except queue.Empty:
            continue
        if event is None:
            break
        message, clip, final = event
        lines.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        yield clip if clip else gr.skip(), "\n".join(lines), final if final else gr.skip()


def build_app():
    with gr.Blocks(title="Autopilot Movie Studio") as app:
        gr.Markdown("# Autopilot Movie Studio\nSequential five-second shots · ComfyUI · Progressive playback")
        with gr.Row():
            with gr.Column(scale=2):
                plot = gr.Textbox(label="Movie Plot / Storyboard Prompt", lines=10,
                                  placeholder="Enter a plot, or one shot per line with matching shot count.")
                image = gr.Image(label="Character Reference Image", type="filepath", sources=["upload"])
                count = gr.Slider(1, 720, value=12, step=1, label="Shot count (720 × 5 seconds = 1 hour)")
                shutdown = gr.Checkbox(value=False, label="Stop Vast.ai instance 10 seconds after final assembly")
                start = gr.Button("Start Autopilot Generation", variant="primary")
                gr.Markdown("The saved workflow must output five-second video clips. "
                            "Preview is silent, 720p at 24 fps. Keep this page open for streaming. "
                            "Generation continues after a browser disconnect while this server stays running.")
            with gr.Column(scale=3):
                video = gr.Video(label="Live movie preview", streaming=True, autoplay=True,
                                 interactive=False)
                logs = gr.Textbox(label="Real-time console", lines=15, max_lines=20, interactive=False)
                final = gr.File(label="Final movie download", interactive=False)
        start.click(generate, [plot, image, count, shutdown], [video, logs, final],
                    concurrency_limit=1, concurrency_id="movie_gpu", trigger_mode="once")
    return app


if __name__ == "__main__":
    # Only expose this trusted local frontend deliberately; do not use share=True
    # on a GPU host with an enabled shutdown hook.
    build_app().queue(max_size=2).launch(server_name=os.getenv("STUDIO_HOST", "127.0.0.1"),
                                       server_port=int(os.getenv("STUDIO_PORT", "7860")))
