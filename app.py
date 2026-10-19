"""Autonomous AI Movie Studio -- one Python file, Python 3.11+.

Install:
  pip install gradio requests cryptography google-api-python-client google-auth-httplib2 google-auth-oauthlib dropbox
  # Install ffmpeg/ffprobe separately; install/configure vastai for teardown.

Required configuration (IDs refer to your exported ComfyUI API workflow):
  COMFY_WORKFLOW=/workspace/workflow_api.json
  COMFY_PROMPT_NODE=6 COMFY_IMAGE_NODE=10 COMFY_VIDEO_NODE=42
Optional input names: COMFY_PROMPT_INPUT=text COMFY_IMAGE_INPUT=image
Optional seed mapping: COMFY_SEED_NODE=3 COMFY_SEED_INPUT=seed
Image input is optional. Auto-Character Mode additionally needs either:
  COMFY_ANCHOR_WORKFLOW=/workspace/anchor_workflow_api.json  # preferred T2I API graph
  COMFY_ANCHOR_PROMPT_NODE=6 COMFY_ANCHOR_SEED_NODE=3 COMFY_ANCHOR_OUTPUT_NODE=9
  (optional COMFY_ANCHOR_PROMPT_INPUT=text COMFY_ANCHOR_SEED_INPUT=seed)
or AUTO_CHARACTER_CHECKPOINT=an_installed_SD_or_SDXL_checkpoint.safetensors
  A CheckpointLoaderSimple in the video graph can also supply that checkpoint.
  No models are downloaded or guessed. Anchor workflow batch size must be 1.
  COMFY_INPUT_DIR=/workspace/ComfyUI/input
  CHARACTER_SEED=12345  # optional: otherwise one random seed is saved per movie
  All new shots reuse that seed and the same master image. Configure the video
  graph's image loader -> IP-Adapter/FaceID/I2V conditioning -> video output.
  Additional file-loader bindings (not adapter IMAGE sockets) can be configured:
  COMFY_IMAGE_BINDINGS='[{"node":"55","input":"image"}]'
  Same pixels/seed/settings reduce drift but cannot guarantee identical identity.
Consistency stack (all settings are snapshotted in the movie checkpoint):
  CONSISTENCY_CONFIG=/workspace/consistency.json  # optional JSON profile
  IDENTITY_ADAPTER_WEIGHT=0.75  # 0.7–0.8, tunable starting point, not a universal optimum
  CHARACTER_IDENTITY_TOKENS / MOVIE_STYLE_TOKENS / MOVIE_OBJECT_TOKENS override
  the default immutable prompt anchors. scene_id controls adjacent-shot chaining.
  Profile example (replace node IDs with actual exported API node IDs):
  {"identity_weight":0.75,
   "continuity":{"mode":"image","loader_node":"60","loader_input":"image",
                 "target_node":"61","target_input":"start_image","loader_output":0},
   "adapters":[], "motion_seed_span":0, "motion_seed_bindings":[],
   "motion_parameters":[], "parameter_bindings":[]}
  Existing IPAdapter nodes with weight inputs are detected automatically. Explicit
  adapter entries use {"node":"45","parameters":{"weight":0.75},"links":{}}.
  FaceID can additionally set weight_faceidv2=0.75 if that input exists.
  WanVideoAddStandInLatent has NO weight socket; keep its reference VAE connected
  to the master loader, and map ip_image_latent/freq_offset through links/parameters.
  Continuity uses a SEPARATE loader from master identity. Image mode extracts the
  last decoded frame, then the workflow re-encodes it for its I2V conditioning.
  Native mode uses mode=latent, loader_input=latent, target_node/target_input and
  output_node/output_key=latents. The model-specific graph must SELECT and SAVE
  its final temporal latent window, not its full video latent. This app copies
  that file unchanged; it never pretends an MP4 frame is an original latent.
  At a scene boundary, latent mode retains the template's initial latent link.
  Without a continuity mapping, temporal chaining is explicitly logged as off.
  Identity seeds stay fixed. Optional motion_seed_bindings name separate motion
  RNG inputs; small numeric seed deltas do not imply small noise differences.
  For controlled motion-noise amplitude, motion_parameters entries accept
  {"node":"70","input":"noise_aug_strength","base":0.02,"jitter":0.005}.
  All undeclared scalar sampler/model settings remain fixed to the API template.
Paths: STUDIO_DIR=/workspace COMFY_OUTPUT_DIR=/workspace/ComfyUI/output
API: COMFY_URL=http://127.0.0.1:8188
Local OpenAI-compatible LLM: LLM_URL=http://127.0.0.1:11434/v1
  LLM_MODEL=your-local-model  (unset means deterministic structural planning)
  LLM_FAILURE_POLICY=structural (or fail); LLM_BATCH_SIZE=12
Media: MOVIE_WIDTH=1920 MOVIE_HEIGHT=1080 MOVIE_FPS=24
  Workflow must generate five-second silent video clips. Audio is intentionally
  removed to keep every segment compatible. Upscaling cannot create detail.

Cloud backup is optional (provider None by default). Google sign-in setup:
  Enable the Google Drive API in your Google Cloud project, configure its OAuth
  consent screen, and create a Desktop app OAuth client. Download its client JSON.
  GOOGLE_OAUTH_CLIENT_FILE=/workspace/google_oauth_client.json
  GOOGLE_OAUTH_PORT=7862  # separate loopback-only callback listener, started lazily
  With the dashboard on a remote GPU, forward BOTH 7860 and 7862 through SSH.
  Redirect URI: http://127.0.0.1:7862/
  Sign in, then click Complete Google Sign-in. The optional authorization-code
  field accepts the callback's code or the full redirect URL (state is required).
  The obsolete urn:ietf:wg:oauth:2.0:oob flow is deliberately not used.
  Login tokens live in server session memory; starting a job encrypts a credential
  snapshot in its checkpoint so restart recovery can still upload unattended.
  No plaintext credentials.json token file is written.

Legacy programmatic/mounted credentials and Dropbox OAuth JSON remain supported:
  Google: authorized_user JSON (client_id, client_secret, refresh_token,
          token_uri=https://oauth2.googleapis.com/token)
  Dropbox: {"app_key":"...", "app_secret":"...", "refresh_token":"..."}
Supply credentials via the password field, or mount
  STUDIO_CREDENTIALS_FILE=/run/secrets/cloud.json (preferred). Password-field
  credentials are encrypted in state; STUDIO_SECRET_KEY is used if provided,
  otherwise a private persistent key is created at STUDIO_DIR/credentials.key.
Generate a key: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
Never expose the GUI publicly without authenticated HTTPS. Defaults to loopback.
Optional STUDIO_USER and STUDIO_PASSWORD enable Gradio authentication.

Run: python app.py
Tests (no third-party dependencies): python app.py --self-test

Operational contract:
* One process owns one workspace, guarded by an OS advisory file lock.
* state.json is atomically replaced and fsynced; filesystem must be persistent.
* Startup automatically resumes interrupted jobs, including assembly/upload.
* Submission ambiguity is reconciled by a persisted job/shot/attempt marker in
  ComfyUI history/queue. This is at-least-once, not distributed exactly-once.
* Three attempts per shot, exponential backoff; exhausted shots are skipped.
  A partial movie is explicitly marked partial in the manifest; cloud backup
  is attempted only when its provider, destination, and credentials are present.
* A verified local final movie is required for teardown. When backup is active,
  verified cloud delivery is also required; failures retain the movie for retry.
  Local-only jobs still stop the GPU ten seconds after final assembly.
* Upload SDK sessions retry in-process; a restart may re-upload bytes, but reuses
  the same Drive file ID / Dropbox path instead of creating duplicate movies.
* Rebuilding the cumulative MP4 copies all completed bytes each time (quadratic
  aggregate I/O); preview playback itself streams each clip only once.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import queue
import random
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlsplit


LOG = logging.getLogger("movie_studio")
ATTEMPTS = 3
TERMINAL = {"complete", "partial", "shutdown_scheduled"}
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/drive"]
IDENTITY_ANCHOR = ("same face shape, same age, same hairstyle, same outfit silhouette, same accessories, "
                   "same body proportions, matching the master reference character")
STYLE_ANCHOR = ("same color palette, same rendering medium, same texture language, consistent lens and grading, "
                "matching the style of the master reference character")
OBJECT_ANCHOR = ("preserve recurring objects' geometry, markings, materials, scale and colors; "
                 "no unexplained object substitutions")


def shot_prompt(shot: dict[str, Any]) -> str:
    """Keep immutable appearance instructions separate from scene/action text."""
    return (f"SCENE ACTION: {shot.get('action_prompt', shot['visual_prompt'])}\n"
            f"CAMERA: {shot['camera_movement']}. LIGHTING: {shot['lighting_cue']}.\n"
            f"LOCKED IDENTITY: {shot.get('identity_tokens', IDENTITY_ANCHOR)}.\n"
            f"LOCKED STYLE: {shot.get('style_tokens', STYLE_ANCHOR)}.\n"
            f"LOCKED OBJECTS: {shot.get('object_tokens', OBJECT_ANCHOR)}.")


def bind_shot_consistency(shots: list[dict[str, Any]], settings: dict[str, Any], master_seed: int) -> None:
    for shot in shots:
        # Do not silently change a checkpointed prompt or sampling seed on retry.
        shot.setdefault("action_prompt", shot["visual_prompt"])
        shot.setdefault("identity_tokens", settings.get("identity_tokens", IDENTITY_ANCHOR))
        shot.setdefault("style_tokens", settings.get("style_tokens", STYLE_ANCHOR))
        shot.setdefault("object_tokens", settings.get("object_tokens", OBJECT_ANCHOR))
        shot.setdefault("scene_id", f"cut_{shot['shot_id']}")
        shot.setdefault("seed", master_seed)
        span = settings.get("motion_seed_span", 0)
        # A small seed-number change is NOT a small perturbation of Gaussian noise.
        # This is an optional separate motion RNG stream, never the identity seed.
        offset = int.from_bytes(hashlib.sha256(f"{master_seed}:{shot['shot_id']}:motion".encode()).digest()[:8], "big")
        shot.setdefault("motion_seed", (master_seed + (offset % (span + 1))) % (2**63))


def consistency_config(graph: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    filename = os.getenv("CONSISTENCY_CONFIG")
    settings = json.loads(Path(filename).read_text(encoding="utf-8")) if filename else {}
    if not isinstance(settings, dict):
        raise ValueError("CONSISTENCY_CONFIG must contain a JSON object")
    settings.setdefault("identity_tokens", os.getenv("CHARACTER_IDENTITY_TOKENS", IDENTITY_ANCHOR))
    settings.setdefault("style_tokens", os.getenv("MOVIE_STYLE_TOKENS", STYLE_ANCHOR))
    settings.setdefault("object_tokens", os.getenv("MOVIE_OBJECT_TOKENS", OBJECT_ANCHOR))
    for key in ("identity_tokens", "style_tokens", "object_tokens"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Consistency {key} must be a nonempty string")
    weight = float(settings.get("identity_weight", os.getenv("IDENTITY_ADAPTER_WEIGHT", "0.75")))
    if not math.isfinite(weight) or not 0.7 <= weight <= 0.8:
        raise ValueError("IDENTITY_ADAPTER_WEIGHT must be within 0.7–0.8")
    adapters = settings.setdefault("adapters", [])
    if not isinstance(adapters, list):
        raise ValueError("Consistency adapters must be a list")
    configured = {item.get("node") for item in adapters if isinstance(item, dict)}
    for node_id, node in graph.items():
        inputs, kind = node.get("inputs", {}), node.get("class_type", "")
        if node_id not in configured and kind.startswith("IPAdapter") and type(inputs.get("weight")) in {int, float}:
            parameters = {"weight": weight}
            if "weight_faceidv2" in inputs:
                parameters["weight_faceidv2"] = weight
            adapters.append({"node": node_id, "parameters": parameters, "links": {}})
        elif node_id not in configured and kind == "WanVideoAddStandInLatent":
            # This node has ip_image_latent + freq_offset, NOT a weight input.
            adapters.append({"node": node_id, "parameters": {}, "links": {}})

    def validate_bindings(bindings: Any, numeric: bool = False) -> None:
        if not isinstance(bindings, list):
            raise ValueError("Consistency bindings must be JSON lists")
        for item in bindings:
            if (not isinstance(item, dict) or item.get("node") not in graph
                    or item.get("input") not in graph[item["node"]].get("inputs", {})):
                raise ValueError("Consistency binding references an unknown node/input")
            value = graph[item["node"]]["inputs"][item["input"]]
            if isinstance(value, list) or (numeric and type(value) not in {int, float}):
                raise ValueError("Scalar consistency binding cannot overwrite a graph link")

    for item in adapters:
        if not isinstance(item, dict) or item.get("node") not in graph:
            raise ValueError("Adapter mapping references an unknown node")
        node = graph[item["node"]]
        for field, value in item.get("parameters", {}).items():
            validate_bindings([{"node": item["node"], "input": field}])
            if not isinstance(value, (str, int, float, bool)) or (type(value) is float and not math.isfinite(value)):
                raise ValueError("Adapter parameters must be finite scalar values")
            if field in {"weight", "weight_faceidv2"} and (type(value) not in {int, float} or not 0.7 <= value <= 0.8):
                raise ValueError("Identity adapter weights must be within 0.7–0.8")
        for field, link in item.get("links", {}).items():
            if (field not in node.get("inputs", {}) or not isinstance(link, list) or len(link) != 2
                    or link[0] not in graph or type(link[1]) is not int or link[1] < 0):
                raise ValueError("Adapter link must reference an existing compatible workflow output")
    settings.setdefault("parameter_bindings", [])
    validate_bindings(settings["parameter_bindings"])
    for item in settings["parameter_bindings"]:
        if "value" not in item or not isinstance(item["value"], (str, int, float, bool)):
            raise ValueError("Locked parameter binding requires a scalar value")
        if type(item["value"]) is float and not math.isfinite(item["value"]):
            raise ValueError("Locked parameters must be finite")
    settings.setdefault("motion_seed_bindings", [])
    validate_bindings(settings["motion_seed_bindings"], numeric=True)
    settings["motion_seed_span"] = int(settings.get("motion_seed_span", 0))
    if not 0 <= settings["motion_seed_span"] <= 1024:
        raise ValueError("motion_seed_span must be between 0 and 1024")
    settings.setdefault("motion_parameters", [])
    validate_bindings(settings["motion_parameters"], numeric=True)
    for item in settings["motion_parameters"]:
        if (not all(type(item.get(key)) in {int, float} and math.isfinite(item[key]) for key in ("base", "jitter"))
                or not 0 <= item["jitter"] <= 0.05 or item["base"] < item["jitter"]):
            raise ValueError("Motion-noise jitter must be finite, nonnegative, and <=0.05")
    continuity = settings.setdefault("continuity", {})
    if not isinstance(continuity, dict):
        raise ValueError("Continuity settings must be a JSON object")
    if not continuity:
        # Safe discovery only: a distinct LoadImage already feeding an existing
        # start_image socket. Never repurpose the immutable master loader.
        excluded = {config["image_node"]} | {item["node"] for item in config.get("image_bindings", [])}
        candidates = set()
        for item in graph.values():
            value = item.get("inputs", {}).get("start_image")
            pending = [value[0]] if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) else []
            seen = set()
            while pending:
                node = pending.pop()
                if node in seen or node not in graph:
                    continue
                seen.add(node)
                inputs = graph[node].get("inputs", {})
                if node not in excluded and graph[node].get("class_type") == "LoadImage" and isinstance(inputs.get("image"), str):
                    candidates.add(node)
                for link in inputs.values():
                    if isinstance(link, list) and len(link) == 2 and isinstance(link[0], str):
                        pending.append(link[0])
        if len(candidates) == 1:
            continuity.update(mode="image", loader_node=candidates.pop(), loader_input="image")
    continuity.setdefault("mode", "image" if continuity.get("loader_node") else "off")
    if continuity["mode"] not in {"off", "image", "latent"}:
        raise ValueError("Continuity mode must be off, image, or latent")
    if continuity["mode"] != "off":
        loader = continuity.get("loader_node")
        field = continuity.setdefault("loader_input", "image" if continuity["mode"] == "image" else "latent")
        if loader == config["image_node"] or any(b["node"] == loader for b in config.get("image_bindings", [])):
            raise ValueError("Continuity loader must be separate from immutable master-image loaders")
        validate_bindings([{"node": loader, "input": field}])
        if not isinstance(graph[loader]["inputs"][field], str):
            raise ValueError("Continuity loader requires a filename string input")
        if continuity.get("target_node"):
            target = graph.get(continuity["target_node"], {}).get("inputs", {})
            if continuity.get("target_input") not in target:
                raise ValueError("Continuity target must name an existing image/latent input")
        if continuity["mode"] == "latent" and (not continuity.get("target_node") or continuity.get("output_node") not in graph):
            raise ValueError("Latent mode requires target_node/input and a saved final-latent output_node")
        continuity.setdefault("loader_output", 0)
        if type(continuity["loader_output"]) is not int or continuity["loader_output"] < 0:
            raise ValueError("Continuity loader_output must be a nonnegative index")
    # Snapshot all remaining scalar model/sampler parameters, including LoRAs,
    # CFG, steps, scheduler and denoise. Only declared motion bindings may vary.
    settings["scalar_snapshot"] = {node: {key: value for key, value in item.get("inputs", {}).items()
                                        if isinstance(value, (str, int, float, bool))}
                                   for node, item in graph.items()}
    return settings


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        if private:
            os.chmod(temporary, 0o600)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode())


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def dropbox_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            result.update(hashlib.sha256(block).digest())
    return result.hexdigest()


def retry(operation: Callable[[], Any], label: str,
          notice: Callable[[str], None], attempts: int = ATTEMPTS) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            # Avoid leaking tokens, resumable upload URLs, or provider bodies.
            notice(f"{label}: attempt {attempt}/{attempts} failed ({type(exc).__name__}).")
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt + random.uniform(0, 0.5))


def run(args: list[str], timeout: int = 900) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        # These subprocesses only handle local media paths, never credentials.
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-1800:]}")
    return result.stdout


def probe(path: Path) -> float:
    seconds = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=nw=1:nk=1", str(path)]).strip())
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid media duration")
    return seconds


def normalize(source: Path, target: Path, media: dict[str, int]) -> None:
    seconds = probe(source)
    if abs(seconds - 5) > 0.25:
        raise ValueError(f"Workflow generated {seconds:.2f}s; expected five seconds")
    width, height, fps = media["width"], media["height"], media["fps"]
    temporary = target.with_suffix(".pending.mp4")
    # Pad slightly short source clips so every successful shot is exactly 5s.
    filters = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
               "tpad=stop_mode=clone:stop_duration=0.3")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-map", "0:v:0", "-an", "-vf", filters, "-t", "5", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
         "-video_track_timescale", "90000", "-movflags", "+faststart", str(temporary)])
    if abs(probe(temporary) - 5) > 0.1:
        raise RuntimeError("Normalized shot duration is invalid")
    os.replace(temporary, target)
    with target.open("r+b") as handle:
        os.fsync(handle.fileno())
    fsync_directory(target.parent)


def preview(clip: Path) -> Path:
    target = clip.with_suffix(".ts")
    pending = clip.with_suffix(".pending.ts")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip),
         "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(pending)])
    os.replace(pending, target)
    return target


class WorkspaceLock:
    """Kernel lock is released automatically after process death."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.handle = (root / "studio.lock").open("a+b")
        self.handle.write(b"0")
        self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another studio process owns this workspace") from None

    def close(self) -> None:
        self.handle.close()


class StateStore:
    def __init__(self, root: Path):
        self.path = root / "state.json"
        self.lock = threading.RLock()
        self.data: dict[str, Any] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("schema") != 1:
                raise ValueError("Unsupported state schema; preserve state.json for recovery")

    def save(self) -> None:
        with self.lock:
            self.data["updated_at"] = utc()
            shots = self.data.get("shots", [])
            self.data["completed_shot_ids"] = [s["shot_id"] for s in shots if s["status"] == "completed"]
            self.data["pending_shot_ids"] = [s["shot_id"] for s in shots if s["status"] not in {"completed", "failed"}]
            self.data["failed_shot_ids"] = [s["shot_id"] for s in shots if s["status"] == "failed"]
            atomic_json(self.path, self.data)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.data)


def structural_shots(plot: str, count: int) -> list[dict[str, Any]]:
    """Deterministic fallback; handles explicit newline-separated storyboard beats."""
    beats = [line.strip() for line in plot.splitlines() if line.strip()] or [plot]
    phases = ["establish the world", "introduce the objective", "develop the conflict",
              "raise the stakes", "reveal the turning point", "resolve the story"]
    shots = []
    for index in range(count):
        beat_index = min(len(beats) - 1, index * len(beats) // count)
        beat = beats[beat_index]
        shots.append({"shot_id": index + 1,
                      "scene_id": f"scene_{beat_index + 1}",
                      "visual_prompt": f"{beat}. Story beat: {phases[min(5, index * 6 // count)]}. "
                                       f"Sequential moment {index + 1} of {count}. Preserve character identity and costume.",
                      "camera_movement": ["slow dolly in", "tracking shot", "steady medium shot"][index % 3],
                      "lighting_cue": "consistent cinematic lighting matching the setting",
                      "duration_seconds": 5})
    return shots


def plan_shots(plot: str, count: int, notice: Callable[[str], None]) -> list[dict[str, Any]]:
    model = os.getenv("LLM_MODEL")
    if not model:
        notice("Using deterministic structural storyboard planner.")
        return structural_shots(plot, count)
    import requests
    url = os.getenv("LLM_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    batch = max(1, min(30, int(os.getenv("LLM_BATCH_SIZE", "12"))))
    shots: list[dict[str, Any]] = []
    try:
        for start in range(0, count, batch):
            size = min(batch, count - start)
            notice(f"Planning shots {start + 1}–{start + size}/{count} with local LLM...")

            def request_batch() -> list[dict[str, Any]]:
                response = requests.post(url + "/chat/completions", timeout=(10, 240), json={
                    "model": model, "temperature": 0.4,
                    "messages": [{"role": "system", "content":
                        'You are a film director. Return only JSON {"shots": [...]}. '
                        "Each shot has visual_prompt, camera_movement, lighting_cue (nonempty strings), "
                        "duration_seconds=5, and scene_id (stable string shared only by consecutive shots "
                        "in the same time/place). Maintain the same protagonist, costume, recurring objects, and style. "
                        "Move the plot toward its ending over the requested total shot count."},
                        {"role": "user", "content": json.dumps({"plot": plot, "total_shots": count,
                            "start_shot": start + 1, "required_count": size, "previous_shots": shots[-4:]})}]})
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
                values = json.loads(content)["shots"]
                if not isinstance(values, list) or len(values) != size:
                    raise ValueError("LLM returned an incorrect shot count")
                result = []
                for offset, value in enumerate(values):
                    if not isinstance(value, dict) or any(
                        not isinstance(value.get(key), str) or not value[key].strip() or len(value[key]) > 12000
                        for key in ("visual_prompt", "camera_movement", "lighting_cue")
                    ) or value.get("duration_seconds") != 5:
                        raise ValueError("Invalid LLM shot schema")
                    scene_id = value.get("scene_id", f"cut_{start + offset + 1}")
                    if not isinstance(scene_id, str) or not scene_id.strip() or len(scene_id) > 128:
                        raise ValueError("LLM scene_id must be a nonempty string of at most 128 characters")
                    result.append({key: value[key] for key in
                                   ("visual_prompt", "camera_movement", "lighting_cue", "duration_seconds")}
                                  | {"shot_id": start + offset + 1,
                                     "scene_id": scene_id.strip()})
                return result

            shots.extend(retry(request_batch, "Storyboard planning", notice))
        return shots
    except Exception:
        if os.getenv("LLM_FAILURE_POLICY", "structural") == "fail":
            raise
        notice("Local LLM unavailable or invalid output; switching to structural planning.")
        return structural_shots(plot, count)


class ComfyClient:
    def __init__(self, config: dict[str, Any]):
        import requests
        self.session = requests.Session()
        self.config = config
        self.url = config["url"].rstrip("/")

    def request(self, method: str, route: str, **kwargs: Any) -> Any:
        response = self.session.request(method, self.url + route, timeout=(10, 60), **kwargs)
        if not response.ok:
            # API bodies may include prompts/paths; do not print credential-bearing URLs.
            raise RuntimeError(f"ComfyUI {route.split('?')[0]} HTTP {response.status_code}")
        return response.json()

    def upload(self, image: Path, job_id: str) -> str:
        with image.open("rb") as handle:
            result = self.request("POST", "/upload/image", files={
                "image": (f"studio_{job_id}{image.suffix}", handle)},
                data={"type": "input", "overwrite": "true"})
        return "/".join(filter(None, [result.get("subfolder"), result["name"]]))

    @staticmethod
    def marker(entry: list[Any]) -> str | None:
        return entry[3].get("extra_pnginfo", {}).get("studio_marker") if len(entry) > 3 and isinstance(entry[3], dict) else None

    def reconcile(self, marker: str) -> str | None:
        """Recover a POST whose server response or subsequent checkpoint was lost."""
        for prompt_id, item in self.request("GET", "/history").items():
            if self.marker(item.get("prompt", [])) == marker:
                return prompt_id
        pending = self.request("GET", "/queue")
        for item in pending.get("queue_running", []) + pending.get("queue_pending", []):
            if self.marker(item) == marker:
                return str(item[1])
        # Close a queue->history race between the two queries.
        for prompt_id, item in self.request("GET", "/history").items():
            if self.marker(item.get("prompt", [])) == marker:
                return prompt_id
        return None

    def submit(self, workflow: dict[str, Any], shot: dict[str, Any], reference: str, marker: str) -> str:
        graph = copy.deepcopy(workflow)
        settings = self.config.get("consistency", {})
        for node, parameters in settings.get("scalar_snapshot", {}).items():
            graph[node]["inputs"].update(copy.deepcopy(parameters))
        for item in settings.get("parameter_bindings", []):
            graph[item["node"]]["inputs"][item["input"]] = item["value"]
        text = shot_prompt(shot)
        graph[self.config["prompt_node"]]["inputs"][self.config["prompt_input"]] = text
        graph[self.config["image_node"]]["inputs"][self.config["image_input"]] = reference
        for binding in self.config.get("image_bindings", []):
            graph[binding["node"]]["inputs"][binding["input"]] = reference
        if self.config["seed_node"]:
            graph[self.config["seed_node"]]["inputs"][self.config["seed_input"]] = shot["seed"]
        for binding in self.config.get("seed_bindings", []):
            graph[binding["node"]]["inputs"][binding["input"]] = shot["seed"]
        for item in settings.get("adapters", []):
            graph[item["node"]]["inputs"].update(copy.deepcopy(item.get("parameters", {})))
            graph[item["node"]]["inputs"].update(copy.deepcopy(item.get("links", {})))
        for item in settings.get("motion_seed_bindings", []):
            graph[item["node"]]["inputs"][item["input"]] = shot.get("motion_seed", shot["seed"])
        for item in settings.get("motion_parameters", []):
            value = int.from_bytes(hashlib.sha256(f"{shot['seed']}:{shot['shot_id']}:{item['node']}:{item['input']}".encode()).digest()[:8], "big")
            graph[item["node"]]["inputs"][item["input"]] = item["base"] + item["jitter"] * (2 * value / (2**64 - 1) - 1)
        continuity = settings.get("continuity", {})
        filename = shot.get("continuity_input")
        if continuity.get("mode") == "image":
            graph[continuity["loader_node"]]["inputs"][continuity["loader_input"]] = filename or reference
        if filename and continuity.get("mode") == "latent":
            graph[continuity["loader_node"]]["inputs"][continuity["loader_input"]] = filename
        if continuity.get("target_node") and (continuity["mode"] == "image" or filename):
            graph[continuity["target_node"]]["inputs"][continuity["target_input"]] = [
                continuity["loader_node"], continuity["loader_output"]]
        return self.submit_graph(graph, marker)

    def submit_graph(self, graph: dict[str, Any], marker: str) -> str:
        result = self.request("POST", "/prompt", json={"prompt": graph, "client_id": marker,
            "extra_data": {"extra_pnginfo": {"studio_marker": marker}}})
        if result.get("node_errors") or not result.get("prompt_id"):
            raise ValueError("ComfyUI rejected workflow inputs; inspect workflow in ComfyUI")
        return result["prompt_id"]

    def wait(self, prompt_id: str, notice: Callable[[str], None], *,
             output_node: str | None = None, kind: str = "video") -> Path:
        timeout = float(os.getenv("SHOT_TIMEOUT", "7200"))
        deadline, heartbeat = time.monotonic() + timeout, 0.0
        failures = 0
        while time.monotonic() < deadline:
            try:
                record = self.request("GET", f"/history/{prompt_id}").get(prompt_id)
                failures = 0
            except Exception:
                failures += 1
                if failures >= ATTEMPTS:
                    raise
                time.sleep(2 ** failures)
                continue
            if record:
                status = record.get("status", {})
                messages = status.get("messages", [])
                if status.get("status_str") == "error" or any(
                    isinstance(msg, list) and msg and msg[0] in {"execution_error", "execution_interrupted"}
                    for msg in messages
                ):
                    raise ValueError("ComfyUI execution failed or was interrupted")
                if status.get("completed"):
                    return self.output_image(record, output_node) if kind == "image" else self.video(record)
            if time.monotonic() >= heartbeat:
                notice(f"Waiting for ComfyUI prompt {prompt_id}...")
                heartbeat = time.monotonic() + 20
            time.sleep(2)
        # Do not interrupt globally: another application may own the active job.
        raise TimeoutError("ComfyUI shot timed out; its original prompt may remain queued")

    def video(self, record: dict[str, Any]) -> Path:
        node = record.get("outputs", {}).get(self.config["video_node"], {})
        root = Path(self.config["output_dir"]).resolve()
        paths = set()
        for key in ("videos", "gifs", "images"):
            for item in node.get(key, []):
                if not isinstance(item, dict) or item.get("type", "output") != "output":
                    continue
                if Path(item.get("filename", "")).suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                    continue
                path = (root / item.get("subfolder", "") / item["filename"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Output escapes configured ComfyUI output directory")
                paths.add(path)
        if len(paths) != 1:
            raise ValueError("Configured video output node must report exactly one saved video")
        path = paths.pop()
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError("ComfyUI video is not available on the shared filesystem")
        return path

    def output_image(self, record: dict[str, Any], output_node: str | None) -> Path:
        root = Path(self.config["output_dir"]).resolve()
        items = record.get("outputs", {}).get(output_node, {}).get("images", [])
        paths = set()
        for item in items:
            if not isinstance(item, dict) or item.get("type", "output") != "output":
                continue
            path = (root / item.get("subfolder", "") / item.get("filename", "")).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Anchor output escapes ComfyUI output directory")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                paths.add(path)
        if len(paths) != 1:
            raise ValueError("Anchor workflow must save exactly one image; set batch_size=1")
        path = paths.pop()
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError("Generated anchor image is not available on the shared filesystem")
        return path


def load_api_workflow(filename: str, setting: str) -> dict[str, Any]:
    path = Path(filename).expanduser()
    if not path.is_file():
        raise ValueError(f"ComfyUI workflow not found: {path}. Export a tested workflow using ComfyUI's "
                         f"Export (API) / Save (API Format), upload that JSON to the GPU instance, and set "
                         f"{setting} to its full path before starting this app. Setting the variable does not create the file.")
    try:
        graph = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError(f"{path} is not valid UTF-8 JSON. Export the workflow again in ComfyUI API format.") from None
    if (not isinstance(graph, dict) or not graph or "nodes" in graph
            or any(not isinstance(node, dict) or not isinstance(node.get("inputs"), dict)
                   or not isinstance(node.get("class_type"), str) for node in graph.values())):
        raise ValueError(f"{path} is not a ComfyUI API workflow. Use Export (API) / Save (API Format); "
                         "the ordinary canvas JSON is a different format.")
    return graph


def workflow_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = {"url": os.getenv("COMFY_URL", "http://127.0.0.1:8188"),
              "output_dir": str(Path(os.getenv("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output")).resolve()),
              **{key: os.getenv(variable, default) for key, variable, default in [
                  ("prompt_node", "COMFY_PROMPT_NODE", ""), ("image_node", "COMFY_IMAGE_NODE", ""),
                  ("video_node", "COMFY_VIDEO_NODE", ""), ("seed_node", "COMFY_SEED_NODE", ""),
                  ("prompt_input", "COMFY_PROMPT_INPUT", "text"), ("image_input", "COMFY_IMAGE_INPUT", "image"),
                  ("seed_input", "COMFY_SEED_INPUT", "seed")]}}
    graph = load_api_workflow(os.getenv("COMFY_WORKFLOW", "/workspace/workflow_api.json"), "COMFY_WORKFLOW")
    for role in ("prompt", "image", "video", "seed"):
        node = config[role + "_node"]
        if role == "seed" and not node:
            continue
        if node not in graph:
            raise ValueError(f"Configure COMFY_{role.upper()}_NODE")
        if role != "video" and config[role + "_input"] not in graph[node].get("inputs", {}):
            raise ValueError(f"Configure COMFY_{role.upper()}_INPUT")
    config["input_dir"] = str(Path(os.getenv("COMFY_INPUT_DIR", str(Path(config["output_dir"]).parent / "input"))).resolve())
    # Bind file-loader inputs, not IMAGE-valued sockets on IP-Adapter nodes.
    # Adapter/identity nodes must already be connected to these loader outputs.
    config["image_bindings"] = json.loads(os.getenv("COMFY_IMAGE_BINDINGS", "[]"))
    if not isinstance(config["image_bindings"], list):
        raise ValueError("COMFY_IMAGE_BINDINGS must be a JSON list of node/input objects")
    for binding in config["image_bindings"]:
        if (not isinstance(binding, dict) or binding.get("node") not in graph
                or binding.get("input") not in graph[binding["node"]].get("inputs", {})):
            raise ValueError("Invalid COMFY_IMAGE_BINDINGS loader node/input mapping")
    config["seed_bindings"] = [{"node": node, "input": name} for node, item in graph.items()
                               for name, value in item.get("inputs", {}).items()
                               if name in {"seed", "noise_seed"} and type(value) is int]
    if not config["seed_node"] and not config["seed_bindings"]:
        raise ValueError("Video workflow must expose a literal seed/noise_seed, or configure COMFY_SEED_NODE / input")
    config["consistency"] = consistency_config(graph, config)
    for item in config["consistency"]["adapters"]:
        graph[item["node"]]["inputs"].update(copy.deepcopy(item.get("links", {})))
    continuity = config["consistency"]["continuity"]
    if continuity["mode"] == "image" and continuity.get("target_node"):
        graph[continuity["target_node"]]["inputs"][continuity["target_input"]] = [
            continuity["loader_node"], continuity["loader_output"]]
    # Verify each mapped file loader actually contributes to the saved video.
    reachable, pending = set(), [config["video_node"]]
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for value in graph[node].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph:
                pending.append(value[0])
    for binding in [{"node": config["image_node"], "input": config["image_input"]}] + config["image_bindings"]:
        if binding["node"] not in reachable:
            raise ValueError("A configured master-image loader is disconnected from the video output")
        if not isinstance(graph[binding["node"]]["inputs"][binding["input"]], str):
            raise ValueError("Map image filenames to file-loader string inputs, not adapter IMAGE sockets")
    if continuity["mode"] == "image" and continuity["loader_node"] not in reachable:
        raise ValueError("Continuity image loader must contribute to the saved video output")
    for adapter in config["consistency"]["adapters"]:
        if adapter["node"] not in reachable:
            raise ValueError("Configured identity adapter / stand-in node is disconnected from the video output")
        item = graph[adapter["node"]]
        field = "ip_image_latent" if item.get("class_type") == "WanVideoAddStandInLatent" else "image"
        if field not in item.get("inputs", {}):
            continue  # Precomputed-embedding adapters keep their exported graph contract.
        link = item["inputs"][field]
        upstream, pending = set(), [link[0]] if isinstance(link, list) and len(link) == 2 else []
        while pending:
            node = pending.pop()
            if node in upstream or node not in graph:
                continue
            upstream.add(node)
            for value in graph[node].get("inputs", {}).values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    pending.append(value[0])
        masters = {config["image_node"]} | {binding["node"] for binding in config["image_bindings"]}
        if not upstream.intersection(masters):
            raise ValueError("Adapter reference image/latent must be connected to an immutable master-image loader")
    return graph, config


def anchor_workflow_config(video_graph: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Use an exported T2I graph, or build a standard SD/SDXL bootstrap graph.

    Never fabricate an IP-Adapter graph for an incompatible video architecture.
    The video API template remains responsible for actual identity conditioning.
    """
    filename = os.getenv("COMFY_ANCHOR_WORKFLOW")
    if filename:
        graph = load_api_workflow(filename, "COMFY_ANCHOR_WORKFLOW")
    else:
        checkpoint = os.getenv("AUTO_CHARACTER_CHECKPOINT") or next(
            (node.get("inputs", {}).get("ckpt_name") for node in video_graph.values()
             if node.get("class_type") == "CheckpointLoaderSimple"), None)
        if not checkpoint:
            raise ValueError("Auto-Character Mode needs COMFY_ANCHOR_WORKFLOW (exported text-to-image API graph), "
                             "or AUTO_CHARACTER_CHECKPOINT naming an installed SD/SDXL checkpoint.")
        graph = {
            "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 24, "cfg": 6.5,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "multiple people, collage, text, watermark, blurry face",
                                                                  "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "studio_anchor"}},
        }
    config = {"prompt_node": os.getenv("COMFY_ANCHOR_PROMPT_NODE", "6"),
              "prompt_input": os.getenv("COMFY_ANCHOR_PROMPT_INPUT", "text"),
              "seed_node": os.getenv("COMFY_ANCHOR_SEED_NODE", "3"),
              "seed_input": os.getenv("COMFY_ANCHOR_SEED_INPUT", "seed"),
              "output_node": os.getenv("COMFY_ANCHOR_OUTPUT_NODE", "9")}
    for role in ("prompt", "seed", "output"):
        node = config[role + "_node"]
        if node not in graph or (role != "output" and config[role + "_input"] not in graph[node].get("inputs", {})):
            raise ValueError(f"Configure COMFY_ANCHOR_{role.upper()}_NODE / input for your anchor workflow")
    return graph, config


def save_anchor_png(source: Path, target: Path) -> None:
    from PIL import Image, ImageOps
    with Image.open(source) as image:
        image.load()
        if image.width < 64 or image.height < 64:
            raise ValueError("Generated master anchor is too small")
        pixels = ImageOps.exif_transpose(image).convert("RGB")
        encoded = io.BytesIO()
        pixels.save(encoded, format="PNG")
        atomic_bytes(target, encoded.getvalue())


class GoogleDriveOAuth:
    """Session-isolated Desktop OAuth with PKCE and a loopback callback.

    The callback never logs its URL/code. Its state must match an outstanding,
    unexpired attempt. Bare codes are accepted only after a validated callback;
    full redirect URLs can be pasted if the callback tunnel is unavailable.
    """

    TTL = 600
    CALLBACK_PATH = "/"

    def __init__(self, root: Path):
        self.root = root
        self.port = int(os.getenv("GOOGLE_OAUTH_PORT", "7862"))
        if not 1024 <= self.port <= 65535:
            raise ValueError("GOOGLE_OAUTH_PORT must be between 1024 and 65535")
        self.redirect_uri = f"http://127.0.0.1:{self.port}{self.CALLBACK_PATH}"
        self.lock = threading.RLock()
        self.pending: dict[str, dict[str, Any]] = {}
        self.tokens: dict[str, str] = {}
        self.server: ThreadingHTTPServer | None = None

    def _purge(self) -> None:
        expired = [state for state, item in self.pending.items()
                   if time.monotonic() - item["created"] > self.TTL]
        for state in expired:
            del self.pending[state]

    def forget(self, owner: str) -> None:
        with self.lock:
            self.tokens.pop(owner, None)
            for state in [state for state, item in self.pending.items() if item["owner"] == owner]:
                del self.pending[state]

    def serialized_credentials(self, owner: str) -> str:
        with self.lock:
            return self.tokens.get(owner, "")

    def _new_flow(self) -> Any:
        from google_auth_oauthlib.flow import Flow
        filename = Path(os.getenv("GOOGLE_OAUTH_CLIENT_FILE", str(self.root / "google_oauth_client.json")))
        if not filename.is_file():
            raise ValueError("Set GOOGLE_OAUTH_CLIENT_FILE to your downloaded Desktop app OAuth client JSON.")
        config = json.loads(filename.read_text(encoding="utf-8"))
        installed = config.get("installed", {})
        if not installed.get("client_id") or not installed.get("client_secret"):
            raise ValueError("Use a Google OAuth client of type Desktop app, not service-account JSON.")
        # Pin provider endpoints instead of trusting arbitrary URLs in a file.
        safe_config = {"installed": {"client_id": installed["client_id"],
            "client_secret": installed["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [self.redirect_uri]}}
        return Flow.from_client_config(safe_config, scopes=GOOGLE_SCOPES,
                                       redirect_uri=self.redirect_uri, autogenerate_code_verifier=True)

    def _ensure_server(self) -> None:
        if self.server:
            return
        manager = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass  # Request targets contain authorization codes: never log them.

            def do_GET(self) -> None:
                status, message, code = manager.capture_callback(self.path)
                page = ("<!doctype html><meta charset='utf-8'><title>Movie Studio Google Sign-in</title>"
                        f"<h2>Google Drive sign-in</h2><p>{html.escape(message)}</p>")
                if code:
                    page += ("<p>Return to Movie Studio and click <b>Complete Google Sign-in</b>. "
                             "The code field may be left empty. Optional code:</p>"
                             f"<pre>{html.escape(code)}</pre>")
                payload = page.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(payload)

        # Only reachable on the GPU's loopback interface / the user's SSH tunnel.
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, name="google-oauth-callback", daemon=True).start()

    def begin(self, owner: str) -> str:
        with self.lock:
            self._purge()
            self.tokens.pop(owner, None)
            for state in [state for state, item in self.pending.items() if item["owner"] == owner]:
                del self.pending[state]
            if len(self.pending) >= 128:
                raise ValueError("Too many pending sign-in attempts; try again shortly.")
            flow = self._new_flow()
            self._ensure_server()
            url, state = flow.authorization_url(access_type="offline", prompt="consent")
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.netloc != "accounts.google.com":
                raise ValueError("OAuth library returned an unexpected authorization endpoint")
            self.pending[state] = {"owner": owner, "flow": flow, "created": time.monotonic(),
                                   "code": None, "exchanging": False}
            return url

    def capture_callback(self, target: str) -> tuple[int, str, str | None]:
        if len(target) > 16384:
            return 400, "Invalid callback.", None
        parsed = urlsplit(target)
        if parsed.path != self.CALLBACK_PATH:
            return 404, "Not found.", None
        values = parse_qs(parsed.query)
        states, codes = values.get("state", []), values.get("code", [])
        with self.lock:
            self._purge()
            item = self.pending.get(states[0]) if len(states) == 1 else None
            if not item:
                return 400, "Sign-in state is invalid or expired. Start sign-in again.", None
            if values.get("error"):
                return 400, "Google permission was denied. Return to the studio and start sign-in again.", None
            if len(codes) != 1 or not codes[0] or not codes[0].isascii() or len(codes[0]) > 4096:
                return 400, "No valid authorization code received.", None
            if item["exchanging"] or (item["code"] and not secrets.compare_digest(item["code"], codes[0])):
                return 400, "Callback was already used. Start sign-in again.", None
            item["code"] = codes[0]
            return 200, "Google returned a valid authorization response.", codes[0]

    def finish(self, owner: str, pasted: str = "") -> None:
        with self.lock:
            self._purge()
            matches = [(state, item) for state, item in self.pending.items() if item["owner"] == owner]
            if len(matches) != 1:
                raise ValueError("No pending sign-in for this session. Click Sign in with Google Drive first.")
            state, item = matches[0]
            pasted = (pasted or "").strip()
            if pasted.startswith(("http://", "https://")):
                parsed, expected = urlsplit(pasted), urlsplit(self.redirect_uri)
                query = parse_qs(parsed.query)
                incoming_state = query.get("state", [])
                if ((parsed.scheme, parsed.netloc, parsed.path) != (expected.scheme, expected.netloc, expected.path)
                        or len(incoming_state) != 1 or not secrets.compare_digest(incoming_state[0], state)):
                    raise ValueError("Return URL does not match this session's sign-in attempt.")
                status, _, code = self.capture_callback(pasted)
                if status != 200:
                    raise ValueError("Invalid Google return URL. Start sign-in again.")
                pasted = code or ""
            if not item["code"]:
                raise ValueError("Awaiting the Google callback. Forward the callback port, or paste the full "
                                 "return URL from the browser address bar (including state).")
            if pasted and not secrets.compare_digest(pasted, item["code"]):
                raise ValueError("Authorization code does not match this session's callback.")
            if item["exchanging"]:
                raise ValueError("Sign-in exchange is already in progress.")
            item["exchanging"] = True
            flow, code = item["flow"], item["code"]
        try:
            # Pass code directly; no insecure-HTTP override or plaintext token file.
            flow.fetch_token(code=code, timeout=60)
            if not flow.credentials.refresh_token:
                raise ValueError("Google did not grant offline access. Start sign-in again and grant permission.")
            token_json = flow.credentials.to_json()
            with self.lock:
                if self.pending.get(state) is not item:
                    raise ValueError("Sign-in was cancelled or replaced. Try again.")
                self.tokens[owner] = token_json
                del self.pending[state]  # Codes are one-use and never persisted.
        except Exception:
            with self.lock:
                item["exchanging"] = False
            raise


def credential_spec(secret: str, root: Path | None = None) -> dict[str, str]:
    """Persist a mounted-secret path or encrypted JSON, never plaintext tokens."""
    filename = os.getenv("STUDIO_CREDENTIALS_FILE")
    if not secret.strip() and filename:
        return {"file": str(Path(filename).resolve())}
    if not secret.strip():
        raise ValueError("Provide refreshable OAuth JSON or STUDIO_CREDENTIALS_FILE")
    secret = secret.strip()
    payload = json.loads(secret) if secret.startswith("{") else {"access_token": secret}
    if not isinstance(payload, dict):
        raise ValueError("Credentials must be an access token or JSON object")
    from cryptography.fernet import Fernet
    key = os.getenv("STUDIO_SECRET_KEY")
    key_spec = {}
    if not key:
        if root is None:
            raise ValueError("No persistent credential-key directory configured")
        key_path = root / "credentials.key"
        if not key_path.exists():
            atomic_bytes(key_path, Fernet.generate_key())
        key = key_path.read_text(encoding="ascii").strip()
        key_spec = {"key_file": str(key_path.resolve())}
    return {"encrypted": Fernet(key.encode()).encrypt(json.dumps(payload).encode()).decode(), **key_spec}


def credentials(spec: dict[str, str]) -> dict[str, Any]:
    if "file" in spec:
        raw = Path(spec["file"]).read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw.startswith("{") else {"access_token": raw}
    from cryptography.fernet import Fernet
    key = (Path(spec["key_file"]).read_text(encoding="ascii").strip()
           if spec.get("key_file") else os.environ["STUDIO_SECRET_KEY"])
    return json.loads(Fernet(key.encode()).decrypt(spec["encrypted"].encode()))


def cloud_enabled(cloud: dict[str, Any]) -> bool:
    """Provider, destination, and actual credential source are all required."""
    return (cloud.get("provider") in {"Google Drive", "Dropbox"}
            and bool(str(cloud.get("destination", "")).strip())
            and bool(cloud.get("credentials")))


def cloud_client(provider: str, info: dict[str, Any]) -> Any:
    if provider == "Google Drive":
        import httplib2
        import google_auth_httplib2
        from google.oauth2.credentials import Credentials
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        # Fixed OAuth endpoint prevents credential JSON from redirecting secrets.
        info = dict(info, token_uri="https://oauth2.googleapis.com/token")
        if info.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/drive"])
        elif info.get("refresh_token"):
            creds = Credentials.from_authorized_user_info(info)
        else:
            creds = Credentials(token=info.get("access_token") or info["token"])
        transport = google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=120))
        return build("drive", "v3", http=transport, cache_discovery=False)
    if provider != "Dropbox":
        raise ValueError("No supported cloud provider selected")
    import dropbox
    return dropbox.Dropbox(oauth2_access_token=info.get("access_token") or info.get("token"),
                          oauth2_refresh_token=info.get("refresh_token"),
                          app_key=info.get("app_key"), app_secret=info.get("app_secret"),
                          timeout=120, max_retries_on_error=2, max_retries_on_rate_limit=2)


def cloud_preflight(provider: str, destination: str, info: dict[str, Any]) -> None:
    client = cloud_client(provider, info)
    if provider == "Google Drive":
        metadata = client.files().get(fileId=destination or "root", fields="id,mimeType,capabilities(canAddChildren)",
                                      supportsAllDrives=True).execute(num_retries=2)
        if metadata["mimeType"] != "application/vnd.google-apps.folder" or not metadata.get("capabilities", {}).get("canAddChildren"):
            raise ValueError("Drive destination must be a writable folder ID")
    else:
        client.users_get_current_account()
        if destination and destination != "/":
            import dropbox
            if not isinstance(client.files_get_metadata(destination), dropbox.files.FolderMetadata):
                raise ValueError("Dropbox destination must be an existing folder path")


def upload_cloud(store: StateStore, movie: Path, notice: Callable[[str], None]) -> dict[str, Any]:
    """SDK-backed large-file transfers; remote metadata must match local checksum."""
    state = store.data
    if not cloud_enabled(state.get("cloud", {})):
        raise ValueError("Cloud settings are incomplete")
    provider = state["cloud"]["provider"]
    client = cloud_client(provider, credentials(state["cloud"]["credentials"]))
    folder = state["cloud"]["destination"]
    local_size = movie.stat().st_size
    if provider == "Google Drive":
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload
        with store.lock:
            remote_id = state["cloud"].get("file_id")
            if not remote_id:
                remote_id = client.files().generateIds(count=1, space="drive").execute(num_retries=2)["ids"][0]
                state["cloud"]["file_id"] = remote_id
                store.save()
        checksum = digest(movie, "md5")

        def metadata() -> Any:
            try:
                return client.files().get(fileId=remote_id, fields="id,size,md5Checksum", supportsAllDrives=True).execute(num_retries=2)
            except HttpError as exc:
                if exc.resp.status == 404:
                    return None
                raise

        remote = metadata()
        if not remote or remote.get("md5Checksum") != checksum or int(remote.get("size", -1)) != local_size:
            media = MediaFileUpload(str(movie), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
            if remote:
                request = client.files().update(fileId=remote_id, media_body=media, fields="id", supportsAllDrives=True)
            else:
                request = client.files().create(body={"id": remote_id, "name": "final_movie.mp4",
                    "parents": [folder or "root"], "description": f"AI Movie Studio run {state['job_id']}"},
                    media_body=media, fields="id", supportsAllDrives=True)
            result = None
            while result is None:
                progress, result = request.next_chunk(num_retries=2)
                if progress:
                    notice(f"Google Drive upload: {progress.progress():.0%}")
            remote = metadata()
        if not remote or remote.get("md5Checksum") != checksum or int(remote.get("size", -1)) != local_size:
            raise RuntimeError("Google Drive checksum/size verification failed")
        return {"provider": provider, "file_id": remote_id, "checksum": checksum,
                "algorithm": "md5", "bytes": local_size, "verified_at": utc()}

    import dropbox
    path = folder.rstrip("/") + f"/{state['job_id']}_final_movie.mp4"
    checksum = dropbox_hash(movie)
    try:
        remote = client.files_get_metadata(path)
    except dropbox.exceptions.ApiError as exc:
        if exc.error.is_path() and exc.error.get_path().is_not_found():
            remote = None
        else:
            raise
    if not remote or getattr(remote, "content_hash", None) != checksum or getattr(remote, "size", -1) != local_size:
        chunk_size = 8 * 1024 * 1024
        with movie.open("rb") as handle:
            session = client.files_upload_session_start(handle.read(chunk_size))
            cursor = dropbox.files.UploadSessionCursor(session_id=session.session_id, offset=handle.tell())
            commit = dropbox.files.CommitInfo(path=path, mode=dropbox.files.WriteMode.overwrite, autorename=False)
            while local_size - handle.tell() > chunk_size:
                block = handle.read(chunk_size)
                client.files_upload_session_append_v2(block, cursor)
                cursor.offset = handle.tell()
                notice(f"Dropbox upload: {cursor.offset / local_size:.0%}")
            client.files_upload_session_finish(handle.read(chunk_size), cursor, commit)
        remote = client.files_get_metadata(path)
    if getattr(remote, "content_hash", None) != checksum or getattr(remote, "size", -1) != local_size:
        raise RuntimeError("Dropbox checksum/size verification failed")
    return {"provider": provider, "path": path, "checksum": checksum,
            "algorithm": "dropbox-content-hash", "bytes": local_size, "verified_at": utc()}


class Studio:
    def __init__(self, root: Path, acquire_lock: bool = True):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace_lock = WorkspaceLock(self.root) if acquire_lock else None
        self.store = StateStore(self.root)
        self.guard = threading.Lock()
        self.thread: threading.Thread | None = None
        self.lines: deque[str] = deque(maxlen=300)
        self.live_lock = threading.Lock()
        self.streams: list[str] = []
        self.assembly_error: BaseException | None = None

    def notice(self, message: str) -> None:
        LOG.info(message)
        with self.live_lock:
            self.lines.append(f"[{utc()}] {message}")

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def launch(self) -> None:
        if self.running():
            raise RuntimeError("Studio is already running")
        self.thread = threading.Thread(target=self.pipeline, name="movie-coordinator", daemon=False)
        self.thread.start()

    def resume(self) -> None:
        if self.store.data and self.store.data.get("status") not in TERMINAL:
            self.notice("Interrupted checkpoint detected; automatically resuming.")
            self.launch()

    def start(self, plot: str, image: str | None, minutes: float, provider: str,
              destination: str, secret: str, shutdown: bool = True,
              google_credentials: str = "") -> None:
        with self.guard:
            if self.running():
                raise ValueError("A movie is already running; attach to the current job")
            if self.store.data and self.store.data.get("status") not in TERMINAL:
                self.launch()
                return
            if not plot.strip():
                raise ValueError("Provide a movie plot")
            count = int(round(float(minutes) * 60 / 5))
            if not 1 <= count <= 720:
                raise ValueError("Duration must be between five seconds and 60 minutes")
            provider = provider or "None"
            secret = secret or ""
            if provider not in {"None", "Google Drive", "Dropbox"}:
                raise ValueError("Select a cloud provider")
            destination = (destination or "").strip()
            if provider == "Google Drive" and google_credentials:
                destination = destination or "root"
                secret = google_credentials
            if provider == "Dropbox" and destination != "/":
                destination = destination.rstrip("/")
            graph, config = workflow_config()
            anchor_graph, anchor_config = anchor_workflow_config(graph) if not image else ({}, {})
            master_seed = int(os.getenv("CHARACTER_SEED", str(uuid.uuid4().int % (2**63))))
            if not 0 <= master_seed < 2**63:
                raise ValueError("CHARACTER_SEED must be between 0 and 2**63-1")
            cloud = {"provider": provider, "destination": destination, "credentials": {}, "status": "pending"}
            if provider != "None" and destination and (secret.strip() or os.getenv("STUDIO_CREDENTIALS_FILE")):
                try:
                    cloud["credentials"] = credential_spec(secret, self.root)
                except Exception as exc:
                    # Optional backup configuration must never prevent rendering.
                    cloud.update(status="error", error=type(exc).__name__, reason="Credential configuration failed")
            else:
                cloud.update(status="skipped", reason="Provider, destination, or credentials not supplied")
            job_id = uuid.uuid4().hex
            folder = self.root / "runs" / job_id
            folder.mkdir(parents=True, mode=0o700)
            reference = None
            if image:
                suffix = Path(image).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    raise ValueError("Use PNG, JPEG, WebP, or BMP character reference")
                reference = folder / ("reference" + suffix)
                atomic_bytes(reference, Path(image).read_bytes())
            anchor = {"mode": "provided" if image else "auto", "seed": master_seed,
                      "status": "ready" if image else "pending", "attempts": 0,
                      "workflow": anchor_graph, "config": anchor_config}
            if reference:
                anchor.update(image_path=str(reference), image_sha256=digest(reference))
            width = int(os.getenv("MOVIE_WIDTH", "1920"))
            height = int(os.getenv("MOVIE_HEIGHT", "1080"))
            fps = int(os.getenv("MOVIE_FPS", "24"))
            if min(width, height) < 64 or width % 2 or height % 2 or not 1 <= fps <= 120:
                raise ValueError("Media dimensions must be even and >=64; fps must be 1–120")
            if self.store.path.exists():
                atomic_bytes(Path(self.store.data["folder"]) / "state.archive.json", self.store.path.read_bytes())
            self.store.data = {"schema": 1, "job_id": job_id, "folder": str(folder), "created_at": utc(),
                "status": "planning", "plot": plot, "shot_count": count, "shots": [],
                "reference": str(reference) if reference else None, "anchor": anchor, "master_seed": master_seed,
                "workflow": graph, "comfy": config, "media": {"width": width, "height": height, "fps": fps},
                "cloud": cloud, "shutdown": True, "master_shot_ids": []}
            self.store.save()
            with self.live_lock:
                self.lines.clear()
                self.streams.clear()
            self.launch()

    def checkpoint_shot(self, shot: dict[str, Any], **changes: Any) -> None:
        with self.store.lock:
            shot.update(changes)
            self.store.save()

    def ensure_master_anchor(self, client: ComfyClient) -> Path:
        """Create once, checkpoint before shots, reuse the exact same pixels.

        A run-private image is authoritative. The required fixed input filename
        is a convenience alias; every clip gets a job-unique upload of the same
        cached anchor, avoiding collisions with other users of ComfyUI.
        """
        state = self.store.data
        if state.get("reference") and not state.get("anchor"):
            # Backward compatibility with pre-Auto-Character checkpoints.
            return Path(state["reference"])
        anchor = state["anchor"]
        if anchor["mode"] == "provided":
            reference = Path(anchor["image_path"])
            if not reference.is_file() or digest(reference) != anchor["image_sha256"]:
                raise RuntimeError("Uploaded master reference is missing/corrupt; restore it before resuming")
            self.notice("Using uploaded character reference as the fixed master anchor.")
            return reference
        cached = Path(state["folder"]) / "master_character.png"
        input_dir = Path(state["comfy"].get("input_dir", str(Path(state["comfy"]["output_dir"]).parent / "input")))
        alias = input_dir / "auto_master_character.png"
        expected_hash = anchor.get("image_sha256")
        if (expected_hash and (not cached.exists() or digest(cached) != expected_hash)
                and alias.is_file() and digest(alias) == expected_hash):
            atomic_bytes(cached, alias.read_bytes())
        if expected_hash and cached.is_file() and digest(cached) == expected_hash:
            if not alias.is_file() or digest(alias) != expected_hash:
                atomic_bytes(alias, cached.read_bytes())
            self.notice("Auto-Character anchor recovered from checkpoint; reusing identical master image.")
            return cached
        self.notice(f"Auto-Character Mode: generating master identity/style anchor with locked seed {anchor['seed']}.")
        first = state["shots"][0]
        description = ("Create the single primary protagonist for this movie as a clear cinematic character reference. "
                       "One subject, unobstructed face, visible hairstyle and distinctive wardrobe, three-quarter portrait. "
                       "Choose and establish a coherent visual rendering style and palette for the entire movie. "
                       f"Movie plot: {state['plot']} First shot: {first['visual_prompt']}. "
                       f"Lighting style: {first['lighting_cue']}.")
        graph = copy.deepcopy(anchor["workflow"])
        config = anchor["config"]
        graph[config["prompt_node"]]["inputs"][config["prompt_input"]] = description
        # Keep every literal sampler seed in the anchor graph fixed too.
        for node in graph.values():
            for key, value in list(node.get("inputs", {}).items()):
                if key in {"seed", "noise_seed"} and type(value) is int:
                    node["inputs"][key] = anchor["seed"]
        graph[config["seed_node"]]["inputs"][config["seed_input"]] = anchor["seed"]
        while anchor.get("attempts", 0) < ATTEMPTS or anchor.get("status") in {"submitting", "submitted"}:
            if anchor.get("status") not in {"submitting", "submitted"}:
                attempt = anchor.get("attempts", 0) + 1
                anchor.update(attempts=attempt, status="submitting",
                              marker=anchor.get("reconcile_marker") or f"{state['job_id']}:anchor:{attempt}",
                              prompt_id=anchor.get("retry_prompt_id"))
                self.store.save()
            prompt_id = anchor.get("prompt_id")
            try:
                if not prompt_id:
                    prompt_id = client.reconcile(anchor["marker"]) or client.submit_graph(graph, anchor["marker"])
                anchor.update(prompt_id=prompt_id, status="submitted")
                self.store.save()
                source = client.wait(prompt_id, self.notice, output_node=config["output_node"], kind="image")
                candidate = cached.with_name("master_character.candidate.png")
                save_anchor_png(source, candidate)
                checksum = digest(candidate)
                if expected_hash and checksum != expected_hash and any(s["status"] == "completed" for s in state["shots"]):
                    raise RuntimeError("Original anchor could not be recovered exactly; refusing to change identity mid-movie")
                # Preserve a private durable copy before exposing the fixed alias.
                atomic_bytes(cached, candidate.read_bytes())
                atomic_bytes(alias, cached.read_bytes())
                anchor.update(status="ready", image_path=str(cached), image_sha256=checksum,
                              description=description, completed_at=utc())
                state["reference"] = str(cached)
                self.store.save()
                self.notice(f"Auto-Character master saved: {alias}. All shots reuse this anchor and locked seed.")
                return cached
            except Exception as exc:
                anchor.update(status="retry", last_error=type(exc).__name__,
                              reconcile_marker=anchor["marker"] if not prompt_id else None,
                              retry_prompt_id=prompt_id if not isinstance(exc, ValueError) else None)
                self.store.save()
                self.notice(f"Master anchor attempt {anchor['attempts']}/3 failed ({type(exc).__name__}).")
                if anchor["attempts"] < ATTEMPTS:
                    time.sleep(2 ** anchor["attempts"])
        raise RuntimeError("Master anchor generation exhausted three attempts; no movie shots were rendered")

    def render_shot(self, client: ComfyClient, shot: dict[str, Any], reference: str) -> Path | None:
        """Persist every attempt before submitting. Reuse IDs after network errors."""
        state = self.store.data
        folder = Path(state["folder"])
        raw = folder / f"raw_{shot['shot_id']:04d}.mp4"
        if shot.get("status") == "rendered" and raw.is_file() and shot.get("raw_sha256") == digest(raw):
            self.capture_continuity(client, shot, raw)
            return raw
        # A lost checkpoint after copying raw output can be recovered from its
        # persisted prompt ID; raw is copied again, never counted twice.
        while shot.get("attempts", 0) < ATTEMPTS or shot.get("status") in {"submitting", "submitted"}:
            continuing = shot.get("status") in {"submitting", "submitted"}
            if not continuing:
                attempt = shot.get("attempts", 0) + 1
                marker = shot.get("reconcile_marker") or f"{state['job_id']}:{shot['shot_id']}:{attempt}"
                self.checkpoint_shot(shot, attempts=attempt, marker=marker,
                                     prompt_id=shot.get("retry_prompt_id"), status="submitting")
            prompt_id = shot.get("prompt_id")
            try:
                if not prompt_id:
                    prompt_id = client.reconcile(shot["marker"])
                if not prompt_id:
                    # A failed POST retains its marker for the next attempt.
                    prompt_id = client.submit(state["workflow"], shot, reference, shot["marker"])
                self.checkpoint_shot(shot, prompt_id=prompt_id, status="submitted")
                self.notice(f"Rendering Shot {shot['shot_id']}/{state['shot_count']} (attempt {shot['attempts']}/3)")
                source = client.wait(prompt_id, self.notice)
                pending = raw.with_suffix(".pending.mp4")
                shutil.copyfile(source, pending)
                probe(pending)
                with pending.open("r+b") as handle:
                    os.fsync(handle.fileno())
                os.replace(pending, raw)
                self.checkpoint_shot(shot, status="rendered", raw_sha256=digest(raw))
                self.capture_continuity(client, shot, raw)
                return raw
            except Exception as exc:
                self.notice(f"Shot {shot['shot_id']} attempt {shot['attempts']}/3 failed ({type(exc).__name__}).")
                # Remove only this studio's pending prompt. Global interrupt is
                # deliberately avoided; a running timed-out job may finish later.
                terminal_execution = isinstance(exc, ValueError)
                if prompt_id and terminal_execution:
                    try:
                        client.request("POST", "/queue", json={"delete": [prompt_id]})
                    except Exception:
                        self.notice("Could not remove studio prompt from pending queue.")
                self.checkpoint_shot(shot, status="retry", last_error=type(exc).__name__, prompt_id=None,
                                     reconcile_marker=shot["marker"] if not prompt_id else None,
                                     retry_prompt_id=prompt_id if not terminal_execution else None)
                if shot["attempts"] < ATTEMPTS:
                    time.sleep(2 ** shot["attempts"] + random.uniform(0, 0.5))
        self.checkpoint_shot(shot, status="failed")
        self.notice(f"Shot {shot['shot_id']} exhausted three attempts; skipping and continuing.")
        return None

    def capture_continuity(self, client: ComfyClient, shot: dict[str, Any], raw: Path) -> None:
        settings = self.store.data.get("comfy", {}).get("consistency", {}).get("continuity", {})
        mode = settings.get("mode", "off")
        if mode == "off":
            return
        cached = shot.get("tail_condition", {})
        if (cached.get("source_sha256") == shot.get("raw_sha256") and cached.get("path")
                and Path(cached["path"]).is_file() and digest(Path(cached["path"])) == cached.get("sha256")):
            return
        folder = Path(self.store.data["folder"])
        if mode == "image":
            target = folder / f"tail_{shot['shot_id']:04d}.png"
            temporary = target.with_name(target.stem + ".pending.png")

            def extract() -> None:
                # Decode the final second through EOF, then reverse that small
                # window. The first output is the actual last decoded frame.
                run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-1",
                     "-i", str(raw), "-map", "0:v:0", "-vf", "reverse", "-frames:v", "1", "-an", str(temporary)])
                if not temporary.is_file() or temporary.stat().st_size < 8:
                    raise RuntimeError("Final-frame extraction produced no image")
                atomic_bytes(target, temporary.read_bytes())

            retry(extract, f"Shot {shot['shot_id']} final-frame extraction", self.notice)
        else:
            # Native latent mode NEVER infers tensors from encoded video. The
            # graph must explicitly select/save its model-specific final latent.
            record = client.request("GET", f"/history/{shot['prompt_id']}").get(shot["prompt_id"], {})
            items = record.get("outputs", {}).get(settings["output_node"], {}).get(settings.get("output_key", "latents"), [])
            root = Path(self.store.data["comfy"]["output_dir"]).resolve()
            paths = set()
            for item in items:
                if not isinstance(item, dict) or item.get("type", "output") != "output":
                    continue
                source = (root / item.get("subfolder", "") / item.get("filename", "")).resolve()
                if not source.is_relative_to(root):
                    raise ValueError("Latent exporter returned a path outside ComfyUI output")
                if source.suffix.lower() in {".latent", ".safetensors"} and source.is_file():
                    paths.add(source)
            if len(paths) != 1:
                raise ValueError("Native continuity exporter must report exactly one final-latent file")
            source = paths.pop()
            target = folder / f"tail_{shot['shot_id']:04d}{source.suffix}"
            atomic_bytes(target, source.read_bytes())
        self.checkpoint_shot(shot, tail_condition={"mode": mode, "path": str(target), "sha256": digest(target),
                            "source_sha256": shot["raw_sha256"], "captured_at": utc()})

    def prepare_continuity(self, client: ComfyClient, shot: dict[str, Any]) -> None:
        state = self.store.data
        settings = state.get("comfy", {}).get("consistency", {}).get("continuity", {})
        mode = settings.get("mode", "off")
        if mode == "off":
            return
        if shot.get("consistency_prepared") and (shot.get("status") in {"submitting", "submitted", "rendered"}
                                                or shot.get("retry_prompt_id") or shot.get("reconcile_marker")):
            return  # Retry/reconciliation must retain the original API payload.
        previous = next((item for item in state["shots"] if item["shot_id"] == shot["shot_id"] - 1), None)
        linked = (previous and previous.get("scene_id") == shot.get("scene_id")
                  and previous.get("status") in {"rendered", "completed"})
        if not linked:
            self.checkpoint_shot(shot, continuity_input=None, continuity_from=None, consistency_prepared=True)
            self.notice(f"Shot {shot['shot_id']}: scene boundary / unavailable predecessor; reset temporal conditioning.")
            return
        tail = previous.get("tail_condition", {})
        if not tail or not Path(tail.get("path", "")).is_file() or digest(Path(tail["path"])) != tail.get("sha256"):
            raw = Path(state["folder"]) / f"raw_{previous['shot_id']:04d}.mp4"
            if not raw.is_file() or digest(raw) != previous.get("raw_sha256"):
                raise RuntimeError("Cannot recover preceding shot's continuity condition without its verified raw clip")
            self.capture_continuity(client, previous, raw)
            tail = previous["tail_condition"]
        if tail["mode"] != mode:
            raise RuntimeError("Continuity mode changed within a checkpointed movie")
        path = Path(tail["path"])
        if mode == "image":
            filename = retry(lambda: client.upload(path, f"{state['job_id']}_tail_{previous['shot_id']}"),
                             "Continuity frame upload", self.notice)
        else:
            filename = f"studio_{state['job_id']}_tail_{previous['shot_id']}{path.suffix}"
            atomic_bytes(Path(state["comfy"]["input_dir"]) / filename, path.read_bytes())
        self.checkpoint_shot(shot, continuity_input=filename, continuity_from=previous["shot_id"],
                            continuity_sha256=tail["sha256"], consistency_prepared=True)
        self.notice(f"Shot {shot['shot_id']}: {mode} continuity chained from Shot {previous['shot_id']}; master identity unchanged.")

    def assemble(self) -> None:
        """Called only by the consumer. Reconcile playlist from durable shot state."""
        snapshot = self.store.snapshot()
        clips = [shot for shot in snapshot["shots"] if shot["status"] == "completed"]
        if not clips:
            return
        folder = Path(snapshot["folder"])
        playlist = folder / "filelist.txt"
        # Atomic replacement acts like append during normal execution, and also
        # removes orphaned/duplicate entries after an interrupted write.
        atomic_bytes(playlist, "".join(f"file 'shot_{shot['shot_id']:04d}.mp4'\n" for shot in clips).encode())
        pending = self.root / "current_movie_stream.pending.mp4"
        master = self.root / "current_movie_stream.mp4"
        # FFmpeg is a subprocess in this dedicated consumer thread. The producer
        # keeps rendering and UI keeps yielding while this process runs.
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1",
             "-i", str(playlist), "-c", "copy", "-movflags", "+faststart", str(pending)], timeout=1800)
        if abs(probe(pending) - 5 * len(clips)) > max(0.25, len(clips) * 0.002):
            raise RuntimeError("Master duration verification failed")
        with pending.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(pending, master)
        fsync_directory(self.root)
        with self.store.lock:
            self.store.data["master_shot_ids"] = [s["shot_id"] for s in clips]
            self.store.save()
        self.notice(f"Master updated: {len(clips)} completed shots, {len(clips) * 5} seconds.")

    def consume(self, work: queue.Queue) -> None:
        """Single consumer preserves shot order; failures never kill it silently."""
        try:
            self.assemble()  # Rebuild durable completed clips on restart.
            while True:
                item = work.get()
                try:
                    if item is None:
                        return
                    shot, raw = item
                    clip = Path(self.store.data["folder"]) / f"shot_{shot['shot_id']:04d}.mp4"
                    try:
                        retry(lambda: normalize(raw, clip, self.store.data["media"]),
                              f"Normalize shot {shot['shot_id']}", self.notice)
                        transport = retry(lambda: preview(clip), "Prepare preview", self.notice)
                    except Exception as exc:
                        self.checkpoint_shot(shot, status="failed", last_error=type(exc).__name__)
                        self.notice(f"Shot {shot['shot_id']} media processing failed; skipping.")
                        continue
                    self.checkpoint_shot(shot, status="completed", clip_sha256=digest(clip), completed_at=utc())
                    with self.live_lock:
                        self.streams.append(str(transport))
                    retry(self.assemble, "Master assembly", self.notice)
                finally:
                    work.task_done()
        except BaseException as exc:
            self.assembly_error = exc
            self.notice(f"Assembly consumer stopped ({type(exc).__name__}); checkpoint retained.")
            # Drain so bounded producer cannot deadlock; pipeline detects error.
            while True:
                item = work.get()
                work.task_done()
                if item is None:
                    return

    def recover_clips(self) -> None:
        for shot in self.store.data["shots"]:
            if shot["status"] != "completed":
                continue
            clip = Path(self.store.data["folder"]) / f"shot_{shot['shot_id']:04d}.mp4"
            if not clip.exists() or digest(clip) != shot.get("clip_sha256"):
                self.notice(f"Shot {shot['shot_id']} checkpoint file missing/corrupt; recovering output.")
                self.checkpoint_shot(shot, status="submitted" if shot.get("prompt_id") else "pending", attempts=0)
                continue
            transport = clip.with_suffix(".ts")
            if not transport.exists():
                transport = preview(clip)
            with self.live_lock:
                self.streams.append(str(transport))

    def backup_final_movie(self, final: Path) -> None:
        """Optional post-processing: errors are persisted/logged, never re-raised.

        Failed delivery retains the local movie and checkpoint for retry. GPU
        shutdown is gated on verified delivery when a backup is configured.
        """
        state = self.store.data
        cloud = state.setdefault("cloud", {"provider": "None"})
        if cloud.get("status") == "verified" and state.get("upload_receipt", {}).get("verified_at"):
            self.notice("Cloud backup verified: checksum already verified.")
            return
        if not cloud_enabled(cloud):
            cloud.update(status="skipped", reason="Provider None or destination/credentials missing")
            state.pop("upload_receipt", None)
            self.store.save()
            self.notice("Cloud backup SKIPPED: optional settings are not complete. Movie saved locally.")
            return
        self.notice(f"Cloud backup started: {cloud['provider']}.")
        try:
            cloud["status"] = "uploading"
            state["status"] = "uploading"
            self.store.save()
            # Authenticate at delivery time so a short-lived access token is not
            # rejected hours before it would actually be used.
            retry(lambda: cloud_preflight(cloud["provider"], cloud["destination"],
                                          credentials(cloud["credentials"])), "Cloud authorization", self.notice)
            receipt = retry(lambda: upload_cloud(self.store, final, self.notice),
                            "Cloud upload/verification", self.notice)
            if not receipt.get("verified_at"):
                raise RuntimeError("Cloud uploader returned no verified receipt")
            state["upload_receipt"] = receipt
            cloud.update(status="verified", verified_at=receipt["verified_at"])
            self.store.save()
            self.notice(f"Cloud backup SUCCESS: {cloud['provider']} checksum and size VERIFIED.")
        except Exception as exc:
            state.pop("upload_receipt", None)
            cloud.update(status="error", error=type(exc).__name__, error_at=utc())
            self.store.save()
            self.notice(f"Cloud backup ERROR ({type(exc).__name__}): all attempts finished. "
                        "Local movie retained; GPU shutdown withheld until delivery succeeds.")

    def complete_movie(self, final: Path) -> None:
        state = self.store.data
        self.backup_final_movie(final)
        with self.store.lock:
            state.update(status="movie_complete", completed_at=utc())
            self.store.save()
        atomic_json(Path(state["folder"]) / "completion.json", {
            "job_id": state["job_id"], "completed_shot_ids": state["completed_shot_ids"],
            "failed_shot_ids": state["failed_shot_ids"], "partial": state["partial"],
            "final_sha256": state["final_sha256"], "cloud_status": state["cloud"]["status"],
            "upload_receipt": state.get("upload_receipt"), "completed_at": state["completed_at"]})
        self.notice(f"Movie processing complete at {state['completed_at']}; "
                    f"cloud backup: {state['cloud']['status']}.")

    def pipeline(self) -> None:
        work: queue.Queue = queue.Queue(maxsize=3)
        consumer: threading.Thread | None = None
        self.assembly_error = None
        final_ready = False
        stage = "dependency checks"
        try:
            state = self.store.data
            for binary in ("ffmpeg", "ffprobe"):
                if not shutil.which(binary):
                    self.notice(f"Missing required executable: {binary}. Install it on PATH.")
                    raise RuntimeError(f"Install {binary} on PATH")
            # A restart during upload/teardown resumes finalization directly.
            final = Path(state.get("final_file", str(self.root / "final_movie.mp4")))
            if state.get("movie_ready") and final.is_file() and digest(final) == state.get("final_sha256"):
                final_ready = True
                stage = "resumed optional backup/finalization"
                self.notice("Verified final movie checkpoint found; resuming post-processing.")
                self.complete_movie(final)
                return
            if not state["shots"]:
                stage = "storyboard planning"
                shots = plan_shots(state["plot"], state["shot_count"], self.notice)
                for shot in shots:
                    shot.update(status="pending", attempts=0,
                                seed=state.get("master_seed", uuid.uuid4().int % (2**63)))
                bind_shot_consistency(shots, state.get("comfy", {}).get("consistency", {}), state.get("master_seed", 0))
                with self.store.lock:
                    state["shots"] = shots
                    state["status"] = "rendering"
                    self.store.save()
            else:
                with self.store.lock:
                    bind_shot_consistency(state["shots"], state.get("comfy", {}).get("consistency", {}), state.get("master_seed", 0))
                    state["status"] = "rendering"
                    self.store.save()
            with self.live_lock:
                self.streams.clear()
            stage = "shot recovery, rendering, and assembly"
            self.recover_clips()
            consumer = threading.Thread(target=self.consume, args=(work,), name="media-consumer", daemon=False)
            consumer.start()
            client = ComfyClient(state["comfy"])
            try:
                remaining = [s for s in state["shots"] if s["status"] not in {"completed", "failed"}]
                if remaining:
                    profile = state.get("comfy", {}).get("consistency", {})
                    self.notice(f"Consistency stack: {len(profile.get('adapters', []))} adapter/stand-in mappings; "
                                f"temporal chaining={profile.get('continuity', {}).get('mode', 'off')}; locked identity/style/object prompts.")
                    if profile.get("continuity", {}).get("mode", "off") == "off":
                        self.notice("Temporal chaining is disabled: configure a separate start-frame/latent loader in CONSISTENCY_CONFIG.")
                    master = self.ensure_master_anchor(client)
                    reference = retry(lambda: client.upload(master, state["job_id"]),
                                      "Reference image upload", self.notice)
                    for shot in remaining:
                        if self.assembly_error:
                            raise RuntimeError("Assembly failed; stopping producer")
                        try:
                            self.prepare_continuity(client, shot)
                        except Exception as exc:
                            self.checkpoint_shot(shot, status="failed", last_error=type(exc).__name__)
                            self.notice(f"Shot {shot['shot_id']}: continuity preparation failed "
                                        f"({type(exc).__name__}); skipping this shot and resetting the next chain.")
                            continue
                        raw = self.render_shot(client, shot, reference)
                        if raw:
                            work.put((shot, raw))
            finally:
                client.session.close()
                work.put(None)
                consumer.join()
                consumer = None
            if self.assembly_error:
                raise RuntimeError("Assembly failed; restart will recover durable clips")
            completed = [s for s in state["shots"] if s["status"] == "completed"]
            failed = [s for s in state["shots"] if s["status"] == "failed"]
            if not completed:
                raise RuntimeError("All shots failed; no movie to upload")
            stage = "final movie validation"
            final = self.root / "final_movie.mp4"
            pending = self.root / "final_movie.pending.mp4"
            shutil.copyfile(self.root / "current_movie_stream.mp4", pending)
            if abs(probe(pending) - len(completed) * 5) > 0.5:
                raise RuntimeError("Final movie duration is invalid")
            with pending.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(pending, final)
            fsync_directory(self.root)
            with self.store.lock:
                state.update(status="finalizing", movie_ready=True, final_file=str(final), final_sha256=digest(final),
                             partial=bool(failed), rendering_finished_at=utc())
                self.store.save()
            final_ready = True
            self.notice(f"Rendering finished: {len(completed)} successful, {len(failed)} failed. "
                        f"{'PARTIAL ' if failed else ''}Movie saved locally ({len(completed) * 5}s).")
            stage = "optional cloud backup/finalization"
            self.complete_movie(final)
        except Exception as exc:
            # Error class is safe to log; provider exceptions can contain secrets.
            self.notice(f"Pipeline stopped during {stage} ({type(exc).__name__}). "
                        + ("Final movie retained; automatic teardown will still run."
                           if final_ready else "Checkpoint retained; shutdown withheld until final movie is ready."))
            LOG.error("Failure category %s during %s; stack (exception payload omitted):\n%s",
                      type(exc).__name__, stage, "".join(traceback.format_tb(exc.__traceback__)))
            with self.store.lock:
                self.store.data.update(status="recoverable_error", last_error=type(exc).__name__,
                                       last_error_stage=stage, error_at=utc())
                self.store.save()
        finally:
            if consumer and consumer.is_alive():
                work.put(None)
                consumer.join()
            if final_ready:
                cloud = self.store.data.get("cloud", {})
                if cloud.get("status") == "error":
                    self.notice("Automatic shutdown withheld: cloud delivery is not verified. "
                                "Attach / Resume Checkpoint retries delivery without rendering again.")
                    with self.store.lock:
                        self.store.data["status"] = "backup_error"
                        self.store.save()
                else:
                    try:
                        self.teardown()
                    except Exception as exc:
                        self.notice(f"Automatic shutdown ERROR ({type(exc).__name__}): check CONTAINER_ID, "
                                    "vastai CLI credentials, and shutdown.log. Movie remains saved locally.")
                        with self.store.lock:
                            self.store.data.update(status="shutdown_error", shutdown_error=type(exc).__name__)
                            self.store.save()

    def teardown(self) -> None:
        state = self.store.data
        final = Path(state.get("final_file", str(self.root / "final_movie.mp4")))
        if not state.get("movie_ready") or not final.is_file() or digest(final) != state.get("final_sha256"):
            raise RuntimeError("Refusing shutdown before a verified local final movie exists")
        cloud = state.get("cloud", {})
        if cloud.get("provider") in {"Google Drive", "Dropbox"} and cloud.get("status") != "skipped":
            if cloud.get("status") != "verified" or not state.get("upload_receipt", {}).get("verified_at"):
                raise RuntimeError("Refusing shutdown until configured cloud backup is checksum verified")
        instance = validate_shutdown()
        self.notice(f"Safe teardown started at {utc()}; GPU stop scheduled in 10 seconds.")
        with self.store.lock:
            state.update(status="shutdown_pending", shutdown_requested_at=utc())
            self.store.save()
        for handler in LOG.handlers:
            handler.flush()
            stream = getattr(handler, "stream", None)
            if stream and stream is not sys.stderr:
                os.fsync(stream.fileno())
        os.sync()
        # Numeric instance ID, trusted PATH, fixed command, safely quoted log path.
        # flock makes repeated scheduling after a crash harmless on Linux.
        folder = Path(state["folder"])
        shell = (f"flock -n {shlex.quote(str(folder / 'teardown.lock'))} "
                 f"sh -c {shlex.quote(f'sleep 10 && vastai stop instance {instance}')}")
        code = os.system(f"nohup sh -c {shlex.quote(shell)} > "
                         f"{shlex.quote(str(folder / 'shutdown.log'))} 2>&1 < /dev/null &")
        if code != 0:
            raise RuntimeError("Could not start shutdown scheduler")
        with self.store.lock:
            state.update(status="shutdown_scheduled", shutdown_scheduled_at=utc())
            self.store.save()
        # Scheduling success is not a claim that Vast.ai already stopped.


def validate_shutdown() -> str:
    instance = os.getenv("CONTAINER_ID", "")
    if os.name != "posix" or not re.fullmatch(r"[0-9]+", instance):
        raise ValueError("Teardown requires Linux and numeric CONTAINER_ID")
    for binary in ("vastai", "flock", "nohup"):
        if not shutil.which(binary):
            raise ValueError(f"Teardown requires {binary} on PATH")
    return instance


def build_app(studio: Studio) -> Any:
    import gradio as gr
    oauth = GoogleDriveOAuth(studio.root)

    def begin_google(owner: str) -> tuple[str, str, str]:
        try:
            url = oauth.begin(owner)
            link = (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                    '🔑 Continue to Google Drive sign-in</a>')
            return link, "", "Open the Google link, grant permission, then click Complete Google Sign-in."
        except Exception as exc:
            detail = str(exc) if type(exc) is ValueError else f"Sign-in setup failed ({type(exc).__name__})."
            return "", "", detail + " Check GOOGLE_OAUTH_CLIENT_FILE and google-auth-oauthlib."

    def finish_google(owner: str, code: str) -> tuple[str, str]:
        try:
            oauth.finish(owner, code)
            return "", "Google Drive connected. Offline access is ready for automatic movie upload."
        except Exception as exc:
            detail = str(exc) if type(exc) is ValueError else f"Google token exchange failed ({type(exc).__name__})."
            return "", detail + " Start sign-in again if the code expired."

    def monitor() -> Iterator[tuple[Any, str, Any, str]]:
        cursor = 0
        delivered_final = None
        # Newly attached browsers replay completed chunks once, then follow live.
        yield None, "Attaching to studio...", None, "Connecting"
        while True:
            with studio.live_lock:
                logs = "\n".join(studio.lines)
                clip = studio.streams[cursor] if cursor < len(studio.streams) else None
                if clip:
                    cursor += 1
            state = studio.store.snapshot()
            status = state.get("status", "idle")
            label = f"{status} | {len(state.get('completed_shot_ids', []))}/{state.get('shot_count', 0)} completed"
            # Do not accidentally expose a prior job's final movie while running.
            final = state.get("final_file") if state.get("movie_ready") else None
            final_output = gr.skip()
            if final and state.get("final_sha256") != delivered_final:
                final_output = final
                delivered_final = state.get("final_sha256")
            yield clip if clip else gr.skip(), logs, final_output, label
            if not studio.running() and not clip:
                break
            time.sleep(0.5)

    def start(plot: str, image: str | None, minutes: float, provider: str,
              destination: str, secret: str, owner: str) -> Iterator[Any]:
        try:
            google_token = oauth.serialized_credentials(owner) if provider == "Google Drive" else ""
            # Resuming an existing job uses its encrypted checkpoint credentials.
            resuming = studio.store.data and studio.store.data.get("status") not in TERMINAL
            if provider == "Google Drive" and not google_token and not resuming:
                raise ValueError("Sign in with Google Drive and complete sign-in before starting a Drive-backed movie.")
            studio.start(plot, image, minutes, provider, destination,
                         secret if provider == "Dropbox" else "", google_credentials=google_token)
        except Exception as exc:
            # Validation errors contain no OAuth payloads; never surface provider exceptions.
            if isinstance(exc, (ValueError, FileNotFoundError)):
                raise gr.Error(str(exc)) from None
            raise gr.Error(f"Could not start studio ({type(exc).__name__}); check configuration.") from None
        yield from monitor()

    with gr.Blocks(title="Autonomous AI Movie Studio") as app:
        owner = gr.State(value=lambda: secrets.token_urlsafe(32), time_to_live=86400,
                         delete_callback=oauth.forget)
        gr.Markdown("# Autonomous AI Movie Studio\nPersistent production · Progressive playback · Optional cloud backup")
        with gr.Row():
            with gr.Column(scale=2):
                plot = gr.Textbox(label="Movie plot / storyboard", lines=8)
                # gr.Image already accepts None; modern Gradio has no optional=True argument.
                image = gr.Image(label="Character reference (optional — leave blank for Auto-Character Mode)",
                                 value=None, type="filepath", sources=["upload"])
                gr.Markdown("No image? The studio generates one master character from Shot 1, then reuses it "
                            "with locked seeds for every clip. Your video workflow must connect the master "
                            "image loader to its IP-Adapter, FaceID, or other identity-conditioning nodes. "
                            "This reduces drift; generated character consistency is not guaranteed.")
                minutes = gr.Slider(1, 60, value=1, step=1, label="Target duration (minutes)")
                with gr.Accordion("📦 Cloud Storage Backup (Optional, Recommended)", open=False):
                    provider = gr.Dropdown(["None", "Google Drive", "Dropbox"], value="None", label="Cloud Provider")
                    destination = gr.Textbox(label="Cloud Destination Folder ID / Path",
                                             placeholder="Google: blank = My Drive root; Dropbox: /Movies or /")
                    with gr.Group(visible=False) as google_controls:
                        google_signin = gr.Button("🔑 Sign in with Google Drive")
                        google_link = gr.HTML()
                        google_code = gr.Textbox(label="Enter Google Authorization Code", type="password",
                            placeholder="Optional after callback; also accepts the full Google return URL")
                        google_finish = gr.Button("Complete Google Sign-in")
                        google_status = gr.Textbox(label="Google Drive connection", interactive=False)
                        gr.Markdown("Sign-in opens Google in a new tab. After permission, return here and complete sign-in. "
                                    "For a remote GPU, forward callback port 7862 as well as app port 7860 "
                                    "(or your configured ports). If the return page cannot load, paste its full URL above. "
                                    "Google OAuth client setup is required once on the server.")
                    secret = gr.Textbox(label="Dropbox Access Token / OAuth JSON", type="password", visible=False,
                                        placeholder="Dropbox only; leave blank to use STUDIO_CREDENTIALS_FILE")
                    gr.Markdown("Provider None disables cloud backup. The final movie is saved locally first. "
                                "With backup enabled, checksum-verified upload is required before the ten-second GPU stop. "
                                "Failed uploads retain the movie and checkpoint for retry.")
                button = gr.Button("Start Autonomous Production", variant="primary")
                attach = gr.Button("Attach / Resume Checkpoint")
                gr.Markdown("Setup requires an API workflow and persistent storage. Cloud backup is optional. "
                            "Three failed attempts skip a shot; a partial movie is clearly reported. "
                            "Autoplay may require browser interaction. Preview and final video are silent.")
            with gr.Column(scale=3):
                status = gr.Textbox(label="Production status", interactive=False)
                video = gr.Video(label="Progressive movie preview", streaming=True, autoplay=True, interactive=False)
                console = gr.Textbox(label="Live production console (UTC)", lines=18, max_lines=20, interactive=False)
                final = gr.File(label="Final movie", interactive=False)
        outputs = [video, console, final, status]
        provider.change(lambda selected: (gr.update(visible=selected == "Google Drive"),
                                          gr.update(visible=selected == "Dropbox")),
                        inputs=provider, outputs=[google_controls, secret], queue=False)
        google_signin.click(begin_google, inputs=owner, outputs=[google_link, google_code, google_status],
                            concurrency_limit=4)
        google_finish.click(finish_google, inputs=[owner, google_code], outputs=[google_code, google_status],
                            concurrency_limit=4)
        button.click(start, [plot, image, minutes, provider, destination, secret, owner], outputs,
                     concurrency_limit=1, concurrency_id="start_job", trigger_mode="once")

        def resume_monitor() -> Iterator[Any]:
            with studio.guard:
                if not studio.running():
                    studio.resume()
            yield from monitor()

        attach.click(resume_monitor, outputs=outputs, concurrency_limit=4)
        app.load(monitor, outputs=outputs, concurrency_limit=4)
    return app


def self_test() -> None:
    """Deterministic resilience tests; never contacts a GPU, cloud, or shell."""
    import tempfile
    import unittest
    from unittest.mock import Mock, patch

    class Tests(unittest.TestCase):
        def test_workflow_setup_errors_and_api_export(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "workflow_api.json"
                with self.assertRaisesRegex(ValueError, "Setting the variable does not create the file"):
                    load_api_workflow(str(path), "COMFY_WORKFLOW")
                path.write_text('{"nodes": []}')
                with self.assertRaisesRegex(ValueError, "ordinary canvas JSON"):
                    load_api_workflow(str(path), "COMFY_WORKFLOW")
                path.write_text('{bad json')
                with self.assertRaisesRegex(ValueError, "not valid UTF-8 JSON"):
                    load_api_workflow(str(path), "COMFY_WORKFLOW")
                graph = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "test"}}}
                path.write_text(json.dumps(graph), encoding="utf-8-sig")
                self.assertEqual(load_api_workflow(str(path), "COMFY_WORKFLOW"), graph)

        def test_continuity_discovery_never_repurposes_master(self):
            graph = {"master": {"class_type": "LoadImage", "inputs": {"image": "master.png"}},
                     "tail": {"class_type": "LoadImage", "inputs": {"image": "tail.png"}},
                     "i2v": {"inputs": {"start_image": ["tail", 0]}}}
            with patch.dict(os.environ, {"CONSISTENCY_CONFIG": ""}):
                profile = consistency_config(graph, {"image_node": "master"})
                self.assertEqual(profile["continuity"]["loader_node"], "tail")
                graph["i2v"]["inputs"]["start_image"] = ["master", 0]
                profile = consistency_config(graph, {"image_node": "master"})
                self.assertEqual(profile["continuity"]["mode"], "off")

        def test_prompt_anchors_survive_scene_changes_and_resume(self):
            shots = structural_shots("Arrival\nDeparture", 4)
            bind_shot_consistency(shots, {"identity_tokens": "fixed face", "style_tokens": "fixed palette",
                                          "object_tokens": "brass compass"}, 123)
            before = shot_prompt(shots[0])
            bind_shot_consistency(shots, {"identity_tokens": "changed configuration"}, 999)
            self.assertEqual(shot_prompt(shots[0]), before)
            self.assertEqual([shot["seed"] for shot in shots], [123] * 4)
            for shot in shots:
                self.assertIn("LOCKED IDENTITY: fixed face", shot_prompt(shot))
                self.assertIn("LOCKED STYLE: fixed palette", shot_prompt(shot))
                self.assertIn("LOCKED OBJECTS: brass compass", shot_prompt(shot))
            self.assertEqual(shots[0]["scene_id"], shots[1]["scene_id"])
            self.assertNotEqual(shots[1]["scene_id"], shots[2]["scene_id"])

        def test_adapter_schema_and_separate_continuity_loader(self):
            graph = {"master": {"inputs": {"image": "master.png"}},
                     "face": {"class_type": "IPAdapterFaceID", "inputs": {"weight": 1.0, "weight_faceidv2": 1.0}},
                     "wan": {"class_type": "WanVideoAddStandInLatent", "inputs": {"freq_offset": 1}}}
            with patch.dict(os.environ, {"CONSISTENCY_CONFIG": "", "IDENTITY_ADAPTER_WEIGHT": "0.75"}):
                config = consistency_config(graph, {"image_node": "master"})
            face = next(a for a in config["adapters"] if a["node"] == "face")
            wan = next(a for a in config["adapters"] if a["node"] == "wan")
            self.assertEqual(face["parameters"], {"weight": 0.75, "weight_faceidv2": 0.75})
            self.assertNotIn("weight", wan["parameters"])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "profile.json"
                for profile in [{"adapters": [{"node": "wan", "parameters": {"weight": 0.75}}]},
                                {"continuity": {"mode": "image", "loader_node": "master"}}]:
                    path.write_text(json.dumps(profile))
                    with patch.dict(os.environ, {"CONSISTENCY_CONFIG": str(path)}), self.assertRaises(ValueError):
                        consistency_config(graph, {"image_node": "master"})

        def test_scene_continuity_reuses_checkpoint_and_resets_at_cuts(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                frame = Path(tmp) / "tail_0001.png"
                frame.write_bytes(b"last-frame")
                shots = structural_shots("Arrival\nDeparture", 4)
                for shot in shots:
                    shot.update(status="pending", seed=123)
                shots[0].update(status="completed", tail_condition={"mode": "image", "path": str(frame), "sha256": digest(frame)})
                studio.store.data = {"schema": 1, "folder": tmp, "job_id": "test", "shots": shots,
                    "comfy": {"consistency": {"continuity": {"mode": "image"}}}}
                client = Mock()
                client.upload.return_value = "previous-frame.png"
                studio.prepare_continuity(client, shots[1])
                self.assertEqual(shots[1]["continuity_from"], 1)
                self.assertEqual(shots[1]["continuity_input"], "previous-frame.png")
                studio.checkpoint_shot(shots[1], status="submitted")
                reloaded = Studio(Path(tmp), acquire_lock=False)
                reloaded.prepare_continuity(client, reloaded.store.data["shots"][1])
                client.upload.assert_called_once()
                studio.prepare_continuity(client, shots[2])
                self.assertIsNone(shots[2]["continuity_from"])
                shots[0]["status"] = "failed"
                shots[1].update(status="pending", consistency_prepared=False)
                studio.prepare_continuity(client, shots[1])
                self.assertIsNone(shots[1]["continuity_input"])

        def test_native_latent_continuity_copies_exact_export(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "output").mkdir()
                source = root / "output" / "last.latent"
                source.write_bytes(b"model-specific-final-latent")
                raw = root / "raw_0001.mp4"
                raw.write_bytes(b"video")
                shots = structural_shots("One scene", 2)
                shots[0].update(status="rendered", raw_sha256=digest(raw), prompt_id="prompt")
                shots[1].update(status="pending")
                studio = Studio(root, acquire_lock=False)
                studio.store.data = {"schema": 1, "folder": tmp, "job_id": "test", "shots": shots,
                    "comfy": {"input_dir": str(root / "input"), "output_dir": str(root / "output"),
                              "consistency": {"continuity": {"mode": "latent", "output_node": "export"}}}}
                client = Mock()
                client.request.return_value = {"prompt": {"outputs": {"export": {"latents": [
                    {"filename": "last.latent", "type": "output"}]}}}}
                studio.capture_continuity(client, shots[0], raw)
                studio.prepare_continuity(client, shots[1])
                restored = root / "input" / shots[1]["continuity_input"]
                self.assertEqual(restored.read_bytes(), source.read_bytes())
                self.assertEqual(shots[1]["continuity_from"], 1)
                client.upload.assert_not_called()

        def test_continuity_payload_preserves_master_and_parameter_locks(self):
            graph = {"prompt": {"inputs": {"text": ""}}, "master": {"inputs": {"image": "master.png"}},
                     "tail": {"inputs": {"image": "tail.png"}},
                     "adapter": {"inputs": {"weight": 1.0, "image": ["master", 0]}},
                     "sampler": {"inputs": {"seed": 0, "steps": 24, "noise_aug_strength": 0.02}}}
            settings = {"scalar_snapshot": {"sampler": {"steps": 24}},
                "adapters": [{"node": "adapter", "parameters": {"weight": 0.75}}],
                "motion_parameters": [{"node": "sampler", "input": "noise_aug_strength", "base": 0.02, "jitter": 0.005}],
                "continuity": {"mode": "image", "loader_node": "tail", "loader_input": "image"}}
            client = ComfyClient.__new__(ComfyClient)
            client.config = {"prompt_node": "prompt", "prompt_input": "text", "image_node": "master", "image_input": "image",
                "seed_node": "sampler", "seed_input": "seed", "consistency": settings}
            submitted = []
            client.submit_graph = lambda graph, marker: submitted.append(graph) or "prompt"
            shots = structural_shots("One scene", 2)
            bind_shot_consistency(shots, {}, 123)
            shots[1]["continuity_input"] = "prior-end-frame.png"
            for shot in shots:
                client.submit(graph, shot, "immutable-master.png", "marker")
            self.assertEqual(submitted[0]["tail"]["inputs"]["image"], "immutable-master.png")
            self.assertEqual(submitted[1]["tail"]["inputs"]["image"], "prior-end-frame.png")
            for payload in submitted:
                self.assertEqual(payload["master"]["inputs"]["image"], "immutable-master.png")
                self.assertEqual(payload["adapter"]["inputs"]["weight"], 0.75)
                self.assertEqual(payload["sampler"]["inputs"]["seed"], 123)
                self.assertEqual(payload["sampler"]["inputs"]["steps"], 24)
                self.assertTrue(0.015 <= payload["sampler"]["inputs"]["noise_aug_strength"] <= 0.025)
            client.submit(graph, shots[1], "immutable-master.png", "retry")
            self.assertEqual(submitted[1], submitted[2])

        def test_native_latent_payload_bootstraps_before_chaining(self):
            graph = {"text": {"inputs": {"text": ""}}, "master": {"inputs": {"image": "master.png"}},
                     "tail": {"inputs": {"latent": "unused.latent"}}, "empty": {"inputs": {}},
                     "sampler": {"inputs": {"seed": 0, "latent_image": ["empty", 0]}}}
            client = ComfyClient.__new__(ComfyClient)
            client.config = {"prompt_node": "text", "prompt_input": "text", "image_node": "master", "image_input": "image",
                "seed_node": "sampler", "seed_input": "seed", "consistency": {"continuity": {
                    "mode": "latent", "loader_node": "tail", "loader_input": "latent", "loader_output": 0,
                    "target_node": "sampler", "target_input": "latent_image"}}}
            payloads = []
            client.submit_graph = lambda graph, marker: payloads.append(graph) or "prompt"
            shots = structural_shots("One scene", 2)
            bind_shot_consistency(shots, {}, 123)
            client.submit(graph, shots[0], "fixed-master.png", "first")
            shots[1]["continuity_input"] = "previous-final.latent"
            client.submit(graph, shots[1], "fixed-master.png", "second")
            self.assertEqual(payloads[0]["sampler"]["inputs"]["latent_image"], ["empty", 0])
            self.assertEqual(payloads[1]["sampler"]["inputs"]["latent_image"], ["tail", 0])
            self.assertEqual(payloads[1]["tail"]["inputs"]["latent"], "previous-final.latent")
            self.assertEqual(payloads[1]["master"]["inputs"]["image"], "fixed-master.png")

        def test_auto_anchor_created_once_and_recovered(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                studio = Studio(root, acquire_lock=False)
                graph, config = anchor_workflow_config({"loader": {
                    "class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "installed-test.safetensors"}}})
                studio.store.data = {"schema": 1, "job_id": "test", "folder": tmp, "plot": "A traveler finds a city.",
                    "shots": [dict(structural_shots("A traveler arrives.", 1)[0], status="pending")],
                    "comfy": {"input_dir": str(root / "input"), "output_dir": str(root / "output")},
                    "anchor": {"mode": "auto", "seed": 123, "status": "pending", "attempts": 0,
                               "workflow": graph, "config": config}}
                source = root / "source.png"
                source.write_bytes(b"generated-image")
                submitted = []

                class Client:
                    def reconcile(self, marker):
                        return None

                    def submit_graph(self, graph, marker):
                        submitted.append(graph)
                        return "anchor-prompt"

                    def wait(self, prompt_id, notice, **kwargs):
                        self.kind = kwargs["kind"]
                        return source

                client = Client()
                with patch(__name__ + ".save_anchor_png", side_effect=lambda src, dst: atomic_bytes(dst, b"master-pixels")):
                    anchor = studio.ensure_master_anchor(client)
                self.assertEqual(client.kind, "image")
                self.assertEqual(submitted[0][config["seed_node"]]["inputs"][config["seed_input"]], 123)
                alias = root / "input" / "auto_master_character.png"
                self.assertEqual(alias.read_bytes(), anchor.read_bytes())
                # Simulate another ComfyUI application overwriting the convenience alias.
                alias.write_bytes(b"changed")
                recovered = Studio(root, acquire_lock=False)
                self.assertEqual(recovered.ensure_master_anchor(client), anchor)
                self.assertEqual(alias.read_bytes(), b"master-pixels")
                anchor.write_bytes(b"corrupted private copy")
                recovered.ensure_master_anchor(client)
                self.assertEqual(anchor.read_bytes(), b"master-pixels")
                self.assertEqual(len(submitted), 1)

        def test_uploaded_reference_bypasses_auto_generation(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                image = Path(tmp) / "ref.png"
                image.write_bytes(b"uploaded-image")
                with patch(__name__ + ".workflow_config", return_value=({}, {})), \
                     patch(__name__ + ".anchor_workflow_config") as auto_config, patch.object(studio, "launch"):
                    studio.start("A traveler arrives.", str(image), 1, "None", "", "")
                client = Mock()
                reference = studio.ensure_master_anchor(client)
                self.assertEqual(reference.read_bytes(), b"uploaded-image")
                auto_config.assert_not_called()
                client.submit_graph.assert_not_called()

        def test_optional_image_starts_auto_mode(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                with patch(__name__ + ".workflow_config", return_value=({}, {})), \
                     patch(__name__ + ".anchor_workflow_config", return_value=({"template": {}}, {})), \
                     patch.object(studio, "launch"):
                    studio.start("A traveler arrives.", None, 1, "None", "", "")
                self.assertIsNone(studio.store.data["reference"])
                self.assertEqual(studio.store.data["anchor"]["mode"], "auto")
                self.assertEqual(studio.store.data["anchor"]["seed"], studio.store.data["master_seed"])

        def test_all_image_loaders_and_seeds_receive_same_anchor(self):
            client = ComfyClient.__new__(ComfyClient)
            client.config = {"prompt_node": "6", "prompt_input": "text", "image_node": "10", "image_input": "image",
                "seed_node": "3", "seed_input": "seed", "image_bindings": [{"node": "11", "input": "image"}],
                "seed_bindings": [{"node": "12", "input": "noise_seed"}]}
            graphs = []

            def request(method, route, **kwargs):
                graphs.append(kwargs["json"]["prompt"])
                return {"prompt_id": "clip"}

            client.request = request
            original = {"6": {"inputs": {"text": "original"}}, "10": {"inputs": {"image": "old"}},
                        "11": {"inputs": {"image": "old"}}, "3": {"inputs": {"seed": 0}},
                        "12": {"inputs": {"noise_seed": 0}}}
            for shot in structural_shots("A traveler arrives.", 2):
                client.submit(original, dict(shot, seed=123), "studio_job_master.png", "marker")
            for graph in graphs:
                self.assertEqual(graph["10"]["inputs"]["image"], "studio_job_master.png")
                self.assertEqual(graph["11"]["inputs"]["image"], "studio_job_master.png")
                self.assertEqual(graph["3"]["inputs"]["seed"], 123)
                self.assertEqual(graph["12"]["inputs"]["noise_seed"], 123)
            self.assertEqual(original["10"]["inputs"]["image"], "old")

        def oauth_fixture(self, tmp):
            from types import SimpleNamespace
            manager = GoogleDriveOAuth(Path(tmp))
            state = secrets.token_urlsafe(32)
            flow = SimpleNamespace(
                authorization_url=lambda **kwargs: ("https://accounts.google.com/o/oauth2/v2/auth?state=" + state, state),
                fetch_token=Mock(), credentials=SimpleNamespace(refresh_token="fake-refresh",
                    to_json=lambda: json.dumps({"token": "fake-access", "refresh_token": "fake-refresh"})))
            manager._new_flow = lambda: flow
            manager._ensure_server = lambda: None
            return manager, flow, state

        def test_google_callback_and_session_isolation(self):
            with tempfile.TemporaryDirectory() as tmp:
                manager, flow, state = self.oauth_fixture(tmp)
                manager.begin("alice")
                status, _, _ = manager.capture_callback(manager.CALLBACK_PATH + "?state=" + state + "&code=accepted")
                self.assertEqual(status, 200)
                manager.finish("alice", "accepted")
                flow.fetch_token.assert_called_once_with(code="accepted", timeout=60)
                self.assertTrue(manager.serialized_credentials("alice"))
                self.assertEqual(manager.serialized_credentials("bob"), "")
                self.assertFalse((Path(tmp) / "credentials.json").exists())
                with self.assertRaises(ValueError):
                    manager.finish("alice", "accepted")  # No replay of a used code.
                manager.forget("alice")
                self.assertEqual(manager.serialized_credentials("alice"), "")

        def test_google_unvalidated_code_is_rejected(self):
            with tempfile.TemporaryDirectory() as tmp:
                manager, flow, state = self.oauth_fixture(tmp)
                manager.begin("alice")
                with self.assertRaises(ValueError):
                    manager.finish("alice", "unvalidated-code")
                status, _, _ = manager.capture_callback(manager.CALLBACK_PATH + "?state=forged&code=accepted")
                self.assertEqual(status, 400)
                with self.assertRaises(ValueError):
                    manager.finish("bob", manager.redirect_uri + "?state=" + state + "&code=accepted")
                flow.fetch_token.assert_not_called()

        def test_google_full_return_url_and_expiry(self):
            with tempfile.TemporaryDirectory() as tmp:
                manager, flow, state = self.oauth_fixture(tmp)
                manager.begin("alice")
                manager.finish("alice", manager.redirect_uri + "?state=" + state + "&code=accepted")
                flow.fetch_token.assert_called_once()
                manager.begin("alice")
                manager.pending[state]["created"] -= manager.TTL + 1
                status, _, _ = manager.capture_callback(manager.CALLBACK_PATH + "?state=" + state + "&code=accepted")
                self.assertEqual(status, 400)

        def test_google_root_destination_from_session(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                image = Path(tmp) / "ref.png"
                image.write_bytes(b"reference")
                with patch(__name__ + ".workflow_config", return_value=({}, {})), \
                     patch.object(studio, "launch"), \
                     patch(__name__ + ".credential_spec", return_value={"encrypted": "fake"}):
                    studio.start("A traveler arrives.", str(image), 1, "Google Drive", "", "",
                                 google_credentials='{"refresh_token":"fake"}')
                self.assertEqual(studio.store.data["cloud"]["destination"], "root")
                self.assertTrue(cloud_enabled(studio.store.data["cloud"]))

        def test_atomic_checkpoint(self):
            with tempfile.TemporaryDirectory() as tmp:
                store = StateStore(Path(tmp))
                store.data = {"schema": 1, "shots": [{"shot_id": 1, "status": "completed"},
                    {"shot_id": 2, "status": "pending"}, {"shot_id": 3, "status": "failed"}]}
                store.save()
                loaded = StateStore(Path(tmp)).data
                self.assertEqual(loaded["completed_shot_ids"], [1])
                self.assertEqual(loaded["pending_shot_ids"], [2])
                self.assertEqual(loaded["failed_shot_ids"], [3])
                self.assertFalse((Path(tmp) / "state.json.tmp").exists())

        def test_planner(self):
            shots = structural_shots("A traveler finds a lost city.", 720)
            self.assertEqual([s["shot_id"] for s in shots], list(range(1, 721)))
            self.assertEqual(sum(s["duration_seconds"] for s in shots), 3600)

        def test_dropbox_block_hash(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "movie"
                data = b"a" * (4 * 1024 * 1024) + b"tail"
                path.write_bytes(data)
                expected = hashlib.sha256(hashlib.sha256(data[:4194304]).digest() +
                                          hashlib.sha256(b"tail").digest()).hexdigest()
                self.assertEqual(dropbox_hash(path), expected)

        def test_retry_budget(self):
            calls = []

            def failing():
                calls.append(1)
                raise OSError("simulated")

            with patch("time.sleep"):
                with self.assertRaises(OSError):
                    retry(failing, "test", lambda _: None)
            self.assertEqual(len(calls), 3)

        def test_exhausted_shot_skips(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                shot = {"shot_id": 1, "status": "pending", "attempts": 0}
                studio.store.data = {"schema": 1, "job_id": "test", "folder": tmp,
                                     "shot_count": 1, "shots": [shot], "workflow": {}}

                calls = []

                class BrokenClient:
                    def reconcile(self, marker):
                        return None

                    def submit(self, *args):
                        calls.append(1)
                        raise ConnectionError("simulated")

                with patch("time.sleep"):
                    self.assertIsNone(studio.render_shot(BrokenClient(), shot, "image.png"))
                self.assertEqual(shot["attempts"], 3)
                self.assertEqual(shot["status"], "failed")
                self.assertEqual(len(calls), 3)

        def test_ambiguous_submission_reconciles(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                shot = {"shot_id": 1, "status": "pending", "attempts": 0}
                studio.store.data = {"schema": 1, "job_id": "test", "folder": tmp,
                                     "shots": [shot], "shot_count": 1, "workflow": {}}
                source = Path(tmp) / "source.mp4"
                source.write_bytes(b"fake")
                markers = []

                class Client:
                    def reconcile(self, marker):
                        return "accepted" if marker in markers else None

                    def submit(self, workflow, shot, reference, marker):
                        markers.append(marker)
                        raise ConnectionError("Server accepted prompt, response lost")

                    def wait(self, prompt_id, notice):
                        return source

                with patch("time.sleep"), patch(__name__ + ".probe", return_value=5):
                    self.assertTrue(studio.render_shot(Client(), shot, "ref").is_file())
                self.assertEqual(len(markers), 1)
                self.assertEqual(shot["attempts"], 2)

        def test_recovery_reuses_prompt(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                shot = {"shot_id": 1, "status": "submitted", "attempts": 1, "prompt_id": "existing"}
                studio.store.data = {"schema": 1, "job_id": "test", "folder": tmp, "shots": [shot], "shot_count": 1}
                source = Path(tmp) / "source.mp4"
                source.write_bytes(b"fake clip")

                class Client:
                    def wait(inner, prompt_id, notice):
                        self.assertEqual(prompt_id, "existing")
                        return source

                with patch(__name__ + ".probe", return_value=5):
                    result = studio.render_shot(Client(), shot, "image.png")
                self.assertTrue(result.is_file())
                self.assertEqual(shot["attempts"], 1)
                self.assertEqual(shot["status"], "rendered")

        def test_shutdown_gate(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                with patch("os.system") as system:
                    with self.assertRaises(RuntimeError):
                        studio.teardown()
                    system.assert_not_called()

        def test_shutdown_does_not_require_backup(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                final = Path(tmp) / "final_movie.mp4"
                final.write_bytes(b"verified local movie")
                studio.store.data = {"schema": 1, "folder": tmp, "movie_ready": True,
                    "final_file": str(final), "final_sha256": digest(final), "shots": [],
                    "cloud": {"provider": "None", "status": "skipped"}}
                with patch(__name__ + ".validate_shutdown", return_value="12345"), \
                     patch("os.sync", create=True), patch("os.system", return_value=0) as system:
                    studio.teardown()
                self.assertIn("sleep 10 && vastai stop instance 12345", system.call_args.args[0])
                self.assertEqual(studio.store.data["status"], "shutdown_scheduled")

        def test_empty_cloud_config_never_blocks_start(self):
            configurations = [("None", "", ""), ("Google Drive", "", ""),
                              ("Dropbox", "/Movies", ""), ("Google Drive", "", "token")]
            for provider, destination, secret in configurations:
                with self.subTest(provider=provider, destination=destination, secret=bool(secret)), \
                     tempfile.TemporaryDirectory() as tmp:
                    studio = Studio(Path(tmp), acquire_lock=False)
                    reference = Path(tmp) / "ref.png"
                    reference.write_bytes(b"reference")
                    with patch(__name__ + ".workflow_config", return_value=({}, {})), \
                         patch.object(studio, "launch"), \
                         patch(__name__ + ".credential_spec") as credential_writer, \
                         patch.dict(os.environ, {"STUDIO_CREDENTIALS_FILE": ""}):
                        studio.start("A traveler arrives.", str(reference), 1, provider, destination, secret)
                    self.assertEqual(studio.store.data["cloud"]["status"], "skipped")
                    credential_writer.assert_not_called()

        def test_completed_clip_integrity_recovery(self):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                clip = folder / "shot_0001.mp4"
                clip.write_bytes(b"completed clip")
                clip.with_suffix(".ts").write_bytes(b"transport")
                studio = Studio(folder, acquire_lock=False)
                shot = {"shot_id": 1, "status": "completed", "clip_sha256": digest(clip),
                        "prompt_id": "original", "attempts": 1}
                studio.store.data = {"schema": 1, "folder": tmp, "shots": [shot]}
                studio.store.save()
                studio.recover_clips()
                self.assertEqual(shot["status"], "completed")
                self.assertEqual(len(studio.streams), 1)
                clip.write_bytes(b"corrupted")
                studio.recover_clips()
                self.assertEqual(shot["status"], "submitted")
                self.assertEqual(studio.store.data["pending_shot_ids"], [1])

        def test_exclusive_workspace_lock(self):
            with tempfile.TemporaryDirectory() as tmp:
                first = WorkspaceLock(Path(tmp))
                try:
                    with self.assertRaises(RuntimeError):
                        WorkspaceLock(Path(tmp))
                finally:
                    first.close()
                second = WorkspaceLock(Path(tmp))
                second.close()

        def test_pipeline_failure_isolation(self):
            from contextlib import ExitStack
            from types import SimpleNamespace
            for outcome in ("verified", "cloud_error", "none", "empty_credentials", "assembly_error", "planned"):
                with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                    studio = Studio(Path(tmp), acquire_lock=False)
                    shots = structural_shots("A traveler arrives.", 2)
                    for shot in shots:
                        shot.update(status="pending", attempts=0)
                    studio.store.data = {"schema": 1, "job_id": "test", "folder": tmp,
                        "shot_count": 2, "shots": [] if outcome == "planned" else shots,
                        "plot": "A traveler arrives.", "master_seed": 246, "workflow": {}, "comfy": {},
                        "reference": "ref", "media": {}, "shutdown": True,
                        "cloud": {"provider": "None" if outcome in {"none", "planned"} else "Dropbox",
                                  "destination": "/Movies",
                                  "credentials": {} if outcome == "empty_credentials" else {"file": "mock"}},
                        "master_shot_ids": []}
                    source = Path(tmp) / "source.mp4"
                    source.write_bytes(b"video")

                    class Client:
                        def __init__(self, _):
                            self.session = SimpleNamespace(close=lambda: None)

                        def upload(self, *args):
                            return "ref"

                        def reconcile(self, marker):
                            return None

                        def submit(self, workflow, shot, reference, marker):
                            return str(shot["shot_id"])

                        def wait(self, prompt_id, notice):
                            if prompt_id == "1":
                                raise ValueError("failed GPU execution")
                            return source

                        def request(self, *args, **kwargs):
                            return {}

                    def fake_normalize(source, clip, media):
                        clip.write_bytes(b"normalized")

                    def fake_preview(clip):
                        transport = clip.with_suffix(".ts")
                        transport.write_bytes(b"transport")
                        return transport

                    def fake_assemble():
                        if outcome == "assembly_error":
                            raise RuntimeError("disk failure")
                        (Path(tmp) / "current_movie_stream.mp4").write_bytes(b"master")

                    def fake_upload(*args):
                        upload_calls.append(1)
                        if outcome == "cloud_error":
                            raise OSError("network outage")
                        return {"verified_at": utc(), "checksum": "verified"}

                    upload_calls = []
                    with ExitStack() as stack:
                        for target, value in [("ComfyClient", Client), ("credentials", lambda _: {}),
                            ("plan_shots", lambda plot, count, notice: structural_shots(plot, count)),
                            ("cloud_preflight", lambda *args: None), ("probe", lambda _: 5),
                            ("normalize", fake_normalize), ("preview", fake_preview), ("upload_cloud", fake_upload)]:
                            stack.enter_context(patch(__name__ + "." + target, value))
                        stack.enter_context(patch("shutil.which", return_value="installed"))
                        stack.enter_context(patch("time.sleep"))
                        stack.enter_context(patch.object(studio, "assemble", fake_assemble))
                        teardown = stack.enter_context(patch.object(studio, "teardown"))
                        studio.pipeline()
                        shots = studio.store.data["shots"]
                        if outcome == "planned":
                            self.assertEqual([s["seed"] for s in shots], [246, 246])
                        if outcome != "assembly_error":
                            if outcome == "cloud_error":
                                teardown.assert_not_called()
                                self.assertEqual(studio.store.data["status"], "backup_error")
                            else:
                                teardown.assert_called_once()
                            self.assertEqual(shots[0]["status"], "failed")
                            self.assertEqual(shots[1]["status"], "completed")
                            self.assertTrue(studio.store.data["partial"])
                            self.assertEqual(studio.store.data["cloud"]["status"],
                                             {"verified": "verified", "cloud_error": "error",
                                              "none": "skipped", "empty_credentials": "skipped", "planned": "skipped"}[outcome])
                            self.assertTrue(studio.store.data["movie_ready"])
                            self.assertEqual(len(upload_calls),
                                             {"verified": 1, "cloud_error": 3,
                                              "none": 0, "empty_credentials": 0, "planned": 0}[outcome])
                        else:
                            teardown.assert_not_called()
                            self.assertEqual(studio.store.data["status"], "recoverable_error")

    # Expected failure simulations should not look like real production crashes.
    with patch.object(LOG, "disabled", True):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = Path(os.getenv("STUDIO_DIR", "/workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    handler = RotatingFileHandler(root / "studio.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    os.chmod(root / "studio.log", 0o600)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
    LOG.addHandler(handler)
    LOG.addHandler(logging.StreamHandler())
    username, password = os.getenv("STUDIO_USER"), os.getenv("STUDIO_PASSWORD")
    if bool(username) != bool(password):
        raise ValueError("Set both STUDIO_USER and STUDIO_PASSWORD")
    studio = Studio(root)
    app = build_app(studio)
    studio.resume()
    app.queue(max_size=16, default_concurrency_limit=4).launch(
        server_name=os.getenv("STUDIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("STUDIO_PORT", "7860")),
        auth=(username, password) if username else None,
        blocked_paths=[str(root / "state.json"), str(root / "studio.log"), str(root / "credentials.key"),
                       str(Path(os.getenv("GOOGLE_OAUTH_CLIENT_FILE", str(root / "google_oauth_client.json"))).resolve()),
                       str(Path(os.getenv("STUDIO_CREDENTIALS_FILE", str(root / ".credentials"))).resolve())])


if __name__ == "__main__":
    main()
