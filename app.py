"""Autonomous AI Movie Studio -- one Python file, Python 3.11+.

Install:
  pip install gradio requests cryptography google-api-python-client google-auth-httplib2 dropbox
  # Install ffmpeg/ffprobe separately; install/configure vastai for teardown.

Required configuration (IDs refer to your exported ComfyUI API workflow):
  COMFY_WORKFLOW=/workspace/workflow_api.json
  COMFY_PROMPT_NODE=6 COMFY_IMAGE_NODE=10 COMFY_VIDEO_NODE=42
Optional input names: COMFY_PROMPT_INPUT=text COMFY_IMAGE_INPUT=image
Optional seed mapping: COMFY_SEED_NODE=3 COMFY_SEED_INPUT=seed
Paths: STUDIO_DIR=/workspace COMFY_OUTPUT_DIR=/workspace/ComfyUI/output
API: COMFY_URL=http://127.0.0.1:8188
Local OpenAI-compatible LLM: LLM_URL=http://127.0.0.1:11434/v1
  LLM_MODEL=your-local-model  (unset means deterministic structural planning)
  LLM_FAILURE_POLICY=structural (or fail); LLM_BATCH_SIZE=12
Media: MOVIE_WIDTH=1920 MOVIE_HEIGHT=1080 MOVIE_FPS=24
  Workflow must generate five-second silent video clips. Audio is intentionally
  removed to keep every segment compatible. Upscaling cannot create detail.

Cloud backup is optional (provider None by default); incomplete settings skip it.
Credentials may be a plain API access token, Google service-account JSON, or
refreshable OAuth JSON (recommended for a long-running movie):
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
* A verified local final movie is required for teardown. Cloud backup success,
  failure, or omission never prevents the requested ten-second GPU stop hook.
* Upload SDK sessions retry in-process; a restart may re-upload bytes, but reuses
  the same Drive file ID / Dropbox path instead of creating duplicate movies.
* Rebuilding the cumulative MP4 copies all completed bytes each time (quadratic
  aggregate I/O); preview playback itself streams each clip only once.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import queue
import random
import re
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
from typing import Any, Callable, Iterator


LOG = logging.getLogger("movie_studio")
ATTEMPTS = 3
TERMINAL = {"complete", "partial", "shutdown_scheduled"}


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
        beat = beats[min(len(beats) - 1, index * len(beats) // count)]
        shots.append({"shot_id": index + 1,
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
                        "and duration_seconds=5. Maintain the same protagonist, costume, and locations. "
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
                    result.append({key: value[key] for key in
                                   ("visual_prompt", "camera_movement", "lighting_cue", "duration_seconds")}
                                  | {"shot_id": start + offset + 1})
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
        text = f"{shot['visual_prompt']} Camera: {shot['camera_movement']}. Lighting: {shot['lighting_cue']}."
        graph[self.config["prompt_node"]]["inputs"][self.config["prompt_input"]] = text
        graph[self.config["image_node"]]["inputs"][self.config["image_input"]] = reference
        if self.config["seed_node"]:
            graph[self.config["seed_node"]]["inputs"][self.config["seed_input"]] = shot["seed"]
        result = self.request("POST", "/prompt", json={"prompt": graph, "client_id": marker,
            "extra_data": {"extra_pnginfo": {"studio_marker": marker}}})
        if result.get("node_errors") or not result.get("prompt_id"):
            raise ValueError("ComfyUI rejected workflow inputs; inspect workflow in ComfyUI")
        return result["prompt_id"]

    def wait(self, prompt_id: str, notice: Callable[[str], None]) -> Path:
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
                    return self.video(record)
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


def workflow_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = {"url": os.getenv("COMFY_URL", "http://127.0.0.1:8188"),
              "output_dir": str(Path(os.getenv("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output")).resolve()),
              **{key: os.getenv(variable, default) for key, variable, default in [
                  ("prompt_node", "COMFY_PROMPT_NODE", ""), ("image_node", "COMFY_IMAGE_NODE", ""),
                  ("video_node", "COMFY_VIDEO_NODE", ""), ("seed_node", "COMFY_SEED_NODE", ""),
                  ("prompt_input", "COMFY_PROMPT_INPUT", "text"), ("image_input", "COMFY_IMAGE_INPUT", "image"),
                  ("seed_input", "COMFY_SEED_INPUT", "seed")]}}
    graph = json.loads(Path(os.getenv("COMFY_WORKFLOW", "/workspace/workflow_api.json")).read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or "nodes" in graph:
        raise ValueError("Workflow must be exported in API format")
    for role in ("prompt", "image", "video", "seed"):
        node = config[role + "_node"]
        if role == "seed" and not node:
            continue
        if node not in graph:
            raise ValueError(f"Configure COMFY_{role.upper()}_NODE")
        if role != "video" and config[role + "_input"] not in graph[node].get("inputs", {}):
            raise ValueError(f"Configure COMFY_{role.upper()}_INPUT")
    return graph, config


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

    def start(self, plot: str, image: str, minutes: float, provider: str,
              destination: str, secret: str, shutdown: bool = True) -> None:
        with self.guard:
            if self.running():
                raise ValueError("A movie is already running; attach to the current job")
            if self.store.data and self.store.data.get("status") not in TERMINAL:
                self.launch()
                return
            if not plot.strip() or not image:
                raise ValueError("Provide both a movie plot and character reference image")
            count = int(round(float(minutes) * 60 / 5))
            if not 1 <= count <= 720:
                raise ValueError("Duration must be between five seconds and 60 minutes")
            provider = provider or "None"
            secret = secret or ""
            if provider not in {"None", "Google Drive", "Dropbox"}:
                raise ValueError("Select a cloud provider")
            destination = (destination or "").strip()
            if provider == "Dropbox" and destination != "/":
                destination = destination.rstrip("/")
            graph, config = workflow_config()
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
            suffix = Path(image).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                raise ValueError("Use PNG, JPEG, WebP, or BMP character reference")
            reference = folder / ("reference" + suffix)
            atomic_bytes(reference, Path(image).read_bytes())
            width = int(os.getenv("MOVIE_WIDTH", "1920"))
            height = int(os.getenv("MOVIE_HEIGHT", "1080"))
            fps = int(os.getenv("MOVIE_FPS", "24"))
            if min(width, height) < 64 or width % 2 or height % 2 or not 1 <= fps <= 120:
                raise ValueError("Media dimensions must be even and >=64; fps must be 1–120")
            if self.store.path.exists():
                atomic_bytes(Path(self.store.data["folder"]) / "state.archive.json", self.store.path.read_bytes())
            self.store.data = {"schema": 1, "job_id": job_id, "folder": str(folder), "created_at": utc(),
                "status": "planning", "plot": plot, "shot_count": count, "shots": [], "reference": str(reference),
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

    def render_shot(self, client: ComfyClient, shot: dict[str, Any], reference: str) -> Path | None:
        """Persist every attempt before submitting. Reuse IDs after network errors."""
        state = self.store.data
        folder = Path(state["folder"])
        raw = folder / f"raw_{shot['shot_id']:04d}.mp4"
        if shot.get("status") == "rendered" and raw.is_file() and shot.get("raw_sha256") == digest(raw):
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

        Once all attempts finish, teardown runs even if backup failed. The local
        MP4 and manifest remain on the instance's persistent disk for recovery.
        """
        state = self.store.data
        cloud = state.setdefault("cloud", {"provider": "None"})
        if cloud.get("status") in {"verified", "error"}:
            self.notice(f"Cloud backup {cloud['status']}: "
                        + ("checksum already verified." if cloud["status"] == "verified"
                           else "configuration/upload failed; continuing to teardown."))
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
                        "Local movie retained; continuing to automatic shutdown.")

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
                    shot.update(status="pending", attempts=0, seed=uuid.uuid4().int % (2**63))
                with self.store.lock:
                    state["shots"] = shots
                    state["status"] = "rendering"
                    self.store.save()
            else:
                with self.store.lock:
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
                    reference = retry(lambda: client.upload(Path(state["reference"]), state["job_id"]),
                                      "Reference image upload", self.notice)
                    for shot in remaining:
                        if self.assembly_error:
                            raise RuntimeError("Assembly failed; stopping producer")
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

    def start(plot: str, image: str, minutes: float, provider: str,
              destination: str, secret: str) -> Iterator[Any]:
        try:
            studio.start(plot, image, minutes, provider, destination, secret)
        except Exception as exc:
            # Validation errors contain no OAuth payloads; never surface provider exceptions.
            if isinstance(exc, (ValueError, FileNotFoundError)):
                raise gr.Error(str(exc)) from None
            raise gr.Error(f"Could not start studio ({type(exc).__name__}); check configuration.") from None
        yield from monitor()

    with gr.Blocks(title="Autonomous AI Movie Studio") as app:
        gr.Markdown("# Autonomous AI Movie Studio\nPersistent production · Progressive playback · Optional cloud backup")
        with gr.Row():
            with gr.Column(scale=2):
                plot = gr.Textbox(label="Movie plot / storyboard", lines=8)
                image = gr.Image(label="Character reference", type="filepath", sources=["upload"])
                minutes = gr.Slider(1, 60, value=1, step=1, label="Target duration (minutes)")
                with gr.Accordion("📦 Cloud Storage Backup (Optional, Recommended)", open=False):
                    provider = gr.Dropdown(["None", "Google Drive", "Dropbox"], value="None", label="Cloud Provider")
                    destination = gr.Textbox(label="Cloud Destination Folder ID / Path",
                                             placeholder="Drive folder ID, or Dropbox /Movies (use / for Dropbox root)")
                    secret = gr.Textbox(label="API Access Token / Service Account JSON", type="password",
                                        placeholder="Optional; token, service-account JSON, or refreshable OAuth JSON")
                    gr.Markdown("Leave provider as None or settings empty to skip backup. "
                                "A configured STUDIO_CREDENTIALS_FILE can supply credentials. "
                                "Refreshable OAuth credentials are recommended for long jobs. "
                                "The final movie is saved locally before backup. Shutdown follows in 10 seconds "
                                "after backup succeeds, fails, or is skipped.")
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
        button.click(start, [plot, image, minutes, provider, destination, secret], outputs,
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
    from unittest.mock import patch

    class Tests(unittest.TestCase):
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
            for outcome in ("verified", "cloud_error", "none", "empty_credentials", "assembly_error"):
                with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                    studio = Studio(Path(tmp), acquire_lock=False)
                    shots = structural_shots("A traveler arrives.", 2)
                    for shot in shots:
                        shot.update(status="pending", attempts=0)
                    studio.store.data = {"schema": 1, "job_id": "test", "folder": tmp,
                        "shot_count": 2, "shots": shots, "workflow": {}, "comfy": {},
                        "reference": "ref", "media": {}, "shutdown": True,
                        "cloud": {"provider": "None" if outcome == "none" else "Dropbox",
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
                            ("cloud_preflight", lambda *args: None), ("probe", lambda _: 5),
                            ("normalize", fake_normalize), ("preview", fake_preview), ("upload_cloud", fake_upload)]:
                            stack.enter_context(patch(__name__ + "." + target, value))
                        stack.enter_context(patch("shutil.which", return_value="installed"))
                        stack.enter_context(patch("time.sleep"))
                        stack.enter_context(patch.object(studio, "assemble", fake_assemble))
                        teardown = stack.enter_context(patch.object(studio, "teardown"))
                        studio.pipeline()
                        if outcome != "assembly_error":
                            teardown.assert_called_once()
                            self.assertEqual(shots[0]["status"], "failed")
                            self.assertEqual(shots[1]["status"], "completed")
                            self.assertTrue(studio.store.data["partial"])
                            self.assertEqual(studio.store.data["cloud"]["status"],
                                             {"verified": "verified", "cloud_error": "error",
                                              "none": "skipped", "empty_credentials": "skipped"}[outcome])
                            self.assertTrue(studio.store.data["movie_ready"])
                            self.assertEqual(len(upload_calls),
                                             {"verified": 1, "cloud_error": 3,
                                              "none": 0, "empty_credentials": 0}[outcome])
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
                       str(Path(os.getenv("STUDIO_CREDENTIALS_FILE", str(root / ".credentials"))).resolve())])


if __name__ == "__main__":
    main()
