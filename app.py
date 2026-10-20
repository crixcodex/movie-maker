"""Autonomous AI Movie Studio -- one Python file, Python 3.11+.

Install:
  pip install gradio requests cryptography google-api-python-client google-auth-httplib2 google-auth-oauthlib dropbox
  # Install ffmpeg/ffprobe separately; install/configure vastai for teardown.

Automatic first-run setup (default when no workflow file exists):
  Discovers the local ComfyUI API (including Vast's internal port 18188), creates
  a core Wan 2.2 TI2V workflow, and downloads missing official public weights.
  Downloads are resumable and SHA-256 verified; first run requires about 18 GB
  of model storage plus space for clips. A CUDA GPU with sufficient memory is
  required. The core profile uses native reference conditioning and locked
  prompts/seeds; independent adapters/temporal chaining need a custom workflow.
  python app.py --prepare  # optional: provision without rendering/shutdown

Optional custom configuration (IDs refer to your exported ComfyUI API workflow):
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
  Custom anchor workflow batch size must be 1. Automatic mode reuses the Wan
  TI2V model for its text-to-image master anchor without another model download.
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
New-job scene ledger:
  Plain prose uses a local LLM story-bible extractor when LLM_MODEL is configured,
  otherwise a conservative structural cast parser. Optional JSON input accepts
  plot, characters, objects, creative_constraints and an exact storyboard.
  Per-character appearance, wardrobe, narrative psychology, deterministic seed,
  root state [x,y,z,vx,vy,vz,mass], and per-scene lighting are checkpointed.
  Newtonian trajectories and energy residuals audit the AUTHORED plan only.
  Object attachments are kinematic constraints, not solved joint dynamics.
  Lighting uses a simple positive-semidefinite proxy, NOT global illumination.
  No quantum solver, pixel entanglement, PINN training, or drift elimination is
  claimed. A PNG is a reference image, not a latent tensor. SDR grading does not
  recreate a camera's log response or photochemical film emulsion.
  All cast anchors are generated/cached; independent image conditioning needs
  consistency character_bindings=[{character_id,node,input}] in a custom graph.
  require_multi_character_conditioning=true rejects missing cast loaders.
  STUDIO_REQUIRE_NATIVE_LATENTS=1 rejects unsupported temporal conditioning.
  Native continuity may specify tensor_contract={tensor_key:'latent_tensor',
    shape:[1,48,1,44,80],dtype:'F16'} (EXAMPLE ONLY; use actual model dimensions).
  Contracted safetensors windows are checked for shape, dtype, NaN/Inf, checksum,
  and model/VAE configuration signature. They are passed unchanged to the next
  same-scene boundary; no incompatible latent is fabricated from an MP4.
Follow-up queue:
  python app.py --enqueue next_movie_30min.json --minutes 30
  Requests persist in production_queue.json. Successful jobs archive their own
  final_movie.mp4 before handing off. GPU stop is deferred until the queue drains;
  backup failures still block both handoff and shutdown. Queued jobs are text-only,
  local-storage jobs with sound/grading; no credentials enter the queue.
  Structured plot render_settings may freeze width/height, crf (12–23), grain
  (0–5), preset (medium/slow/slower), maxrate_mbps + bufsize_mbit and
  retain_raw_copies. The supplied 30-minute film targets native-width 1280x536,
  CRF 18, light grain, slow encoding and a 5 Mbps video ceiling for this instance's
  limited disk. retain_raw_copies=false removes only verified redundant private
  raw copies after assembly; ComfyUI originals, prompt IDs, graded clips, PCM
  audio and final archives remain. Native temporal-continuity jobs retain raw.
Paths: STUDIO_DIR=/workspace COMFY_OUTPUT_DIR=/workspace/ComfyUI/output
API: COMFY_URL=http://127.0.0.1:8188
Local OpenAI-compatible LLM: LLM_URL=http://127.0.0.1:11434/v1
  LLM_MODEL=your-local-model  (unset means deterministic structural planning)
  LLM_FAILURE_POLICY=structural (or fail); LLM_BATCH_SIZE=12
Media: MOVIE_WIDTH=1920 MOVIE_HEIGHT=1080 MOVIE_FPS=24
  Workflow must generate five-second video clips. Source audio is replaced with
  context-selected procedural sound layers (rain/wind/water/fire/city/engines/
  footsteps), mixed to stereo 48 kHz. This is an ambient fallback, not a neural
  Foley model, speech generator, or frame-aligned event detector.
  New movies default to 2.39:1 center crop (1920x804 at width 1920), subtle luma
  grain, restrained teal/orange grading, and mild spatial/temporal denoising.
  MOVIE_CRF=16 POST_THREADS=4; H.264 medium/grain, stereo AAC 320k, 48 kHz.
  Set MOVIE_WIDTH=3840 for a 3840x1606 scope export (more CPU/storage).
  Upscaling cannot create missing source detail. Both sound and cinematic finish
  can be disabled in the GUI. Checkpoints freeze their finishing profile; legacy
  jobs stay silent/non-graded. PCM segments are concatenated before master AAC
  encoding to prevent cumulative encoder-delay drift across long movies.

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
Real FFmpeg integration test (no GPU/cloud/shutdown): python app.py --media-self-test

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
import wave
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
            f"LOCKED OBJECTS: {shot.get('object_tokens', OBJECT_ANCHOR)}."
            + ("\nPLANNED CONTINUITY (conditioning instructions, not measured video): "
               + shot["world_state"]["conditioning"] if shot.get("world_state") else ""))


def story_input(text: str) -> dict[str, Any]:
    """Plain prose is supported; optional JSON supplies an explicit cast ledger."""
    if text.lstrip().startswith("{"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Structured story input must be valid JSON") from None
        if not isinstance(value, dict) or not isinstance(value.get("plot"), str) or not value["plot"].strip():
            raise ValueError("Structured story input requires a nonempty plot string")
        return value
    return {"plot": text}


def vector3(value: Any, limit: float = 1e6) -> list[float]:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(v) not in {int, float} or not math.isfinite(v) or abs(v) > limit for v in value)):
        raise ValueError("World vectors must contain three finite, bounded numbers")
    return [float(v) for v in value]


def build_story_bible(plot: str, seed: int, notice: Callable[[str], None]) -> dict[str, Any]:
    """Semantic LLM extraction when configured; conservative structural fallback.

    This is an authored scene ledger, not a learned physical world model. A
    numerical v_c contains [position_m, velocity_m_s, mass_kg]; identity,
    wardrobe and narrative psychology remain named, auditable attributes.
    """
    source = story_input(plot)
    origin = "structured" if "characters" in source else "structural"
    if "characters" not in source and os.getenv("LLM_MODEL"):
        try:
            import requests
            response = requests.post(os.getenv("LLM_URL", "http://127.0.0.1:11434/v1").rstrip("/") + "/chat/completions",
                timeout=(10, 180), json={"model": os.environ["LLM_MODEL"], "temperature": 0,
                "messages": [{"role": "system", "content":
                    'Extract a story bible as JSON {"characters":[{"id":"protagonist","name":"...",'
                    '"appearance":"...","wardrobe":"...","psychology":"narrative goal and temperament"}],'
                    '"objects":[{"id":"prop_1","name":"...","appearance":"...","mass_kg":1}]}. '
                    "Put the protagonist first. At most 8 characters and 32 recurring objects. "
                    "Use only details supported by the plot; mark unspecified details as unspecified. "
                    "Do not invent diagnoses. No physics predictions or code."},
                    {"role": "user", "content": source["plot"]}]})
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            extracted = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content))
            if not isinstance(extracted, dict) or not isinstance(extracted.get("characters"), list) or not extracted["characters"]:
                raise ValueError("Invalid extracted cast")
            # Validate model output through the same strict ledger parser below.
            source = {"plot": source["plot"], **{k: extracted[k] for k in ("characters", "objects") if k in extracted}}
            origin = "local_llm"
        except Exception:
            notice("Story-bible LLM unavailable; using a conservative structural cast ledger.")
    if not source.get("characters"):
        names = re.findall(r"(?:named|called|character:|protagonist:)\s+([A-Z][a-zA-Z-]{1,40})", source["plot"])
        for pair in re.findall(r"\b([A-Z][a-z]{1,30})\s+(?:and|meets|follows|helps|chases)\s+([A-Z][a-z]{1,30})\b", source["plot"]):
            names.extend(pair)
        names = list(dict.fromkeys(names)) or ["Primary protagonist"]
        source["characters"] = [{"id": f"character_{i+1}", "name": name} for i, name in enumerate(names)]
    def text(value: Any, default: str) -> str:
        if value is None:
            return default
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError("Story attributes must be nonempty strings of at most 2000 characters")
        return value.strip()
    def entities(values: Any, limit: int, character: bool) -> list[dict[str, Any]]:
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(f"Story ledger supports at most {limit} entities in this category")
        result = []
        identifiers = set()
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                raise ValueError("Each story entity must be a JSON object")
            entity_id = item.get("id", f"{'character' if character else 'object'}_{index+1}")
            if not isinstance(entity_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", entity_id) or entity_id in identifiers:
                raise ValueError("Story entity IDs must be unique safe identifiers")
            identifiers.add(entity_id)
            if (not character and item.get("attached_to") is not None
                    and (not isinstance(item["attached_to"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", item["attached_to"]))):
                raise ValueError("Object attachment must use a safe character identifier")
            mass = item.get("mass_kg", 70 if character else 1)
            if type(mass) not in {float, int} or not math.isfinite(mass) or not 0.001 <= mass <= 1e9:
                raise ValueError("Entity mass must be finite and between 0.001 and 1e9 kg")
            position = vector3(item.get("position_m", [index * 1.5, 0, 0 if character else 3]))
            velocity = vector3(item.get("velocity_m_s", [0, 0, 0]), 1000)
            if position[1] < 0:
                raise ValueError("Entity starts below the authored ground plane")
            result.append({"id": entity_id, "name": text(item.get("name"), entity_id),
                "appearance": text(item.get("appearance"), "follow the plot and immutable master reference"),
                "wardrobe": text(item.get("wardrobe"), "same clothing, silhouette, materials and accessories throughout") if character else None,
                "psychology": text(item.get("psychology"), "narrative motivation unspecified; maintain coherent reactions") if character else None,
                "mass_kg": float(mass), "mass_source": "authored" if "mass_kg" in item else "planning_assumption",
                "position_m": position, "velocity_m_s": velocity,
                "attached_to": item.get("attached_to") if not character else None,
                "attachment_offset_m": vector3(item.get("attachment_offset_m", [0, 0.9, 0])) if not character else None,
                "state_vector": position + velocity + [float(mass)],
                "seed": int.from_bytes(hashlib.sha256(f"{seed}:{entity_id}".encode()).digest()[:8], "big") % (2**63)})
        return result
    try:
        characters = entities(source["characters"], 8, True)
        objects = entities(source.get("objects", []), 32, False)
        if {c["id"] for c in characters}.intersection(o["id"] for o in objects):
            raise ValueError("Character and object IDs must be globally unique")
        if any(o.get("attached_to") is not None and o["attached_to"] not in {c["id"] for c in characters} for o in objects):
            raise ValueError("Object attachment must name an existing character ID")
    except ValueError:
        if origin != "local_llm":
            raise
        notice("Story-bible LLM attributes failed validation; falling back to a primary character ledger.")
        return build_story_bible(json.dumps({"plot": source["plot"], "characters": [{"id": "protagonist", "name": "Primary protagonist"}]}), seed, notice)
    characters[0]["seed"] = seed
    return {"version": 1, "origin": origin, "plot": source["plot"],
            "creative_constraints": text(source.get("creative_constraints"), "coherent cinematic style matching the immutable master reference"), "characters": characters,
            "objects": objects, "primary_character_id": characters[0]["id"],
            "scope": "authored planning constraints; not measurements or guarantees about generated pixels"}


def advance_body(position: list[float], velocity: list[float], mass: float, seconds: float,
                 ballistic: bool = False) -> dict[str, Any]:
    """Analytic Newtonian trajectory, with a perfectly inelastic ground contact.

    H_classical=p^2/(2m)+m*g*y is an audit quantity. No Schrödinger solver,
    quantum entanglement, PINN training, or diffusion-gradient guidance occurs.
    """
    position, velocity = vector3(position), vector3(velocity, 1000)
    if not math.isfinite(seconds) or not 0 < seconds <= 5 or not math.isfinite(mass) or mass <= 0:
        raise ValueError("Invalid trajectory duration or mass")
    if position[1] < 0 or (not ballistic and abs(velocity[1]) > 1e-9):
        raise ValueError("Invalid grounded trajectory")
    g = 9.80665
    acceleration = [0.0, -g if ballistic else 0.0, 0.0]
    duration = seconds
    contact = False
    if ballistic:
        impact = (velocity[1] + math.sqrt(velocity[1] ** 2 + 2 * g * position[1])) / g
        if impact <= seconds:
            duration, contact = max(0.0, impact), True
    endpoint = [position[i] + velocity[i] * seconds for i in range(3)]
    endpoint[1] = max(0.0, position[1] + velocity[1] * duration + acceleration[1] * duration**2 / 2)
    final_velocity = [velocity[i] + acceleration[i] * duration for i in range(3)]
    dissipated = mass * final_velocity[1] ** 2 / 2 if contact else 0.0
    if contact:
        endpoint[1] = 0.0
        final_velocity[1] = 0.0
    initial_energy = mass * sum(v*v for v in velocity) / 2 + mass * g * position[1]
    final_energy = mass * sum(v*v for v in final_velocity) / 2 + mass * g * endpoint[1]
    residual = final_energy + dissipated - initial_energy
    if not math.isfinite(residual) or abs(residual) > max(1e-5, abs(initial_energy) * 1e-9):
        raise ValueError("Planned trajectory failed its mechanical energy balance")
    return {"position_m": endpoint, "velocity_m_s": final_velocity, "mass_kg": mass,
            "state_vector": endpoint + final_velocity + [mass], "ground_contact": contact,
            "dissipated_energy_j": dissipated, "mechanical_energy_j": final_energy,
            "energy_residual_j": residual, "rest_energy_j": mass * 299792458**2}


def bind_world_state(shots: list[dict[str, Any]], bible: dict[str, Any]) -> None:
    """Freeze authored per-scene physics/lighting instructions before any POST."""
    previous: dict[str, Any] | None = None
    for shot in shots:
        if shot.get("world_state"):
            previous = shot
            continue  # Never change a checkpointed prompt or simulated boundary.
        same_scene = previous is not None and previous["scene_id"] == shot["scene_id"]
        prior = previous["world_state"]["entities"] if same_scene else {}
        scene_text = (shot["visual_prompt"] + " " + shot["lighting_cue"]).lower()
        light = (copy.deepcopy(previous["world_state"]["lighting"]) if same_scene else
                 {"temperature_k": 3200 if re.search(r"\b(night|candle|tungsten)\b", scene_text) else 5600,
                  "key_direction_world": [-0.4, 0.8, 0.4472135955], "ambient": 0.2, "key_intensity": 0.8})
        direction = light["key_direction_world"]
        # Positive semidefinite lighting proxy, NOT simulated global illumination.
        light["illumination_proxy"] = [[light["ambient"] * int(i == j) + light["key_intensity"] * direction[i] * direction[j]
                                        for j in range(3)] for i in range(3)]
        entities = {}
        for entity in bible["characters"] + bible["objects"]:
            start = prior.get(entity["id"], {}).get("end", entity)
            position = start["position_m"][:]
            velocity = start["velocity_m_s"][:]
            # Prescribed root motion needs external forces: report the impulse
            # rather than falsely asserting conservation for an accelerated actor.
            command = None
            mentioned = entity in bible["characters"] and (len(bible["characters"]) == 1 or entity["name"].lower() in scene_text)
            if mentioned:
                command = (0.0 if re.search(r"\b(stands|waits|stops|remains still)\b", scene_text) else
                           4.5 if re.search(r"\b(run|runs|running)\b", scene_text) else
                           1.3 if re.search(r"\b(walk|walks|walking|stroll|strolls|strolling)\b", scene_text) else None)
            requested_velocity = velocity[:] if command is None else [command, velocity[1], velocity[2]]
            ballistic = bool(re.search(r"\b(falling|freefall|free-fall)\b", scene_text)) and mentioned
            if not ballistic:
                requested_velocity[1] = 0.0
            end = advance_body(position, requested_velocity, entity["mass_kg"], 5, ballistic)
            impulse = [entity["mass_kg"] * (requested_velocity[i] - velocity[i]) for i in range(3)]
            entities[entity["id"]] = {"start_position_m": position, "start_velocity_m_s": requested_velocity,
                                      "authored_impulse_ns": impulse, "end": end}
            if entity.get("attached_to"):
                parent = entities[entity["attached_to"]]
                offset = entity["attachment_offset_m"]
                attached = entities[entity["id"]]
                attached["start_position_m"] = [parent["start_position_m"][i] + offset[i] for i in range(3)]
                attached["start_velocity_m_s"] = parent["start_velocity_m_s"][:]
                attached["authored_impulse_ns"] = [entity["mass_kg"] * (attached["start_velocity_m_s"][i] - velocity[i]) for i in range(3)]
                attached["end"]["position_m"] = [parent["end"]["position_m"][i] + offset[i] for i in range(3)]
                attached["end"]["velocity_m_s"] = parent["end"]["velocity_m_s"][:]
                attached["end"]["state_vector"] = attached["end"]["position_m"] + attached["end"]["velocity_m_s"] + [entity["mass_kg"]]
                attached["end"]["mechanical_energy_j"] = entity["mass_kg"] * (
                    sum(v*v for v in attached["end"]["velocity_m_s"]) / 2 + 9.80665 * attached["end"]["position_m"][1])
                attached["end"]["energy_residual_j"] = None
                attached["end"]["dissipated_energy_j"] = None
                attached["attachment"] = {"parent": entity["attached_to"], "offset_m": offset,
                                           "model": "kinematic constraint; joint forces and work not solved"}
        identities = "; ".join(f"{c['id']} ({c['name']}): {c['appearance']}; wardrobe: {c['wardrobe']}; motivation: {c['psychology']}"
                               for c in bible["characters"])
        props = "; ".join(f"{o['id']}: {o['appearance']}, fixed mass {o['mass_kg']} kg" for o in bible["objects"])
        motion = "; ".join(f"{name}: root position {item['start_position_m']} -> {item['end']['position_m']} m; "
                           f"root velocity {item['start_velocity_m_s']} m/s" for name, item in entities.items())
        shot["world_state"] = {"version": 1, "source": "authored_simulation_not_video_measurement",
            "entities": entities, "lighting": light, "scene_boundary": not same_scene,
            "conditioning": (f"CAST: {identities}. RECURRING OBJECTS: {props or 'preserve established props'}. "
                f"PLANNED ROOT MOTION: {motion}. Gravity 9.80665 m/s^2; supported bodies stay on the ground, "
                "objects keep shape and mass, no unexplained duplication; fluid motion remains coherent. "
                f"World-space key direction {direction}, color temperature {light['temperature_k']} K; "
                "keep shadow direction, reflections and exposure consistent within the scene. "
                f"GLOBAL DIRECTION: {bible.get('creative_constraints', '')}")}
        previous = shot


def workflow_capabilities(config: dict[str, Any]) -> dict[str, Any]:
    profile = config.get("consistency", {})
    continuity = profile.get("continuity", {})
    mode = continuity.get("mode", "off")
    return {"identity_reference": True, "adapter_count": len(profile.get("adapters", [])),
            "continuity": mode, "native_latent_contract": bool(continuity.get("tensor_contract")),
            "multi_character_bindings": len(config.get("character_bindings", [])),
            "physical_control": "textual planning only", "quantum_solver": False, "pinn": False,
            "global_illumination_solver": False, "guaranteed_identity_or_physics": False,
            "grading": "display-referred SDR; not camera-log reconstruction"}


def inspect_terminal_latent(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Validate a bounded safetensors terminal window without pickle or torch.

    No tensor arithmetic or silent reshaping is performed. A final VAE latent is
    not the sampler's final noise state; the exported graph must define which
    representation its next-shot conditioning node actually consumes.
    """
    import array
    shape = contract.get("shape")
    dtype = contract.get("dtype")
    if (not isinstance(shape, list) or len(shape) != 5 or shape[0] != 1
            or any(type(n) is not int or n <= 0 for n in shape) or shape[2] > 8
            or math.prod(shape) > 64 * 1024**2 or dtype not in {"F16", "BF16", "F32"}):
        raise ValueError("Terminal latent contract needs bounded [1,C,T,H,W] shape (T<=8) and F16/BF16/F32 dtype")
    key = contract.get("tensor_key", "latent_tensor")
    if not isinstance(key, str) or not key or len(key) > 128:
        raise ValueError("Invalid terminal latent tensor key")
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 2 <= header_size <= 1024**2:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(header_size))
        item = header.get(key, {})
        offsets = item.get("data_offsets")
        if (item.get("shape") != shape or item.get("dtype") != dtype or not isinstance(offsets, list)
                or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets)):
            raise ValueError("Exported terminal tensor does not match the configured shape/dtype")
        length = math.prod(shape) * (4 if dtype == "F32" else 2)
        if offsets[1] - offsets[0] != length or 8 + header_size + offsets[1] > path.stat().st_size:
            raise ValueError("Terminal tensor offsets or byte count are invalid")
        stream.seek(8 + header_size + offsets[0])
        raw = stream.read(length)
    values = array.array("I" if dtype == "F32" else "H", raw)
    if sys.byteorder != "little":
        values.byteswap()
    exponent = {"F32": 0x7f800000, "F16": 0x7c00, "BF16": 0x7f80}[dtype]
    if any(value & exponent == exponent for value in values):
        raise ValueError("Terminal tensor contains NaN or infinity")
    return {"shape": shape, "dtype": dtype, "tensor_key": key, "finite": True}


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


def consistency_config(graph: dict[str, Any], config: dict[str, Any],
                       settings_override: dict[str, Any] | None = None) -> dict[str, Any]:
    filename = os.getenv("CONSISTENCY_CONFIG")
    settings = (copy.deepcopy(settings_override) if settings_override is not None else
                json.loads(Path(filename).read_text(encoding="utf-8")) if filename else {})
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


def validate_finished_media(path: Path, seconds: float, media: dict[str, Any]) -> dict[str, Any]:
    """Check both streams before cloud delivery or the GPU shutdown gate."""
    metadata = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    streams = metadata.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    settings = media["postproduction"]
    height = (max(2, 2 * round(media["width"] / settings["aspect"] / 2))
              if settings["cinematic"] else media["height"])
    if (video.get("codec_name") != "h264" or video.get("width") != media["width"]
            or video.get("height") != height or video.get("pix_fmt") != "yuv420p"):
        raise RuntimeError("Finished video codec or dimensions do not match the frozen movie profile")
    if abs(float(video.get("duration", 0)) - seconds) > 0.1:
        raise RuntimeError("Finished video timeline duration is invalid")
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if settings["audio"]:
        if (len(audio) != 1 or audio[0].get("codec_name") != "aac"
                or int(audio[0].get("sample_rate", 0)) != 48000 or audio[0].get("channels") != 2
                or abs(float(audio[0].get("duration", 0)) - seconds) > 0.1
                or abs(float(audio[0].get("start_time", 0))) > 0.05):
            raise RuntimeError("Finished AAC soundtrack is missing or out of sync")
    elif audio:
        raise RuntimeError("Unexpected soundtrack in an audio-disabled movie")
    return metadata


def postproduction_config(audio: bool = True, cinematic: bool = True) -> dict[str, Any]:
    """Freeze finishing parameters per movie; legacy checkpoints remain unchanged."""
    crf = int(os.getenv("MOVIE_CRF", "16"))
    threads = int(os.getenv("POST_THREADS", "4"))
    if not 12 <= crf <= 23 or not 1 <= threads <= 32:
        raise ValueError("MOVIE_CRF must be 12–23 and POST_THREADS must be 1–32")
    return {"version": 1, "audio": bool(audio), "cinematic": bool(cinematic),
            "aspect": 2.39, "grain": 3, "crf": crf, "preset": "medium", "threads": threads,
            "audio_rate": 48000, "audio_bitrate": "320k", "audio_channels": 2,
            "loudness_lufs": -23, "true_peak_db": -2}


def validated_render_settings(value: Any) -> dict[str, Any]:
    """Bounded per-movie delivery settings for an authored production request."""
    if not isinstance(value, dict) or set(value) - {"width", "height", "crf", "grain", "preset", "retain_raw_copies", "maxrate_mbps", "bufsize_mbit"}:
        raise ValueError("Unknown or invalid per-movie render settings")
    for key in ("width", "height"):
        if key in value and (type(value[key]) is not int or not 64 <= value[key] <= 3840 or value[key] % 2):
            raise ValueError("Delivery dimensions must be even integers between 64 and 3840")
    for key, low, high in (("crf", 12, 23), ("grain", 0, 5)):
        if key in value and (type(value[key]) is not int or not low <= value[key] <= high):
            raise ValueError(f"Render {key} must be an integer between {low} and {high}")
    if "preset" in value and value["preset"] not in {"medium", "slow", "slower"}:
        raise ValueError("Delivery preset must be medium, slow or slower")
    if "retain_raw_copies" in value and type(value["retain_raw_copies"]) is not bool:
        raise ValueError("retain_raw_copies must be boolean")
    for key, ceiling in (("maxrate_mbps", 50), ("bufsize_mbit", 100)):
        if key in value and (type(value[key]) is not int or not 1 <= value[key] <= ceiling):
            raise ValueError("Invalid bounded video bitrate settings")
    if ("maxrate_mbps" in value) != ("bufsize_mbit" in value):
        raise ValueError("Video maxrate and buffer size must be supplied together")
    return copy.deepcopy(value)


def audio_plan(shot: dict[str, Any]) -> dict[str, Any]:
    """Deterministic procedural fallback, not an event-aligned neural Foley model.

    Only the scene/action/lighting fields select sounds: identity/style anchors
    and user text never become FFmpeg expressions or executable commands.
    """
    text = " ".join(str(shot.get(key, "")) for key in
                    ("action_prompt", "visual_prompt", "lighting_cue")).lower()
    vocabulary = {
        "rain": r"\b(rain\w*|drizzl\w*|downpour|storm\w*)\b",
        "wind": r"\b(wind\w*|breeze|gust\w*|blizzard|desert)\b",
        "mechanical": r"\b(engine\w*|motor\w*|machin\w*|spaceship|spacecraft|factory|industrial|generator|robot\w*|train|subway)\b",
        "water": r"\b(ocean|sea|surf|wave\w*|river|stream|waterfall|beach)\b",
        "fire": r"\b(fire|flame\w*|campfire|burning)\b",
        "city": r"\b(city|street\w*|traffic|urban|downtown|highway)\b",
        "footsteps": r"\b(walk\w*|footstep\w*|running|runs|run|stroll\w*|hiking)\b",
    }
    cues = [name for name, pattern in vocabulary.items() if re.search(pattern, text)]
    seed = int.from_bytes(hashlib.sha256(
        f"{shot.get('seed', 0)}:{shot.get('shot_id', 0)}:sound-v1".encode()).digest()[:4], "big") % 2147483647
    plan = {"version": 1, "mode": "procedural", "cues": cues or ["room_tone"], "seed": seed}
    if shot.get("world_state"):
        plan.update(version=2, pan=-0.65 if re.search(r"\b(?:from|on|to) (?:the )?left\b", text)
                    else 0.65 if re.search(r"\b(?:from|on|to) (?:the )?right\b", text) else 0.0)
    return plan


def sound_layers(plan: dict[str, Any]) -> list[str]:
    """Return trusted lavfi sources. Each layer has its own deterministic seed."""
    seed = plan["seed"]
    def noise(color: str, level: float, offset: int) -> str:
        return f"anoisesrc=r=48000:d=5:c={color}:a={level}:s={(seed + offset) % 2147483647}"
    # A quiet bed always prevents an empty mix. Additional sources stay distinct
    # until mixing so weather, low mechanical tones and movement can coexist.
    layers = [noise("pink", 0.035, 0) + ",highpass=f=80,lowpass=f=650"]
    for cue in plan["cues"]:
        if cue == "rain":
            layers.append(noise("white", 0.18, 1) + ",highpass=f=650,lowpass=f=11000")
        elif cue == "wind":
            layers.append(noise("pink", 0.26, 2) + ",highpass=f=50,lowpass=f=1700,tremolo=f=0.35:d=0.65")
        elif cue == "mechanical":
            layers.extend(["sine=f=55:r=48000:d=5,volume=0.65,tremolo=f=2.3:d=0.12",
                           "sine=f=110:r=48000:d=5,volume=0.22",
                           noise("brown", 0.16, 3) + ",highpass=f=35,lowpass=f=420"])
        elif cue == "water":
            layers.append(noise("pink", 0.24, 4) + ",highpass=f=120,lowpass=f=5500,tremolo=f=0.22:d=0.8")
        elif cue == "fire":
            layers.append(noise("white", 0.16, 5) + ",highpass=f=180,lowpass=f=4500,tremolo=f=11:d=0.85")
        elif cue == "city":
            layers.append(noise("brown", 0.22, 6) + ",highpass=f=55,lowpass=f=1200,tremolo=f=0.7:d=0.35")
        elif cue == "footsteps":
            layers.append(noise("brown", 0.35, 7) +
                          r",highpass=f=85,lowpass=f=1100,volume='if(lt(mod(t,0.62),0.085),0.75,0.015)':eval=frame")
    return layers


def validate_pcm(path: Path) -> None:
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()) != (48000, 2, 2, 240000):
            raise ValueError("Sound segment must contain exactly five seconds of stereo 48 kHz PCM")


def generate_sound(shot: dict[str, Any], target: Path, settings: dict[str, Any],
                   notice: Callable[[str], None]) -> dict[str, Any]:
    plan = audio_plan(shot)
    sources = sound_layers(plan)
    notice(f"Shot {shot['shot_id']}: sound design — {', '.join(plan['cues'])}; mixing {len(sources)} layers.")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
    for source in sources:
        args += ["-f", "lavfi", "-i", source]
    labels = []
    filters = []
    for index in range(len(sources)):
        # Modest asymmetric levels give the layers width without phase reversal.
        left, right = ((0.90, 1.0) if index % 2 else (1.0, 0.90))
        if index and "pan" in plan:
            angle = (plan["pan"] + 1) * math.pi / 4
            left, right = math.cos(angle), math.sin(angle)  # Equal-power stereo pan.
        filters.append(f"[{index}:a]pan=stereo|c0={left}*c0|c1={right}*c0[layer{index}]")
        labels.append(f"[layer{index}]")
    filters.append("".join(labels) + f"amix=inputs={len(sources)}:duration=longest:normalize=0,"
        f"loudnorm=I={settings['loudness_lufs']}:TP={settings['true_peak_db']}:LRA=7,"
        "aresample=48000,afade=t=in:st=0:d=0.05,afade=t=out:st=4.88:d=0.12,"
        "apad=whole_len=240000,atrim=end_sample=240000,asetpts=N/SR/TB[mix]")
    pending = target.with_suffix(".pending.wav")
    args += ["-filter_complex", ";".join(filters), "-map", "[mix]", "-c:a", "pcm_s16le",
             "-ar", "48000", "-ac", "2", str(pending)]
    # The procedural engine needs no model, credentials, network or license.
    # A deterministic clean room-tone retry is the fallback if a cue fails.
    try:
        run(args, timeout=120)
        validate_pcm(pending)
    except Exception as exc:
        notice(f"Shot {shot['shot_id']}: layered sound failed ({type(exc).__name__}); using a clean ambient bed.")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             f"anoisesrc=r=48000:d=5:c=pink:a=0.05:s={plan['seed']}",
             "-af", "highpass=f=80,lowpass=f=700,afade=t=in:d=0.05,afade=t=out:st=4.88:d=0.12",
             "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-t", "5", str(pending)], timeout=120)
        validate_pcm(pending)
        plan["fallback"] = "ambient_bed"
    with pending.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(pending, target)
    return plan


def video_filters(media: dict[str, Any], seed: int = 0) -> str:
    width, height, fps = media["width"], media["height"], media["fps"]
    settings = media.get("postproduction", {})
    if settings.get("cinematic"):
        scope_height = max(2, 2 * round(width / settings["aspect"] / 2))
        filters = (f"scale={width}:{scope_height}:force_original_aspect_ratio=increase:flags=lanczos,"
                   f"crop={width}:{scope_height},setsar=1,fps={fps},"
                   "tpad=stop_mode=clone:stop_duration=0.3,"
                   "hqdn3d=0.6:0.4:1.2:0.8,"
                   "curves=master='0/0 0.18/0.14 0.5/0.51 0.82/0.87 1/1',"
                   "colorbalance=rs=-0.02:gs=0.008:bs=0.025:rh=0.025:gh=0.008:bh=-0.018:pl=1,"
                   "eq=saturation=1.04,"
                   f"noise=c0s={settings['grain']}:c0f=t+u:c0_seed={seed % 2147483647},format=yuv420p")
    else:
        filters = (f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                   "tpad=stop_mode=clone:stop_duration=0.3")
    return filters


def normalize(source: Path, target: Path, media: dict[str, Any],
              shot: dict[str, Any] | None = None,
              notice: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    seconds = probe(source)
    if abs(seconds - 5) > 0.25:
        raise ValueError(f"Workflow generated {seconds:.2f}s; expected five seconds")
    fps = media["fps"]
    settings = media.get("postproduction", {})
    details: dict[str, Any] = {}
    audio = target.with_suffix(".wav")
    if settings.get("audio"):
        if not shot:
            raise ValueError("Sound design requires the checkpointed shot description")
        details["audio_plan"] = generate_sound(shot, audio, settings, notice)
        details["audio_sha256"] = digest(audio)
    temporary = target.with_suffix(".pending.mp4")
    # Pad slightly short source clips so every successful shot is exactly 5s.
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_threads", "2", "-i", str(source)]
    if settings.get("audio"):
        args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac",
                 "-b:a", settings["audio_bitrate"], "-ar", "48000", "-ac", "2"]
    else:
        args += ["-map", "0:v:0", "-an"]
    args += ["-vf", video_filters(media, (shot or {}).get("seed", 0)), "-t", "5", "-c:v", "libx264",
             "-threads", str(settings.get("threads", 4)), "-preset", settings.get("preset", "veryfast"),
             "-crf", str(settings.get("crf", 18)), "-pix_fmt", "yuv420p", "-r", str(fps),
             "-video_track_timescale", "90000"]
    if settings:
        args += ["-tune", "grain" if settings.get("cinematic") else "film", "-profile:v", "high",
                 "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    if settings.get("maxrate_mbps"):
        args += ["-maxrate", f"{settings['maxrate_mbps']}M", "-bufsize", f"{settings['bufsize_mbit']}M"]
    args += ["-movflags", "+faststart", str(temporary)]
    run(args)
    if abs(probe(temporary) - 5) > 0.1:
        raise RuntimeError("Normalized shot duration is invalid")
    os.replace(temporary, target)
    with target.open("r+b") as handle:
        os.fsync(handle.fileno())
    fsync_directory(target.parent)
    return details


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


class MovieQueue:
    """Durable text-only follow-up queue; cloud credentials never enter this file."""
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "production_queue.json"

    def transaction(self, operation: Callable[[list[dict[str, Any]]], Any]) -> Any:
        for attempt in range(40):
            try:
                lock = WorkspaceLock(self.root / "queue_guard")
                break
            except RuntimeError:
                if attempt == 39:
                    raise
                time.sleep(0.05)
        try:
            items = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
            if not isinstance(items, list):
                raise ValueError("Invalid production queue; retain the file for recovery")
            result = operation(items)
            atomic_json(self.path, items)
            return result
        finally:
            lock.close()

    def enqueue(self, plot: str, minutes: float = 30) -> dict[str, Any]:
        if not isinstance(plot, str) or not plot.strip() or len(plot) > 250000 or not 5 / 60 <= minutes <= 60:
            raise ValueError("Queued movie needs a plot and duration between five seconds and 60 minutes")
        validated_render_settings(story_input(plot).get("render_settings", {}))
        fingerprint = hashlib.sha256((plot + f"\n{minutes:.6f}").encode()).hexdigest()
        def update(items: list[dict[str, Any]]) -> dict[str, Any]:
            existing = next((item for item in items if item.get("fingerprint") == fingerprint
                             and item.get("status") in {"pending", "starting", "running"}), None)
            if existing:
                return copy.deepcopy(existing)
            item = {"id": uuid.uuid4().hex, "plot": plot, "minutes": minutes, "fingerprint": fingerprint,
                    "status": "pending", "created_at": utc(), "provider": "None"}
            items.append(item)
            return copy.deepcopy(item)
        return self.transaction(update)

    def claim(self, completed_id: str | None = None) -> dict[str, Any] | None:
        def update(items: list[dict[str, Any]]) -> dict[str, Any] | None:
            for item in items:
                if completed_id and item["id"] == completed_id:
                    item.update(status="completed", completed_at=utc())
            item = next((item for item in items if item.get("status") in {"pending", "starting"}), None)
            if item:
                item.update(status="starting", claimed_at=utc())
            return copy.deepcopy(item)
        return self.transaction(update)

    def started(self, request_id: str, job_id: str) -> None:
        def update(items: list[dict[str, Any]]) -> None:
            item = next(item for item in items if item["id"] == request_id)
            item.update(status="running", job_id=job_id, started_at=utc())
        self.transaction(update)


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
    source = story_input(plot)
    if "storyboard" in source:
        board = source["storyboard"]
        if not isinstance(board, list) or not board or len(board) > 720:
            raise ValueError("Structured storyboard must contain 1–720 scenes")
        shots = []
        seen = set()
        for scene in board:
            if not isinstance(scene, dict) or not isinstance(scene.get("shots"), list) or not scene["shots"]:
                raise ValueError("Each storyboard scene needs a nonempty shots list")
            scene_id = scene.get("scene_id")
            if not isinstance(scene_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", scene_id) or scene_id in seen:
                raise ValueError("Storyboard scene IDs must be unique safe identifiers")
            seen.add(scene_id)
            for item in scene["shots"]:
                item = {"visual_prompt": item} if isinstance(item, str) else item
                if not isinstance(item, dict):
                    raise ValueError("Storyboard shots must be prompt strings or objects")
                shot = {"shot_id": len(shots) + 1, "scene_id": scene_id,
                    "visual_prompt": item.get("visual_prompt"),
                    "camera_movement": item.get("camera_movement", "restrained cinematic camera movement"),
                    "lighting_cue": item.get("lighting_cue", scene.get("lighting_cue", "consistent motivated lighting")),
                    "duration_seconds": item.get("duration_seconds", 5)}
                if (shot["duration_seconds"] != 5 or any(not isinstance(shot.get(k), str) or not shot[k].strip()
                    or len(shot[k]) > 12000 for k in ("visual_prompt", "camera_movement", "lighting_cue"))):
                    raise ValueError("Storyboard shots require finite five-second durations and nonempty bounded prompts")
                shots.append(shot)
        if len(shots) != count:
            raise ValueError(f"Storyboard has {len(shots)} shots; selected duration requires {count} five-second shots")
        notice(f"Using authored storyboard: {len(board)} scenes, {count} sequential shots.")
        return shots
    plot = source["plot"]
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
        for binding in self.config.get("character_bindings", []):
            filename = shot.get("character_references", {}).get(binding["character_id"])
            if not filename:
                raise ValueError("Missing checkpointed cast reference for a configured character loader")
            graph[binding["node"]]["inputs"][binding["input"]] = filename
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


def workflow_config(graph_override: dict[str, Any] | None = None,
                    mappings: dict[str, Any] | None = None,
                    profile: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = {"url": os.getenv("COMFY_URL", "http://127.0.0.1:8188"),
              "output_dir": str(Path(os.getenv("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output")).resolve()),
              **{key: os.getenv(variable, default) for key, variable, default in [
                  ("prompt_node", "COMFY_PROMPT_NODE", ""), ("image_node", "COMFY_IMAGE_NODE", ""),
                  ("video_node", "COMFY_VIDEO_NODE", ""), ("seed_node", "COMFY_SEED_NODE", ""),
                  ("prompt_input", "COMFY_PROMPT_INPUT", "text"), ("image_input", "COMFY_IMAGE_INPUT", "image"),
                  ("seed_input", "COMFY_SEED_INPUT", "seed")]}}
    if mappings:
        config.update(mappings)
    graph = (copy.deepcopy(graph_override) if graph_override is not None else
             load_api_workflow(os.getenv("COMFY_WORKFLOW", "/workspace/workflow_api.json"), "COMFY_WORKFLOW"))
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
    config["image_bindings"] = (copy.deepcopy(mappings.get("image_bindings", [])) if mappings else
                                json.loads(os.getenv("COMFY_IMAGE_BINDINGS", "[]")))
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
    config["consistency"] = consistency_config(graph, config, profile)
    config["model_signature"] = hashlib.sha256(json.dumps({node: item for node, item in graph.items()
        if any(name in item.get("class_type", "").lower() for name in ("unetloader", "vaeloader", "checkpointloader", "loraloader"))},
        sort_keys=True).encode()).hexdigest()
    config["character_bindings"] = copy.deepcopy(config["consistency"].get("character_bindings", []))
    if not isinstance(config["character_bindings"], list):
        raise ValueError("character_bindings must be a list")
    bound_loaders = {config["image_node"]} | {b["node"] for b in config["image_bindings"]}
    for binding in config["character_bindings"]:
        if (not isinstance(binding, dict) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", str(binding.get("character_id", "")))
                or binding.get("node") not in graph
                or not isinstance(graph[binding["node"]].get("inputs", {}).get(binding.get("input")), str)
                or binding["node"] in bound_loaders):
            raise ValueError("Each character binding must map a distinct file loader and a safe character ID")
        bound_loaders.add(binding["node"])
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
    for binding in config["character_bindings"]:
        if binding["node"] not in reachable or binding["node"] == continuity.get("loader_node"):
            raise ValueError("Character reference loader must reach the video output and be separate from the temporal loader")
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


class SetupError(RuntimeError):
    """An actionable provisioning error safe to display without provider secrets."""


# Official model artifacts used by Comfy-Org's video_wan2_2_5B_ti2v template.
# Downloads use a pinned repository commit and the Hub's LFS SHA-256 metadata.
AUTO_MODELS = (
    ("UNETLoader", "unet_name", "diffusion_models", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"),
    ("CLIPLoader", "clip_name", "text_encoders", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    ("VAELoader", "vae_name", "vae", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/vae/wan2.2_vae.safetensors"),
)


def model_choices(info: dict[str, Any], node: str, field: str) -> list[str]:
    definition = info.get(node, {}).get("input", {}).get("required", {}).get(field, [])
    return definition[0] if definition and isinstance(definition[0], list) else []


def provision_model(models: Path, spec: tuple[str, ...], notice: Callable[[str], None]) -> str:
    """Resume interrupted downloads; promote only checksum-verified weights.

    A .part file is never visible to ComfyUI as a model. Metadata is saved before
    transfer, so a retry/restart continues the exact same immutable artifact.
    """
    import requests
    _, _, category, repo, remote_file = spec
    target = models / category / Path(remote_file).name
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix(target.suffix + ".source.json")
    metadata = {}
    if receipt.is_file():
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
    if not metadata:
        response = requests.get(f"https://huggingface.co/api/models/{repo}",
                                params={"blobs": "true"}, timeout=(15, 90))
        response.raise_for_status()
        result = response.json()
        entry = next((s for s in result.get("siblings", []) if s["rfilename"] == remote_file), {})
        metadata = {"repo": repo, "file": remote_file, "revision": result.get("sha"),
                    "sha256": entry.get("lfs", {}).get("sha256"), "size": entry.get("size")}
        if (not re.fullmatch(r"[a-f0-9]{40}", metadata.get("revision") or "")
                or not re.fullmatch(r"[a-f0-9]{64}", metadata.get("sha256") or "")
                or type(metadata.get("size")) is not int or metadata["size"] <= 0):
            raise SetupError(f"Official model metadata has no verifiable checksum: {target.name}")
        atomic_json(receipt, metadata)
    if (metadata.get("repo") != repo or metadata.get("file") != remote_file
            or not re.fullmatch(r"[a-f0-9]{40}", str(metadata.get("revision", "")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(metadata.get("sha256", "")))
            or type(metadata.get("size")) is not int or metadata["size"] <= 0):
        raise SetupError(f"Invalid saved model download metadata: {target.name}")
    if target.is_file():
        notice(f"Verifying installed model {target.name}...")
        if target.stat().st_size == metadata["size"] and digest(target) == metadata["sha256"]:
            return target.name
        raise SetupError(f"Existing model {target.name} failed checksum verification; retained for investigation.")
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > metadata["size"]:
        raise SetupError(f"Oversized partial model download: {target.name}")
    remaining = metadata["size"] - offset
    if shutil.disk_usage(target.parent).free < remaining + 2 * 1024**3:
        raise SetupError(f"Not enough disk space for {target.name}: need {remaining / 1024**3:.1f} GiB plus 2 GiB reserve.")
    url = f"https://huggingface.co/{repo}/resolve/{metadata['revision']}/{remote_file}"
    if remaining:
        notice(f"Downloading {target.name}: {offset / metadata['size']:.0%} of {metadata['size'] / 1024**3:.1f} GiB.")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            if offset and response.status_code == 200:
                offset = 0  # Server ignored Range; restart rather than append corrupt data.
            elif response.status_code == 206:
                expected = f"bytes {offset}-"
                if not response.headers.get("Content-Range", "").startswith(expected):
                    raise SetupError("Model server returned an incorrect resume range")
            elif response.status_code != 200:
                raise SetupError("Unexpected model download response")
            heartbeat = time.monotonic()
            with partial.open("ab" if offset else "wb") as handle:
                for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not block:
                        continue
                    handle.write(block)
                    offset += len(block)
                    if offset > metadata["size"]:
                        raise SetupError("Model download exceeded its declared size")
                    if time.monotonic() - heartbeat >= 10:
                        notice(f"Downloading {target.name}: {offset / metadata['size']:.0%}")
                        heartbeat = time.monotonic()
                handle.flush()
                os.fsync(handle.fileno())
    notice(f"Checking SHA-256 for {target.name}...")
    if partial.stat().st_size != metadata["size"] or digest(partial) != metadata["sha256"]:
        # Quarantine bad bytes so the next retry can start cleanly.
        os.replace(partial, partial.with_suffix(".corrupt"))
        raise SetupError(f"Downloaded model checksum mismatch: {target.name}")
    os.replace(partial, target)
    fsync_directory(target.parent)
    notice(f"Model installed and verified: {target.name}")
    return target.name


def automatic_workflow(root: Path, notice: Callable[[str], None]) -> dict[str, Any]:
    """Provision a core Wan 2.2 profile without custom nodes or manual node IDs.

    Native I2V reuses the immutable master image for each shot. This model has no
    separate identity adapter input: do not falsely enable frame chaining, which
    would replace its only reference with a progressively drifting last frame.
    """
    import requests
    deadline = time.monotonic() + 7200
    while True:
        try:
            lock = WorkspaceLock(root / "setup")
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise SetupError("Another setup process is still preparing the models; retry Attach / Resume Checkpoint later.") from None
            notice("Waiting for the existing model setup process; its downloads are continuing...")
            time.sleep(10)
    try:
        requested = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
        candidates = [requested]
        if urlsplit(requested).hostname in {"localhost", "127.0.0.1"}:
            candidates += ["http://127.0.0.1:18188", "http://127.0.0.1:8188"]
        info, url = {}, requested
        for candidate in dict.fromkeys(candidates):
            try:
                response = requests.get(candidate + "/object_info", timeout=(3, 30))
                response.raise_for_status()
                value = response.json()
                if isinstance(value, dict) and "KSampler" in value:
                    info, url = value, candidate
                    break
            except (requests.RequestException, ValueError):
                continue
        if not info:
            raise SetupError("ComfyUI is unavailable. Automatic setup checked the local API ports 8188 and 18188; no compatible API responded.")
        notice(f"Connected to ComfyUI at {url}. Preparing automatic Wan 2.2 movie workflow.")
        required = {"UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode", "LoadImage", "KSampler",
                    "ModelSamplingSD3", "Wan22ImageToVideoLatent", "VAEDecode", "SaveImage", "CreateVideo", "SaveVideo"}
        missing = sorted(required - info.keys())
        if missing:
            raise SetupError("ComfyUI is missing required core nodes: " + ", ".join(missing))
        comfy_root = Path(os.getenv("COMFY_ROOT", "/workspace/ComfyUI")).resolve()
        models = comfy_root / "models"
        names = []
        for spec in AUTO_MODELS:
            filename = Path(spec[4]).name
            installed = next((name for name in model_choices(info, spec[0], spec[1])
                              if Path(name.replace("\\", "/")).name == filename), None)
            local = models / spec[2] / filename
            if installed and not local.with_suffix(local.suffix + ".source.json").exists():
                notice(f"Reusing installed model: {installed}")
                names.append(installed)
            else:
                names.append(retry(lambda spec=spec: provision_model(models, spec, notice),
                                   "Automatic model installation", notice))
        # Re-read model lists: ComfyUI updates directory caches without a restart.
        fresh = requests.get(url + "/object_info", timeout=(10, 60))
        fresh.raise_for_status()
        info = fresh.json()
        for spec, name in zip(AUTO_MODELS, names):
            if name not in model_choices(info, spec[0], spec[1]):
                raise SetupError(f"ComfyUI cannot see installed model {name}; its model directory differs from COMFY_ROOT.")
        graph = build_automatic_wan_graph(info, names)
        graph, config = workflow_config(graph, {
            "url": url, "output_dir": str(comfy_root / "output"),
            "prompt_node": "6", "prompt_input": "text", "image_node": "56", "image_input": "image",
            "video_node": "58", "seed_node": "3", "seed_input": "seed", "image_bindings": []},
            {"continuity": {"mode": "off"}})
        # Same TI2V model supports text-only anchor generation: no extra model or
        # separate SDXL configuration required when the user leaves image blank.
        anchor = copy.deepcopy(graph)
        for node in ("56", "57", "58"):
            anchor.pop(node)
        anchor["55"]["inputs"].pop("start_image")
        anchor["55"]["inputs"]["length"] = 1
        anchor["9"] = {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": "studio/auto_master_character"}}
        result = {"workflow": graph, "comfy": config, "anchor_workflow": anchor,
                  "anchor_config": {"prompt_node": "6", "prompt_input": "text", "seed_node": "3",
                                    "seed_input": "seed", "output_node": "9"}, "prepared_at": utc()}
        atomic_json(root / "automatic_workflow.json", result)
        notice("Automatic setup complete. Native reference conditioning and locked prompts/seeds enabled; separate adapter/temporal chaining unavailable in this core profile.")
        return result
    finally:
        lock.close()


def build_automatic_wan_graph(info: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """API translation of Comfy-Org's tested Wan 2.2 5B TI2V template.

    Source: github.com/Comfy-Org/workflow_templates, templates/
    video_wan2_2_5B_ti2v.json. 121 frames at 24fps normalize to exactly five seconds.
    """
    save = {"video": ["57", 0], "filename_prefix": "studio/shot", "format": "auto"}
    schema = info["SaveVideo"].get("input", {}).get("required", {})
    if schema.get("format", [None])[0] == "COMFY_DYNAMICCOMBO_V3":
        save["format.codec"] = "auto"  # Dynamic API inputs use dotted child names.
    elif "codec" in schema:
        save["codec"] = "auto"
    return {
        "37": {"class_type": "UNETLoader", "inputs": {"unet_name": names[0], "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": names[1], "type": "wan", "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": names[2]}},
        "48": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["37", 0], "shift": 8.0}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["38", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, distorted face, extra limbs, text, watermark, static shot", "clip": ["38", 0]}},
        "56": {"class_type": "LoadImage", "inputs": {"image": "auto_master_character.png"}},
        "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["39", 0],
                "width": 1280, "height": 704, "length": 121, "batch_size": 1, "start_image": ["56", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["48", 0], "positive": ["6", 0],
               "negative": ["7", 0], "latent_image": ["55", 0], "seed": 0, "steps": 20,
               "cfg": 5.0, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "57": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 24.0}},
        "58": {"class_type": "SaveVideo", "inputs": save},
    }


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
        self.handoff_thread: threading.Thread | None = None
        self.lines: deque[str] = deque(maxlen=300)
        self.live_lock = threading.Lock()
        self.streams: list[str] = []
        self.assembly_error: BaseException | None = None

    def notice(self, message: str) -> None:
        LOG.info(message)
        with self.live_lock:
            self.lines.append(f"[{utc()}] {message}")

    def publish_media(self, source: Path) -> Path:
        """Serve only media, never grant Gradio access to private checkpoints.

        Hard links avoid duplicating large movies on the same filesystem. A
        copy fallback handles filesystems that do not support hard links.
        """
        if source.suffix.lower() not in {".mp4", ".ts"}:
            raise ValueError("Only finished video media can be published")
        target = self.root / "public_media" / self.store.data["job_id"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and os.path.samefile(source, target):
            return target
        pending = target.with_suffix(target.suffix + ".pending")
        pending.unlink(missing_ok=True)
        try:
            os.link(source, pending)
        except OSError:
            shutil.copyfile(source, pending)
        os.replace(pending, target)
        return target

    def running(self) -> bool:
        return bool((self.thread and self.thread.is_alive()) or (self.handoff_thread and self.handoff_thread.is_alive()))

    def launch(self) -> None:
        if self.running() and threading.current_thread() is not self.handoff_thread:
            raise RuntimeError("Studio is already running")
        self.thread = threading.Thread(target=self.pipeline, name="movie-coordinator", daemon=False)
        self.thread.start()

    def resume(self) -> None:
        if self.store.data and self.store.data.get("status") not in TERMINAL:
            self.notice("Interrupted checkpoint detected; automatically resuming.")
            self.launch()
        elif not self.running() and MovieQueue(self.root).path.exists():
            self.schedule_next_queued()

    def schedule_next_queued(self) -> bool:
        """Commit a handoff before withholding teardown; retry safely after crash."""
        queue_store = MovieQueue(self.root)
        if not queue_store.path.exists():
            return False
        request = queue_store.claim(self.store.data.get("queued_request_id"))
        if not request:
            return False
        with self.store.lock:
            if self.store.data:
                self.store.data.update(status="partial" if self.store.data.get("partial") else "complete",
                                       next_queued_request=request["id"])
                self.store.save()
        previous = self.thread
        def handoff() -> None:
            try:
                if previous and previous is not threading.current_thread():
                    previous.join()
                self.start(request["plot"], None, request["minutes"], "None", "", "",
                           queued_request_id=request["id"])
                queue_store.started(request["id"], self.store.data["job_id"])
            except Exception as exc:
                self.notice(f"Queued movie handoff failed ({type(exc).__name__}); request retained for restart recovery. GPU stop withheld.")
            finally:
                if self.handoff_thread is threading.current_thread():
                    self.handoff_thread = None
        self.notice(f"Movie archived. Queued production {request['id']} will start next; GPU shutdown deferred.")
        self.handoff_thread = threading.Thread(target=handoff, name="movie-handoff", daemon=False)
        self.handoff_thread.start()
        return True

    def start(self, plot: str, image: str | None, minutes: float, provider: str,
              destination: str, secret: str, shutdown: bool = True,
              google_credentials: str = "", sound_design: bool = True, cinematic: bool = True,
              queued_request_id: str | None = None) -> None:
        with self.guard:
            if self.running() and threading.current_thread() is not self.handoff_thread:
                raise ValueError("A movie is already running; attach to the current job")
            if self.store.data and self.store.data.get("status") not in TERMINAL:
                self.launch()
                return
            if not plot.strip():
                raise ValueError("Provide a movie plot")
            story_input(plot)  # Validate structured input before creating a job.
            render_settings = validated_render_settings(story_input(plot).get("render_settings", {}))
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
            # Provision missing workflow/models in the coordinator, never inside
            # the Gradio request thread. The job is checkpointed before downloads.
            setup_pending = False
            try:
                graph, config = workflow_config()
            except ValueError:
                if Path(os.getenv("COMFY_WORKFLOW", "/workspace/workflow_api.json")).is_file():
                    raise  # Preserve errors in explicitly supplied custom workflows.
                setup_pending = True
                graph, config = {}, {}
            if setup_pending:
                anchor_graph, anchor_config = {}, {}
            else:
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
            width, height = render_settings.get("width", width), render_settings.get("height", height)
            fps = int(os.getenv("MOVIE_FPS", "24"))
            if min(width, height) < 64 or width % 2 or height % 2 or not 1 <= fps <= 120:
                raise ValueError("Media dimensions must be even and >=64; fps must be 1–120")
            if self.store.path.exists():
                atomic_bytes(Path(self.store.data["folder"]) / "state.archive.json", self.store.path.read_bytes())
            finishing = postproduction_config(sound_design, cinematic)
            finishing.update({k: v for k, v in render_settings.items() if k in {"crf", "grain", "preset", "maxrate_mbps", "bufsize_mbit"}})
            self.store.data = {"schema": 1, "job_id": job_id, "folder": str(folder), "created_at": utc(),
                "status": "setting_up" if setup_pending else "planning", "setup_pending": setup_pending,
                "story_engine_version": 1,
                "queued_request_id": queued_request_id,
                "plot": plot, "shot_count": count, "shots": [],
                "reference": str(reference) if reference else None, "anchor": anchor, "master_seed": master_seed,
                "workflow": graph, "comfy": config, "media": {"width": width, "height": height, "fps": fps,
                    "retain_raw_copies": render_settings.get("retain_raw_copies", True),
                    "postproduction": finishing},
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

    def ensure_master_anchor(self, client: ComfyClient, character: dict[str, Any] | None = None) -> Path:
        """Create once, checkpoint before shots, reuse the exact same pixels.

        A run-private image is authoritative. The required fixed input filename
        is a convenience alias; every clip gets a job-unique upload of the same
        cached anchor, avoiding collisions with other users of ComfyUI.
        """
        state = self.store.data
        if character is None and state.get("reference") and not state.get("anchor"):
            # Backward compatibility with pre-Auto-Character checkpoints.
            return Path(state["reference"])
        anchor = character["anchor"] if character is not None else state["anchor"]
        if anchor["mode"] == "provided":
            reference = Path(anchor["image_path"])
            if not reference.is_file() or digest(reference) != anchor["image_sha256"]:
                raise RuntimeError("Uploaded master reference is missing/corrupt; restore it before resuming")
            self.notice("Using uploaded character reference as the fixed master anchor.")
            return reference
        basename = f"character_{character['id']}" if character is not None else "master_character"
        cached = Path(state["folder"]) / (basename + ".png")
        input_dir = Path(state["comfy"].get("input_dir", str(Path(state["comfy"]["output_dir"]).parent / "input")))
        alias = input_dir / (f"auto_{basename}.png" if character is not None else "auto_master_character.png")
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
        identity = character or next(iter(state.get("story_bible", {}).get("characters", [])), None)
        if identity:
            description = (f"Create a single cinematic character reference for {identity['name']}. "
                           f"Appearance: {identity['appearance']}. Wardrobe: {identity['wardrobe']}. "
                           "One subject, clear unobstructed face, visible hairstyle, distinctive outfit, three-quarter portrait. "
                           f"Narrative temperament: {identity['psychology']}. "
                           f"Movie context: {state.get('story_bible', {}).get('plot', state['plot'])}. "
                           f"Opening scene: {first['visual_prompt']}; lighting: {first['lighting_cue']}.")
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
                              marker=anchor.get("reconcile_marker") or (f"{state['job_id']}:anchor:{character['id']}:{attempt}"
                                  if character is not None else f"{state['job_id']}:anchor:{attempt}"),
                              prompt_id=anchor.get("retry_prompt_id"))
                self.store.save()
            prompt_id = anchor.get("prompt_id")
            try:
                if not prompt_id:
                    prompt_id = client.reconcile(anchor["marker"]) or client.submit_graph(graph, anchor["marker"])
                anchor.update(prompt_id=prompt_id, status="submitted")
                self.store.save()
                source = client.wait(prompt_id, self.notice, output_node=config["output_node"], kind="image")
                candidate = cached.with_name(basename + ".candidate.png")
                save_anchor_png(source, candidate)
                checksum = digest(candidate)
                if expected_hash and checksum != expected_hash and any(s["status"] == "completed" for s in state["shots"]):
                    raise RuntimeError("Original anchor could not be recovered exactly; refusing to change identity mid-movie")
                # Preserve a private durable copy before exposing the fixed alias.
                atomic_bytes(cached, candidate.read_bytes())
                atomic_bytes(alias, cached.read_bytes())
                anchor.update(status="ready", image_path=str(cached), image_sha256=checksum,
                              description=description, completed_at=utc())
                if character is None:
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

    def ensure_cast_references(self, client: ComfyClient, master: Path) -> dict[str, str]:
        """Generate/cache every cast anchor; inject only supported loader sockets."""
        state = self.store.data
        bible = state.get("story_bible")
        if not bible:
            return {}
        characters = bible["characters"]
        primary = bible["primary_character_id"]
        bindings = state["comfy"].get("character_bindings", [])
        identifiers = {c["id"] for c in characters}
        if any(binding["character_id"] not in identifiers for binding in bindings):
            raise SetupError("Workflow character bindings reference an ID absent from the story bible")
        bound = {b["character_id"] for b in bindings} | {primary}
        missing = identifiers - bound
        if missing and state["comfy"].get("consistency", {}).get("require_multi_character_conditioning"):
            raise SetupError("Strict cast conditioning requires independent loaders for: " + ", ".join(sorted(missing)))
        references = {}
        for character in characters:
            if character["id"] == primary:
                character["anchor"] = copy.deepcopy(state["anchor"])
                path = master
            else:
                if "anchor" not in character:
                    template = state.get("auto_anchor_template", {})
                    if not template:
                        graph, config = anchor_workflow_config(state["workflow"])
                        template = {"workflow": graph, "config": config}
                    character["anchor"] = {"mode": "auto", "seed": character["seed"], "status": "pending",
                        "attempts": 0, "workflow": copy.deepcopy(template["workflow"]), "config": copy.deepcopy(template["config"])}
                    self.store.save()
                path = self.ensure_master_anchor(client, character)
            references[character["id"]] = retry(lambda path=path, cid=character["id"]:
                client.upload(path, f"{state['job_id']}_cast_{cid}"), "Cast reference upload", self.notice)
        if missing:
            self.notice("Cast anchors generated, but this workflow has no independent image conditioning for "
                        + ", ".join(sorted(missing)) + "; those characters currently receive text constraints only.")
        with self.store.lock:
            for shot in state["shots"]:
                shot.setdefault("character_references", copy.deepcopy(references))
            self.store.save()
        atomic_json(Path(state["folder"]) / "story_bible.json", bible)
        return references

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
            if settings.get("tensor_contract"):
                tensor_info = inspect_terminal_latent(source, settings["tensor_contract"])
            else:
                tensor_info = {"contract_validated": False}
            target = folder / f"tail_{shot['shot_id']:04d}{source.suffix}"
            atomic_bytes(target, source.read_bytes())
        self.checkpoint_shot(shot, tail_condition={"mode": mode, "path": str(target), "sha256": digest(target),
                            "source_sha256": shot["raw_sha256"], "captured_at": utc(),
                            "model_signature": self.store.data["comfy"].get("model_signature"),
                            "tensor_info": tensor_info if mode == "latent" else None})

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
        if tail.get("model_signature") != state["comfy"].get("model_signature"):
            raise ValueError("Continuity model/VAE signature mismatch; refusing incompatible boundary conditioning")
        path = Path(tail["path"])
        if mode == "image":
            filename = retry(lambda: client.upload(path, f"{state['job_id']}_tail_{previous['shot_id']}"),
                             "Continuity frame upload", self.notice)
        else:
            if settings.get("tensor_contract"):
                inspect_terminal_latent(path, settings["tensor_contract"])
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
        settings = snapshot["media"].get("postproduction", {})
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(playlist)]
        if settings.get("audio"):
            audio_list = folder / "audio_filelist.txt"
            for shot in clips:
                path = folder / f"shot_{shot['shot_id']:04d}.wav"
                validate_pcm(path)
                if digest(path) != shot.get("audio_sha256"):
                    raise RuntimeError(f"Shot {shot['shot_id']} soundtrack failed integrity verification")
            atomic_bytes(audio_list, "".join(f"file 'shot_{shot['shot_id']:04d}.wav'\n" for shot in clips).encode())
            # Decode lossless PCM segments and encode AAC once for this entire
            # master, preventing per-segment AAC priming delays from adding up.
            args += ["-f", "concat", "-safe", "1", "-i", str(audio_list), "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", settings["audio_bitrate"], "-ar", "48000", "-ac", "2",
                     "-af", f"atrim=end_sample={len(clips) * 240000},asetpts=N/SR/TB", "-t", str(len(clips) * 5)]
        else:
            args += ["-map", "0:v:0", "-c:v", "copy", "-an"]
        args += ["-movflags", "+faststart", str(pending)]
        run(args, timeout=1800)
        if abs(probe(pending) - 5 * len(clips)) > max(0.25, len(clips) * 0.002):
            raise RuntimeError("Master duration verification failed")
        if settings:
            validate_finished_media(pending, 5 * len(clips), snapshot["media"])
        with pending.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(pending, master)
        fsync_directory(self.root)
        with self.store.lock:
            self.store.data["master_shot_ids"] = [s["shot_id"] for s in clips]
            self.store.save()
        self.notice(f"Master updated: {len(clips)} completed shots, {len(clips) * 5} seconds"
                    + (", synchronized stereo AAC soundtrack." if settings.get("audio") else "."))

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
                        media = self.store.data["media"]
                        if media.get("postproduction"):
                            self.notice(f"Shot {shot['shot_id']}: cinematic finishing and sound design.")
                            details = retry(lambda: normalize(raw, clip, media, shot, self.notice),
                                            f"Finish shot {shot['shot_id']}", self.notice)
                        else:
                            # Preserve old checkpoints' silent/non-graded look.
                            retry(lambda: normalize(raw, clip, media), f"Normalize shot {shot['shot_id']}", self.notice)
                            details = {}
                        transport = retry(lambda: preview(clip), "Prepare preview", self.notice)
                    except Exception as exc:
                        self.checkpoint_shot(shot, status="failed", last_error=type(exc).__name__)
                        self.notice(f"Shot {shot['shot_id']} media processing failed; skipping.")
                        continue
                    self.checkpoint_shot(shot, status="completed", clip_sha256=digest(clip), completed_at=utc(), **details)
                    public_transport = self.publish_media(transport)
                    with self.live_lock:
                        self.streams.append(str(public_transport))
                    retry(self.assemble, "Master assembly", self.notice)
                    self.release_duplicate_raw(shot, raw)
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

    def release_duplicate_raw(self, shot: dict[str, Any], raw: Path) -> None:
        """Opt-in disk policy: release only this job's redundant private raw copy.

        The ComfyUI original and prompt ID remain available for recovery. Native
        continuity jobs keep their raw copies. Graded clips and PCM remain durable.
        """
        state = self.store.data
        if (state.get("media", {}).get("retain_raw_copies", True)
                or state.get("comfy", {}).get("consistency", {}).get("continuity", {}).get("mode", "off") != "off"):
            return
        expected = Path(state["folder"]) / f"raw_{shot['shot_id']:04d}.mp4"
        clip = expected.with_name(f"shot_{shot['shot_id']:04d}.mp4")
        if (raw.resolve() != expected.resolve() or shot.get("status") != "completed"
                or not clip.is_file() or digest(clip) != shot.get("clip_sha256")):
            raise ValueError("Refusing to release raw data before a verified finished clip exists")
        raw.unlink(missing_ok=True)
        self.checkpoint_shot(shot, private_raw_released_at=utc())

    def recover_clips(self) -> None:
        for shot in self.store.data["shots"]:
            if shot["status"] != "completed":
                continue
            clip = Path(self.store.data["folder"]) / f"shot_{shot['shot_id']:04d}.mp4"
            audio = clip.with_suffix(".wav")
            sound = self.store.data.get("media", {}).get("postproduction", {}).get("audio", False)
            if (not clip.exists() or digest(clip) != shot.get("clip_sha256")
                    or (sound and (not audio.is_file() or digest(audio) != shot.get("audio_sha256")))):
                self.notice(f"Shot {shot['shot_id']} media checkpoint missing/corrupt; recovering output.")
                raw = clip.with_name(f"raw_{shot['shot_id']:04d}.mp4")
                reusable = raw.is_file() and digest(raw) == shot.get("raw_sha256")
                self.checkpoint_shot(shot, status="rendered" if reusable else "submitted" if shot.get("prompt_id") else "pending", attempts=0)
                continue
            transport = clip.with_suffix(".ts")
            if not transport.exists():
                transport = preview(clip)
            public_transport = self.publish_media(transport)
            with self.live_lock:
                self.streams.append(str(public_transport))

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
        # Keep every film under its own job, even as root/final_movie.mp4 is
        # replaced by a later queued film. Hard links use no extra video storage.
        archive = Path(state["folder"]) / "final_movie.mp4"
        if final.resolve() != archive.resolve():
            if archive.exists():
                if digest(archive) != state["final_sha256"]:
                    raise RuntimeError("Existing archived movie has a different checksum")
            else:
                try:
                    os.link(final, archive)
                except OSError:
                    shutil.copyfile(final, archive)
                fsync_directory(archive.parent)
            final = archive
            state["final_file"] = str(archive)
            self.store.save()
        self.backup_final_movie(final)
        with self.store.lock:
            state.update(status="movie_complete", completed_at=utc())
            self.store.save()
        atomic_json(Path(state["folder"]) / "completion.json", {
            "job_id": state["job_id"], "completed_shot_ids": state["completed_shot_ids"],
            "failed_shot_ids": state["failed_shot_ids"], "partial": state["partial"],
            "final_sha256": state["final_sha256"], "cloud_status": state["cloud"]["status"],
            "upload_receipt": state.get("upload_receipt"), "completed_at": state["completed_at"],
            "postproduction": state.get("media", {}).get("postproduction", {"version": 0, "audio": False, "cinematic": False})})
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
            if state.get("setup_pending"):
                stage = "automatic ComfyUI setup"
                self.notice("First-run setup: preparing the workflow and checking required models. Downloads resume automatically after interruption.")
                prepared = automatic_workflow(self.root, self.notice)
                with self.store.lock:
                    state.update(workflow=prepared["workflow"], comfy=prepared["comfy"],
                                 setup_pending=False, status="planning")
                    state["auto_anchor_template"] = {"workflow": prepared["anchor_workflow"], "config": prepared["anchor_config"]}
                    if state["anchor"]["mode"] == "auto":
                        state["anchor"].update(workflow=prepared["anchor_workflow"], config=prepared["anchor_config"])
                    self.store.save()
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
            if state.get("story_engine_version"):
                stage = "story bible and authored continuity planning"
                if not state.get("story_bible"):
                    state["story_bible"] = build_story_bible(state["plot"], state.get("master_seed", 0), self.notice)
                    self.store.save()
                with self.store.lock:
                    bind_world_state(state["shots"], state["story_bible"])
                    state["capabilities"] = workflow_capabilities(state["comfy"])
                    self.store.save()
                atomic_json(Path(state["folder"]) / "story_bible.json", state["story_bible"])
                atomic_json(Path(state["folder"]) / "continuity_plan.json", {
                    "capabilities": state["capabilities"], "shots": [{"shot_id": s["shot_id"],
                     "scene_id": s["scene_id"], "world_state": s["world_state"]} for s in state["shots"]]})
                self.notice(f"Story ledger: {len(state['story_bible']['characters'])} characters; "
                            f"{len(state['story_bible']['objects'])} recurring objects. Physics and lighting are authored constraints, not pixel-level guarantees.")
                continuity = state["comfy"].get("consistency", {}).get("continuity", {})
                strict = (state["comfy"].get("consistency", {}).get("require_native_latents", False)
                          or os.getenv("STUDIO_REQUIRE_NATIVE_LATENTS", "0") == "1")
                if strict and (continuity.get("mode") != "latent" or not continuity.get("tensor_contract")):
                    raise SetupError("Strict native continuity needs a model-compatible terminal latent exporter, loader, and tensor_contract. The automatic Wan core profile cannot provide that contract.")
            with self.live_lock:
                self.streams.clear()
            stage = "shot recovery, rendering, and assembly"
            finishing = state["media"].get("postproduction", {})
            if finishing:
                self.notice(f"Finishing profile: sound={'procedural stereo' if finishing['audio'] else 'off'}, "
                            f"cinematic scope={'on' if finishing['cinematic'] else 'off'}, H.264 CRF {finishing['crf']}.")
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
                    if state.get("story_engine_version"):
                        self.ensure_cast_references(client, master)
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
            pending.unlink(missing_ok=True)
            try:
                os.link(self.root / "current_movie_stream.mp4", pending)
            except OSError:
                shutil.copyfile(self.root / "current_movie_stream.mp4", pending)
            if abs(probe(pending) - len(completed) * 5) > 0.5:
                raise RuntimeError("Final movie duration is invalid")
            if state["media"].get("postproduction"):
                validate_finished_media(pending, len(completed) * 5, state["media"])
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
            if isinstance(exc, SetupError):
                self.notice(str(exc))
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
                        if not self.schedule_next_queued():
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
        stop_command = f"sleep 10 && vastai stop instance {instance}"
        if os.getenv("CONTAINER_API_KEY"):
            # Vast's in-container CLI is not pre-authenticated. Expand the
            # credential only inside the child shell; never put it in logs/state.
            stop_command += ' --api-key "$CONTAINER_API_KEY"'
        shell = (f"flock -n {shlex.quote(str(folder / 'teardown.lock'))} "
                 f"sh -c {shlex.quote(stop_command)}")
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
        observed_job = None
        # Newly attached browsers replay completed chunks once, then follow live.
        yield None, "Attaching to studio...", None, "Connecting"
        while True:
            state = studio.store.snapshot()
            if state.get("job_id") != observed_job:
                observed_job = state.get("job_id")
                cursor, delivered_final = 0, None
            with studio.live_lock:
                logs = "\n".join(studio.lines)
                clip = studio.streams[cursor] if cursor < len(studio.streams) else None
                if clip:
                    cursor += 1
            status = state.get("status", "idle")
            label = f"{status} | {len(state.get('completed_shot_ids', []))}/{state.get('shot_count', 0)} completed"
            # Do not accidentally expose a prior job's final movie while running.
            final = state.get("final_file") if state.get("movie_ready") else None
            final_output = gr.skip()
            if final and state.get("final_sha256") != delivered_final:
                final_output = str(studio.publish_media(Path(final)))
                delivered_final = state.get("final_sha256")
            yield clip if clip else gr.skip(), logs, final_output, label
            if not studio.running() and not clip:
                break
            time.sleep(0.5)

    def start(plot: str, image: str | None, minutes: float, provider: str,
              destination: str, secret: str, owner: str, sound: bool, cinematic: bool) -> Iterator[Any]:
        try:
            google_token = oauth.serialized_credentials(owner) if provider == "Google Drive" else ""
            # Resuming an existing job uses its encrypted checkpoint credentials.
            resuming = studio.store.data and studio.store.data.get("status") not in TERMINAL
            if provider == "Google Drive" and not google_token and not resuming:
                studio.notice("Google Drive is not authorized; optional backup will be skipped for this job.")
            studio.start(plot, image, minutes, provider, destination,
                         secret if provider == "Dropbox" else "", google_credentials=google_token,
                         sound_design=sound, cinematic=cinematic)
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
                with gr.Accordion("Sound design & cinematic finishing", open=False):
                    sound = gr.Checkbox(value=True, label="Automatic ambient sound design (stereo AAC)")
                    cinematic = gr.Checkbox(value=True, label="Cinematic grade, film grain & 2.39:1 center crop")
                    gr.Markdown("Sound uses context-selected procedural layers such as rain, wind and mechanical hums. "
                                "It does not generate dialogue or detect exact on-screen events. "
                                "The widescreen center crop can trim the top and bottom of the source. "
                                "Finishing choices are saved with the movie; resumed jobs keep their original settings.")
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
                gr.Markdown("Missing workflows and public model weights are prepared automatically. Cloud backup is optional. "
                            "Three failed attempts skip a shot; a partial movie is clearly reported. "
                            "Autoplay with sound may require browser interaction. Source resolution limits recoverable detail.")
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
        button.click(start, [plot, image, minutes, provider, destination, secret, owner, sound, cinematic], outputs,
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
        def test_authored_delivery_settings_are_bounded_and_checkpointed(self):
            overrides = {"width": 1280, "height": 720, "crf": 18, "grain": 1, "preset": "slow",
                         "retain_raw_copies": False, "maxrate_mbps": 5, "bufsize_mbit": 4}
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                with patch(__name__ + ".workflow_config", return_value=({}, {})), \
                     patch(__name__ + ".anchor_workflow_config", return_value=({}, {})), patch.object(studio, "launch"):
                    studio.start(json.dumps({"plot": "A film", "render_settings": overrides}), None, 30, "None", "", "")
                media = studio.store.data["media"]
                self.assertEqual(media["width"], 1280)
                self.assertFalse(media["retain_raw_copies"])
                self.assertEqual(media["postproduction"]["maxrate_mbps"], 5)
                self.assertEqual(media["postproduction"]["crf"], 18)
            for invalid in ({"crf": 99}, {"width": "1280"}, {"maxrate_mbps": 5}, {"retain_raw_copies": 0}):
                with self.assertRaises(ValueError):
                    validated_render_settings(invalid)

        def test_raw_copy_release_preserves_original_and_finished_media(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw, clip, backend = root / "raw_0001.mp4", root / "shot_0001.mp4", root / "backend_original.mp4"
                raw.write_bytes(b"raw")
                clip.write_bytes(b"finished")
                backend.write_bytes(b"original")
                shot = {"shot_id": 1, "status": "completed", "clip_sha256": digest(clip)}
                studio = Studio(root, acquire_lock=False)
                studio.store.data = {"schema": 1, "folder": tmp, "shots": [shot], "media": {"retain_raw_copies": False}}
                with self.assertRaises(ValueError):
                    studio.release_duplicate_raw(shot, backend)
                self.assertTrue(backend.exists())
                studio.release_duplicate_raw(shot, raw)
                self.assertFalse(raw.exists())
                self.assertEqual(clip.read_bytes(), b"finished")
                self.assertEqual(backend.read_bytes(), b"original")

        def test_authored_storyboard_duration_and_scene_boundaries(self):
            plot = json.dumps({"plot": "A traveler enters then leaves", "storyboard": [
                {"scene_id": "arrival", "shots": ["Enters the station", "Waits by a window"]},
                {"scene_id": "departure", "shots": ["Walks out"]}]})
            shots = plan_shots(plot, 3, lambda _: None)
            self.assertEqual([s["scene_id"] for s in shots], ["arrival", "arrival", "departure"])
            with self.assertRaises(ValueError):
                plan_shots(plot, 4, lambda _: None)

        def test_verified_final_archives_before_queue_handoff_and_withholds_stop(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = root / "run"
                folder.mkdir()
                final = root / "final_movie.mp4"
                final.write_bytes(b"validated-test-movie")
                studio = Studio(root, acquire_lock=False)
                studio.store.data = {"schema": 1, "job_id": "job", "folder": str(folder), "shots": [], "partial": False,
                    "movie_ready": True, "final_file": str(final), "final_sha256": digest(final), "cloud": {"provider": "None"}}
                with patch("shutil.which", return_value="installed"), patch.object(studio, "schedule_next_queued", return_value=True) as handoff, \
                     patch.object(studio, "teardown") as stop:
                    studio.pipeline()
                handoff.assert_called_once()
                stop.assert_not_called()
                self.assertEqual((folder / "final_movie.mp4").read_bytes(), final.read_bytes())
                self.assertEqual(studio.store.data["final_file"], str(folder / "final_movie.mp4"))

        def test_persistent_queue_is_idempotent_and_recovers_claims(self):
            with tempfile.TemporaryDirectory() as tmp:
                queue_store = MovieQueue(Path(tmp))
                first = queue_store.enqueue("A cinematic coastal story", 30)
                self.assertEqual(first["id"], queue_store.enqueue("A cinematic coastal story", 30)["id"])
                claimed = MovieQueue(Path(tmp)).claim()
                self.assertEqual(claimed["id"], first["id"])
                self.assertEqual(MovieQueue(Path(tmp)).claim()["id"], first["id"])
                queue_store.started(first["id"], "job")
                self.assertIsNone(queue_store.claim(first["id"]))
                self.assertEqual(json.loads(queue_store.path.read_text())[0]["status"], "completed")

        def test_queue_handoff_starts_next_job_without_teardown(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                studio.store.data = {"schema": 1, "job_id": "old", "status": "movie_complete", "shots": [], "partial": False}
                MovieQueue(Path(tmp)).enqueue("Next film", 30)
                def start(*args, **kwargs):
                    studio.store.data = {"job_id": "new", "queued_request_id": kwargs["queued_request_id"]}
                with patch.object(studio, "start", side_effect=start) as start_call, patch.object(studio, "teardown") as stop:
                    self.assertTrue(studio.schedule_next_queued())
                    thread = studio.handoff_thread
                    if thread:
                        thread.join(timeout=5)
                    start_call.assert_called_once()
                    stop.assert_not_called()
                item = json.loads(MovieQueue(Path(tmp)).path.read_text())[0]
                self.assertEqual(item["status"], "running")
                self.assertEqual(item["job_id"], "new")

        def test_attached_prop_follows_character_root_without_claiming_joint_dynamics(self):
            bible = build_story_bible(json.dumps({"plot": "Mara walks.", "characters": [{"id": "mara", "name": "Mara"}],
                "objects": [{"id": "beacon", "name": "Beacon", "attached_to": "mara", "attachment_offset_m": [0, 1, 0]}]}), 1, lambda _: None)
            shots = structural_shots("Mara walks.", 2)
            bind_world_state(shots, bible)
            for shot in shots:
                entities = shot["world_state"]["entities"]
                self.assertEqual(entities["mara"]["end"]["position_m"][0], entities["beacon"]["end"]["position_m"][0])
                self.assertEqual(entities["beacon"]["end"]["position_m"][1], 1)
                self.assertIsNone(entities["beacon"]["end"]["energy_residual_j"])

        def test_world_ledger_and_motion_boundaries_are_deterministic(self):
            source = json.dumps({"plot": "Mara walks across a station.\nMara waits at a lighthouse.",
                "characters": [{"id": "mara", "name": "Mara", "wardrobe": "blue coat", "psychology": "resolve and compassion"}],
                "objects": [{"id": "beacon", "name": "Beacon", "appearance": "brass cylinder", "mass_kg": 2}]})
            bible = build_story_bible(source, 123, lambda _: None)
            self.assertEqual(bible["characters"][0]["seed"], 123)
            shots = structural_shots(story_input(source)["plot"], 4)
            bind_world_state(shots, bible)
            first, second = shots[0]["world_state"], shots[1]["world_state"]
            self.assertEqual(first["entities"]["mara"]["end"]["position_m"], second["entities"]["mara"]["start_position_m"])
            self.assertEqual(first["lighting"], second["lighting"])
            self.assertTrue(shots[2]["world_state"]["scene_boundary"])
            frozen = json.dumps(shots, sort_keys=True)
            bible["characters"][0]["wardrobe"] = "changed outfit"
            bind_world_state(shots, bible)
            self.assertEqual(json.dumps(shots, sort_keys=True), frozen)
            self.assertIn("blue coat", shot_prompt(shots[0]))
            self.assertIn("brass cylinder", shot_prompt(shots[0]))

        def test_classical_energy_balance_and_inelastic_ground_contact(self):
            free = advance_body([0, 100, 0], [2, 0, 0], 3, 2, ballistic=True)
            self.assertAlmostEqual(free["position_m"][0], 4)
            self.assertAlmostEqual(free["position_m"][1], 100 - 9.80665 * 2)
            self.assertAlmostEqual(free["energy_residual_j"], 0)
            impact = advance_body([0, 1, 0], [2, 0, 0], 3, 5, ballistic=True)
            self.assertTrue(impact["ground_contact"])
            self.assertEqual(impact["position_m"], [10.0, 0.0, 0.0])
            self.assertEqual(impact["velocity_m_s"], [2.0, 0.0, 0.0])
            self.assertAlmostEqual(impact["dissipated_energy_j"], 3 * 9.80665)
            self.assertAlmostEqual(impact["energy_residual_j"], 0)
            self.assertEqual(impact["mass_kg"], 3)
            with self.assertRaises(ValueError):
                advance_body([0, float("nan"), 0], [0, 0, 0], 3, 5)

        def test_story_ledger_rejects_unsafe_or_conflicting_ids(self):
            for source in [{"plot": "A story", "characters": [{"id": "../secret"}]},
                           {"plot": "A story", "characters": [{"id": "a"}], "objects": [{"id": "a"}]},
                           {"plot": "A story", "characters": [{"mass_kg": float("nan")}]}]:
                with self.assertRaises(ValueError):
                    build_story_bible(json.dumps(source), 0, lambda _: None)
            with patch.dict(os.environ, {"LLM_MODEL": ""}):
                bible = build_story_bible("Mara meets Ivo at a station.", 123, lambda _: None)
            self.assertEqual([c["name"] for c in bible["characters"]], ["Mara", "Ivo"])

        def test_terminal_tensor_contract_rejects_wrong_shape_and_nonfinite_values(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "boundary.latent"
                contract = {"shape": [1, 1, 1, 1, 1], "dtype": "F16"}
                header = json.dumps({"latent_tensor": {"dtype": "F16", "shape": contract["shape"], "data_offsets": [0, 2]}}).encode()
                path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00")
                self.assertTrue(inspect_terminal_latent(path, contract)["finite"])
                with self.assertRaises(ValueError):
                    inspect_terminal_latent(path, {**contract, "shape": [1, 2, 1, 1, 1]})
                path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x7e")
                with self.assertRaisesRegex(ValueError, "NaN"):
                    inspect_terminal_latent(path, contract)

        def test_strict_native_policy_blocks_unsupported_workflow_before_rendering(self):
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"LLM_MODEL": ""}):
                studio = Studio(Path(tmp), acquire_lock=False)
                studio.store.data = {"schema": 1, "job_id": "test", "folder": tmp, "plot": "A traveler walks through rain.",
                    "story_engine_version": 1, "master_seed": 123, "shot_count": 2, "shots": [],
                    "media": {}, "comfy": {"consistency": {"require_native_latents": True}}, "cloud": {"provider": "None"}}
                with patch("shutil.which", return_value="installed"), patch.object(studio, "teardown") as teardown, \
                     patch(__name__ + ".ComfyClient") as client:
                    studio.pipeline()
                client.assert_not_called()
                teardown.assert_not_called()
                self.assertEqual(studio.store.data["status"], "recoverable_error")
                self.assertTrue(studio.store.data["shots"][0]["world_state"])
                self.assertFalse(studio.store.data["capabilities"]["quantum_solver"])

        def test_cast_reference_injection_does_not_replace_master(self):
            graph = {"p": {"inputs": {"text": ""}}, "master": {"inputs": {"image": "master"}},
                     "second": {"inputs": {"image": "second"}}, "s": {"inputs": {"seed": 0}}}
            client = ComfyClient.__new__(ComfyClient)
            client.config = {"prompt_node": "p", "prompt_input": "text", "image_node": "master", "image_input": "image",
                "seed_node": "s", "seed_input": "seed", "character_bindings": [{"character_id": "ivo", "node": "second", "input": "image"}]}
            submitted = []
            client.submit_graph = lambda graph, marker: submitted.append(graph) or "prompt"
            shot = structural_shots("Mara meets Ivo", 1)[0]
            shot.update(seed=123, character_references={"ivo": "ivo.png"})
            client.submit(graph, shot, "mara.png", "marker")
            self.assertEqual(submitted[0]["master"]["inputs"]["image"], "mara.png")
            self.assertEqual(submitted[0]["second"]["inputs"]["image"], "ivo.png")

        def test_sound_plan_uses_scene_context_and_stable_seeds(self):
            shot = {"shot_id": 2, "seed": 42, "visual_prompt": "Walking through rain beside a spaceship engine in the wind",
                    "identity_tokens": "ocean fire"}
            plan = audio_plan(shot)
            self.assertEqual(plan, audio_plan(copy.deepcopy(shot)))
            self.assertEqual(set(plan["cues"]), {"rain", "mechanical", "wind", "footsteps"})
            self.assertGreater(len(sound_layers(plan)), len(plan["cues"]))
            self.assertNotIn("water", plan["cues"])
            self.assertNotEqual(plan["seed"], audio_plan({**shot, "shot_id": 3})["seed"])
            self.assertEqual(audio_plan({"shot_id": 1, "visual_prompt": "A portrait indoors"})["cues"], ["room_tone"])

        def test_new_movies_freeze_finishing_options_without_migrating_legacy(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                with patch(__name__ + ".workflow_config", return_value=({}, {})), \
                     patch(__name__ + ".anchor_workflow_config", return_value=({}, {})), \
                     patch.object(studio, "launch"):
                    studio.start("A traveler", None, 1, "None", "", "", sound_design=False, cinematic=True)
                profile = StateStore(Path(tmp)).data["media"]["postproduction"]
                self.assertFalse(profile["audio"])
                self.assertTrue(profile["cinematic"])
                self.assertEqual(profile["crf"], 16)
                legacy = {"width": 640, "height": 360, "fps": 24}
                self.assertNotIn("curves", video_filters(legacy))
                self.assertNotIn("noise=", video_filters(legacy))
                finished = video_filters({**legacy, "postproduction": profile})
                self.assertIn("crop=640:268", finished)
                self.assertIn("curves=", finished)
                self.assertIn("colorbalance=", finished)
                self.assertIn("noise=", finished)

        def test_sound_generation_has_a_validated_ambient_fallback(self):
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "sound.wav"
                def fake_run(args, **kwargs):
                    if "-filter_complex" in args:
                        raise RuntimeError("cue filter failed")
                    with wave.open(args[-1], "wb") as audio:
                        audio.setnchannels(2)
                        audio.setsampwidth(2)
                        audio.setframerate(48000)
                        audio.writeframes(b"\x00" * 960000)
                    return ""
                with patch(__name__ + ".run", side_effect=fake_run):
                    plan = generate_sound({"shot_id": 1, "visual_prompt": "Rain"}, target,
                                          postproduction_config(), lambda _: None)
                validate_pcm(target)
                self.assertEqual(plan["fallback"], "ambient_bed")

        def test_finished_movie_requires_synchronized_aac_audio(self):
            settings = {"width": 1920, "height": 1080, "fps": 24, "postproduction": postproduction_config()}
            streams = [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                        "height": 804, "pix_fmt": "yuv420p", "duration": "10.0"},
                       {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000",
                        "channels": 2, "duration": "10.0", "start_time": "0.0"}]
            for change in ("missing", "short", "late", "wrong_codec", "valid"):
                sample = copy.deepcopy(streams)
                if change == "missing":
                    sample.pop()
                elif change == "short":
                    sample[1]["duration"] = "9.0"
                elif change == "late":
                    sample[1]["start_time"] = "0.5"
                elif change == "wrong_codec":
                    sample[1]["codec_name"] = "mp3"
                with patch(__name__ + ".run", return_value=json.dumps({"streams": sample})):
                    if change == "valid":
                        validate_finished_media(Path("movie.mp4"), 10, settings)
                    else:
                        with self.assertRaises(RuntimeError):
                            validate_finished_media(Path("movie.mp4"), 10, settings)

        def test_public_media_does_not_expose_checkpoint_files(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                studio.store.data = {"job_id": "test-job"}
                source = Path(tmp) / "movie.mp4"
                source.write_bytes(b"finished movie")
                public = studio.publish_media(source)
                self.assertEqual(public.read_bytes(), source.read_bytes())
                self.assertTrue(public.is_relative_to(Path(tmp) / "public_media"))
                self.assertEqual(studio.publish_media(source), public)
                with self.assertRaises(ValueError):
                    studio.publish_media(Path(tmp) / "state.json")

        def test_missing_workflow_is_checkpointed_for_background_setup(self):
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp), acquire_lock=False)
                with patch.dict(os.environ, {"COMFY_WORKFLOW": str(Path(tmp) / "missing.json")}), \
                     patch.object(studio, "launch") as launch:
                    studio.start("A traveler discovers a garden", None, 1, "None", "", "")
                launch.assert_called_once()
                recovered = StateStore(Path(tmp)).data
                self.assertTrue(recovered["setup_pending"])
                self.assertEqual(recovered["status"], "setting_up")
                self.assertEqual(recovered["anchor"]["mode"], "auto")

        def test_automatic_graph_supports_new_and_legacy_video_schema(self):
            names = [Path(spec[4]).name for spec in AUTO_MODELS]
            for definition, field in [({"format": ["COMFY_DYNAMICCOMBO_V3"]}, "format.codec"),
                                      ({"format": [["auto"]], "codec": [["auto"]]}, "codec")]:
                info = {"SaveVideo": {"input": {"required": definition}}}
                graph = build_automatic_wan_graph(info, names)
                self.assertEqual(graph["58"]["inputs"][field], "auto")
                with patch.dict(os.environ, {"CONSISTENCY_CONFIG": "missing-legacy-profile.json"}):
                    _, config = workflow_config(graph, {"prompt_node": "6", "prompt_input": "text",
                        "image_node": "56", "image_input": "image", "video_node": "58", "seed_node": "3",
                        "seed_input": "seed"}, {"continuity": {"mode": "off"}})
                self.assertEqual(config["consistency"]["continuity"]["mode"], "off")
                self.assertEqual(graph["55"]["inputs"]["start_image"], ["56", 0])

        def test_automatic_setup_creates_independent_text_only_anchor(self):
            from types import SimpleNamespace
            names = [Path(spec[4]).name for spec in AUTO_MODELS]
            info = {item["class_type"]: {} for item in build_automatic_wan_graph(
                {"SaveVideo": {"input": {"required": {"format": ["COMFY_DYNAMICCOMBO_V3"]}}}}, names).values()}
            info["SaveImage"] = {}
            info["SaveVideo"] = {"input": {"required": {"format": ["COMFY_DYNAMICCOMBO_V3"]}}}
            for spec, name in zip(AUTO_MODELS, names):
                info[spec[0]] = {"input": {"required": {spec[1]: [[name]]}}}
            response = Mock()
            response.json.return_value = info
            with tempfile.TemporaryDirectory() as tmp:
                with patch.dict(sys.modules, {"requests": SimpleNamespace(get=Mock(return_value=response), RequestException=OSError)}), \
                     patch.dict(os.environ, {"COMFY_ROOT": str(Path(tmp) / "comfy"),
                                             "CONSISTENCY_CONFIG": "irrelevant-legacy-profile.json"}):
                    result = automatic_workflow(Path(tmp), lambda _: None)
                self.assertNotIn("start_image", result["anchor_workflow"]["55"]["inputs"])
                self.assertEqual(result["anchor_workflow"]["55"]["inputs"]["length"], 1)
                self.assertEqual(result["workflow"]["55"]["inputs"]["length"], 121)
                self.assertNotIn("56", result["anchor_workflow"])
                self.assertEqual(result["anchor_workflow"]["9"]["class_type"], "SaveImage")
                self.assertTrue((Path(tmp) / "automatic_workflow.json").is_file())

        def test_model_download_resumes_and_verifies_before_promotion(self):
            from types import SimpleNamespace
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec = AUTO_MODELS[0]
                target = root / spec[2] / Path(spec[4]).name
                target.parent.mkdir()
                payload = b"verified model bytes"
                atomic_json(target.with_suffix(target.suffix + ".source.json"), {
                    "repo": spec[3], "file": spec[4], "revision": "a" * 40,
                    "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})
                part = target.with_suffix(target.suffix + ".part")
                part.write_bytes(payload[:4])
                response = Mock(status_code=206, headers={"Content-Range": f"bytes 4-{len(payload)-1}/{len(payload)}"})
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                response.iter_content.return_value = [payload[4:]]
                get = Mock(return_value=response)
                with patch.dict(sys.modules, {"requests": SimpleNamespace(get=get)}):
                    result = provision_model(root, spec, lambda _: None)
                self.assertEqual(result, target.name)
                self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=4-")
                self.assertEqual(target.read_bytes(), payload)
                self.assertFalse(part.exists())
                # Corrupt complete files must not be silently trusted or replaced.
                target.write_bytes(b"corrupt")
                with patch.dict(sys.modules, {"requests": SimpleNamespace(get=get)}), self.assertRaises(SetupError):
                    provision_model(root, spec, lambda _: None)

        def test_model_checksum_failure_quarantines_partial_file(self):
            from types import SimpleNamespace
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec = AUTO_MODELS[0]
                target = root / spec[2] / Path(spec[4]).name
                target.parent.mkdir()
                atomic_json(target.with_suffix(target.suffix + ".source.json"), {
                    "repo": spec[3], "file": spec[4], "revision": "a" * 40,
                    "sha256": hashlib.sha256(b"good").hexdigest(), "size": 4})
                target.with_suffix(target.suffix + ".part").write_bytes(b"evil")
                with patch.dict(sys.modules, {"requests": SimpleNamespace()}), self.assertRaises(SetupError):
                    provision_model(root, spec, lambda _: None)
                self.assertFalse(target.exists())
                self.assertFalse(target.with_suffix(target.suffix + ".part").exists())
                self.assertTrue(target.with_suffix(target.suffix + ".corrupt").exists())

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
                studio.store.data = {"schema": 1, "job_id": "recovery-test", "folder": tmp, "shots": [shot]}
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


def media_self_test() -> None:
    """Real two-shot FFmpeg test: no ComfyUI, credentials, cloud or teardown.

    Verifies effect syntax, audible PCM, AAC duration, progressive concatenation,
    widescreen dimensions, preview audio, and recovery without a GPU re-render.
    """
    import array
    import tempfile
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"Media integration test needs {binary} on PATH")
    LOG.setLevel(logging.INFO)
    LOG.addHandler(logging.StreamHandler())
    with tempfile.TemporaryDirectory(prefix="studio-media-test-") as temporary:
        root = Path(temporary)
        source = root / "source.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=320x180:rate=24:duration=5", "-c:v", "libx264", "-threads", "2",
             "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(source)])
        studio = Studio(root, acquire_lock=False)
        shots = structural_shots("Rain and wind around a humming spaceship engine.\nWalking beside a city river with a campfire.", 2)
        for shot in shots:
            raw = root / f"raw_{shot['shot_id']:04d}.mp4"
            shutil.copyfile(source, raw)
            shot.update(seed=1234, status="rendered", attempts=1, raw_sha256=digest(raw))
        bind_world_state(shots, build_story_bible(json.dumps({"plot": "Synthetic media test",
            "characters": [{"id": "subject", "name": "Test subject"}]}), 1234, studio.notice))
        media = {"width": 640, "height": 360, "fps": 24, "postproduction": postproduction_config()}
        studio.store.data = {"schema": 1, "job_id": "media-test", "folder": str(root),
                             "shots": shots, "media": media, "master_shot_ids": []}
        studio.store.save()
        work: queue.Queue = queue.Queue()
        for shot in shots:
            work.put((shot, root / f"raw_{shot['shot_id']:04d}.mp4"))
        work.put(None)
        studio.consume(work)
        if studio.assembly_error or any(s["status"] != "completed" for s in shots):
            raise RuntimeError("Media integration failed: " + "\n".join(studio.lines))
        metadata = validate_finished_media(root / "current_movie_stream.mp4", 10, media)
        for shot in shots:
            if shot["audio_plan"].get("fallback"):
                raise RuntimeError("Layered audio unexpectedly used its fallback")
            with wave.open(str(root / f"shot_{shot['shot_id']:04d}.wav"), "rb") as audio:
                samples = array.array("h", audio.readframes(audio.getnframes()))
                if sys.byteorder != "little":
                    samples.byteswap()
                peak = max(abs(value) for value in samples) / 32768
                if not 0.001 < peak < 0.95:
                    raise RuntimeError("Synthesized sound is silent or clipping")
            preview_info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                                          str(root / f"shot_{shot['shot_id']:04d}.ts")]))
            if not any(s.get("codec_name") == "aac" for s in preview_info["streams"]):
                raise RuntimeError("Streaming preview lost its soundtrack")
        # Losing a PCM checkpoint should regenerate finishing from durable raw
        # video, not rerender the shot or mix old and new processing profiles.
        (root / "shot_0001.wav").unlink()
        studio.recover_clips()
        if shots[0]["status"] != "rendered" or shots[1]["status"] != "completed":
            raise RuntimeError("Audio checkpoint recovery failed")
        # Also exercise optional sound/grade toggles on real files.
        for audio_enabled, grade_enabled in ((False, True), (True, False), (False, False)):
            settings = {**media, "postproduction": postproduction_config(audio_enabled, grade_enabled)}
            target = root / f"toggle_{int(audio_enabled)}_{int(grade_enabled)}.mp4"
            normalize(source, target, settings, shots[0], studio.notice)
            validate_finished_media(target, 5, settings)
        hd = {**media, "width": 1920, "height": 1080}
        normalize(source, root / "hd.mp4", hd, shots[0], studio.notice)
        validate_finished_media(root / "hd.mp4", 5, hd)
        video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        print(f"Media integration PASSED: two shots, 10.0s, {video['width']}x{video['height']}, "
              "H.264 + stereo AAC 48 kHz; 1920x804 export, layered sound, scope grade/grain, TS previews, toggles and recovery verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--media-self-test", action="store_true", help="Run real FFmpeg finishing tests without GPU/cloud/shutdown")
    parser.add_argument("--prepare", action="store_true", help="Prepare automatic workflows/models without rendering or shutdown")
    parser.add_argument("--enqueue", metavar="PLOT_FILE", help="Queue a text/structured JSON plot after the active production")
    parser.add_argument("--minutes", type=float, default=30, help="Duration for --enqueue (default: 30)")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.media_self_test:
        media_self_test()
        return
    root = Path(os.getenv("STUDIO_DIR", "/workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.enqueue:
        request = MovieQueue(root).enqueue(Path(args.enqueue).read_text(encoding="utf-8-sig"), args.minutes)
        print(f"Queued {request['id']}: {request['minutes']} minutes; status={request['status']}; cloud backup=None")
        return
    LOG.setLevel(logging.INFO)
    handler = RotatingFileHandler(root / "studio.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    os.chmod(root / "studio.log", 0o600)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
    LOG.addHandler(handler)
    LOG.addHandler(logging.StreamHandler())
    if args.prepare:
        automatic_workflow(root, lambda message: LOG.info(message))
        return
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
        allowed_paths=[str(root / "public_media")],
        blocked_paths=[str(root / "state.json"), str(root / "production_queue.json"), str(root / "studio.log"), str(root / "credentials.key"),
                       str(Path(os.getenv("GOOGLE_OAUTH_CLIENT_FILE", str(root / "google_oauth_client.json"))).resolve()),
                       str(Path(os.getenv("STUDIO_CREDENTIALS_FILE", str(root / ".credentials"))).resolve())])


if __name__ == "__main__":
    main()
