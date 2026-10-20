"""Capability-driven cinematic planning and the official MiniMax H3 V2 adapter.

This module has no import-time downloads, model loads, network calls, or SDK
dependencies. Documentation verification is distinct from account availability
and from successful generation. See orchestration_audit.json for the baseline.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.parse import urlsplit


H3_SOURCE = "https://platform.minimax.io/docs/api-reference/video-generation-v2-create"
H3_LOCAL_SOURCE = "https://github.com/MiniMax-AI/MiniMax-H3"
H3_BASE = "https://api.minimax.io"


@dataclass(frozen=True)
class Capability:
    id: str
    backend: str
    availability: str
    modes: tuple[str, ...]
    durations: tuple[int, ...]
    images: int
    videos: int = 0
    audios: int = 0
    combined: int = 1
    video_seconds: int = 0
    audio_seconds: int = 0
    resolutions: tuple[str, ...] = ()
    fps: int = 24
    native_audio: bool = False
    evidence: str = ""
    reason: str = ""
    real_generation_verified: bool = False

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def registry(config: dict[str, Any] | None = None, *, allow_remote: bool = True) -> list[Capability]:
    config = config or {}
    enabled = os.getenv("MINIMAX_ENABLED", "0") == "1" and allow_remote
    state = ("configured_unverified" if enabled and os.getenv("MINIMAX_API_KEY") else
             "awaiting_credentials" if enabled else "disabled_by_policy")
    reason = ("Configured credentials; verify account connectivity before generation."
              if state == "configured_unverified" else "Set MINIMAX_ENABLED=1 and MINIMAX_API_KEY on the server.")
    result = [Capability("MiniMax-H3", "minimax", state, ("text", "endpoints", "reference"),
                         tuple(range(4, 16)), 9, 3, 3, 12, 15, 15, ("768P", "2K"),
                         native_audio=True, evidence=H3_SOURCE, reason=reason),
              Capability("MiniMax-H3-Max", "minimax", state, ("text", "endpoints", "reference"),
                         tuple(range(5, 16)), 9, 3, 3, 12, 15, 15, ("480P", "768P"),
                         native_audio=True, evidence=H3_SOURCE, reason=reason)]
    # The existing custom graph is preserved as an opaque tested integration.
    for model, modes in (("MiniMax-H3-Base-FL2VA",("text","endpoints")),
                         ("MiniMax-H3-Base-Ref2VA",("reference",))):
        result.append(Capability(model,"local_h3","unverified_local_installation",modes,tuple(range(4,16)),
            9 if "reference" in modes else 2, videos=3 if "reference" in modes else 0,
            audios=3 if "reference" in modes else 0,combined=12 if "reference" in modes else 2,
            video_seconds=15,audio_seconds=15,native_audio=True,evidence=H3_LOCAL_SOURCE,
            reason="Local variant requires verified weights, compatible runtime and an audited workflow; registered nodes alone are insufficient. No automatic large download."))
    # Its duration remains five seconds unless an explicit audited profile exists.
    ready = bool(config.get("video_node"))
    result.append(Capability("existing-comfy-workflow", "comfyui", "configured" if ready else "setup_required",
                  ("reference",), (5,), 1 + len(config.get("character_bindings", [])),
                  combined=1 + len(config.get("character_bindings", [])), evidence="existing exported API graph",
                  reason="Separate temporal conditioning uses only the graph's declared continuity mapping."))
    return result


def choose_model(caps: list[Capability], policy: str = "auto", seconds: int | None = None) -> tuple[Capability, str]:
    if policy not in {"auto", "local", "h3", "h3-max"}:
        raise ValueError("Unknown generation policy")
    eligible = [c for c in caps if c.availability in {"configured", "configured_unverified", "verified_access"}
                and (seconds is None or seconds in c.durations)]
    if policy != "local":
        h3 = next((c for c in eligible if c.backend == "minimax"
                   and (policy != "h3-max" or c.id == "MiniMax-H3-Max")), None)
        if h3:
            return h3, "Preferred compatible H3 provider; real generation remains unverified until completion."
    if policy not in {"h3","h3-max"}:
        local = next((c for c in caps if c.backend == "comfyui" and (seconds is None or seconds in c.durations)), None)
        if local:
            return local, "H3 unavailable, disabled, or incompatible; retain the existing ComfyUI workflow."
    raise ValueError("No compatible backend is available for this request; inspect model readiness")


class ReferenceBudget:
    """Select relevant approved assets; critical references are never truncated."""
    def __init__(self, capability: Capability):
        self.cap = capability

    def select(self, assets: list[dict[str, Any]], mode: str,
               characters: list[str] | None = None) -> dict[str, Any]:
        if mode not in self.cap.modes:
            raise ValueError("Conditioning mode is unsupported by the selected model")
        selected, omitted, seen = [], [], set()
        counts = {"image": 0, "video": 0, "audio": 0}
        totals = {"video": 0.0, "audio": 0.0}
        roles = set()
        if any(not isinstance(a,dict) or not math.isfinite(float(a.get("priority",0))) for a in assets):
            raise ValueError("References require finite priorities and object records")
        for asset in sorted(assets, key=lambda a: (not a.get("critical", False), -float(a.get("priority", 0)))):
            a = copy.deepcopy(asset)
            role, kind = a.get("role"), a.get("kind")
            why = None
            if role == "planning":
                why = "Internal planning frame, not a submitted reference"
            elif characters is not None and a.get("character_id") and a["character_id"] not in characters:
                why = "Character is absent from this shot"
            elif not a.get("approved", False):
                why = "Asset is not approved"
            if why:
                if a.get("critical") and role != "planning" and why == "Asset is not approved":
                    raise ValueError("A required reference needs approval")
                omitted.append({"id": a.get("id"), "reason": why})
                continue
            if kind not in counts:
                raise ValueError("Unknown reference media type")
            if mode == "endpoints" and (kind != "image" or role not in {"first_frame", "last_frame"}):
                raise ValueError("Endpoint mode cannot mix in identity, video, or audio references")
            if mode == "reference" and role not in {"reference_image", "reference_video", "reference_audio"}:
                raise ValueError("Reference mode cannot include temporal endpoints")
            if mode == "text":
                raise ValueError("Text-only mode does not accept media")
            identity = a.get("sha256") or a.get("id")
            if not identity:
                raise ValueError("References need a stable asset ID")
            if (identity, role) in seen:
                continue
            limit = (1 if role in {"first_frame", "last_frame"} else
                     {"image": self.cap.images, "video": self.cap.videos, "audio": self.cap.audios}[kind])
            seconds = float(a.get("duration_seconds", 0))
            if kind != "image" and (not math.isfinite(seconds) or not 2 <= seconds <= 15):
                raise ValueError("Video/audio references must be 2–15 seconds")
            full = ((role in {"first_frame", "last_frame"} and role in roles)
                    or (role not in {"first_frame", "last_frame"} and counts[kind] >= limit)
                    or len(selected) >= (2 if mode == "endpoints" else self.cap.combined)
                    or (kind != "image" and totals[kind] + seconds > getattr(self.cap, kind + "_seconds")))
            if full:
                if a.get("critical"):
                    raise ValueError("Required references exceed the verified budget; split the shot or change backend")
                omitted.append({"id": a["id"], "reason": "Lower-priority optional reference exceeds the budget"})
                continue
            roles.add(role); seen.add((identity, role)); counts[kind] += 1
            if kind != "image":
                totals[kind] += seconds
            a["selection_reason"] = a.get("selection_reason", "Approved asset relevant to the shot")
            a["submission_order"] = len(selected)
            selected.append(a)
        return {"model": self.cap.id, "mode": mode, "selected": selected, "omitted": omitted,
                "used": counts, "limits": self.cap.snapshot(), "validated": True}


def distribute(total: int, count: int, low: int, high: int) -> list[int]:
    if count < 1 or not count * low <= total <= count * high:
        raise ValueError("Requested duration cannot fit the model's shot limits")
    base, remainder = divmod(total, count)
    return [base + (i < remainder) for i in range(count)]


class CinematicBrain:
    """Auditable structural director; does not masquerade as an LLM/vision model."""
    def __init__(self, capabilities: list[Capability], policy: str = "auto",
                 notice: Callable[[str],None] = lambda _:None):
        self.cap, self.routing_reason = choose_model(capabilities, policy)
        self.notice = notice

    def plan(self, source: dict[str, Any], seconds: int, film_id: str = "draft") -> dict[str, Any]:
        if not isinstance(source.get("plot"), str) or not source["plot"].strip() or not 5 <= seconds <= 3600:
            raise ValueError("Provide a story and a runtime between five seconds and one hour")
        authored = source.get("storyboard")
        if authored:
            scenes = copy.deepcopy(authored)
        else:
            beats = [b.strip() for b in source["plot"].splitlines() if b.strip()]
            if len(beats) == 1:
                beats = [b.strip() for b in re.split(r"(?<=[.!?])\s+(?=[A-Z])", beats[0]) if b.strip()]
            scenes = [{"scene_id": f"scene_{i+1:03d}", "description": beat} for i, beat in enumerate(beats)]
        if not isinstance(scenes, list) or not 1 <= len(scenes) <= 720:
            raise ValueError("Invalid scene list")
        step = 5 if self.cap.durations == (5,) else 1
        low, high = min(self.cap.durations), max(self.cap.durations)
        if seconds % step:
            raise ValueError("The existing ComfyUI workflow needs a runtime divisible by five seconds")
        if authored:
            scene_seconds = [sum(int((x if isinstance(x, dict) else {}).get("duration_seconds", 5)) for x in s["shots"]) for s in scenes]
            if sum(scene_seconds) != seconds:
                raise ValueError("Authored storyboard duration does not equal the selected runtime")
        else:
            units = seconds // step
            base = math.ceil(low / step)
            if units < base * len(scenes):
                raise ValueError("Too many story beats for this runtime; combine scenes or increase duration")
            weights = [1 + len(re.findall(r"\b(then|while|runs|chases|fights|asks|replies|says)\b", s["description"], re.I)) for s in scenes]
            remaining = units - base * len(scenes)
            shares = [remaining * w / sum(weights) for w in weights]
            allocated = [base + int(x) for x in shares]
            for i in sorted(range(len(scenes)), key=lambda j: shares[j] % 1, reverse=True)[:units-sum(allocated)]:
                allocated[i] += 1
            scene_seconds = [x * step for x in allocated]
        tasks, scene_records, seen = [], [], set()
        cast = source.get("characters", [])
        for index, (scene, duration) in enumerate(zip(scenes, scene_seconds)):
            sid = scene.get("scene_id", f"scene_{index+1:03d}")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid) or sid in seen:
                raise ValueError("Scene IDs must be unique safe identifiers")
            seen.add(sid)
            description = scene.get("description", scene.get("title", source["plot"]))
            if not isinstance(description,str) or not description.strip():
                raise ValueError("Scene description must be nonempty text")
            text = description.lower()
            dialogue = bool(re.search(r'\b(dialogue|asks|replies|conversation|says)\b|["“]', text))
            action = bool(re.search(r"\b(chase|chases|runs|fight|fights|crashes|then|hands over)\b", text))
            present = [c.get("id", f"character_{i+1}") for i,c in enumerate(cast)
                       if len(cast) == 1 or c.get("name", "\x00").lower() in text]
            coverage = (["master", "over_the_shoulder", "reverse_angle", "reaction"] if dialogue else
                        ["wide_action", "medium_action", "reaction"] if action else ["continuous"])
            needed = max(math.ceil(duration / high), min(len(coverage), duration // low))
            durations = ([5] * (duration // 5) if step == 5 else distribute(duration, needed, low, high))
            authored_shots = scene.get("shots")
            if authored_shots:
                durations = []
                for shot in authored_shots:
                    original = shot if isinstance(shot, dict) else {"visual_prompt": shot}
                    d = original.get("duration_seconds", 5)
                    if type(d) is not int or d < low:
                        raise ValueError("Authored shot duration is unsupported")
                    n = math.ceil(d / high)
                    durations.extend([(part, original) for part in distribute(d, n, low, high)])
            else:
                durations = [(d, None) for d in durations]
            scene_tasks = []
            for position, (d, original) in enumerate(durations):
                number = len(tasks) + 1
                framing = (original or {}).get("framing", coverage[position % len(coverage)])
                original = original or {}
                prompt = original.get("visual_prompt", description)
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
                    raise ValueError("Shot prompt must be nonempty and at most 12000 characters")
                shot_present = original.get("characters_present")
                known = {c.get("id",f"character_{i+1}") for i,c in enumerate(cast)}
                if shot_present is None:
                    shot_present = ([c.get("id",f"character_{i+1}") for i,c in enumerate(cast)
                        if c.get("name","\x00").lower() in prompt.lower()] if original else present)
                    if not shot_present and len(cast) == 1:
                        shot_present = list(known)
                if not isinstance(shot_present,list) or any(c not in known for c in shot_present):
                    raise ValueError("Shot cast must reference existing character IDs")
                uid = hashlib.sha256(f"{film_id}:{sid}:{position}".encode()).hexdigest()[:24]
                # A continuous take may use sequential generation tasks when the
                # actual backend cannot generate its entire duration at once.
                logical = f"{sid}:take_{0 if coverage == ['continuous'] else position}"
                task = {"shot_id": number, "shot_uid": uid, "scene_id": sid, "sequence_id": f"sequence_{index//8+1}",
                    "logical_take_id": logical, "visual_prompt": prompt, "duration_seconds": d,
                    "camera_movement": original.get("camera_movement", "locked camera" if framing == "continuous" else "restrained cinematic movement"),
                    "lighting_cue": original.get("lighting_cue", scene.get("lighting_cue", "preserve scene lighting and weather")),
                    "framing": framing, "characters_present": shot_present, "narrative_purpose": description,
                    "starting_state": original.get("starting_state", "preserve established scene geography"),
                    "ending_state": original.get("ending_state", "complete only this shot's stated action"),
                    "dialogue": original.get("dialogue", ""), "approval": "pending",
                    "model_route": {"model": self.cap.id, "backend": self.cap.backend,
                                    "mode": "reference", "reason": self.routing_reason},
                    "keyframe_strategy": "approved identity references; new angles may use new compositions",
                    "dependency_ids": [tasks[-1]["shot_uid"]] if tasks and tasks[-1]["scene_id"] == sid and coverage == ["continuous"] else [],
                    "continuity_policy": "continue" if coverage == ["continuous"] and position else "new_composition",
                    "validation_required": ["duration", "dimensions", "audio_sync", "decode"],
                    "visual_assessment": "not_measured; compatible vision evaluator or human review required"}
                tasks.append(task); scene_tasks.append(uid)
            scene_records.append({"scene_id": sid, "duration_seconds": duration, "task_ids": scene_tasks,
                "coverage": coverage, "decision": "Dialogue coverage" if dialogue else "Action decomposition" if action else
                "One continuous take; split into generation tasks only where duration requires it"})
        plan = {"version": 1, "film_id": film_id, "planner": "deterministic_structural_director",
                "original_story": source["plot"], "duration_seconds": seconds, "scenes": scene_records,
                "shots": tasks, "characters": copy.deepcopy(cast), "objects": copy.deepcopy(source.get("objects", [])),
                "capability": self.cap.snapshot(), "limitations": ["No vision identity measurement is installed by this planner.",
                 "Independent camera composition requires a compatible reference-conditioned backend or approved keyframes."]}
        self.validate(plan)
        if not authored and os.getenv("LLM_MODEL"):
            plan = self.refine(plan)
        return plan

    def refine(self, plan: dict[str,Any]) -> dict[str,Any]:
        """Use the existing compatible narrative endpoint; preserve task topology.

        A failed batch never leaves a half-rewritten production plan. Requests
        contain narrative metadata only, never cloud/provider credentials.
        """
        import requests
        result = copy.deepcopy(plan)
        fields = {"visual_prompt","camera_movement","lighting_cue","starting_state","ending_state","dialogue","characters_present"}
        batch = max(1,min(30,int(os.getenv("LLM_BATCH_SIZE","12"))))
        known = {c["id"] for c in plan["characters"]}
        try:
            for offset in range(0,len(result["shots"]),batch):
                current = result["shots"][offset:offset+batch]
                self.notice(f"Narrative director: refining tasks {offset+1}–{offset+len(current)} with the configured LLM.")
                response = requests.post(os.getenv("LLM_URL","http://127.0.0.1:11434/v1").rstrip('/')+'/chat/completions',
                    timeout=(10,180),json={"model":os.environ["LLM_MODEL"],"temperature":0,"messages":[
                        {"role":"system","content":'Return JSON {"shots":[...]} with the same ordered shot_id values. '
                         'Direct only each assigned action beat, never replay the whole story in each shot. Preserve the ending, '
                         'scene geography, wardrobe, eyelines, screen direction, and established props. Use only known character IDs. '
                         'Write feasible motion fitting duration_seconds. Keep dialogue distinct per take, do not repeat it. '
                         'Allowed fields: shot_id, visual_prompt, camera_movement, lighting_cue, starting_state, ending_state, '
                         'dialogue, characters_present. No invented technical guarantees or new characters.'},
                        {"role":"user","content":json.dumps({"story":plan["original_story"],"cast":plan["characters"],
                         "scene_coverage":plan["scenes"],"assigned_tasks":current},ensure_ascii=False)}]})
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content'].strip()
                updates = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',content))['shots']
                if not isinstance(updates,list) or [s.get('shot_id') for s in updates] != [s['shot_id'] for s in current]:
                    raise ValueError('Narrative response changed task order')
                for shot,update in zip(current,updates):
                    if set(update) - fields - {'shot_id'} or any(c not in known for c in update.get('characters_present',[])):
                        raise ValueError('Narrative response references unknown fields/cast')
                    shot.update({k:v for k,v in update.items() if k in fields})
            self.validate(result)
            result['planner'] = 'structural_coverage_with_configured_llm_direction'
            result['narrative_model'] = os.environ['LLM_MODEL']
            return result
        except Exception:
            if os.getenv('LLM_FAILURE_POLICY','structural') == 'fail':
                raise ValueError('Narrative refinement failed validation; production not submitted') from None
            self.notice('Narrative refinement unavailable/invalid; the complete structural plan was retained.')
            return plan

    @staticmethod
    def validate(plan: dict[str, Any]) -> None:
        shots = plan.get("shots", [])
        if not isinstance(shots, list) or not 1 <= len(shots) <= 900:
            raise ValueError("Production plan must contain 1–900 shots")
        ids = set()
        for index, shot in enumerate(shots, 1):
            if not isinstance(shot,dict):
                raise ValueError("Shot must be an object")
            if (shot.get("shot_id") != index or not re.fullmatch(r"[a-f0-9]{24}",str(shot.get("shot_uid","")))
                    or shot.get("shot_uid") in ids):
                raise ValueError("Shot IDs/order are invalid or duplicated")
            ids.add(shot.get("shot_uid"))
            if type(shot.get("duration_seconds")) is not int or not 4 <= shot["duration_seconds"] <= 15:
                raise ValueError("Each generation task must be 4–15 seconds")
            if not isinstance(shot.get("visual_prompt"), str) or not shot["visual_prompt"].strip():
                raise ValueError("Every shot needs a visual prompt")
            for field in ("visual_prompt","framing","camera_movement","lighting_cue","starting_state","ending_state","dialogue"):
                if field in shot and (not isinstance(shot[field],str) or len(shot[field]) > 12000):
                    raise ValueError("Shot text fields must be bounded strings")
            if not isinstance(shot.get("model_route"),dict) or not isinstance(shot.get("characters_present"),list):
                raise ValueError("Each shot needs a model route and cast list")
            if any(not isinstance(c,str) for c in shot["characters_present"]):
                raise ValueError("Cast IDs must be strings")
            if any(dep not in ids - {shot["shot_uid"]} for dep in shot.get("dependency_ids",[])):
                raise ValueError("Dependencies must reference earlier shots")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",str(shot.get("scene_id",""))):
                raise ValueError("Shot scene ID is invalid")
        if sum(s["duration_seconds"] for s in shots) != plan.get("duration_seconds"):
            raise ValueError("Plan duration does not match the requested film runtime")


def compile_motion(shot: dict[str, Any], identities: str = "", style: str = "") -> str:
    parts = [f"Framing: {shot.get('framing', 'cinematic')}", f"Initial state: {shot.get('starting_state','establish scene')}",
             f"Action: {shot['visual_prompt']}", f"Camera: {shot.get('camera_movement','steady')}",
             f"Lighting: {shot.get('lighting_cue','consistent')}", f"Terminal state: {shot.get('ending_state','finish action')}"]
    if identities:
        parts.append("Locked cast identity: " + identities)
    if style:
        parts.append("Locked style: " + style)
    if shot.get("dialogue"):
        parts.append("Exact dialogue, once: " + str(shot["dialogue"]))
    return ". ".join(parts)


class ProviderError(RuntimeError):
    def __init__(self, category: str, *, ambiguous: bool = False):
        super().__init__(category)
        self.category, self.ambiguous = category, ambiguous


class MiniMaxH3:
    """Official V2 API. No automatic replay of ambiguous paid POST requests."""
    def __init__(self, capability: Capability, session: Any = None):
        if session is None:
            import requests
            session = requests.Session()
        self.session, self.cap = session, capability

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        key = os.getenv("MINIMAX_API_KEY", "")
        if not key:
            raise ProviderError("credentials_missing")
        for attempt in range(3):
            try:
                response = self.session.request(method, H3_BASE + path,
                    headers={"Authorization": "Bearer " + key}, json=payload, timeout=(10, 90), allow_redirects=False)
            except Exception:
                if method == "POST":
                    raise ProviderError("submission_unknown", ambiguous=True) from None
                if attempt == 2:
                    raise ProviderError("network_unavailable") from None
                time.sleep(2 ** attempt)
                continue
            code = response.status_code
            if code == 429:
                if attempt == 2:
                    raise ProviderError("rate_limited")
                time.sleep(min(30, 2 ** (attempt+1)))
                continue
            if code >= 500:
                if method == "POST":
                    raise ProviderError("submission_unknown", ambiguous=True)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            if not 200 <= code < 300:
                raise ProviderError({401:"authentication_failed",403:"permission_denied",402:"insufficient_balance",
                                     400:"invalid_request",422:"request_rejected"}.get(code,"provider_http_error"))
            try:
                body = response.json()
                if not isinstance(body, dict) or body.get("error"):
                    raise ValueError()
                return body
            except (ValueError, TypeError):
                raise ProviderError("invalid_provider_response", ambiguous=method == "POST") from None
        raise ProviderError("provider_unavailable")

    def verify_access(self) -> dict[str, Any]:
        body = self.request("GET", "/v2/query/video_generation?page_num=1&page_size=1")
        if not isinstance(body.get("items"), list):
            raise ProviderError("unexpected_task_list_schema")
        return {"availability":"verified_access", "real_generation_verified":False}

    def payload(self, shot: dict[str, Any], manifest: dict[str, Any], roots: list[Path],
                resolution: str = "768P") -> dict[str, Any]:
        if shot["duration_seconds"] not in self.cap.durations or resolution not in self.cap.resolutions:
            raise ValueError("Unsupported H3 duration/resolution")
        # Revalidate, even when a caller supplies an edited manifest.
        manifest = ReferenceBudget(self.cap).select(manifest.get("selected", []), manifest["mode"])
        content = [{"type":"text", "text":shot.get("compiled_prompt") or compile_motion(shot)}]
        size = 0
        for asset in manifest["selected"]:
            path = Path(asset["path"]).resolve()
            if not any(path.is_relative_to(root.resolve()) for root in roots) or not path.is_file():
                raise ValueError("Reference must be an existing file inside this project")
            size += path.stat().st_size * 4 // 3 + 4
            ceiling = {"image":30,"video":50,"audio":15}[asset["kind"]] * 1024**2
            if path.stat().st_size > ceiling or size > 63 * 1024**2:
                raise ValueError("Reference data exceeds the provider request budget")
            suffix = path.suffix.lower()
            mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".webp":"image/webp",
                    ".mp4":"video/mp4",".mov":"video/quicktime",".wav":"audio/wav",".mp3":"audio/mpeg"}.get(suffix)
            if not mime or not mime.startswith(asset["kind"] + "/"):
                raise ValueError("Unsupported reference file format")
            if asset["kind"] == "image":
                from PIL import Image
                with Image.open(path) as image:
                    w,h = image.size
                    image.verify()
                if not 256 <= min(w,h) or max(w,h) > 5760 or not 0.4 <= w/h <= 2.5:
                    raise ValueError("H3 reference dimensions are outside the documented limits")
            else:
                metadata = json.loads(subprocess.run(["ffprobe","-v","error","-show_streams","-show_format",
                    "-of","json",str(path)],check=True,capture_output=True,text=True,timeout=30).stdout)
                actual = float(metadata.get("format",{}).get("duration",0))
                if not 2 <= actual <= 15 or abs(actual-float(asset["duration_seconds"])) > 0.1:
                    raise ValueError("Reference duration metadata does not match the media")
                if asset["kind"] == "video":
                    video = next((s for s in metadata.get("streams",[]) if s.get("codec_type")=="video"),{})
                    rate = video.get("avg_frame_rate","0/1").split("/")
                    fps = float(rate[0])/float(rate[1]) if len(rate)==2 and float(rate[1]) else 0
                    if video.get("codec_name") not in {"h264","hevc"} or not 23.976 <= fps <= 60:
                        raise ValueError("Reference video codec/frame rate is unsupported")
            raw = path.read_bytes()
            if asset.get("sha256") and hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                raise ValueError("Approved reference changed after selection")
            data = "data:" + mime + ";base64," + base64.b64encode(raw).decode()
            kind = asset["kind"] + "_url"
            content.append({"type":kind, kind:{"url":data}, "role":asset["role"]})
        return {"model":self.cap.id,"content":content,"duration":shot["duration_seconds"],
                "resolution":resolution,"ratio":"adaptive" if manifest["mode"]=="endpoints" else "16:9"}

    def create(self, payload: dict[str, Any], endpoint: str = "/v2/video_generation") -> str:
        if endpoint not in {"/v2/video_generation", "/v2/h3_context_ir", "/v2/video_regeneration"}:
            raise ValueError("Unsupported official H3 task endpoint")
        task = self.request("POST", endpoint, payload).get("task_id")
        if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task):
            raise ProviderError("submission_unknown", ambiguous=True)
        return task

    def wait(self, task_id: str, notice: Callable[[str], None], timeout: float = 7200) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
            raise ValueError("Invalid provider task ID")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.request("GET", "/v2/query/video_generation/" + task_id).get("task", {})
            state = task.get("status")
            if state == "succeeded":
                return task
            if state in {"failed", "cancelled"}:
                raise ProviderError("provider_task_" + state)
            if state not in {"queued", "running"}:
                raise ProviderError("unexpected_task_status")
            notice(f"H3 task {task_id}: {state}; provider does not expose a completion percentage.")
            time.sleep(5)
        raise ProviderError("provider_task_timeout")

    def download(self, task: dict[str, Any], target: Path) -> None:
        url = task.get("content", {}).get("url", "")
        # Never send the API credential to output CDNs; reject local/metadata
        # URLs and untrusted redirect destinations rather than enabling SSRF.
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None,443}:
            raise ProviderError("unsafe_output_url")
        trusted = ("minimax.io","minimax.chat","hailuoai.com","hailuoai.video")
        if not any(host == domain or host.endswith("."+domain) for domain in trusted):
            raise ProviderError("unverified_output_cdn")
        import ipaddress
        addresses = socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ProviderError("unsafe_output_address")
        import requests
        pending = target.with_suffix(".pending.mp4")
        try:
            with requests.get(url,stream=True,timeout=(10,120),allow_redirects=False) as response:
                if response.status_code != 200:
                    raise ProviderError("output_download_failed")
                total = 0
                with pending.open("wb") as handle:
                    for block in response.iter_content(1024**2):
                        total += len(block)
                        if total > 2*1024**3 or shutil.disk_usage(target.parent).free < 256*1024**2:
                            raise ProviderError("insufficient_output_storage")
                        handle.write(block)
                    handle.flush();os.fsync(handle.fileno())
                if total == 0:
                    raise ProviderError("empty_output")
            os.replace(pending,target)
        finally:
            pending.unlink(missing_ok=True)


def dependency_report(manifest: Path, *, install: bool = False,
                      notice: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Extend startup provisioning without changing existing model installers.

    Never upgrade an installed incompatible package automatically. The existing
    requirements file remains the sole Python dependency manifest.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:
        if not install:
            return {"ready":False,"generation_ready":False,"packages":[{"name":"packaging","state":"missing"}],"tools":{}}
        subprocess.run([sys.executable,"-m","pip","install","packaging>=24,<27"],check=True,capture_output=True,timeout=300)
        from packaging.requirements import Requirement
    aliases = {"google-api-python-client":"googleapiclient", "google-auth-httplib2":"google_auth_httplib2",
               "google-auth-oauthlib":"google_auth_oauthlib", "Pillow":"PIL"}
    report = {"python":sys.version.split()[0], "packages":[], "tools":{}, "ready":True}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.split("#",1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        name = req.name
        try:
            version = importlib.metadata.version(name)
            state = "ready" if version in req.specifier else "version_conflict"
        except importlib.metadata.PackageNotFoundError:
            version,state = None,"missing"
        if state == "missing" and install:
            notice("Installing missing dependency: " + name)
            try:
                subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",line],
                               check=True,capture_output=True,timeout=600)
                version = importlib.metadata.version(name)
                state = "ready" if version in req.specifier else "version_conflict"
            except (OSError,subprocess.SubprocessError,importlib.metadata.PackageNotFoundError):
                state = "installation_failed"
        if state == "ready":
            module = aliases.get(name,name.replace("-","_"))
            if importlib.util.find_spec(module) is None:
                state = "import_unavailable"
        essential = name in {"gradio","requests","cryptography","packaging"}
        report["packages"].append({"name":name,"version":version,"state":state,"essential":essential})
        if essential and state != "ready":
            report["ready"] = False
    for binary in ("ffmpeg","ffprobe"):
        report["tools"][binary] = "ready" if shutil.which(binary) else "missing"
    report["generation_ready"] = report["ready"] and all(v == "ready" for v in report["tools"].values())
    return report
