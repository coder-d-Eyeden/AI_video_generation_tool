"""
REST endpoints for AI video generation (Gemini Omni 1.1 Flash), clip library,
AI/manual editing and final timeline merging.
"""
import base64
import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend import config, media_store
from backend.job_manager import Job, job_manager
from backend.models import (
    ClipAIEditRequest,
    ClipTrimRequest,
    MergeRequest,
    PlanVideoRequest,
    ScenePrompt,
    SceneVideoRequest,
    VideoGenerationOptions,
)
from backend.omni_video_adapter import OmniFlashVideoAdapter, OmniVideoError
from backend.tts_service import TTSService
from backend.video_editor import (
    AUDIO_MODES,
    TRANSITIONS,
    RenderSettings,
    SegmentSpec,
    VideoEditError,
    frame_size,
    merge_timeline,
    trim_clip,
)

router = APIRouter(prefix="/api")

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def _adapter(opts) -> OmniFlashVideoAdapter:
    try:
        return OmniFlashVideoAdapter(
            api_key=opts.api_key,
            model_name=opts.model_name,
            aspect_ratio=opts.aspect_ratio,
            resolution=opts.resolution,
        )
    except OmniVideoError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


def _job_response(job: Job) -> dict:
    return job.to_dict()


def _generate_scene_clip(adapter: OmniFlashVideoAdapter, scene: ScenePrompt, opts: VideoGenerationOptions, job: Job, base: int, span: int) -> dict:
    def progress(msg: str) -> None:
        job.update(base + span // 3, f"Scene {scene.scene_number}: {msg}")

    result = adapter.generate_scene_video(
        visual_prompt=scene.visual_prompt,
        camera_angle=scene.camera_angle,
        duration_seconds=scene.duration_seconds,
        reference_images=[img.model_dump() for img in opts.reference_images],
        on_progress=progress,
    )
    if not result.success:
        raise OmniVideoError(f"Scene {scene.scene_number} failed: {result.error_message}")
    job.update(base + span, f"Scene {scene.scene_number}: saving clip")
    record = media_store.save_bytes(
        "clip",
        base64.b64decode(result.media_base64),
        ".mp4",
        source="omni",
        label=f"Scene {scene.scene_number}: {scene.scene_title}",
        scene_number=scene.scene_number,
        scene=scene.model_dump(),
        generation=result.metadata,
    )
    return media_store.public(record)


@router.get("/video/config")
async def video_config():
    return {
        "model": config.OMNI_MODEL_NAME,
        "api_key_configured": bool(config.get_omni_api_key()),
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": ["360p", "720p", "1080p", "4k"],
        "default_aspect_ratio": config.OMNI_DEFAULT_ASPECT_RATIO,
        "default_resolution": config.OMNI_DEFAULT_RESOLUTION,
        "transitions": TRANSITIONS,
        "audio_modes": AUDIO_MODES,
    }


@router.post("/video/generate-scene")
async def generate_scene_video(request: SceneVideoRequest):
    adapter = _adapter(request)
    scene = request.scene

    def task(job: Job):
        clip = _generate_scene_clip(adapter, scene, request, job, 0, 95)
        job.update(100, f"Scene {scene.scene_number} clip ready")
        return {"clips": [clip]}

    job = job_manager.submit("generate_scene", task, meta={"scene_numbers": [scene.scene_number]})
    return _job_response(job)


@router.post("/video/generate-all")
async def generate_all_videos(request: PlanVideoRequest):
    adapter = _adapter(request)
    scenes = [s for s in request.plan.scenes if not request.scene_numbers or s.scene_number in request.scene_numbers]
    if not scenes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No scenes selected for video generation.")

    def task(job: Job):
        clips, failures = [], []
        span = 100 // len(scenes)
        for i, scene in enumerate(scenes):
            try:
                clips.append(_generate_scene_clip(adapter, scene, request, job, i * span, span))
            except Exception as e:
                failures.append({"scene_number": scene.scene_number, "error": str(e)})
            job.result = {"clips": clips, "failures": failures}
        if not clips:
            raise OmniVideoError("; ".join(f["error"] for f in failures))
        job.update(100, f"Generated {len(clips)}/{len(scenes)} scene clips")
        return {"clips": clips, "failures": failures}

    job = job_manager.submit("generate_all", task, meta={"scene_numbers": [s.scene_number for s in scenes]})
    return _job_response(job)


@router.get("/jobs")
async def list_jobs():
    return job_manager.list()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job.to_dict()


# ---------------------------------------------------------------------- #
# Clip library
# ---------------------------------------------------------------------- #
@router.get("/clips")
async def list_clips():
    return media_store.list_media("clip")


@router.post("/clips/upload")
async def upload_clip(file: UploadFile = File(...), label: Optional[str] = Form(default=None)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _VIDEO_EXTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported video type '{ext}'.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded video is empty.")
    record = media_store.save_bytes("clip", data, ext, source="upload", label=label or file.filename)
    if not record.get("has_video"):
        media_store.delete("clip", record["id"])
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File does not contain a readable video stream.")
    return record


@router.delete("/clips/{clip_id}")
async def delete_clip(clip_id: str):
    if not media_store.delete("clip", clip_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip not found.")
    return {"deleted": clip_id}


def _require(kind: str, media_id: str) -> dict:
    rec = media_store.get(kind, media_id)
    if not rec or not os.path.exists(rec["path"]):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{kind.title()} '{media_id}' not found.")
    return rec


@router.post("/clips/{clip_id}/trim")
async def trim_clip_endpoint(clip_id: str, request: ClipTrimRequest):
    src = _require("clip", clip_id)
    out = media_store.new_media_path("clip", ".mp4")
    try:
        trim_clip(src["path"], out, request.start, request.end)
    except VideoEditError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return media_store.public(media_store.register(
        "clip", out, source="trim", parent_id=clip_id, scene_number=src.get("scene_number"),
        scene=src.get("scene"), label=f"{src.get('label', clip_id)} (trimmed {request.start}-{request.end or 'end'}s)",
    ))


@router.post("/clips/{clip_id}/ai-edit")
async def ai_edit_clip(clip_id: str, request: ClipAIEditRequest):
    src = _require("clip", clip_id)
    if request.mode not in ("edit", "extend"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="mode must be 'edit' or 'extend'.")
    adapter = _adapter(request)

    def task(job: Job):
        with open(src["path"], "rb") as f:
            video_bytes = f.read()
        notify = lambda msg: job.update(40, msg)
        if request.mode == "extend":
            result = adapter.extend_video(video_bytes, request.instruction, request.extend_seconds, on_progress=notify)
        else:
            result = adapter.edit_video(video_bytes, request.instruction, on_progress=notify)
        if not result.success:
            raise OmniVideoError(result.error_message)
        record = media_store.save_bytes(
            "clip", base64.b64decode(result.media_base64), ".mp4",
            source=f"omni-{request.mode}", parent_id=clip_id, scene_number=src.get("scene_number"),
            scene=src.get("scene"), label=f"{src.get('label', clip_id)} (AI {request.mode})",
            instruction=request.instruction, generation=result.metadata,
        )
        job.update(100, f"AI {request.mode} complete")
        return {"clips": [media_store.public(record)]}

    job = job_manager.submit(f"ai_{request.mode}", task, meta={"clip_id": clip_id})
    return _job_response(job)


# ---------------------------------------------------------------------- #
# Assets (background music)
# ---------------------------------------------------------------------- #
@router.get("/assets")
async def list_assets():
    return media_store.list_media("asset")


@router.post("/assets/upload")
async def upload_asset(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _AUDIO_EXTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported audio type '{ext}'.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio is empty.")
    return media_store.save_bytes("asset", data, ext, source="upload", label=file.filename)


# ---------------------------------------------------------------------- #
# Merge / final render
# ---------------------------------------------------------------------- #
@router.get("/finals")
async def list_finals():
    return media_store.list_media("final")


@router.post("/video/merge")
async def merge_videos(request: MergeRequest):
    if not request.segments:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Timeline is empty. Add clips before merging.")
    for seg in request.segments:
        if seg.audio_mode not in AUDIO_MODES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid audio_mode '{seg.audio_mode}'.")
    if request.transition not in TRANSITIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid transition '{request.transition}'.")
    clip_paths = {seg.clip_id: _require("clip", seg.clip_id)["path"] for seg in request.segments}
    music_path = _require("asset", request.background_music_id)["path"] if request.background_music_id else None

    def task(job: Job):
        tts_files = []
        try:
            specs = []
            for i, seg in enumerate(request.segments):
                vo_path = None
                if seg.audio_mode in ("voiceover", "mix") and (seg.voiceover_text or "").strip():
                    job.update(2, f"Synthesizing Bengali voiceover {i + 1}/{len(request.segments)}...")
                    vo_path = os.path.join(config.TEMP_DIR, f"vo_{uuid.uuid4().hex}.mp3")
                    TTSService.synthesize_to_file(seg.voiceover_text, vo_path)
                    tts_files.append(vo_path)
                specs.append(SegmentSpec(
                    path=clip_paths[seg.clip_id],
                    trim_start=seg.trim_start,
                    trim_end=seg.trim_end,
                    speed=seg.speed,
                    overlay_text=seg.overlay_text,
                    voiceover_path=vo_path,
                    narration_start=seg.narration_start,
                    audio_mode=seg.audio_mode,
                    original_volume=seg.original_volume,
                    voiceover_volume=seg.voiceover_volume,
                ))
            width, height = frame_size(request.aspect_ratio, request.resolution)
            settings = RenderSettings(
                width=width,
                height=height,
                transition=request.transition,
                transition_duration=request.transition_duration,
                background_music_path=music_path,
                background_music_volume=request.background_music_volume,
                fade_in_out=request.fade_in_out,
            )
            out = media_store.new_media_path("final", ".mp4")
            info = merge_timeline(specs, out, settings, on_progress=lambda p, m: job.update(p, m))
            record = media_store.register(
                "final", out, label=request.title or "Final merged video",
                timeline=request.model_dump(), segments=info["segments"],
            )
            job.update(100, f"Final video ready ({info['duration']}s)")
            return {"final": media_store.public(record)}
        finally:
            for p in tts_files:
                if os.path.exists(p):
                    os.remove(p)

    job = job_manager.submit("merge", task, meta={"segments": len(request.segments)})
    return _job_response(job)
