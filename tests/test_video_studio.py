"""
Tests for Omni 1.1 Flash adapter, ffmpeg editing/merging engine, media library
and video studio REST endpoints. No real API calls are made.
"""
import base64
import os
import subprocess
import time
from types import SimpleNamespace

import pytest

from backend import config


@pytest.fixture(autouse=True)
def media_root(tmp_path, monkeypatch):
    root = str(tmp_path / "media")
    monkeypatch.setattr(config, "MEDIA_ROOT", root)
    monkeypatch.setattr(config, "CLIPS_DIR", os.path.join(root, "clips"))
    monkeypatch.setattr(config, "FINAL_DIR", os.path.join(root, "final"))
    monkeypatch.setattr(config, "ASSETS_DIR", os.path.join(root, "assets"))
    monkeypatch.setattr(config, "TEMP_DIR", os.path.join(root, "tmp"))
    config.ensure_media_dirs()
    return root


def _make_clip(path, seconds=2, size="640x360", audio=True):
    from backend.video_editor import get_ffmpeg_exe
    cmd = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac"] if audio else []
    subprocess.run(cmd + [str(path)], check=True)
    return str(path)


# ---------------------------------------------------------------------- #
# Omni adapter
# ---------------------------------------------------------------------- #
class FakeInteractions:
    def __init__(self, final_status="completed", video_bytes=b"fake-mp4"):
        self.created = []
        self.final_status = final_status
        self.video_bytes = video_bytes
        self.polls = 0

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id="int-1", status="in_progress")

    def get(self, interaction_id):
        self.polls += 1
        video = SimpleNamespace(data=base64.b64encode(self.video_bytes).decode(), uri=None, mime_type="video/mp4")
        return SimpleNamespace(
            id=interaction_id, status=self.final_status, output_video=video,
            errors=[SimpleNamespace(message="safety block")] if self.final_status == "failed" else None,
        )


def _adapter(fake):
    from backend.omni_video_adapter import OmniFlashVideoAdapter
    return OmniFlashVideoAdapter(client=SimpleNamespace(interactions=fake), poll_interval=0)


def test_omni_default_model_is_omni_1_1_flash():
    assert config.OMNI_MODEL_NAME == "gemini-omni-1.1-flash"


def test_omni_adapter_requires_api_key(monkeypatch):
    from backend.omni_video_adapter import OmniFlashVideoAdapter, OmniVideoError
    monkeypatch.setattr(config, "OMNI_API_KEY", "")
    monkeypatch.delenv("OMNI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(OmniVideoError):
        OmniFlashVideoAdapter()


def test_omni_text_to_video_request_and_polling():
    fake = FakeInteractions(video_bytes=b"mp4-bytes")
    res = _adapter(fake).generate_scene_video("Students at Presidency University", "Slow dolly in", 10)
    assert res.success, res.error_message
    assert base64.b64decode(res.media_base64) == b"mp4-bytes"
    req = fake.created[0]
    assert req["model"] == "gemini-omni-1.1-flash"
    assert req["generation_config"] == {"video_config": {"task": "text_to_video"}}
    assert req["response_format"]["type"] == "video"
    assert req["response_format"]["duration"] == "10s"
    assert req["background"] is True
    assert "CAMERA DIRECTION: Slow dolly in" in req["input"][-1]["text"]
    assert fake.polls == 1


def test_omni_reference_images_switch_task():
    fake = FakeInteractions()
    img = {"base64": base64.b64encode(b"jpg").decode(), "mime_type": "image/jpeg"}
    _adapter(fake).generate_scene_video("Library scene", reference_images=[img])
    req = fake.created[0]
    assert req["generation_config"]["video_config"]["task"] == "reference_to_video"
    assert req["input"][0]["type"] == "image"


def test_omni_edit_sends_video_and_handles_failure():
    fake = FakeInteractions(final_status="failed")
    res = _adapter(fake).edit_video(b"src-video", "Make it golden hour")
    assert not res.success
    assert "safety block" in res.error_message
    req = fake.created[0]
    assert req["generation_config"]["video_config"]["task"] == "edit"
    assert req["input"][0]["type"] == "video"
    assert "duration" not in req["response_format"]


# ---------------------------------------------------------------------- #
# Editing engine
# ---------------------------------------------------------------------- #
def test_merge_timeline_with_transitions_voiceover_and_overlay(tmp_path):
    from backend.video_editor import RenderSettings, SegmentSpec, merge_timeline, probe
    a = _make_clip(tmp_path / "a.mp4", 3)
    b = _make_clip(tmp_path / "b.mp4", 2, size="360x640", audio=False)
    segs = [
        SegmentSpec(a, trim_start=0.5, trim_end=2.5, overlay_text=["প্রতি বছর ₹১২,০০০", "Apply online"]),
        SegmentSpec(b, speed=2.0, audio_mode="original"),
        SegmentSpec(a, audio_mode="mute"),
    ]
    out = str(tmp_path / "final.mp4")
    info = merge_timeline(segs, out, RenderSettings(width=640, height=360, transition="fade", transition_duration=0.4))
    meta = probe(out)
    assert meta["has_audio"] and meta["width"] == 640 and meta["height"] == 360
    # 2.0 + 1.0 + 3.0 minus two 0.4s crossfades
    assert abs(info["duration"] - 5.2) < 0.15
    assert abs(meta["duration"] - 5.2) < 0.25


def test_merge_hard_cut_with_background_music(tmp_path):
    from backend.video_editor import RenderSettings, SegmentSpec, merge_timeline, probe
    a = _make_clip(tmp_path / "a.mp4", 2)
    music = _make_clip(tmp_path / "music.mp4", 1)
    out = str(tmp_path / "final.mp4")
    info = merge_timeline([SegmentSpec(a), SegmentSpec(a)], out,
                          RenderSettings(width=640, height=360, transition="none", background_music_path=music))
    assert abs(info["duration"] - 4.0) < 0.15
    assert probe(out)["has_audio"]


def test_trim_rejects_invalid_range(tmp_path):
    from backend.video_editor import VideoEditError, trim_clip
    a = _make_clip(tmp_path / "a.mp4", 2)
    with pytest.raises(VideoEditError):
        trim_clip(a, str(tmp_path / "t.mp4"), 1.5, 1.0)


def test_frame_size():
    from backend.video_editor import frame_size
    assert frame_size("16:9", "720p") == (1280, 720)
    assert frame_size("9:16", "1080p") == (1080, 1920)


# ---------------------------------------------------------------------- #
# REST endpoints
# ---------------------------------------------------------------------- #
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def _wait_job(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError("job timed out")


def test_video_config_endpoint(client):
    data = client.get("/api/video/config").json()
    assert data["model"] == "gemini-omni-1.1-flash"
    assert "crossfade" not in data["transitions"] and "fade" in data["transitions"]


def test_generate_scene_without_key_returns_400(client, monkeypatch):
    monkeypatch.setattr(config, "OMNI_API_KEY", "")
    monkeypatch.delenv("OMNI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    scene = {
        "scene_number": 1, "scene_title": "Overview", "visual_prompt": "Wide shot", "camera_angle": "Drone",
        "voiceover_bengali": "নমস্কার", "no_lip_sync_enforcement": "B-roll only",
    }
    resp = client.post("/api/video/generate-scene", json={"scene": scene})
    assert resp.status_code == 400
    assert "API key is missing" in resp.json()["detail"]


def test_generate_scene_job_saves_clip(client, tmp_path, monkeypatch):
    from backend import omni_video_adapter
    clip_bytes = open(_make_clip(tmp_path / "gen.mp4", 2), "rb").read()
    fake = FakeInteractions(video_bytes=clip_bytes)
    monkeypatch.setattr(omni_video_adapter.OmniFlashVideoAdapter, "client", property(lambda self: SimpleNamespace(interactions=fake)))
    monkeypatch.setattr(config, "OMNI_POLL_INTERVAL_SECONDS", 0)
    scene = {
        "scene_number": 2, "scene_title": "Benefits", "visual_prompt": "Students in library", "camera_angle": "Pan",
        "voiceover_bengali": "নমস্কার", "no_lip_sync_enforcement": "B-roll only",
    }
    resp = client.post("/api/video/generate-scene", json={"scene": scene, "api_key": "test-key-123"})
    assert resp.status_code == 200
    job = _wait_job(client, resp.json()["id"])
    assert job["status"] == "completed", job["error"]
    clip = job["result"]["clips"][0]
    assert clip["scene_number"] == 2 and clip["has_video"]
    assert client.get(clip["url"]).status_code == 200
    assert any(c["id"] == clip["id"] for c in client.get("/api/clips").json())


def test_upload_trim_merge_flow(client, tmp_path):
    path = _make_clip(tmp_path / "up.mp4", 3)
    with open(path, "rb") as f:
        up = client.post("/api/clips/upload", files={"file": ("up.mp4", f, "video/mp4")})
    assert up.status_code == 200
    clip_id = up.json()["id"]

    trimmed = client.post(f"/api/clips/{clip_id}/trim", json={"start": 0.5, "end": 2.0})
    assert trimmed.status_code == 200
    assert abs(trimmed.json()["duration"] - 1.5) < 0.15

    resp = client.post("/api/video/merge", json={
        "title": "Test", "resolution": "360p", "transition": "dissolve", "transition_duration": 0.3,
        "segments": [{"clip_id": clip_id, "overlay_text": ["Hello"]}, {"clip_id": trimmed.json()["id"], "speed": 1.5}],
    })
    assert resp.status_code == 200
    job = _wait_job(client, resp.json()["id"], timeout=120)
    assert job["status"] == "completed", job["error"]
    final = job["result"]["final"]
    assert final["width"] == 640 and final["height"] == 360
    assert client.get(final["url"]).status_code == 200


def test_merge_validation(client):
    assert client.post("/api/video/merge", json={"segments": []}).status_code == 400
    resp = client.post("/api/video/merge", json={"segments": [{"clip_id": "missing"}]})
    assert resp.status_code == 404
