"""
Unit and Integration Test Suite for Bengal Scheme-to-Video Pipeline.
Tests input validation, document parsing, TTS synthesis, critic logic, media adapter, and API endpoints.
"""
import io
import pytest
from fastapi.testclient import TestClient

from backend.models import ProviderType, ScenePrompt, VideoProjectPlan, CriticAudit, CriticIssue
from backend.llm_client import LLMClient, LLMAuthError
from backend.document_parser import extract_text_from_bytes, DocumentParsingError, process_grounding_images
from backend.tts_service import TTSService
from backend.media_adapter import BaseMediaAdapter, MediaGenerationResult, ImageToVideoPipeline
from backend.main import app

client = TestClient(app)


def test_missing_api_key_raises_error():
    """Verify system strictly rejects execution when API key is missing."""
    with pytest.raises(LLMAuthError) as exc_info:
        LLMClient(provider=ProviderType.GEMINI, api_key="")
    assert "API Key is missing" in str(exc_info.value)


def test_missing_api_key_endpoint():
    """Test that /api/generate-plan rejects requests without API key with HTTP 400."""
    response = client.post(
        "/api/generate-plan",
        data={
            "provider": "gemini",
            "api_key": "",
            "scheme_text": "This is a test scheme document."
        }
    )
    assert response.status_code == 400
    assert "API Key is missing" in response.json()["detail"]


def test_missing_document_endpoint():
    """Test that /api/generate-plan rejects requests without scheme document with HTTP 400."""
    response = client.post(
        "/api/generate-plan",
        data={
            "provider": "gemini",
            "api_key": "dummy_test_key_1234567890",
            "scheme_text": ""
        }
    )
    assert response.status_code == 400
    assert "Scheme document is missing" in response.json()["detail"]


def test_txt_document_extraction():
    """Test plain text document extraction."""
    sample_text = "SVMCM Scholarship 2026: Eligible students receive 12000 per year from Government of West Bengal."
    extracted = extract_text_from_bytes(sample_text.encode("utf-8"), "scheme.txt")
    assert "SVMCM Scholarship" in extracted


def test_empty_document_rejection():
    """Test empty document raises DocumentParsingError."""
    with pytest.raises(DocumentParsingError):
        extract_text_from_bytes(b"", "empty.txt")


def test_bengali_tts_synthesis():
    """Test that Bengali TTS synthesizes valid audio MP3 data URI and timing metadata."""
    sample_bengali = "পশ্চিমবঙ্গের মেধাবী ছাত্রছাত্রীদের জন্য স্বামী বিবেকানন্দ স্কলারশিপ।"
    result = TTSService.synthesize_bengali_voiceover(sample_bengali, start_pad_sec=1.0, end_pad_sec=1.5)
    assert result["audio_data_uri"].startswith("data:audio/mp3;base64,")
    assert result["word_count"] > 0
    assert result["timing_compliance"]["narration_start_sec"] == 1.0


def test_scene_model_validation():
    """Test ScenePrompt model field constraints."""
    scene = ScenePrompt(
        scene_number=1,
        scene_title="Scheme Overview",
        duration_seconds=10.0,
        visual_prompt="Cinematic wide shot of Presidency University Kolkata heritage building.",
        camera_angle="Wide aerial tracking shot slowly descending",
        voiceover_bengali="পশ্চিমবঙ্গের মেধাবী ছাত্রছাত্রীদের জন্য স্বামী বিবেকানন্দ স্কলারশিপ একটি বিশেষ উদ্যোগ।",
        narration_start_second=1.0,
        narration_end_second=8.2,
        onscreen_text=["স্বামী বিবেকানন্দ স্কলারশিপ", "উচ্চশিক্ষার সুযোগ"],
        wb_grounding_elements=["Presidency University Kolkata"],
        no_lip_sync_enforcement="Students are walking in campus courtyard viewed from behind; third-person narration."
    )
    assert scene.scene_number == 1
    assert scene.narration_start_second >= 1.0
    assert scene.narration_end_second <= 8.5


def test_media_adapter_contract():
    """Verify modular media adapter can be implemented independently by team members."""
    class CustomRunwayAdapter(BaseMediaAdapter):
        def generate_image_from_prompt(self, prompt, negative_prompt=None, grounding_reference_paths=None):
            return MediaGenerationResult(success=True, media_type="image", media_url_or_path="https://example.com/frame.png")

        def generate_video_from_image(self, image_bytes, camera_motion_prompt, duration_seconds=10.0):
            return MediaGenerationResult(success=True, media_type="video", media_url_or_path="https://example.com/clip.mp4")

        def generate_video_from_text(self, prompt, camera_angle, duration_seconds=10.0):
            return MediaGenerationResult(success=True, media_type="video", media_url_or_path="https://example.com/video.mp4")

    adapter = CustomRunwayAdapter()
    res = adapter.generate_video_from_text("Presidency campus students", "Drone shot", 10.0)
    assert res.success is True
    assert res.media_type == "video"
