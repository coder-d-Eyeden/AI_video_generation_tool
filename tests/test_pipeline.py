"""
Unit and Integration Test Suite for Bengal Scheme-to-Video Pipeline.
Tests input validation, document parsing, TTS synthesis, critic logic, media adapter, and API endpoints.
"""
import io
import json
import pytest
from fastapi.testclient import TestClient

from backend.models import ProviderType, ScenePrompt, VideoProjectPlan, CriticAudit, CriticIssue
from backend.llm_client import (
    LLMClient,
    LLMAuthError,
    LLMExecutionError,
    DEFAULT_GROQ_MODEL,
    discover_groq_models,
    filter_chat_models,
    _GROQ_MODEL_CACHE,
)
from backend.document_parser import extract_text_from_bytes, DocumentParsingError, process_grounding_images
from backend.tts_service import TTSService
from backend.media_adapter import BaseMediaAdapter, MediaGenerationResult, ImageToVideoPipeline
from backend.critic_agent import CriticAgent
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


# ============================================================
# AI Critic Agent (Groq) wiring
# ============================================================

@pytest.fixture(autouse=True)
def _clear_groq_model_cache():
    """Model discovery is cached per key, so isolate it between tests."""
    _GROQ_MODEL_CACHE.clear()
    yield
    _GROQ_MODEL_CACHE.clear()


def _build_scene(n: int) -> ScenePrompt:
    return ScenePrompt(
        scene_number=n,
        scene_title=f"Scene {n}",
        duration_seconds=10.0,
        visual_prompt="Wide shot of Presidency University library stacks, students seen from behind.",
        camera_angle="Slow dolly forward, 35mm",
        voiceover_bengali="পশ্চিমবঙ্গের মেধাবী ছাত্রছাত্রীদের জন্য এই বিশেষ সুযোগটি।",
        narration_start_second=1.0,
        narration_end_second=8.0,
        onscreen_text=["৬০% নম্বর"],
        wb_grounding_elements=["National Library Alipore"],
        no_lip_sync_enforcement="All characters are filmed from behind; third-person narrator only."
    )


def _build_plan(scene_count: int = 5) -> VideoProjectPlan:
    return VideoProjectPlan(
        scheme_title="SVMCM Scholarship",
        total_duration_seconds=50.0,
        global_guidelines="No lip sync, WB grounded.",
        scenes=[_build_scene(n) for n in range(1, scene_count + 1)]
    )


def _groq_response_payload(plan: VideoProjectPlan) -> str:
    return json.dumps({
        "compliance_score": 96,
        "passed_checks": True,
        "summary_notes": "All scenes compliant after minor tightening.",
        "issues_found": [{
            "scene_number": 1,
            "category": "timing_margin",
            "severity": "warning",
            "description": "Narration started at 0.5s.",
            "resolution": "Shifted narration start to 1.0s."
        }],
        "refined_scenes": [s.model_dump() for s in plan.scenes]
    })


class _FakeModels:
    def __init__(self, model_ids):
        self._model_ids = model_ids

    def list(self):
        return [type("M", (), {"id": mid})() for mid in self._model_ids]


class _FakeCompletions:
    def __init__(self, content: str, available_models=None):
        self.content = content
        self.available_models = available_models
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.available_models is not None and kwargs.get("model") not in self.available_models:
            raise RuntimeError(
                f"Error code: 404 - {{'error': {{'message': "
                f"\"The model `{kwargs.get('model')}` has been decommissioned\"}}}}"
            )
        message = type("Msg", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


class _FakeOpenAI:
    """Stands in for the OpenAI SDK pointed at Groq's compatible endpoint."""

    # Live chat models on Groq as of Sept 2026 -- every Llama ID is retired.
    LIVE_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "whisper-large-v3",
        "meta-llama/llama-prompt-guard-2-86m",
    ]

    def __init__(self, content: str, available_models=None, model_ids=None):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content, available_models)})()
        self.models = _FakeModels(self.LIVE_MODELS if model_ids is None else model_ids)


def test_critic_defaults_to_groq_chat_model(monkeypatch):
    """The Critic Agent must default to Groq's current flagship, not the scene provider."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    critic = CriticAgent(api_key="gsk_dummy_key_1234567890")
    assert critic.model_name == "openai/gpt-oss-120b"
    assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-120b"
    assert critic.llm_client.provider == ProviderType.GROQ


def test_no_retired_llama_ids_remain_in_config():
    """Groq shut down every Llama ID; none may linger as a default or fallback."""
    from backend.llm_client import GROQ_CANDIDATE_MODELS
    configured = [DEFAULT_GROQ_MODEL] + GROQ_CANDIDATE_MODELS
    assert not any("llama" in m.lower() for m in configured), configured


def test_critic_key_falls_back_to_env_var(monkeypatch):
    """With no key passed in, the Critic Agent reads GROQ_API_KEY from the environment."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_key_1234567890")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    critic = CriticAgent()
    assert critic.api_key == "gsk_env_key_1234567890"
    assert critic.model_name == "openai/gpt-oss-20b"


def test_critic_missing_groq_key_raises(monkeypatch):
    """A missing Groq key fails loudly with a message naming the critic requirement."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    critic = CriticAgent()
    with pytest.raises(LLMAuthError) as exc_info:
        _ = critic.llm_client
    assert "Groq API Key is missing" in str(exc_info.value)


def test_model_discovery_filters_non_chat_models(monkeypatch):
    """Speech-to-text and guard models must never be selected for a chat audit."""
    fake = _FakeOpenAI("")
    monkeypatch.setattr("backend.llm_client.OpenAI", lambda **kw: fake)

    discovered = discover_groq_models("gsk_dummy_key_1234567890", use_cache=False)

    assert "openai/gpt-oss-120b" in discovered
    assert "whisper-large-v3" not in discovered
    assert not any("guard" in m for m in discovered)


def test_model_discovery_rejects_bad_key(monkeypatch):
    """A bad key during discovery surfaces as an auth error, not a silent empty list."""
    class _Unauthorized:
        def __init__(self, **kw):
            self.models = type("M", (), {"list": lambda self: (_ for _ in ()).throw(
                RuntimeError("Error code: 401 - invalid_api_key")
            )})()

    monkeypatch.setattr("backend.llm_client.OpenAI", _Unauthorized)
    with pytest.raises(LLMAuthError):
        discover_groq_models("gsk_bad_key_1234567890", use_cache=False)


def test_critic_calls_groq_endpoint(monkeypatch):
    """End-to-end: the audit request is sent to Groq and returns a refined plan."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    plan = _build_plan()
    fake = _FakeOpenAI(_groq_response_payload(plan))

    captured = {}

    def _factory(api_key=None, base_url=None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return fake

    monkeypatch.setattr("backend.llm_client.OpenAI", _factory)

    refined = CriticAgent(api_key="gsk_dummy_key_1234567890").audit_and_refine(
        current_plan=plan,
        scheme_document_text="SVMCM Scholarship 2026 for West Bengal students."
    )

    call = fake.chat.completions.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["response_format"] == {"type": "json_object"}
    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert refined.is_refined_by_critic is True
    assert refined.critic_audit.compliance_score == 96
    assert len(refined.critic_audit.issues_found) == 1


def test_critic_recovers_from_retired_model(monkeypatch):
    """A 404 from the requested model must fall back to a live model, not abort the audit."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    plan = _build_plan()
    # Every model except one live chat model is treated as decommissioned.
    fake = _FakeOpenAI(
        _groq_response_payload(plan),
        available_models=["qwen/qwen3.8-27b"],
    )
    monkeypatch.setattr("backend.llm_client.OpenAI", lambda **kw: fake)

    refined = CriticAgent(
        api_key="gsk_dummy_key_1234567890",
        model_name="llama-3.3-70b-versatile",
    ).audit_and_refine(
        current_plan=plan,
        scheme_document_text="SVMCM Scholarship 2026 for West Bengal students.",
    )

    attempted = [c["model"] for c in fake.chat.completions.calls]
    assert attempted[0] == "llama-3.3-70b-versatile"
    assert attempted[-1] == "qwen/qwen3.8-27b"
    assert refined.is_refined_by_critic is True


def test_groq_client_rejects_grounding_images():
    """Groq chat models are text-only, so grounding photos must be rejected with a clear error."""
    groq_client = LLMClient(provider=ProviderType.GROQ, api_key="gsk_dummy_key_1234567890")
    with pytest.raises(LLMExecutionError) as exc_info:
        groq_client.generate_json(
            system_prompt="s",
            user_prompt="u",
            grounding_images=[{"filename": "college.jpg", "mime_type": "image/jpeg", "base64": "aGk="}]
        )
    assert "does not accept image inputs" in str(exc_info.value)


def test_run_critic_requires_no_scene_provider_key():
    """/api/run-critic audits via Groq, so the scene generator key is no longer required."""
    response = client.post("/api/run-critic", json={
        "critic_api_key": "",
        "current_plan": _build_plan().model_dump(),
        "scheme_document_text": "",
    })
    # 400 (missing scheme text) rather than 401/500 proves no scene key was demanded.
    assert response.status_code == 400
    assert "scheme document text is required" in response.json()["detail"]
