"""
Data models for Video Scene Prompts, Critic Agent, and Request/Response schemas.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"


class ScenePrompt(BaseModel):
    scene_number: int = Field(..., description="Scene index 1 to 6")
    scene_title: str = Field(..., description="E.g., Scheme Overview, Benefits, Eligibility, Required Documents, How to Apply, Helpline & Website")
    duration_seconds: float = Field(default=10.0, description="Duration in seconds (typically 9-10s)")
    visual_prompt: str = Field(..., description="Production video generation prompt with cinematic camera movement, West Bengal grounding, NO lip sync")
    camera_angle: str = Field(..., description="Precise camera angle, focal length and movement description")
    voiceover_bengali: str = Field(..., description="Voiceover script in natural, everyday Bengali spoken by ordinary people")
    narration_start_second: float = Field(default=1.0, description="Narration start offset (strictly >= 1.0s to avoid audio overlap)")
    narration_end_second: float = Field(default=8.5, description="Narration end offset (must end 1-2s before scene ends)")
    onscreen_text: List[str] = Field(default_factory=list, description="Micro-prompting bullet points and highlighted on-screen text")
    wb_grounding_elements: List[str] = Field(default_factory=list, description="Specific West Bengal college/library/landmark or local context used")
    no_lip_sync_enforcement: str = Field(..., description="Visual composition rule ensuring no character is shown speaking or lip-syncing")


class CriticIssue(BaseModel):
    scene_number: int
    category: str = Field(..., description="lip_sync | wb_grounding | timing_margin | bengali_naturalness | document_accuracy | onscreen_micro_prompt")
    severity: str = Field(..., description="info | warning | error")
    description: str
    resolution: str


class CriticAudit(BaseModel):
    compliance_score: int = Field(..., description="Score out of 100")
    passed_checks: bool
    issues_found: List[CriticIssue] = Field(default_factory=list)
    summary_notes: str


class VideoProjectPlan(BaseModel):
    is_valid_scheme: bool = True
    error_message: Optional[str] = None
    scheme_title: str
    total_duration_seconds: float = Field(default=50.0, description="Total video duration between 45-55 seconds")
    global_guidelines: str
    detected_grounding_references: List[str] = Field(default_factory=list)
    scenes: List[ScenePrompt] = Field(default_factory=list)
    critic_audit: Optional[CriticAudit] = None
    is_refined_by_critic: bool = False


class GenerateRequest(BaseModel):
    provider: ProviderType
    api_key: str = Field(..., min_length=10, description="User's Gemini or OpenAI API key")
    scheme_text: Optional[str] = None
    additional_notes: Optional[str] = None


class RefineRequest(BaseModel):
    provider: ProviderType
    api_key: str = Field(..., min_length=10)
    current_plan: VideoProjectPlan
    scheme_document_text: str
    grounding_notes: Optional[str] = None


class SynthesizeAudioRequest(BaseModel):
    bengali_text: str
    start_pad_sec: float = 1.0
    end_pad_sec: float = 1.5


# --------------------------------------------------------------------------
# Video generation, editing & merging
# --------------------------------------------------------------------------
class ReferenceImage(BaseModel):
    filename: Optional[str] = None
    mime_type: str = "image/jpeg"
    base64: str


class VideoGenerationOptions(BaseModel):
    api_key: Optional[str] = Field(default=None, description="Optional override; falls back to OMNI_API_KEY in backend/config.py")
    model_name: Optional[str] = None
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    reference_images: List[ReferenceImage] = Field(default_factory=list)


class SceneVideoRequest(VideoGenerationOptions):
    scene: ScenePrompt


class PlanVideoRequest(VideoGenerationOptions):
    plan: VideoProjectPlan
    scene_numbers: Optional[List[int]] = Field(default=None, description="Subset of scenes to generate; all when omitted")


class ClipAIEditRequest(BaseModel):
    instruction: str = Field(..., min_length=3)
    mode: str = Field(default="edit", description="edit | extend")
    extend_seconds: float = 5.0
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    aspect_ratio: str = "16:9"
    resolution: str = "720p"


class ClipTrimRequest(BaseModel):
    start: float = 0.0
    end: Optional[float] = None


class TimelineSegment(BaseModel):
    clip_id: str
    trim_start: float = 0.0
    trim_end: Optional[float] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    overlay_text: List[str] = Field(default_factory=list)
    voiceover_text: Optional[str] = None
    narration_start: float = 1.0
    audio_mode: str = Field(default="original", description="original | voiceover | mix | mute")
    original_volume: float = 1.0
    voiceover_volume: float = 1.0


class MergeRequest(BaseModel):
    segments: List[TimelineSegment]
    title: Optional[str] = None
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    transition: str = "fade"
    transition_duration: float = Field(default=0.5, ge=0.0, le=2.0)
    background_music_id: Optional[str] = None
    background_music_volume: float = Field(default=0.15, ge=0.0, le=1.0)
    fade_in_out: bool = True
