"""
Media Generation Adapter Interface.
Designed for modular extension: enables team members to independently attach:
1. Text-to-Image models (e.g. Imagen 3, Stable Diffusion, Midjourney)
2. Image-to-Video models (e.g. Runway Gen-3, Luma Dream Machine, SVD, Kling AI)
3. Direct Text-to-Video models (e.g. Sora, Pika, Kling)
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class MediaGenerationResult(BaseModel):
    success: bool
    media_type: str  # "image" | "video"
    media_url_or_path: Optional[str] = None
    media_base64: Optional[str] = None
    metadata: Dict[str, Any] = {}
    error_message: Optional[str] = None


class BaseMediaAdapter(ABC):
    """
    Abstract contract for video and image generation engines.
    Team members can implement this class for specific cloud APIs or local models.
    """

    @abstractmethod
    def generate_image_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        grounding_reference_paths: Optional[List[str]] = None
    ) -> MediaGenerationResult:
        """Generates a keyframe image from the scene visual prompt."""
        pass

    @abstractmethod
    def generate_video_from_image(
        self,
        image_bytes: bytes,
        camera_motion_prompt: str,
        duration_seconds: float = 10.0
    ) -> MediaGenerationResult:
        """Animates a keyframe image into video with camera motion."""
        pass

    @abstractmethod
    def generate_video_from_text(
        self,
        prompt: str,
        camera_angle: str,
        duration_seconds: float = 10.0
    ) -> MediaGenerationResult:
        """Direct text-to-video generation."""
        pass


class ImageToVideoPipeline:
    """
    Orchestrates the two-stage pipeline:
    Stage 1: Scene Prompt -> Keyframe Image (with WB Grounding)
    Stage 2: Keyframe Image -> 10-second Scene Video
    """
    def __init__(self, media_adapter: BaseMediaAdapter):
        self.adapter = media_adapter

    def produce_scene_clip(
        self,
        visual_prompt: str,
        camera_angle: str,
        duration_seconds: float = 10.0,
        grounding_references: Optional[List[str]] = None
    ) -> MediaGenerationResult:
        """
        Executes two-stage generation: Image first, then Video from that image.
        """
        # Step 1: Keyframe generation
        img_res = self.adapter.generate_image_from_prompt(
            prompt=visual_prompt,
            grounding_reference_paths=grounding_references
        )
        if not img_res.success:
            return img_res

        # Step 2: Animate keyframe
        if img_res.media_base64:
            import base64
            raw_img = base64.b64decode(img_res.media_base64)
            return self.adapter.generate_video_from_image(
                image_bytes=raw_img,
                camera_motion_prompt=camera_angle,
                duration_seconds=duration_seconds
            )

        return img_res
