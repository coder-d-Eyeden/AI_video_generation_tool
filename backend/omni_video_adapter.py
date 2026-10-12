"""
Gemini Omni 1.1 Flash video adapter.

Implements BaseMediaAdapter on top of the google-genai Interactions API:
  - text_to_video   : storyboard scene prompt -> clip
  - image_to_video  : grounding photo / keyframe + prompt -> clip
  - edit            : existing clip + natural-language instruction -> edited clip
  - extend          : existing clip + prompt -> longer clip
"""
import base64
import time
from typing import Any, Callable, Dict, List, Optional

from backend import config
from backend.media_adapter import BaseMediaAdapter, MediaGenerationResult


class OmniVideoError(Exception):
    """Raised when Omni video generation fails."""


_PENDING_STATUSES = {"in_progress", "queued"}

ProgressCallback = Optional[Callable[[str], None]]


class OmniFlashVideoAdapter(BaseMediaAdapter):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        client: Any = None,
        poll_interval: Optional[float] = None,
        max_wait_seconds: Optional[float] = None,
    ):
        self.api_key = config.get_omni_api_key(api_key)
        if not self.api_key and client is None:
            raise OmniVideoError(
                "Omni video API key is missing. Paste it into OMNI_API_KEY in backend/config.py, "
                "set the OMNI_API_KEY environment variable, or enter it in the Video Studio panel."
            )
        self.model_name = model_name or config.OMNI_MODEL_NAME
        self.aspect_ratio = aspect_ratio or config.OMNI_DEFAULT_ASPECT_RATIO
        self.resolution = resolution or config.OMNI_DEFAULT_RESOLUTION
        self.poll_interval = config.OMNI_POLL_INTERVAL_SECONDS if poll_interval is None else poll_interval
        self.max_wait_seconds = max_wait_seconds or config.OMNI_MAX_WAIT_SECONDS
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------ #
    # BaseMediaAdapter contract
    # ------------------------------------------------------------------ #
    def generate_image_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        grounding_reference_paths: Optional[List[str]] = None,
    ) -> MediaGenerationResult:
        return MediaGenerationResult(
            success=False,
            media_type="image",
            error_message=f"{self.model_name} is a video model; use generate_video_from_text or generate_video_from_image.",
        )

    def generate_video_from_text(
        self,
        prompt: str,
        camera_angle: str = "",
        duration_seconds: float = 10.0,
        on_progress: ProgressCallback = None,
    ) -> MediaGenerationResult:
        return self._run(
            task="text_to_video",
            prompt=self._compose_prompt(prompt, camera_angle),
            media=[],
            duration_seconds=duration_seconds,
            on_progress=on_progress,
        )

    def generate_video_from_image(
        self,
        image_bytes: bytes,
        camera_motion_prompt: str,
        duration_seconds: float = 10.0,
        mime_type: str = "image/jpeg",
        on_progress: ProgressCallback = None,
    ) -> MediaGenerationResult:
        return self._run(
            task="image_to_video",
            prompt=camera_motion_prompt,
            media=[self._media_part("image", image_bytes, mime_type)],
            duration_seconds=duration_seconds,
            on_progress=on_progress,
        )

    # ------------------------------------------------------------------ #
    # Omni-specific capabilities
    # ------------------------------------------------------------------ #
    def generate_scene_video(
        self,
        visual_prompt: str,
        camera_angle: str = "",
        duration_seconds: float = 10.0,
        reference_images: Optional[List[Dict[str, Any]]] = None,
        on_progress: ProgressCallback = None,
    ) -> MediaGenerationResult:
        """Text-to-video, or reference-to-video when grounding photos are supplied."""
        media = [
            self._media_part("image", base64.b64decode(img["base64"]), img.get("mime_type", "image/jpeg"))
            for img in (reference_images or [])
        ]
        return self._run(
            task="reference_to_video" if media else "text_to_video",
            prompt=self._compose_prompt(visual_prompt, camera_angle),
            media=media,
            duration_seconds=duration_seconds,
            on_progress=on_progress,
        )

    def edit_video(
        self,
        video_bytes: bytes,
        instruction: str,
        mime_type: str = "video/mp4",
        on_progress: ProgressCallback = None,
    ) -> MediaGenerationResult:
        return self._run(
            task="edit",
            prompt=instruction,
            media=[self._media_part("video", video_bytes, mime_type)],
            duration_seconds=None,
            on_progress=on_progress,
        )

    def extend_video(
        self,
        video_bytes: bytes,
        prompt: str,
        duration_seconds: float = 5.0,
        mime_type: str = "video/mp4",
        on_progress: ProgressCallback = None,
    ) -> MediaGenerationResult:
        return self._run(
            task="extend",
            prompt=prompt,
            media=[self._media_part("video", video_bytes, mime_type)],
            duration_seconds=duration_seconds,
            on_progress=on_progress,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose_prompt(prompt: str, camera_angle: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            raise OmniVideoError("Video prompt is empty.")
        camera_angle = (camera_angle or "").strip()
        if camera_angle and camera_angle.lower() not in prompt.lower():
            prompt += f"\n\nCAMERA DIRECTION: {camera_angle}"
        return prompt

    @staticmethod
    def _media_part(kind: str, raw: bytes, mime_type: str) -> Dict[str, Any]:
        return {"type": kind, "data": base64.b64encode(raw).decode("utf-8"), "mime_type": mime_type}

    def _response_format(self, duration_seconds: Optional[float]) -> Dict[str, Any]:
        fmt = {
            "type": "video",
            "aspect_ratio": self.aspect_ratio,
            "resolution": self.resolution,
            "delivery": "inline",
        }
        if duration_seconds:
            fmt["duration"] = f"{int(round(duration_seconds))}s"
        return fmt

    def _run(
        self,
        task: str,
        prompt: str,
        media: List[Dict[str, Any]],
        duration_seconds: Optional[float],
        on_progress: ProgressCallback,
    ) -> MediaGenerationResult:
        notify = on_progress or (lambda _msg: None)
        contents = list(media) + [{"type": "text", "text": prompt}]
        try:
            notify(f"Submitting {task} request to {self.model_name}...")
            interaction = self.client.interactions.create(
                model=self.model_name,
                input=contents,
                response_format=self._response_format(duration_seconds),
                generation_config={"video_config": {"task": task}},
                background=True,
            )
            interaction = self._wait(interaction, notify)
            video_bytes, mime = self._extract_video(interaction)
        except OmniVideoError as e:
            return MediaGenerationResult(success=False, media_type="video", error_message=str(e))
        except Exception as e:
            msg = str(e)
            if "API_KEY_INVALID" in msg or "401" in msg or "PERMISSION_DENIED" in msg:
                msg = f"Omni API key rejected: {msg}"
            return MediaGenerationResult(success=False, media_type="video", error_message=msg)

        return MediaGenerationResult(
            success=True,
            media_type="video",
            media_base64=base64.b64encode(video_bytes).decode("utf-8"),
            metadata={
                "model": self.model_name,
                "task": task,
                "mime_type": mime,
                "interaction_id": getattr(interaction, "id", None),
                "aspect_ratio": self.aspect_ratio,
                "resolution": self.resolution,
            },
        )

    def _wait(self, interaction: Any, notify: Callable[[str], None]) -> Any:
        started = time.time()
        while getattr(interaction, "status", None) in _PENDING_STATUSES:
            elapsed = time.time() - started
            if elapsed > self.max_wait_seconds:
                raise OmniVideoError(f"Timed out after {int(elapsed)}s waiting for {self.model_name}.")
            notify(f"Rendering with {self.model_name}... {int(elapsed)}s elapsed")
            time.sleep(self.poll_interval)
            interaction = self.client.interactions.get(interaction.id)

        status = getattr(interaction, "status", None)
        if status != "completed":
            errors = getattr(interaction, "errors", None) or []
            detail = "; ".join(str(getattr(e, "message", e)) for e in errors) or "no error details returned"
            raise OmniVideoError(f"Omni video generation ended with status '{status}': {detail}")
        return interaction

    def _extract_video(self, interaction: Any):
        video = getattr(interaction, "output_video", None)
        if video is None:
            raise OmniVideoError("Omni response contained no video output.")
        mime = getattr(video, "mime_type", None) or "video/mp4"
        data = getattr(video, "data", None)
        if data:
            return base64.b64decode(data), mime
        uri = getattr(video, "uri", None)
        if uri:
            return self.client.files.download(file=uri), mime
        raise OmniVideoError("Omni video output had neither inline data nor a download URI.")
