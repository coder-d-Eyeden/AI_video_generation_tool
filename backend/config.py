"""
Central configuration for video generation, editing and media storage.
"""
import os

# ---------------------------------------------------------------------------
# Gemini Omni 1.1 Flash (video generation / AI editing)
# Paste your Google AI Studio API key between the quotes below.
# It can also be supplied via the OMNI_API_KEY / GEMINI_API_KEY environment
# variables, or per-request from the UI (UI value takes priority).
# ---------------------------------------------------------------------------
OMNI_API_KEY = ""

OMNI_MODEL_NAME = os.environ.get("OMNI_MODEL_NAME", "gemini-omni-1.1-flash")

OMNI_DEFAULT_ASPECT_RATIO = "16:9"   # "16:9" | "9:16"
OMNI_DEFAULT_RESOLUTION = "720p"     # "360p" | "720p" | "1080p" | "4k"
OMNI_DEFAULT_DURATION_SECONDS = 10
OMNI_POLL_INTERVAL_SECONDS = 10
OMNI_MAX_WAIT_SECONDS = 15 * 60

# ---------------------------------------------------------------------------
# Media storage
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", os.path.join(PROJECT_ROOT, "output"))
CLIPS_DIR = os.path.join(MEDIA_ROOT, "clips")
FINAL_DIR = os.path.join(MEDIA_ROOT, "final")
ASSETS_DIR = os.path.join(MEDIA_ROOT, "assets")
TEMP_DIR = os.path.join(MEDIA_ROOT, "tmp")

# Optional: TTF/OTF font with Bengali glyphs for on-screen text overlays.
# Leave blank to auto-detect (Noto Sans Bengali, Nirmala UI, Vrinda, ...).
OVERLAY_FONT_PATH = os.environ.get("OVERLAY_FONT_PATH", "")

# Final render settings
RENDER_FPS = 24
RENDER_AUDIO_SAMPLE_RATE = 44100


def get_omni_api_key(override: str = None) -> str:
    """Resolves the Omni API key: request override > config constant > env vars."""
    for candidate in (override, OMNI_API_KEY, os.environ.get("OMNI_API_KEY"), os.environ.get("GEMINI_API_KEY")):
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


def ensure_media_dirs() -> None:
    for d in (CLIPS_DIR, FINAL_DIR, ASSETS_DIR, TEMP_DIR):
        os.makedirs(d, exist_ok=True)
