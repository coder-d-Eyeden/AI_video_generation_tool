"""
FFmpeg-based video editing & merging engine.

Each timeline segment is first normalized (trim, speed, scale/pad, fps, text
overlay, audio track: original / Bengali voiceover / mix / mute) into an
identically-encoded intermediate, then all segments are stitched with either
hard cuts or xfade transitions. Optional background music and global fades
are applied in the final pass.
"""
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from backend import config

TRANSITIONS = ["none", "fade", "fadeblack", "fadewhite", "dissolve", "wipeleft", "wiperight", "slideleft", "slideright", "circleopen", "circleclose", "radial", "smoothleft"]
AUDIO_MODES = ["original", "voiceover", "mix", "mute"]

_RESOLUTION_HEIGHTS = {"360p": 360, "480p": 480, "720p": 720, "1080p": 1080, "4k": 2160}

_BUNDLED_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

_FONT_CANDIDATES = [
    os.path.join(_BUNDLED_FONT_DIR, "NotoSansBengali-Bold.ttf"),
    os.path.join(_BUNDLED_FONT_DIR, "NotoSansBengali-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "C:/Windows/Fonts/Nirmala.ttf",
    "C:/Windows/Fonts/NirmalaB.ttf",
    "C:/Windows/Fonts/vrinda.ttf",
    "/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc",
    "/System/Library/Fonts/KohinoorBangla.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


class VideoEditError(Exception):
    """Raised when an ffmpeg editing operation fails."""


@dataclass
class SegmentSpec:
    path: str
    trim_start: float = 0.0
    trim_end: Optional[float] = None
    speed: float = 1.0
    overlay_text: List[str] = field(default_factory=list)
    voiceover_path: Optional[str] = None
    narration_start: float = 1.0
    audio_mode: str = "original"
    original_volume: float = 1.0
    voiceover_volume: float = 1.0


@dataclass
class RenderSettings:
    width: int = 1280
    height: int = 720
    fps: int = config.RENDER_FPS
    transition: str = "fade"
    transition_duration: float = 0.5
    background_music_path: Optional[str] = None
    background_music_volume: float = 0.15
    fade_in_out: bool = True


# ---------------------------------------------------------------------- #
# ffmpeg helpers
# ---------------------------------------------------------------------- #
def get_ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise VideoEditError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.") from e


def run_ffmpeg(args: List[str]) -> None:
    cmd = [get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoEditError(f"ffmpeg failed: {proc.stderr.strip()[-1500:]}")


def probe(path: str) -> dict:
    """Returns duration, audio presence and frame size by parsing `ffmpeg -i` output."""
    if not os.path.exists(path):
        raise VideoEditError(f"File not found: {path}")
    proc = subprocess.run([get_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True)
    info = proc.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if not m:
        raise VideoEditError(f"Could not read media duration of {os.path.basename(path)}.")
    duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    size = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})", info)
    return {
        "duration": round(duration, 3),
        "has_audio": bool(re.search(r"Stream #.*Audio:", info)),
        "has_video": bool(re.search(r"Stream #.*Video:", info)),
        "width": int(size.group(1)) if size else None,
        "height": int(size.group(2)) if size else None,
    }


def frame_size(aspect_ratio: str = "16:9", resolution: str = "720p") -> tuple:
    h = _RESOLUTION_HEIGHTS.get(resolution, 720)
    w = int(round(h * 16 / 9 / 2) * 2)
    return (h, w) if aspect_ratio == "9:16" else (w, h)


def _tmp_path(ext: str) -> str:
    config.ensure_media_dirs()
    return os.path.join(config.TEMP_DIR, f"{uuid.uuid4().hex}{ext}")


def _fmt(x: float) -> str:
    return f"{x:.3f}"


# ---------------------------------------------------------------------- #
# Text overlay
# ---------------------------------------------------------------------- #
def find_overlay_font() -> Optional[str]:
    if config.OVERLAY_FONT_PATH and os.path.exists(config.OVERLAY_FONT_PATH):
        return config.OVERLAY_FONT_PATH
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


_LATIN_FONT_CANDIDATES = [
    os.path.join(_BUNDLED_FONT_DIR, "NotoSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _is_bengali_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x0980 <= cp <= 0x09FF or cp in (0x0964, 0x0965, 0x200C, 0x200D, 0x20B9)


def _script_runs(text: str) -> List[tuple]:
    """Splits text into (is_bengali, chunk) runs; whitespace sticks to the current run."""
    runs: List[list] = []
    for ch in text:
        beng = runs[-1][0] if (ch.isspace() and runs) else _is_bengali_char(ch)
        if runs and runs[-1][0] == beng:
            runs[-1][1] += ch
        else:
            runs.append([beng, ch])
    return [(b, t) for b, t in runs]


def render_text_overlay(lines: List[str], width: int, height: int, out_path: str) -> str:
    """Renders lower-third caption lines onto a transparent PNG.

    Bengali runs use a Bengali font (shaped by raqm when available); Latin digits/punctuation/URLs
    fall back to a Latin font, since Bengali-only fonts lack those glyphs.
    """
    from PIL import Image, ImageDraw, ImageFont, features

    lines = [ln.strip() for ln in lines if ln and ln.strip()][:4]
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not lines:
        img.save(out_path)
        return out_path

    font_size = max(18, height // 22)
    kwargs = {"layout_engine": ImageFont.Layout.RAQM} if features.check("raqm") else {}

    def load(path):
        return ImageFont.truetype(path, font_size, **kwargs) if path else ImageFont.load_default()

    main_path = find_overlay_font()
    latin_path = next((p for p in _LATIN_FONT_CANDIDATES if os.path.exists(p)), None)
    beng_font = load(main_path)
    split_scripts = bool(main_path and latin_path and "bengali" in os.path.basename(main_path).lower())
    latin_font = load(latin_path) if split_scripts else beng_font

    draw = ImageDraw.Draw(img)
    line_runs = []
    for ln in lines:
        runs = [(beng_font if b else latin_font, t) for b, t in _script_runs(ln)] if split_scripts else [(beng_font, ln)]
        line_runs.append([(f, t, draw.textlength(t, font=f)) for f, t in runs])
    widths = [sum(w for _, _, w in runs) for runs in line_runs]

    pad = font_size // 2
    line_h = int(font_size * 1.45)
    box_w = int(min(width * 0.92, max(widths) + pad * 2))
    box_h = line_h * len(lines) + pad
    x0 = (width - box_w) // 2
    y0 = height - box_h - int(height * 0.07)
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=pad, fill=(10, 14, 23, 170))
    for i, runs in enumerate(line_runs):
        tx = (width - widths[i]) / 2
        baseline = y0 + pad // 2 + i * line_h + int(font_size * 1.05)
        for f, t, w in runs:
            draw.text((tx, baseline), t, font=f, fill=(255, 255, 255, 255), anchor="ls")
            tx += w
    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------- #
# Single-clip operations
# ---------------------------------------------------------------------- #
def trim_clip(src: str, out: str, start: float, end: Optional[float] = None) -> str:
    info = probe(src)
    end = info["duration"] if end is None else min(end, info["duration"])
    if start < 0 or end <= start:
        raise VideoEditError(f"Invalid trim range {start}s - {end}s (clip is {info['duration']}s).")
    run_ffmpeg([
        "-ss", _fmt(start), "-i", src, "-t", _fmt(end - start),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out,
    ])
    return out


def extract_thumbnail(src: str, out: str, at: float = 1.0) -> str:
    info = probe(src)
    run_ffmpeg(["-ss", _fmt(min(at, max(0.0, info["duration"] - 0.1))), "-i", src, "-frames:v", "1", "-vf", "scale=320:-2", out])
    return out


def normalize_segment(seg: SegmentSpec, out: str, settings: RenderSettings) -> float:
    """Encodes one timeline segment to the common format. Returns its output duration."""
    if seg.audio_mode not in AUDIO_MODES:
        raise VideoEditError(f"Unknown audio mode '{seg.audio_mode}'.")
    speed = min(2.0, max(0.5, float(seg.speed or 1.0)))
    info = probe(seg.path)
    start = max(0.0, float(seg.trim_start or 0.0))
    end = info["duration"] if seg.trim_end in (None, 0) else min(float(seg.trim_end), info["duration"])
    if end - start < 0.2:
        raise VideoEditError(f"Segment {os.path.basename(seg.path)} is too short after trimming ({start}s-{end}s).")
    dur = (end - start) / speed
    W, H, FPS, SR = settings.width, settings.height, settings.fps, config.RENDER_AUDIO_SAMPLE_RATE

    inputs = ["-i", seg.path]
    idx = 1
    filters = [
        f"[0:v]trim=start={_fmt(start)}:end={_fmt(end)},setpts=(PTS-STARTPTS)/{speed},"
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={FPS},format=yuv420p[vbase]"
    ]
    v_label = "vbase"

    if seg.overlay_text:
        png = render_text_overlay(seg.overlay_text, W, H, _tmp_path(".png"))
        inputs += ["-loop", "1", "-i", png]
        filters.append(f"[vbase][{idx}:v]overlay=0:0:shortest=1,format=yuv420p[vtxt]")
        v_label = "vtxt"
        idx += 1

    afmt = f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo"
    use_original = seg.audio_mode in ("original", "mix") and info["has_audio"]
    use_vo = seg.audio_mode in ("voiceover", "mix") and seg.voiceover_path and os.path.exists(seg.voiceover_path)

    parts = []
    if use_original:
        filters.append(
            f"[0:a]atrim=start={_fmt(start)}:end={_fmt(end)},asetpts=PTS-STARTPTS,atempo={speed},{afmt},volume={seg.original_volume}[aorig]"
        )
        parts.append("[aorig]")
    if use_vo:
        inputs += ["-i", seg.voiceover_path]
        delay_ms = int(max(0.0, seg.narration_start) * 1000)
        filters.append(f"[{idx}:a]{afmt},adelay={delay_ms}|{delay_ms},volume={seg.voiceover_volume}[avo]")
        parts.append("[avo]")
        idx += 1

    if not parts:
        inputs += ["-f", "lavfi", "-t", _fmt(dur), "-i", f"anullsrc=r={SR}:cl=stereo"]
        filters.append(f"[{idx}:a]{afmt}[asil]")
        parts.append("[asil]")
        idx += 1

    if len(parts) == 2:
        filters.append(f"{parts[0]}{parts[1]}amix=inputs=2:duration=longest:dropout_transition=0,volume=2[amixed]")
        a_label = "amixed"
    else:
        a_label = parts[0][1:-1]
    filters.append(f"[{a_label}]apad,atrim=0:{_fmt(dur)},asetpts=PTS-STARTPTS[aout]")

    run_ffmpeg(inputs + [
        "-filter_complex", ";".join(filters),
        "-map", f"[{v_label}]", "-map", "[aout]", "-t", _fmt(dur),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2", out,
    ])
    return dur


# ---------------------------------------------------------------------- #
# Merge
# ---------------------------------------------------------------------- #
def concat_segments(paths: List[str], durations: List[float], out: str, settings: RenderSettings) -> float:
    n = len(paths)
    if n == 0:
        raise VideoEditError("No segments to merge.")
    transition = settings.transition if settings.transition in TRANSITIONS else "fade"
    t = max(0.0, float(settings.transition_duration or 0.0))
    if n > 1 and transition != "none" and t > 0:
        t = min(t, min(durations) / 2.0)
    else:
        transition, t = "none", 0.0

    inputs = []
    for p in paths:
        inputs += ["-i", p]
    filters = []
    if transition == "none":
        streams = "".join(f"[{i}:v][{i}:a]" for i in range(n))
        filters.append(f"{streams}concat=n={n}:v=1:a=1[vcat][acat]")
        total = sum(durations)
    else:
        prev_v, prev_a, offset = "[0:v]", "[0:a]", 0.0
        for i in range(1, n):
            offset += durations[i - 1] - t
            vo, ao = f"[v{i}]", f"[a{i}]"
            filters.append(f"{prev_v}[{i}:v]xfade=transition={transition}:duration={_fmt(t)}:offset={_fmt(offset)}{vo}")
            filters.append(f"{prev_a}[{i}:a]acrossfade=d={_fmt(t)}{ao}")
            prev_v, prev_a = vo, ao
        total = sum(durations) - t * (n - 1)
        filters.append(f"{prev_v}null[vcat]")
        filters.append(f"{prev_a}anull[acat]")

    v_label, a_label = "vcat", "acat"
    if settings.fade_in_out and total > 2:
        fd = 0.6
        filters.append(f"[vcat]fade=t=in:st=0:d={fd},fade=t=out:st={_fmt(total - fd)}:d={fd}[vfaded]")
        filters.append(f"[acat]afade=t=in:st=0:d={fd},afade=t=out:st={_fmt(total - fd)}:d={fd}[afaded]")
        v_label, a_label = "vfaded", "afaded"

    if settings.background_music_path and os.path.exists(settings.background_music_path):
        inputs += ["-stream_loop", "-1", "-i", settings.background_music_path]
        m = n
        fade_st = max(0.0, total - 2.0)
        filters.append(
            f"[{m}:a]aresample={config.RENDER_AUDIO_SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=0:{_fmt(total)},asetpts=PTS-STARTPTS,volume={settings.background_music_volume},"
            f"afade=t=out:st={_fmt(fade_st)}:d=2[bgm]"
        )
        filters.append(f"[{a_label}][bgm]amix=inputs=2:duration=first:dropout_transition=0,volume=2[abgm]")
        a_label = "abgm"

    run_ffmpeg(inputs + [
        "-filter_complex", ";".join(filters),
        "-map", f"[{v_label}]", "-map", f"[{a_label}]", "-t", _fmt(total),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(settings.fps),
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out,
    ])
    return total


def merge_timeline(segments: List[SegmentSpec], out: str, settings: RenderSettings, on_progress=None) -> dict:
    notify = on_progress or (lambda _p, _m: None)
    if not segments:
        raise VideoEditError("Timeline is empty. Add at least one clip to merge.")
    temps, durations = [], []
    try:
        for i, seg in enumerate(segments):
            notify(int(70 * i / len(segments)), f"Processing segment {i + 1}/{len(segments)}...")
            tmp = _tmp_path(".mp4")
            temps.append(tmp)
            durations.append(normalize_segment(seg, tmp, settings))
        notify(75, f"Stitching {len(segments)} segments ({settings.transition} transitions)...")
        total = concat_segments(temps, durations, out, settings)
    finally:
        for p in temps:
            if os.path.exists(p):
                os.remove(p)
    notify(100, "Final video rendered.")
    return {"path": out, "duration": round(total, 2), "segments": len(segments), "width": settings.width, "height": settings.height}
