# Bengal Scheme-to-Video AI Studio

A modular AI-driven prompt engineering and agentic quality inspection platform for the **Government of West Bengal**. It extracts information from official scheme documents and grounding photos, generates **5 to 6 scenes** for a 45–55 second video storyboard, synthesizes authentic Bengali voiceovers, and performs autonomous auditing through an **AI Critic Agent**.

---

## 🏛️ Key Capabilities & Guardrails

1. **5–6 Structured Video Scenes (~8–10s each)**:
   - **Scene 1**: Scheme Overview & Vision
   - **Scene 2**: Benefits & Financial Stipends (DBT)
   - **Scene 3**: Eligibility Criteria (Family income ceiling & academic marks)
   - **Scene 4**: Required Documents (Marksheets, income certificate, bank passbook)
   - **Scene 5**: How to Apply (Step-by-step portal registration & upload)
   - **Scene 6**: Helpline, Contact Details & Official Portal
2. **Strict No Lip-Sync Enforcement**:
   - Prompts mandate candid B-roll cinematography (students walking in corridors, library research, digital form entry). Visuals never show human characters speaking or mouthing words.
3. **Local West Bengal Grounding**:
   - Visuals strictly reference West Bengal institutions (e.g. *Presidency University*, *Jadavpur University*, *Calcutta University*, *National Library Kolkata*, *Bikash Bhavan*).
   - If grounding photos of local colleges/libraries are uploaded, they are passed as multimodal references and strictly integrated into visual prompts.
4. **Narration Audio Timing Margins**:
   - Narration begins strictly at $\ge 1.0\text{s}$ (1.0s–1.2s lead-in).
   - Narration concludes at $\le 8.5\text{s}$ (leaving a 1.5s–2.0s silent buffer before scene transition) to prevent audio overlapping when scenes are stitched.
5. **Natural Everyday Bengali (সহজ ও সাবলীল প্রাত্যহিক বাংলা)**:
   - Everyday conversational Bengali phrasing tailored for students and parents.
6. **Micro-Prompting On-Screen Text**:
   - Prominent Bengali & numerical on-screen cues for quick retention.
7. **AI Critic & Refinement Agent**:
   - Autonomously audits every scene against the 6 core criteria, flags violations, fixes timings, refines Bengali phrasing, and outputs a compliance score (0–100) and resolution log.
8. **Bengali TTS Audio Preview**:
   - Direct voiceover preview player with timing buffer calculation.
9. **AI Video Generation (Gemini Omni 1.1 Flash)**:
   - Turns every generated scene prompt into a real video clip (text-to-video, or reference-to-video using your grounding photos).
   - Generate one scene or all scenes as background jobs, with live progress.
   - AI editing of any clip with a plain-language instruction (Omni `edit`), or continuing a shot (Omni `extend`).
10. **Video Editing & Merging Studio** (FFmpeg):
   - Clip library: upload your own clips, trim, delete.
   - Timeline: reorder, trim, speed (0.5×–2×), Bengali on-screen text, audio mode (`original` / `voiceover` / `mix` / `mute`).
   - Bengali TTS voiceover placed at each scene's narration start, background music with volume, transitions (fade, dissolve, wipes, slides, ...), fade in/out.
   - Renders a final H.264/AAC MP4 (16:9 or 9:16, 360p–4K) for download.
11. **Zero Template / Placeholder Fallback**:
   - System requires a valid API key (Gemini / OpenAI) and genuine scheme document. Generation is blocked with explicit 400 alerts if inputs are missing.

---

## 🧱 Modular Architecture

```
h:/py_vid/
│
├── backend/
│   ├── __init__.py
│   ├── models.py             # Pydantic schemas (ScenePrompt, VideoProjectPlan, CriticAudit)
│   ├── document_parser.py    # Extracts PDF, Word (.docx), TXT & processes grounding photos
│   ├── llm_client.py         # Multi-provider wrapper (Gemini / OpenAI) with strict error handling
│   ├── scene_generator.py    # Directs 5-6 scenes with WB grounding, timing, & Bengali voiceover
│   ├── critic_agent.py       # Autonomous AI Critic Agent that audits and refines prompts
│   ├── tts_service.py        # Bengali Text-to-Speech audio preview generator
│   ├── media_adapter.py      # Abstract media generation interface
│   ├── config.py             # ⚙️ OMNI_API_KEY slot, model name, media folders, render defaults
│   ├── omni_video_adapter.py # Gemini Omni 1.1 Flash: text/reference-to-video, edit, extend
│   ├── video_editor.py       # FFmpeg engine: trim, speed, overlays, VO mixing, transitions, merge
│   ├── media_store.py        # File-backed clip / asset / final-render library (output/)
│   ├── job_manager.py        # Background jobs for long-running generation & rendering
│   ├── video_routes.py       # /api/video, /api/clips, /api/jobs, /api/assets, /api/finals
│   ├── fonts/                # Noto Sans Bengali + Noto Sans (OFL) for burned-in captions
│   └── main.py               # FastAPI application with REST endpoints
│
├── frontend/
│   ├── index.html            # Ultra-modern glassmorphic interface with Bengali font support
│   ├── styles.css            # Dark mode, responsive design system & badges
│   ├── app.js                # State management, live prompt editing, audio playback & exports
│   └── video_studio.js       # Video Production Studio: generation, clip editing, timeline, merge
│
├── tests/
│   ├── test_pipeline.py      # Automated unit tests for all modules and guardrails
│   └── test_video_studio.py  # Omni adapter (mocked), FFmpeg editing/merge and studio API tests
│
├── requirements.txt
├── run.py                    # Server launcher
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

FFmpeg is required for editing/merging. A system `ffmpeg` on `PATH` is used if present; otherwise the bundled binary from `imageio-ffmpeg` is used.

### 2. Add your Omni API key
Paste your key into `backend/config.py`:
```python
OMNI_API_KEY = ""   # <-- paste your Gemini Omni 1.1 Flash API key here
```
Alternatives: set the `OMNI_API_KEY` (or `GEMINI_API_KEY`) environment variable, or paste it into the **Omni API Key** field in the Video Production Studio (sent per request, overrides config). Never commit a real key.

Optional env vars: `OMNI_MODEL_NAME` (default `gemini-omni-1.1-flash`), `MEDIA_ROOT` (default `output/`), `OVERLAY_FONT_PATH`.

### 3. Start the Server
```bash
python run.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 4. Make a video
1. Generate a storyboard from a scheme document (as before).
2. **Video Production Studio → 1.** Click **Generate All Scene Videos** (or per-scene **Generate**).
3. **2. Clip Library** — preview, trim, or use **AI Edit** to change a clip with an instruction.
4. **3. Timeline** — **Auto-build from Scenes**, adjust trims / audio / on-screen text, pick a transition and background music.
5. Click **Merge & Render Final Video**, then download the MP4 from **4. Final Videos**.

### 5. Run Automated Tests
```bash
python -m pytest tests/
```

---

## 🎞️ Video API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/video/config` | Model name, key status, transitions, audio modes |
| POST | `/api/video/generate-scene` | Generate one scene clip (background job) |
| POST | `/api/video/generate-all` | Generate all / selected scenes of a plan (background job) |
| GET | `/api/jobs/{id}` | Poll job progress / result |
| GET, POST, DELETE | `/api/clips`, `/api/clips/upload`, `/api/clips/{id}` | Clip library |
| POST | `/api/clips/{id}/trim` | Create a trimmed copy |
| POST | `/api/clips/{id}/ai-edit` | Omni edit / extend with an instruction (background job) |
| GET, POST | `/api/assets`, `/api/assets/upload` | Background music |
| POST | `/api/video/merge` | Render a timeline into a final MP4 (background job) |
| GET | `/api/finals` | List final renders |

Generated files live in `output/` (`clips/`, `assets/`, `final/`) and are served from `/media/...`.

---

## 🔌 Independent Team Extension (Media Adapter)
`backend/omni_video_adapter.py` is the reference implementation of `BaseMediaAdapter`. Other models can be plugged in the same way:
```python
from backend.media_adapter import BaseMediaAdapter, MediaGenerationResult

class MyVideoModelAdapter(BaseMediaAdapter):
    def generate_image_from_prompt(self, prompt, negative_prompt=None, grounding_reference_paths=None):
        # Plug in Imagen 3, Stable Diffusion, or Midjourney API
        ...

    def generate_video_from_image(self, image_bytes, camera_motion_prompt, duration_seconds=10.0):
        # Plug in Runway Gen-3, Kling, or Luma Dream Machine
        ...
```
