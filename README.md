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
7. **AI Critic & Refinement Agent (Groq)**:
   - Autonomously audits every scene against the 6 core criteria, flags violations, fixes timings, refines Bengali phrasing, and outputs a compliance score (0–100) and resolution log.
   - The Critic always runs on **Groq** (`openai/gpt-oss-120b` by default), independent of the provider that generated the scenes, so the audit never inherits the generator's blind spots.
   - The model list is **discovered at runtime** from `GET /openai/v1/models`, so a Groq retirement degrades to a fallback model instead of a hard failure.
8. **Bengali TTS Audio Preview**:
   - Direct voiceover preview player with timing buffer calculation.
9. **Zero Template / Placeholder Fallback**:
   - System requires a valid API key (Gemini / OpenAI for scenes, Groq for the Critic) and genuine scheme document. Generation is blocked with explicit 400 alerts if inputs are missing.

---

## 🧱 Modular Architecture

```
h:/py_vid/
│
├── backend/
│   ├── __init__.py
│   ├── models.py             # Pydantic schemas (ScenePrompt, VideoProjectPlan, CriticAudit)
│   ├── document_parser.py    # Extracts PDF, Word (.docx), TXT & processes grounding photos
│   ├── llm_client.py         # Multi-provider wrapper (Gemini / OpenAI / Groq) with strict error handling
│   ├── scene_generator.py    # Directs 5-6 scenes with WB grounding, timing, & Bengali voiceover
│   ├── critic_agent.py       # Autonomous AI Critic Agent (Groq) that audits and refines prompts
│   ├── tts_service.py        # Bengali Text-to-Speech audio preview generator
│   ├── media_adapter.py      # Abstract interface for future Text-to-Image and Image-to-Video models
│   └── main.py               # FastAPI application with REST endpoints
│
├── frontend/
│   ├── index.html            # Ultra-modern glassmorphic interface with Bengali font support
│   ├── styles.css            # Dark mode, responsive design system & badges
│   └── app.js                # State management, live prompt editing, audio playback & exports
│
├── tests/
│   └── test_pipeline.py      # Automated unit tests for all modules and guardrails
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

### 2. Configure the Critic Key (Groq)
The AI Critic Agent runs on Groq and needs its own key. Either export it:
```bash
set GROQ_API_KEY=gsk_...
```
or paste it into the **AI Critic Credentials (Groq)** box in the UI, which overrides the environment variable. Pressing **Verify Key** there also loads the chat models your account can currently reach, so you can pick from a live list.

Optional environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq credential used by the Critic | _(required unless pasted in the UI)_ |
| `GROQ_MODEL` | Overrides the Critic's model | `openai/gpt-oss-120b` |

The scene generator uses the provider/key selected in the UI (Gemini or OpenAI) and never the Groq key.

> **Note on Llama:** Groq retired every Llama model ID — `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` were shut down on 2026-08-16, and the Llama 4 Scout/Maverick IDs earlier. Groq's recommended successor to `llama-3.3-70b-versatile` is `openai/gpt-oss-120b`, which is the critic's default. Requesting a retired ID returns HTTP 404, so the client auto-discovers the live model list and falls back rather than failing.

### 3. Start the Server
```bash
python run.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 4. Run Automated Tests
```bash
python -m pytest tests/
```

---

## 🔌 Independent Team Extension (Media Adapter)
Team members working on image/video generation can implement `BaseMediaAdapter` in `backend/media_adapter.py`:
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
