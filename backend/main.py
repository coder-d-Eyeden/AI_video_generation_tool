"""
FastAPI Application for Government Scheme Video Prompt Generator.
Enforces strict input validation: requires valid API key and scheme document.
Zero placeholder/template fallbacks.
"""
import os
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    ProviderType,
    VideoProjectPlan,
    SynthesizeAudioRequest,
    RefineRequest
)
from backend.document_parser import (
    extract_text_from_bytes,
    process_grounding_images,
    DocumentParsingError
)
from backend.llm_client import LLMClient, LLMAuthError, LLMExecutionError
from backend.scene_generator import SceneGenerator
from backend.critic_agent import CriticAgent
from backend.tts_service import TTSService

app = FastAPI(
    title="Bengal Scheme-to-Video AI Studio",
    description="Transforms West Bengal Government scheme documents into 45-55s video prompts with local grounding, Bengali voiceover, and AI Critic validation.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/validate-key")
async def validate_api_key(
    provider: str = Form(...),
    api_key: str = Form(...),
    model_name: Optional[str] = Form(default=None)
):
    """
    Verifies the API key using a lightweight metadata call — NOT a generation request.
    Gemini: calls models.list() which is near-instant.
    OpenAI: calls models.list() which is near-instant.
    This avoids triggering the full retry/backoff chain just to check a key.
    """
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API Key cannot be blank. Please paste your Gemini or OpenAI API key."
        )

    try:
        provider_type = ProviderType(provider.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider '{provider}'. Must be 'gemini' or 'openai'."
        )

    try:
        if provider_type == ProviderType.GEMINI:
            from google import genai
            client = genai.Client(api_key=cleaned_key)
            # models.list() is a fast metadata call — no generation, no quota used
            model_list = [m.name for m in client.models.list()]
            if not model_list:
                raise ValueError("No models returned — key may be invalid.")
            return {
                "success": True,
                "provider": provider_type.value,
                "message": f"API Key verified! {len(model_list)} Gemini models available.",
                "available_models": [m for m in model_list if "gemini" in m.lower()][:10]
            }

        elif provider_type == ProviderType.OPENAI:
            from openai import OpenAI
            client = OpenAI(api_key=cleaned_key)
            model_list = [m.id for m in client.models.list()]
            return {
                "success": True,
                "provider": provider_type.value,
                "message": f"API Key verified! {len(model_list)} OpenAI models available."
            }

    except LLMAuthError as ae:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(ae))
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "invalid" in err_msg.lower() or "API_KEY" in err_msg:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid API Key: {err_msg}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Key validation failed: {err_msg}")


@app.post("/api/generate-plan")
async def generate_plan_endpoint(
    provider: str = Form(default=""),
    api_key: str = Form(default=""),
    model_name: Optional[str] = Form(default=None),
    scheme_text: Optional[str] = Form(default=None),
    additional_notes: Optional[str] = Form(default=None),
    auto_run_critic: bool = Form(default=True),
    scheme_file: Optional[UploadFile] = None,
    grounding_photos: Optional[List[UploadFile]] = None
):
    """
    Main endpoint: extracts scheme details, generates 5-6 scene prompts,
    and runs the AI Critic Agent to audit and refine.
    Strictly fails if API key or scheme document is missing.
    """
    # 1. Validate API Key
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API Key is missing! You must paste your Gemini or OpenAI API key. The system will not generate any output without an API key."
        )

    try:
        provider_type = ProviderType(provider.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider '{provider}'. Choose 'gemini' or 'openai'."
        )

    # 2. Extract and Validate Scheme Document
    document_content = ""
    grounding_images = []
    
    if scheme_file and scheme_file.filename:
        file_bytes = await scheme_file.read()
        try:
            document_content = extract_text_from_bytes(file_bytes, scheme_file.filename)
        except DocumentParsingError as dpe:
            if provider_type == ProviderType.GEMINI and scheme_file.filename.lower().endswith(".pdf"):
                import base64
                # Offload PDF parsing to Gemini natively
                b64_str = base64.b64encode(file_bytes).decode("utf-8")
                grounding_images.append({
                    "filename": scheme_file.filename,
                    "mime_type": "application/pdf",
                    "base64": b64_str
                })
                document_content = f"The scheme document '{scheme_file.filename}' is attached as a PDF file to this prompt. Please read the attached PDF."
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(dpe))

    # Fallback to direct pasted text if no file uploaded
    if not document_content and scheme_text and scheme_text.strip():
        document_content = scheme_text.strip()

    if not document_content or len(document_content.strip()) < 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Scheme document is missing! Please upload a scheme document (PDF, Word, or TXT) or paste the scheme details. The system cannot produce video prompts without scheme source text."
        )

    # 3. Process Grounding Photos (Colleges, libraries, local architecture)
    if grounding_photos:
        raw_images = []
        for photo in grounding_photos:
            if photo and photo.filename:
                photo_bytes = await photo.read()
                raw_images.append((photo.filename, photo_bytes))
        if raw_images:
            try:
                grounding_images.extend(process_grounding_images(raw_images))
            except DocumentParsingError as dpe:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(dpe))

    # 4. Initialize LLM Client
    try:
        llm_client = LLMClient(provider=provider_type, api_key=cleaned_key)
    except LLMAuthError as ae:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(ae))

    # 5. Generate Initial Scene Prompts
    try:
        generator = SceneGenerator(llm_client=llm_client)
        plan = generator.generate_plan(
            scheme_document_text=document_content,
            grounding_images=grounding_images,
            additional_notes=additional_notes,
            model_name=model_name
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except LLMAuthError as ae:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(ae))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Generation failed: {str(e)}")

    # 6. Run AI Critic & Refinement Agent
    if auto_run_critic:
        try:
            import asyncio
            # Brief pause so the Critic call does not immediately hit the same
            # overloaded model endpoint as the generation call above.
            await asyncio.sleep(3)
            critic = CriticAgent(llm_client=llm_client)
            grounding_notes = ""
            if grounding_images:
                grounding_notes = f"Grounding images provided: {[img['filename'] for img in grounding_images]}. Ground visual prompts in these uploaded buildings."
            plan = critic.audit_and_refine(
                current_plan=plan,
                scheme_document_text=document_content,
                grounding_notes=grounding_notes,
                model_name=model_name
            )
        except Exception as e:
            # Critic failed (overload or transient). Return the already-generated
            # plan intact -- user can re-run Critic manually from the UI.
            plan.global_guidelines += f" [Critic skipped: {str(e)[:120]}]"

    return plan


@app.post("/api/run-critic")
async def run_critic_endpoint(request: RefineRequest):
    """Allows manual trigger of AI Critic Agent on an existing or user-edited plan."""
    cleaned_key = (request.api_key or "").strip()
    if not cleaned_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="API Key required.")

    if not request.scheme_document_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Original scheme document text is required for auditing.")

    try:
        llm_client = LLMClient(provider=request.provider, api_key=cleaned_key)
        critic = CriticAgent(llm_client=llm_client)
        refined_plan = critic.audit_and_refine(
            current_plan=request.current_plan,
            scheme_document_text=request.scheme_document_text,
            grounding_notes=request.grounding_notes
        )
        return refined_plan
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Critic audit failed: {str(e)}")


@app.post("/api/synthesize-voice")
async def synthesize_voice_endpoint(request: SynthesizeAudioRequest):
    """Synthesizes Bengali voiceover script to audio MP3 preview."""
    try:
        result = TTSService.synthesize_bengali_voiceover(
            bengali_text=request.bengali_text,
            start_pad_sec=request.start_pad_sec,
            end_pad_sec=request.end_pad_sec
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# Mount frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Bengal Scheme-to-Video AI Studio API is running."}
