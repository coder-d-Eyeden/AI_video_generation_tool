"""
Scene Generator Module.
Generates 5-6 structured scene prompts for a 45-55 second video from a scheme document.
Enforces West Bengal local grounding, Bengali voiceover, timing margins, and NO lip sync.
"""
from typing import List, Dict, Any, Optional
from backend.models import VideoProjectPlan, ScenePrompt, ProviderType
from backend.llm_client import LLMClient, LLMExecutionError


SCENE_GENERATION_SYSTEM_PROMPT = """
You are an expert Video Director and Prompt Engineer for Government Schemes.
Your task is to transform official government scheme documents into a high-impact, professional 45-55 second video storyboard consisting of exactly 5 to 6 scenes.

MANDATORY RULES & CONSTRAINTS:
1. STRICT DOCUMENT ADHERENCE (NO HALLUCINATIONS):
   - YOU MUST BASE YOUR PROMPT EXACTLY ON THE UPLOADED DOCUMENT CONTENT.
   - DO NOT substitute the scheme with any other scheme. For example, if the document is about "PM Vidyalaxmi", you MUST generate prompts for "PM Vidyalaxmi". Do NOT change it to "West Bengal Student Credit Card" or anything else.
   - If the uploaded document is not a valid scheme document or is garbage text, return an error in your JSON response (see schema).

2. SCENE SEQUENCE (Total 5 to 6 scenes, 8 to 10 seconds each):
   - Scene 1: Scheme Overview (Title, purpose, vision)
   - Scene 2: Benefits & Support Amount
   - Scene 3: Eligibility Criteria
   - Scene 4: Required Documents
   - Scene 5: How to Apply
   - Scene 6: Helpline & Official Portal

3. ABSOLUTELY NO LIP-SYNC:
   - The video MUST NEVER show any human face talking or lip-syncing.
   - All characters are shown in natural candid action. The narration is strictly a voice-over (VO).

4. VISUAL GROUNDING (WEST BENGAL):
   - The user requested that IF any colleges or libraries are shown, they MUST be visually set in West Bengal (e.g. West Bengal architecture, Bengali students, etc).
   - Do NOT change the name of the scheme to a West Bengal scheme just because the visuals are in West Bengal.

5. AUDIO & NARRATION:
   - Narration starts at 1.0s or 1.2s. Ends at 8.0s to 8.5s.
   - The voiceover MUST be in natural, everyday Bengali.

6. HIGHLY DETAILED VISUAL PROMPT STRUCTURE:
   The `visual_prompt` string for EACH scene MUST follow this EXACT highly detailed, verbose multi-section layout. DO NOT output a vague prompt.
   Create a 10-second photorealistic, cinematic government awareness video scene about [Topic].
   This is an independent 10-second clip that will later be stitched together.
   SCENE PURPOSE: [Detailed purpose]
   VISUAL SETTING: [Detailed setting in WB]
   VISUAL SEQUENCE: [Detailed shot-by-shot progression]
   NARRATION: The narration MUST be spoken entirely in Bengali. Use this EXACT narration: "[Bengali text]"
   VOICE: Use a natural Bengali-speaking voice. Tone: formal, informative...
   AUDIO: Bengali narration must be the primary audio. Add only subtle realistic ambience...
   ON-SCREEN TEXT: [List English text]
   CAMERA: [Detailed camera instructions]
   CHARACTER CONSISTENCY: Use realistic Bengali students...
   VISUAL STYLE: Photorealistic. Cinematic...
   SYNCHRONIZATION: The visuals should naturally correspond...
   FACTUAL CONSTRAINTS: [Facts from document]
   NEGATIVE CONSTRAINTS: No incorrect English text. No lip sync...
   FINAL PRIORITY: 1. Exact Bengali narration 2. Correct factual information...

You must respond ONLY with a valid JSON object matching the exact schema specified.
"""

SCENE_GENERATION_USER_PROMPT_TEMPLATE = """
SCHEME DOCUMENT TO CONVERT:
\"\"\"
{scheme_text}
\"\"\"

ADDITIONAL NOTES / GROUNDING INPUTS:
{additional_notes}

NUMBER OF GROUNDING IMAGES ATTACHED: {num_images}

Please generate the complete video project plan strictly adhering to all guidelines. Do NOT hallucinate scheme details; use ONLY the provided document exactly as written.

JSON SCHEMA TO RETURN:
{{
  "is_valid_scheme": true,
  "error_message": "Leave empty if valid. If the document is garbage, set is_valid_scheme to false and explain why here.",
  "scheme_title": "Full name of the scheme EXACTLY as in document (e.g. PM Vidyalaxmi)",
  "total_duration_seconds": 50.0,
  "global_guidelines": "Core directorial guidelines covering tone, audio pacing, and strict no-lip-sync rule",
  "detected_grounding_references": [
    "List of WB institutions or uploaded landmarks referenced"
  ],
  "scenes": [
    {{
      "scene_number": 1,
      "scene_title": "Scheme Overview",
      "duration_seconds": 10.0,
      "visual_prompt": "YOUR HIGHLY DETAILED STRUCTURED PROMPT GOES HERE, FORMATTED EXACTLY WITH HEADINGS: 'SCENE PURPOSE:', 'VISUAL SETTING:', 'VISUAL SEQUENCE:', 'NARRATION:', 'VOICE:', 'AUDIO:', 'ON-SCREEN TEXT:', 'CAMERA:', 'CHARACTER CONSISTENCY:', 'VISUAL STYLE:', 'SYNCHRONIZATION:', 'FACTUAL CONSTRAINTS:', 'NEGATIVE CONSTRAINTS:', 'FINAL PRIORITY:'",
      "camera_angle": "E.g., Wide establishing drone shot...",
      "voiceover_bengali": "Bengali voiceover script",
      "narration_start_second": 1.0,
      "narration_end_second": 8.2,
      "onscreen_text": ["English/Bengali text"],
      "wb_grounding_elements": ["Specific WB visual cues"],
      "no_lip_sync_enforcement": "Reason why no lip sync occurs"
    }}
  ]
}}
"""


class SceneGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_plan(
        self,
        scheme_document_text: str,
        grounding_images: Optional[List[Dict[str, Any]]] = None,
        additional_notes: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> VideoProjectPlan:
        """Generates the initial 5-6 scenes from scheme document."""
        if not scheme_document_text or len(scheme_document_text.strip()) < 30:
            raise ValueError("Scheme document text is empty or too short to generate a video plan.")

        num_images = len(grounding_images) if grounding_images else 0
        img_note = ""
        if num_images > 0:
            img_names = [img.get("filename", f"image_{i+1}") for i, img in enumerate(grounding_images)]
            img_note = f"\nUser uploaded {num_images} grounding photo(s): {', '.join(img_names)}. Strictly analyze visual architectural details (building facade, library halls, campus gates) from these images and incorporate them into the visual prompts."

        notes_combined = (additional_notes or "").strip() + img_note

        user_prompt = SCENE_GENERATION_USER_PROMPT_TEMPLATE.format(
            scheme_text=scheme_document_text.strip(),
            additional_notes=notes_combined if notes_combined else "Strict West Bengal context.",
            num_images=num_images
        )

        raw_json = self.llm_client.generate_json(
            system_prompt=SCENE_GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            grounding_images=grounding_images,
            model_name=model_name
        )

        try:
            # Check if the LLM flagged the document as invalid
            is_valid = raw_json.get("is_valid_scheme", True)
            if not is_valid:
                error_msg = raw_json.get("error_message", "The provided text does not appear to be a valid scheme document.")
                raise ValueError(f"AI Validation Error: {error_msg}")

            # Parse into Pydantic model
            scenes = [ScenePrompt(**s) for s in raw_json.get("scenes", [])]
            if len(scenes) < 4:
                raise LLMExecutionError(f"Generated plan has only {len(scenes)} scenes. Minimum 5 scenes required.")

            plan = VideoProjectPlan(
                is_valid_scheme=True,
                error_message=None,
                scheme_title=raw_json.get("scheme_title", "Government Scheme Video"),
                total_duration_seconds=float(raw_json.get("total_duration_seconds", sum(s.duration_seconds for s in scenes))),
                global_guidelines=raw_json.get("global_guidelines", "Documentary style, natural lighting, WB campus grounding, no lip sync."),
                detected_grounding_references=raw_json.get("detected_grounding_references", []),
                scenes=scenes,
                critic_audit=None,
                is_refined_by_critic=False
            )
            return plan

        except Exception as e:
            raise LLMExecutionError(f"Failed to structure generated video plan: {str(e)}")
