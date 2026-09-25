"""
AI Critic & Refinement Agent.
Audits generated video scene prompts against strict constraints:
- Zero lip-sync violation
- West Bengal educational institutions / libraries grounding
- 1s lead-in and 1-2s lead-out narration timing margins
- Natural everyday Bengali phrasing
- Micro-prompting on-screen text
- Scheme document factual precision
Refines the prompts and outputs an audit log with resolution details.
"""
from typing import List, Dict, Any, Optional
from backend.models import VideoProjectPlan, ScenePrompt, CriticAudit, CriticIssue
from backend.llm_client import LLMClient, LLMExecutionError


CRITIC_AGENT_SYSTEM_PROMPT = """
You are an expert AI Video Quality Inspector & Chief Director for Government of West Bengal media productions.
Your mission is to perform a rigorous audit on generated video prompts and refine them to 100% compliance.

INSPECTION CHECKLIST:
1. [LIP-SYNC AUDIT]: Check if any prompt portrays a human talking to the camera or moving lips in sync with speech.
   - VIOLATION: Character talking to camera, speaking into mic, or mouth moving.
   - CORRECTION: Convert to candid documentary B-roll (wide shot, over-the-shoulder, looking at laptop, walking, studying, library stacks) with narrator as third-person voiceover.
2. [WEST BENGAL GROUNDING & IMAGES]:
   - VIOLATION: Generic unspecified colleges or non-Bengal landmarks.
   - CORRECTION: Replace with specific prominent West Bengal colleges or libraries (e.g. Presidency University, Jadavpur University, Calcutta University, National Library Alipore, Bikash Bhavan, St. Xavier's Kolkata) or incorporate uploaded reference photos if specified.
3. [TIMING MARGIN AUDIT]:
   - For every 10s scene: narration MUST start at >= 1.0s (e.g. 1.0s or 1.2s).
   - Narration MUST end at <= 8.5s (leaving at least 1.5s to 2.0s silent buffer for clean audio stitching).
   - Spoken Bengali word count must fit within 6 to 7.2 seconds (approx 16-24 Bengali words).
4. [NATURAL EVERYDAY BENGALI]:
   - VIOLATION: Stiff Sanskritized syntax, textbook grammar errors, or machine-translated literal English phrases.
   - CORRECTION: Convert to natural, modern everyday spoken Bengali that common students and parents across West Bengal speak and immediately understand.
5. [MICRO-PROMPTING FOR ON-SCREEN TEXT]:
   - Ensure crisp, readable bullet phrases and numbers (e.g. "৬০% নম্বর", "আড়াই লক্ষ টাকা", "svmcm.wbhed.gov.in") are explicitly listed.
6. [DOCUMENT FACTUAL ACCURACY]:
   - Confirm benefits, income ceiling, marks criteria, and helpline numbers match the original scheme text.

You must return a JSON response containing:
- compliance_score: integer 0-100
- passed_checks: boolean (true if compliance_score >= 90)
- issues_found: list of identified issues and their resolutions
- summary_notes: concise audit overview
- refined_scenes: fully corrected list of scenes
"""

CRITIC_AGENT_USER_PROMPT = """
ORIGINAL SCHEME DOCUMENT TEXT:
\"\"\"
{scheme_text}
\"\"\"

GROUNDING NOTES / DETECTED ASSETS:
{grounding_notes}

PROPOSED VIDEO PROJECT PLAN TO AUDIT:
\"\"\"
{current_plan_json}
\"\"\"

Please audit every scene meticulously, record all defects/improvements, and output the refined compliant scenes.

JSON SCHEMA TO RETURN:
{{
  "compliance_score": 95,
  "passed_checks": true,
  "summary_notes": "Detailed review summary...",
  "issues_found": [
    {{
      "scene_number": 1,
      "category": "timing_margin",
      "severity": "warning",
      "description": "Narration start was at 0.5s which risked audio overlap upon scene stitching.",
      "resolution": "Shifted narration start to 1.0s and trimmed 2 words to finish cleanly at 8.0s."
    }}
  ],
  "refined_scenes": [
    {{
      "scene_number": 1,
      "scene_title": "Scheme Overview",
      "duration_seconds": 10.0,
      "visual_prompt": "Refined visual prompt with strict WB grounding and no lip sync...",
      "camera_angle": "Camera angle...",
      "voiceover_bengali": "Natural everyday Bengali voiceover...",
      "narration_start_second": 1.0,
      "narration_end_second": 8.0,
      "onscreen_text": ["On screen bullet 1", "On screen bullet 2"],
      "wb_grounding_elements": ["West Bengal landmarks"],
      "no_lip_sync_enforcement": "Explicit rule description"
    }}
  ]
}}
"""


class CriticAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def audit_and_refine(
        self,
        current_plan: VideoProjectPlan,
        scheme_document_text: str,
        grounding_notes: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> VideoProjectPlan:
        """
        Runs the Critic Agent to evaluate and refine the plan iteratively.
        Loops up to 3 times if there are still issues, ensuring the prompt is actually refined.
        Returns a refined VideoProjectPlan containing the CriticAudit log.
        """
        import json
        
        max_iterations = 3
        current_iteration_plan = current_plan
        last_audit = None
        
        for iteration in range(max_iterations):
            plan_dict = current_iteration_plan.model_dump(exclude={"critic_audit"})
            user_prompt = CRITIC_AGENT_USER_PROMPT.format(
                scheme_text=scheme_document_text.strip(),
                grounding_notes=grounding_notes or "West Bengal educational and institutional grounding.",
                current_plan_json=json.dumps(plan_dict, ensure_ascii=False, indent=2)
            )
            
            # Note: In subsequent loops, the model is informed of what it just generated. 
            if iteration > 0 and last_audit:
                user_prompt += f"\n\nPREVIOUS AUDIT RESULTS (You must fix these issues in this iteration):\n{json.dumps(last_audit.model_dump(), indent=2)}"

            raw_response = self.llm_client.generate_json(
                system_prompt=CRITIC_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model_name=model_name
            )

            try:
                issues = [CriticIssue(**issue) for issue in raw_response.get("issues_found", [])]
                passed = bool(raw_response.get("passed_checks", True))
                
                last_audit = CriticAudit(
                    compliance_score=int(raw_response.get("compliance_score", 95)),
                    passed_checks=passed,
                    issues_found=issues,
                    summary_notes=raw_response.get("summary_notes", "Audit completed successfully.")
                )

                refined_scene_dicts = raw_response.get("refined_scenes", [])
                refined_scenes = []
                if refined_scene_dicts and len(refined_scene_dicts) >= len(current_iteration_plan.scenes):
                    for s in refined_scene_dicts:
                        # Enforce timing programmatic guardrail as well
                        start_sec = max(1.0, float(s.get("narration_start_second", 1.0)))
                        duration = float(s.get("duration_seconds", 10.0))
                        end_sec = min(duration - 1.5, float(s.get("narration_end_second", duration - 1.5)))
                        s["narration_start_second"] = start_sec
                        s["narration_end_second"] = end_sec
                        refined_scenes.append(ScenePrompt(**s))
                else:
                    refined_scenes = current_iteration_plan.scenes

                current_iteration_plan = VideoProjectPlan(
                    scheme_title=current_iteration_plan.scheme_title,
                    total_duration_seconds=sum(s.duration_seconds for s in refined_scenes),
                    global_guidelines=current_iteration_plan.global_guidelines,
                    detected_grounding_references=current_iteration_plan.detected_grounding_references,
                    scenes=refined_scenes,
                    critic_audit=last_audit,
                    is_refined_by_critic=True
                )
                
                if passed:
                    break

            except Exception as e:
                if iteration == max_iterations - 1:
                    raise LLMExecutionError(f"Critic agent failed during audit parsing on final iteration: {str(e)}")
                else:
                    continue

        return current_iteration_plan
