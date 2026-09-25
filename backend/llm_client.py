"""
LLM Multi-Provider Client for Google Gemini and OpenAI.
Strictly requires a valid API key and input document.
Resilient retry with exponential backoff + plain-text fallback for large JSON payloads.
Zero placeholder/template fallbacks.
"""
import json
import re
import base64
import time
from typing import List, Dict, Any, Optional

from backend.models import ProviderType


class LLMAuthError(Exception):
    """Raised when API key is missing or invalid."""
    pass


class LLMExecutionError(Exception):
    """Raised when LLM call fails."""
    pass


# Ordered by availability & stability based on live model-list verification.
# gemini-2.5-flash added back (confirmed in user's model list).
GEMINI_CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

# Exponential backoff wait times (seconds) per retry attempt on same model
_BACKOFF_SECONDS = [2, 5, 10]


class LLMClient:
    def __init__(self, provider: ProviderType, api_key: str):
        if not api_key or not api_key.strip():
            raise LLMAuthError(
                "API Key is missing! You must paste a valid Gemini or OpenAI API key. "
                "The system will not generate any output without an API key."
            )
        self.provider = provider
        self.api_key = api_key.strip()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        grounding_images: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the LLM request and returns a parsed JSON dictionary.
        Supports multimodal grounding images if provided.
        """
        if self.provider == ProviderType.GEMINI:
            return self._call_gemini(system_prompt, user_prompt, grounding_images, model_name)
        elif self.provider == ProviderType.OPENAI:
            return self._call_openai(system_prompt, user_prompt, grounding_images, model_name)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        grounding_images: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Build candidate list: user-chosen model first, then fallback pool
        primary_model = (model_name or "gemini-3.5-flash").strip()
        candidates = [primary_model]
        for m in GEMINI_CANDIDATE_MODELS:
            if m not in candidates:
                candidates.append(m)

        # Build contents (images first, then text)
        contents = []
        if grounding_images:
            for img in grounding_images:
                raw_bytes = base64.b64decode(img["base64"])
                part = types.Part.from_bytes(
                    data=raw_bytes,
                    mime_type=img.get("mime_type", "image/jpeg")
                )
                contents.append(part)
        contents.append(user_prompt)

        last_error = None

        for model in candidates:
            # Attempt with structured JSON output first, then plain text fallback
            for use_json_mode in (True, False):
                if use_json_mode:
                    config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.2,
                    )
                else:
                    # Plain text mode: sometimes works when JSON mode quota is exhausted
                    config = types.GenerateContentConfig(
                        system_instruction=system_prompt + "\n\nIMPORTANT: Your entire response must be valid JSON only. No markdown, no explanations.",
                        temperature=0.2,
                    )

                # Retry this model+mode up to 3 times with exponential backoff
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model=model,
                            contents=contents,
                            config=config,
                        )
                        raw_text = response.text or ""
                        return self._extract_json(raw_text)

                    except Exception as e:
                        last_error = e
                        err_msg = str(e)

                        # Fatal auth error — stop immediately
                        if ("API_KEY_INVALID" in err_msg or "401" in err_msg
                                or "PERMISSION_DENIED" in err_msg):
                            raise LLMAuthError(f"Gemini API Key rejected: {err_msg}")

                        is_transient = (
                            "503" in err_msg or "UNAVAILABLE" in err_msg
                            or "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
                            or "overload" in err_msg.lower()
                        )
                        is_not_found = "404" in err_msg or "NOT_FOUND" in err_msg

                        if is_transient and attempt < 2:
                            wait = _BACKOFF_SECONDS[attempt]
                            time.sleep(wait)
                            continue
                        elif is_transient or is_not_found:
                            # Exhausted retries or model missing — try next model/mode
                            break
                        else:
                            # Non-transient, non-auth error — skip to next mode/model
                            break

        # All candidates and modes exhausted
        raise LLMExecutionError(
            f"All candidate Gemini models failed ({', '.join(candidates)}). "
            f"Last error: {str(last_error)}"
        )

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        grounding_images: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            target_model = model_name or "gpt-4o"

            messages = [{"role": "system", "content": system_prompt}]

            user_content = []
            if grounding_images:
                for img in grounding_images:
                    mime = img.get("mime_type", "image/jpeg")
                    b64 = img["base64"]
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}
                    })

            user_content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": user_content})

            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            raw_text = response.choices[0].message.content or ""
            return self._extract_json(raw_text)

        except Exception as e:
            err_msg = str(e)
            if "invalid_api_key" in err_msg or "401" in err_msg:
                raise LLMAuthError(f"OpenAI API Key rejected: {err_msg}")
            raise LLMExecutionError(f"OpenAI generation error: {err_msg}")

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Strips markdown wrappers and parses response to JSON."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as jde:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            raise LLMExecutionError(
                f"Failed to parse LLM response as JSON: {str(jde)}\n"
                f"Response preview: {text[:400]}"
            )
