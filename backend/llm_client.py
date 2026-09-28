""" LLM Multi-Provider Client 
=========================
  Supported providers: 
 - Google Gemini 
 - OpenAI 
 - Groq (GPT-OSS / Qwen chat models) 
  
 Architecture: 
 Scene Generator:
   Gemini / OpenAI 
 Critic Agent:
   Groq (independent of the scene generator's provider)


 Features:
  - Strict API key validation
  - JSON-only responses
  - Gemini model fallback
  - Groq model discovery with runtime retirement handling
  - Exponential backoff for transient errors
  - OpenAI multimodal image support
  - JSON extraction from markdown-wrapped responses
  - No placeholder/template fallbacks """


from openai import OpenAI
import json
import re
import base64
import time
from typing import List, Dict, Any, Optional
import os

from backend.models import ProviderType

# ============================================================ 
# Exceptions 
# ============================================================

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

# ============================================================ 
# Groq configuration
# ============================================================ 
GROQ_BASE_URL = "https://api.groq.com/openai/v1" 

# NOTE (Sept 2026): Groq has retired every Llama model ID.
#   llama-3.3-70b-versatile            -> shut down 08/16/26, see openai/gpt-oss-120b
#   llama-3.1-8b-instant              -> shut down 08/16/26, see openai/gpt-oss-20b
#   meta-llama/llama-4-scout-17b-16e  -> shut down 07/17/26, see openai/gpt-oss-120b
#   meta-llama/llama-4-maverick-17b   -> shut down 03/09/26, see openai/gpt-oss-120b
# Requesting a retired ID returns HTTP 404, so the critic defaults to Groq's
# recommended replacement and discovers the live list at runtime as a safety net.

# Default critic model. Groq's official successor to llama-3.3-70b-versatile.
# Override per-request via model_name, or globally via the GROQ_MODEL env var.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Ordered by audit quality, used when the requested model is unavailable.
GROQ_CANDIDATE_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

# Substrings identifying models that cannot serve a chat completion, so runtime
# discovery never picks up speech-to-text, moderation, or guard models.
_GROQ_NON_CHAT_MARKERS = (
    "whisper",
    "guard",
    "safeguard",
    "orpheus",
    "tts",
    "stt",
    "vision",
    "compound",
    "prompt",
)

# How long a discovered model list stays fresh before re-querying /models.
_GROQ_MODEL_CACHE_TTL_SECONDS = 900

# Environment variable consulted when no Groq key is supplied per-request.
GROQ_API_KEY_ENV = "GROQ_API_KEY"

# Environment variable that overrides DEFAULT_GROQ_MODEL for the critic.
GROQ_MODEL_ENV = "GROQ_MODEL"

# key -> (model_ids, fetched_at_monotonic)
_GROQ_MODEL_CACHE: Dict[str, Any] = {}


def resolve_groq_api_key(api_key: Optional[str] = None) -> str:
    """Returns the per-request Groq key, falling back to the GROQ_API_KEY env var."""
    return (api_key or os.getenv(GROQ_API_KEY_ENV) or "").strip()


def resolve_groq_model(model_name: Optional[str] = None) -> str:
    """Returns the per-request model, falling back to GROQ_MODEL then the default."""
    return (model_name or os.getenv(GROQ_MODEL_ENV) or DEFAULT_GROQ_MODEL).strip()


def filter_chat_models(model_ids: List[str]) -> List[str]:
    """Keeps only chat-capable model IDs, preserving the order Groq returned."""
    return [
        m for m in model_ids
        if not any(marker in m.lower() for marker in _GROQ_NON_CHAT_MARKERS)
    ]


def discover_groq_models(
    api_key: str,
    use_cache: bool = True,
    ttl_seconds: int = _GROQ_MODEL_CACHE_TTL_SECONDS
) -> List[str]:
    """
    Queries Groq's /models endpoint for the account's live chat-capable models.

    Groq retires model IDs on a schedule, so a hardcoded list is guaranteed to
    rot. Results are cached briefly because the critic may loop up to 3 times
    per request and each call would otherwise add a round trip.
    """
    if use_cache:
        cached = _GROQ_MODEL_CACHE.get(api_key)
        if cached:
            model_ids, fetched_at = cached
            if (time.monotonic() - fetched_at) < ttl_seconds:
                return model_ids

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    try:
        model_ids = filter_chat_models([m.id for m in client.models.list()])
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "invalid_api_key" in err_msg.lower() or "authentication" in err_msg.lower():
            raise LLMAuthError(f"Groq API Key rejected: {err_msg}")
        raise LLMExecutionError(f"Failed to list Groq models: {err_msg}")

    _GROQ_MODEL_CACHE[api_key] = (model_ids, time.monotonic())
    return model_ids


def _build_groq_candidates(api_key: str, primary_model: str) -> List[str]:
    """
    Orders candidates as: requested model, curated fallbacks, then whatever the
    account can actually reach. The live list is appended so a retirement only
    degrades us to a slower model instead of a hard failure.
    """
    candidates: List[str] = [primary_model]
    for m in GROQ_CANDIDATE_MODELS:
        if m not in candidates:
            candidates.append(m)

    try:
        for m in discover_groq_models(api_key):
            if m not in candidates:
                candidates.append(m)
    except LLMExecutionError:
        # Discovery is a safety net, not a requirement -- curated list still applies.
        pass

    return candidates



class LLMClient:
    def __init__(self, provider: ProviderType, api_key: str):
        if not api_key or not api_key.strip():
            raise LLMAuthError(
                "API Key is missing! You must paste a valid Gemini, OpenAI, or Groq API key. "
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
        elif self.provider == ProviderType.GROQ:
            return self._call_groq(system_prompt, user_prompt, grounding_images, model_name)
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
    

# ========================================================
# Groq
# ========================================================

   
    def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        grounding_images: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call a chat model hosted through Groq (default: openai/gpt-oss-120b).

        Groq provides an OpenAI-compatible API, so the
        OpenAI Python client can be used with Groq's endpoint.
        """

        if grounding_images:
            raise LLMExecutionError(
                "Groq does not accept image inputs. "
                "Grounding photos must be processed by the Gemini/OpenAI scene generator."
            )

        primary_model = resolve_groq_model(model_name)

        if not primary_model:
            raise LLMExecutionError(
                "No Groq model name was provided."
            )

        # Requested model first, then curated fallbacks, then the live account list
        candidates = _build_groq_candidates(self.api_key, primary_model)
        client = OpenAI(
            api_key=self.api_key,
            base_url=GROQ_BASE_URL
        )
        last_error = None

        for target_model in candidates:
            # Attempt with structured JSON output first, then plain text fallback
            for use_json_mode in (True, False):
                for attempt in range(3):
                    try:
                        if use_json_mode:
                            response_format = {"type": "json_object"}
                            messages = [
                                {
                                    "role": "system",
                                    "content": system_prompt
                                },
                                {
                                    "role": "user",
                                    "content": user_prompt
                                }
                            ]
                        else:
                            # Plain text mode: Groq JSON mode can fail on
                            # long/unicode-heavy payloads such as Bengali text.
                            response_format = None
                            messages = [
                                {
                                    "role": "system",
                                    "content": system_prompt
                                        + "\n\nIMPORTANT: Your entire response must be valid JSON only. "
                                        "No markdown, no explanations."
                                },
                                {
                                    "role": "user",
                                    "content": user_prompt
                                }
                            ]

                        response = client.chat.completions.create(
                            model=target_model,
                            messages=messages,
                            response_format=response_format,
                            temperature=0.2
                        )

                        raw_text = (
                            response.choices[0].message.content
                            or ""
                        )

                        return self._extract_json(raw_text)

                    except Exception as e:
                        last_error = e
                        err_msg = str(e)

                        # Fatal auth error — stop immediately
                        if (
                            "401" in err_msg
                            or "invalid_api_key" in err_msg.lower()
                            or "authentication" in err_msg.lower()
                        ):
                            raise LLMAuthError(
                                f"Groq API Key rejected: {err_msg}"
                            )

                        is_transient = (
                            "429" in err_msg
                            or "500" in err_msg
                            or "502" in err_msg
                            or "503" in err_msg
                            or "rate limit" in err_msg.lower()
                            or "overloaded" in err_msg.lower()
                        )
                        # 404 = retired/unknown model ID, 403 = model not enabled
                        # for this key. Both mean "try the next candidate".
                        is_unavailable = (
                            "404" in err_msg
                            or "403" in err_msg
                            or "model_not_found" in err_msg.lower()
                            or "decommissioned" in err_msg.lower()
                            or "not found" in err_msg.lower()
                        )

                        if is_transient and attempt < 2:
                            wait = _BACKOFF_SECONDS[attempt]
                            time.sleep(wait)
                            continue
                        elif is_transient or is_unavailable:
                            # Exhausted retries or model missing — try next model/mode
                            break
                        else:
                            # Non-transient, non-auth error — skip to next mode/model
                            break

        # All candidates and modes exhausted
        raise LLMExecutionError(
            f"All candidate Groq models failed. Tried: {', '.join(candidates)}. "
            f"Last error: {str(last_error)}. "
            f"Groq retires model IDs periodically -- re-run the 'Verify Key' button in the "
            f"Critic section to see the models currently available on your account."
        )
                
  