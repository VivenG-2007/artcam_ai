"""
Multi-stage AI filter generation pipeline.

Pipeline:
  1. Groq / LLaMA 3 fast draft        (uses official groq SDK)
  2. Validator + compiler smoke test
  3. OpenRouter / DeepSeek repair      (uses openai SDK pointed at OpenRouter)
  4. Validator + compiler smoke test again
  5. Groq / LLaMA 3 strict retry as final model fallback

Using the official SDKs (groq + openai) instead of raw urllib so that
Cloudflare never blocks the requests (proper TLS fingerprint + User-Agent).
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import config  # noqa: F401  – loads .env
from compiler import FilterCompiler
from validator import FilterValidator


DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
DEFAULT_DEEPSEEK_MODEL = os.environ.get(
    "OPENROUTER_DEEPSEEK_MODEL",
    "deepseek/deepseek-chat", # Maps to DeepSeek V3 via OpenRouter
)
DEFAULT_STRICT_GROQ_MODEL = os.environ.get("GROQ_STRICT_MODEL", DEFAULT_GROQ_MODEL)

DEFAULT_APP_URL = os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost:7860")
DEFAULT_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "AI Filter Platform")

BASE_SYSTEM_PROMPT = """You are an expert OpenCV Python filter writer.
The user will describe a visual filter.

You MUST respond with EXACTLY this function signature and NOTHING else
(no explanations, no markdown, no backticks, no comments outside the function):

def apply_filter(frame, landmarks=None):
    # your code here
    return frame

Rules:
- Only import/use cv2 and numpy (as np). Both are pre-imported.
- No os, sys, subprocess, eval, exec, open, socket, requests, urllib, pathlib, or any I/O.
- frame is a BGR uint8 numpy array. Always return a BGR uint8 numpy array.
- landmarks is a dict of exact 2D pixel coordinates, or None if no face is seen. Keys: "nose_tip", "left_eye", "right_eye", "mouth_center", "chin", "left_cheek", "right_cheek", "forehead", "face_bounding_box" (x, y, w, h), and "mesh". Example: cx, cy = landmarks["left_eye"].
- DO NOT draw crude, solid geometric shapes (like giant black rectangles or circles) to simulate objects like sunglasses or hats. You do not have image assets. Instead, use clever OpenCV array math to algorithmicly create effects (like green tinting, pixelation matrices, blurs, glow, or edge detection overlays) around the landmarks!
- For cinematic color grading (like a Matrix green tint), split the BGR channels (`b, g, r = cv2.split(frame)`), mentally scale them using standard NumPy float operations (e.g. `b = b * 0.2`), then `cv2.merge` back. NEVER use `cv2.multiply(array, float)` or `cv2.add` with scalar floats because it crashes OpenCV shape matching! Always use native numpy `*` or `+`, followed by `np.clip(..., 0, 255).astype(np.uint8)`.
- CRITICAL BITWISE FIX: Never do `cv2.bitwise_and(frame, mask)` if mask is 1-channel and frame is 3-channel. Use `cv2.bitwise_and(frame, frame, mask=mask)`!
- CRITICAL: Never use list indexing on landmarks (like `landmarks[1]`). ALWAYS use the string keys provided (like `landmarks["nose_tip"]` or `landmarks["left_eye"]`).
- CRITICAL TYPO GUARD: Ensure you spell it `landmarks` (with a "d"). Never use `landscape` or `landmark`!
- Prefer defensive numpy/OpenCV code that works on arbitrary frame sizes.
- CRITICAL: Avoid np.random unless strictly cast to (H, W, 3) uint8. OpenCV requires exactly matching shapes and dtype=np.uint8 for addWeighted and binary ops.
- CRITICAL UNPACKING FIX: If you need frame dimensions, ALWAYS use `h, w = frame.shape[:2]`. Doing `h, w = frame.shape` will crash because it has 3 channels! 
- CRITICAL: `face_bounding_box` is a 4-tuple `(x, y, w, h)`. NEVER try to unpack it into only 2 variables. Always use `fx, fy, fw, fh = landmarks["face_bounding_box"]`.
- CRITICAL BROADCAST CRASH FIX: You are STRICTLY FORBIDDEN from assigning local patches directly into `frame` slices (e.g., `frame[y1:y2, x1:x2] = patch`) because edge-clipping will throw broadcasting shape mismatch exceptions!
- INSTEAD, use this exact pattern to draw overlays or localized effects seamlessly:
  ```python
  mask = np.zeros(frame.shape[:2], dtype=np.uint8)
  cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1) # define region
  effect = cv2.GaussianBlur(frame, (21, 21), 0) # compute your effect
  # CRITICAL: Spread mask across 3 channels using [:,:,None] for broadcasting!
  frame = np.where(mask[:, :, None] > 0, effect, frame) 
  ```
- Keep all operations heavily mathematically vectorized for speed. Avoid deep nested Python FOR loops (which drop the framerate incredibly).
"""

STRICT_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """
- Do not use helper functions, classes, lambdas, or extra top-level definitions.
- Do not add markdown fences or prose.
- Avoid fragile operations that can raise on empty masks or shape mismatches.
- Prefer defensive numpy/OpenCV code that works on arbitrary frame sizes.
"""


@dataclass
class GenerationAttempt:
    stage: str
    provider: str
    model: str
    success: bool
    code: str = ""
    error: str = ""
    validation_message: str = ""
    latency_ms: int = 0


@dataclass
class GenerationResult:
    session_id: str
    prompt: str
    success: bool
    code: str
    message: str
    final_stage: str
    attempts: List[GenerationAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["attempts"] = [asdict(a) for a in self.attempts]
        return payload


class AIFilterGenerator:
    """Generates filter code through a staged repair pipeline."""

    def __init__(
        self,
        validator: Optional[FilterValidator] = None,
        compiler: Optional[FilterCompiler] = None,
    ) -> None:
        self._validator = validator or FilterValidator()
        self._compiler = compiler or FilterCompiler()

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        groq_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ) -> GenerationResult:
        groq_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "AIzaSyBiylbwgBsORKTChUJ9sZISPDaqYhOspYc")
        session_id = str(uuid.uuid4())
        attempts: List[GenerationAttempt] = []

        if not prompt.strip():
            return GenerationResult(
                session_id=session_id, prompt=prompt, success=False,
                code="", message="Prompt is required.", final_stage="input_validation",
                attempts=attempts,
            )

        if not groq_key:
            return GenerationResult(
                session_id=session_id, prompt=prompt, success=False,
                code="", message="Groq API key is required via input or GROQ_API_KEY env.",
                final_stage="input_validation", attempts=attempts,
            )

        # ── Stage 1: Groq fast draft ───────────────────────────────────────
        fast_code, fast_attempt = self._groq_stage(
            stage="groq_fast_draft",
            model=DEFAULT_GROQ_MODEL,
            api_key=groq_key,
            system_prompt=BASE_SYSTEM_PROMPT,
            user_prompt=f"Create a filter that: {prompt}",
        )
        attempts.append(fast_attempt)

        ok, message = False, ""
        if fast_code:
            ok, message = self._validate_and_test(fast_code)
            fast_attempt.validation_message = message
            if ok:
                fast_attempt.success = True
                # We intentionally DO NOT return here anymore! We pass to DeepSeek V3.

        # ── Stage 2: Gemini or DeepSeek V3 Validation & Enhancement ────
        if gemini_key or openrouter_key:
            if ok:
                # Groq code passed validation, ask Refiner to visibly improve/review it
                repair_context = (
                    f"User request:\n{prompt}\n\n"
                    f"Here is a draft filter that technically passes syntax validation:\n{fast_code}\n\n"
                    "Act as an expert OpenCV developer. Review this filter for visual cinematic quality, "
                    "color grading, mathematical correctness, and algorithmic aesthetics. "
                    "Make necessary improvements to make it look professional (like real cinematic grades or effects). "
                    "Ensure you DO NOT violate the strict execution environment rules.\n"
                    "Return ONLY the updated def apply_filter(frame, landmarks=None): function code."
                )
            else:
                # Groq code was broken, normal repair
                repair_context = self._failure_context(prompt, attempts)

            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/" if gemini_key else "https://openrouter.ai/api/v1"
            use_key = gemini_key if gemini_key else openrouter_key
            use_model = "gemini-2.0-flash" if gemini_key else DEFAULT_DEEPSEEK_MODEL
            use_stage = "gemini_review_and_repair" if gemini_key else "deepseek_review_and_repair"

            repair_code, repair_attempt = self._openrouter_stage(
                stage=use_stage,
                model=use_model,
                api_key=use_key,
                base_url=base_url,
                system_prompt=STRICT_SYSTEM_PROMPT,
                user_prompt=repair_context,
            )
            attempts.append(repair_attempt)

            if repair_code:
                ok2, message2 = self._validate_and_test(repair_code)
                repair_attempt.validation_message = message2
                if ok2:
                    repair_attempt.success = True
                    return self._success_result(
                        session_id, prompt, repair_code,
                        f"Generated by Groq and expertly refined by {use_model}.",
                        use_stage, attempts,
                    )
                elif ok:
                    # Enhancement failed syntax, but Groq worked! Fallback safely.
                    return self._success_result(
                        session_id, prompt, fast_code,
                        f"{use_model} enhancement failed validation. Fell back to working Groq draft.",
                        "groq_fast_draft", attempts,
                    )
        else:
            if ok:
                # No keys for refinement, but Groq worked. Return Groq.
                return self._success_result(
                    session_id, prompt, fast_code,
                    "Generated with Groq fast draft (Enhancement skipped: no Gemini/OpenRouter key).",
                    "groq_fast_draft", attempts,
                )
            attempts.append(GenerationAttempt(
                stage="gemini_review_and_repair", provider="openai",
                model="gemini-2.0-flash", success=False,
                error="Skipped: Neither GEMINI_API_KEY nor OPENROUTER_API_KEY provided.",
            ))

        # ── Stage 3: Groq strict retry ─────────────────────────────────────
        strict_prompt = self._strict_retry_prompt(prompt, attempts)
        strict_code, strict_attempt = self._groq_stage(
            stage="groq_strict_retry",
            model=DEFAULT_STRICT_GROQ_MODEL,
            api_key=groq_key,
            system_prompt=STRICT_SYSTEM_PROMPT,
            user_prompt=strict_prompt,
        )
        attempts.append(strict_attempt)

        if strict_code:
            ok, message = self._validate_and_test(strict_code)
            strict_attempt.validation_message = message
            if ok:
                strict_attempt.success = True
                return self._success_result(
                    session_id, prompt, strict_code,
                    "Groq strict retry passed validation.",
                    "groq_strict_retry", attempts,
                )

        final_message = self._build_failure_message(attempts)
        final_code = self._last_non_empty_code(attempts)
        return GenerationResult(
            session_id=session_id, prompt=prompt, success=False,
            code=final_code, message=final_message,
            final_stage="failed", attempts=attempts,
        )

    # ── SDK call helpers ───────────────────────────────────────────────────────

    def _groq_stage(
        self,
        stage: str,
        model: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, GenerationAttempt]:
        started = time.perf_counter()
        attempt = GenerationAttempt(stage=stage, provider="groq", model=model, success=False)
        try:
            from groq import Groq  # type: ignore
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise RuntimeError("Groq returned empty content.")
            code = self._strip_code_fences(raw)
            attempt.code = code
        except Exception as exc:
            attempt.error = str(exc)
            code = ""

        attempt.latency_ms = int((time.perf_counter() - started) * 1000)
        return code, attempt

    def _openrouter_stage(
        self, stage: str, model: str, api_key: str, base_url: str,
        system_prompt: str, user_prompt: str,
    ) -> Tuple[Optional[str], GenerationAttempt]:
        started = time.perf_counter()
        attempt = GenerationAttempt(stage=stage, provider="openai_compatible", model=model, success=False)
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": DEFAULT_APP_URL,
                    "X-Title": DEFAULT_APP_NAME,
                },
            )
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise RuntimeError("OpenRouter returned empty content.")
            code = self._strip_code_fences(raw)
            attempt.code = code
        except Exception as exc:
            attempt.error = str(exc)
            code = ""

        attempt.latency_ms = int((time.perf_counter() - started) * 1000)
        return code, attempt

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_and_test(self, code: str) -> tuple[bool, str]:
        ok, message = self._validator.validate(code)
        if not ok:
            return False, message
        _, compile_error = self._compiler.compile_and_smoke_test(code)
        if compile_error:
            return False, compile_error
        return True, "ok"

    # ── Prompt builders ────────────────────────────────────────────────────────

    def _failure_context(self, prompt: str, attempts: List[GenerationAttempt]) -> str:
        latest_code = self._last_non_empty_code(attempts)
        latest_issue = self._latest_issue(attempts)
        return (
            f"Original user request:\n{prompt}\n\n"
            f"The previous draft failed validation or smoke testing.\n"
            f"Failure details:\n{latest_issue}\n\n"
            f"Previous code:\n{latest_code}\n\n"
            "Return a corrected apply_filter(frame, landmarks=None) function only."
        )

    def _strict_retry_prompt(self, prompt: str, attempts: List[GenerationAttempt]) -> str:
        latest_issue = self._latest_issue(attempts)
        return (
            f"User request:\n{prompt}\n\n"
            f"Previous attempts failed. You must avoid this issue:\n{latest_issue}\n\n"
            "Write a minimal, robust OpenCV filter implementation.\n"
            "Return only def apply_filter(frame, landmarks=None): ..."
        )

    # ── Static utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        text = code.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()
        return text

    @staticmethod
    def _last_non_empty_code(attempts: List[GenerationAttempt]) -> str:
        for attempt in reversed(attempts):
            if attempt.code.strip():
                return attempt.code
        return ""

    @staticmethod
    def _build_failure_message(attempts: List[GenerationAttempt]) -> str:
        for attempt in reversed(attempts):
            if attempt.validation_message:
                return (
                    "Generation failed after retries. "
                    f"Last validation error: {attempt.validation_message}"
                )
            if attempt.error:
                return (
                    "Generation failed after retries. "
                    f"Last error: {attempt.error}"
                )
        return "Generation failed after retries."

    @staticmethod
    def _latest_issue(attempts: List[GenerationAttempt]) -> str:
        for attempt in reversed(attempts):
            if attempt.validation_message:
                return attempt.validation_message
            if attempt.error:
                return attempt.error
        return "Unknown failure."

    @staticmethod
    def _success_result(
        session_id: str,
        prompt: str,
        code: str,
        message: str,
        final_stage: str,
        attempts: List[GenerationAttempt],
    ) -> GenerationResult:
        return GenerationResult(
            session_id=session_id, prompt=prompt, success=True,
            code=code, message=message, final_stage=final_stage,
            attempts=attempts,
        )
