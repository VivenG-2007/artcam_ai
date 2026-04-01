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
    "deepseek/deepseek-coder",
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
- landmarks is a tuple (x, y, w, h) representing the bounding box of a detected face, or None if no face is visible. It is NOT an array of points.
- Keep it concise, deterministic, and safe for real-time execution.
- Preserve the input frame shape.
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
    ) -> GenerationResult:
        groq_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
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

        if fast_code:
            ok, message = self._validate_and_test(fast_code)
            fast_attempt.validation_message = message
            if ok:
                fast_attempt.success = True
                return self._success_result(
                    session_id, prompt, fast_code,
                    "Generated with Groq fast draft and passed validation.",
                    "groq_fast_draft", attempts,
                )

        # ── Stage 2: OpenRouter / DeepSeek repair ─────────────────────────
        repair_context = self._failure_context(prompt, attempts)

        if openrouter_key:
            repair_code, repair_attempt = self._openrouter_stage(
                stage="deepseek_repair",
                model=DEFAULT_DEEPSEEK_MODEL,
                api_key=openrouter_key,
                system_prompt=STRICT_SYSTEM_PROMPT,
                user_prompt=repair_context,
            )
            attempts.append(repair_attempt)

            if repair_code:
                ok, message = self._validate_and_test(repair_code)
                repair_attempt.validation_message = message
                if ok:
                    repair_attempt.success = True
                    return self._success_result(
                        session_id, prompt, repair_code,
                        "DeepSeek repair passed validation after Groq failure.",
                        "deepseek_repair", attempts,
                    )
        else:
            attempts.append(GenerationAttempt(
                stage="deepseek_repair", provider="openrouter",
                model=DEFAULT_DEEPSEEK_MODEL, success=False,
                error="Skipped: OPENROUTER_API_KEY not provided.",
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
        self,
        stage: str,
        model: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, GenerationAttempt]:
        started = time.perf_counter()
        attempt = GenerationAttempt(stage=stage, provider="openrouter", model=model, success=False)
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
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
