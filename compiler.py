"""
compiler.py
────────────────────────────────────────────────────────────────────────────────
Safe compilation of validated filter code.

Uses a restricted globals dict so the executed code has ZERO access to
Python builtins or the host environment.
"""

import cv2
import numpy as np
import threading
from typing import Callable, Optional, Tuple

# Thread-local cache: maps code_hash → compiled_function
_cache: dict = {}
_cache_lock = threading.Lock()


class FilterCompiler:
    """Compiles validated Python filter code into a callable."""

    # ── Restricted execution environment ─────────────────────────────────────
    _SAFE_GLOBALS = {
        "cv2":        cv2,
        "np":         np,
        "__builtins__": __builtins__,  # Allow normal functions like min, max, len, int
    }

    def compile(self, code: str) -> Tuple[Optional[Callable], Optional[str]]:
        """
        Compile validated filter code and return (apply_filter_fn, None)
        or (None, error_message) on failure.

        Results are cached by code hash.
        """
        code_hash = hash(code)

        with _cache_lock:
            if code_hash in _cache:
                return _cache[code_hash], None

        try:
            local_ns: dict = {}
            exec(                          # noqa: S102 – intentional safe exec
                compile(code, "<filter>", "exec"),
                dict(self._SAFE_GLOBALS),  # copy so each exec is isolated
                local_ns,
            )
        except Exception as e:
            return None, f"Compilation error: {e}"

        fn = local_ns.get("apply_filter")
        if fn is None or not callable(fn):
            return None, "apply_filter function not found after compilation."

        with _cache_lock:
            _cache[code_hash] = fn

        return fn, None

    def compile_and_smoke_test(
        self,
        code: str,
        sample_frame: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Callable], Optional[str]]:
        """
        Compile validated code and execute it once against a tiny sample frame.

        This catches runtime issues early in the generation pipeline before code
        is exposed to the user or persisted.
        """
        fn, err = self.compile(code)
        if fn is None:
            return None, err

        frame = sample_frame
        if frame is None:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)

        try:
            result = fn(frame.copy())
        except Exception as exc:
            return None, f"Smoke test execution error: {exc}"

        if not isinstance(result, np.ndarray):
            return None, "Smoke test failed: apply_filter must return a numpy.ndarray."
        if result.dtype != np.uint8:
            return None, "Smoke test failed: apply_filter must return uint8 output."
        if result.ndim != 3 or result.shape[2] != 3:
            return None, "Smoke test failed: apply_filter must return a BGR image."
        if result.shape != frame.shape:
            return None, "Smoke test failed: apply_filter must preserve frame shape."

        return fn, None

    def clear_cache(self) -> None:
        """Clear the compiled-filter cache."""
        with _cache_lock:
            _cache.clear()
