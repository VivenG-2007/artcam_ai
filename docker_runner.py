"""
docker_runner.py
────────────────────────────────────────────────────────────────────────────────
Sandboxed image-mode filter execution via Docker.

Creates a temporary Docker container with:
  • No network access (--network none)
  • Memory limit (--memory 256m)
  • CPU limit (--cpus 0.5)
  • 5-second execution timeout
  • Read-only host filesystem mount for the script
"""

import os
import json
import base64
import tempfile
import subprocess
import textwrap
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


DOCKER_IMAGE = "python:3.11-slim"   # must have cv2 + numpy; see README for build instructions
TIMEOUT_SEC  = 5
MEMORY_LIMIT = "256m"
CPU_LIMIT    = "0.5"


class DockerRunner:
    """Run a validated filter function inside a Docker sandbox."""

    def run(
        self,
        code: str,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Execute *code* on *image* inside Docker.

        Returns (result_frame, None) or (None, error_message).
        """
        if not self._docker_available():
            return None, "Docker is not available on this system."

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Write input image
            input_path = tmp / "input.png"
            cv2.imwrite(str(input_path), image)

            # 2. Write the runner script
            runner_script = self._build_runner(code)
            script_path = tmp / "runner.py"
            script_path.write_text(runner_script)

            output_path = tmp / "output.png"

            # 3. Build docker command
            cmd = [
                "docker", "run",
                "--rm",
                "--network",  "none",
                "--memory",   MEMORY_LIMIT,
                "--cpus",     CPU_LIMIT,
                "--read-only",
                "--tmpfs",    "/tmp:rw,size=64m",
                "-v", f"{tmpdir}:/workspace:ro",
                "-v", f"{str(output_path.parent)}:/out",
                DOCKER_IMAGE,
                "python", "/workspace/runner.py",
                "/workspace/input.png",
                "/out/output.png",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=TIMEOUT_SEC,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                return None, f"Docker execution timed out after {TIMEOUT_SEC}s."
            except FileNotFoundError:
                return None, "Docker binary not found. Is Docker installed?"

            if result.returncode != 0:
                err = (result.stderr or result.stdout or "Unknown error").strip()
                return None, f"Docker error (exit {result.returncode}): {err[:200]}"

            # 4. Read output
            if not output_path.exists():
                return None, "Docker ran but produced no output image."

            out_frame = cv2.imread(str(output_path))
            if out_frame is None:
                return None, "Could not read Docker output image."

            return out_frame, None

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _docker_available() -> bool:
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=5,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _build_runner(filter_code: str) -> str:
        """Generate the Python script that Docker will execute."""
        safe_code = filter_code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return textwrap.dedent(f"""
import sys
import cv2
import numpy as np

FILTER_CODE = \"\"\"{safe_code}\"\"\"

def main():
    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    frame = cv2.imread(input_path)
    if frame is None:
        print("Cannot read input image", file=sys.stderr)
        sys.exit(1)

    local_ns = {{}}
    exec(
        compile(FILTER_CODE, "<filter>", "exec"),
        {{"cv2": cv2, "np": np, "__builtins__": {{}}}},
        local_ns,
    )

    apply_filter = local_ns.get("apply_filter")
    if apply_filter is None:
        print("apply_filter not found", file=sys.stderr)
        sys.exit(1)

    result = apply_filter(frame)
    cv2.imwrite(output_path, result)

if __name__ == "__main__":
    main()
""").strip()
