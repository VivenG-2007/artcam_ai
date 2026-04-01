"""
validator.py
────────────────────────────────────────────────────────────────────────────────
AST-based security validator for AI-generated filter code.

Pipeline:
  1. Keyword (string) blocking
  2. AST parsing
  3. Structural checks  (exactly one function named apply_filter)
  4. Dangerous node detection
  5. Import whitelist enforcement
"""

import ast
from typing import Tuple

# ── 1. Keyword blocklist ──────────────────────────────────────────────────────
BLOCKED_KEYWORDS = [
    "os", "sys", "subprocess", "eval", "exec",
    "open", "socket", "requests", "__import__",
    "urllib", "pathlib", "importlib",
    "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input",
    "print",  # no stdout pollution in filter functions
]

# ── 2. Dangerous AST node types ───────────────────────────────────────────────
DANGEROUS_NODES = (
    ast.Import,        # only ast.ImportFrom with whitelist is ok
    ast.ClassDef,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.While,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)

# ── 3. Whitelisted import modules ─────────────────────────────────────────────
ALLOWED_IMPORTS = {"cv2", "numpy"}   # "import numpy as np" → module = numpy


class FilterValidator:
    """Validates AI-generated filter code before compilation."""

    # ── Public interface ──────────────────────────────────────────────────────
    def validate(self, code: str) -> Tuple[bool, str]:
        """
        Returns (True, "ok") if code is safe, or (False, reason) if not.
        """
        # Step 1 – keyword scan
        ok, msg = self._check_keywords(code)
        if not ok:
            return False, msg

        # Step 2 – parse
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Step 3 – structural check
        ok, msg = self._check_structure(tree)
        if not ok:
            return False, msg

        # Step 4 – dangerous nodes
        ok, msg = self._check_dangerous_nodes(tree)
        if not ok:
            return False, msg

        # Step 5 – import whitelist
        ok, msg = self._check_imports(tree)
        if not ok:
            return False, msg

        return True, "ok"

    # ── Internal checks ───────────────────────────────────────────────────────
    def _check_keywords(self, code: str) -> Tuple[bool, str]:
        for kw in BLOCKED_KEYWORDS:
            # word-boundary check: "os" should not match "numpy"
            import re
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, code):
                return False, f"Blocked keyword detected: '{kw}'"
        return True, "ok"

    def _check_structure(self, tree: ast.AST) -> Tuple[bool, str]:
        functions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        if len(functions) == 0:
            return False, "Code must define exactly one function named 'apply_filter'."
        if len(functions) > 1:
            return False, f"Only one function allowed, found {len(functions)}."
        if functions[0].name != "apply_filter":
            return False, f"Function must be named 'apply_filter', got '{functions[0].name}'."

        # Must have at least 'frame' parameter
        args = [a.arg for a in functions[0].args.args]
        if "frame" not in args:
            return False, "apply_filter must accept a 'frame' parameter."

        return True, "ok"

    def _check_dangerous_nodes(self, tree: ast.AST) -> Tuple[bool, str]:
        for node in ast.walk(tree):
            if isinstance(node, DANGEROUS_NODES):
                return False, f"Dangerous AST node detected: {type(node).__name__}"

            # Block attribute access to __dunder__ names (e.g. frame.__class__)
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return False, f"Dunder attribute access blocked: '{node.attr}'"

            # Block calls to builtins by name
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_KEYWORDS:
                        return False, f"Blocked function call: '{node.func.id}'"

        return True, "ok"

    def _check_imports(self, tree: ast.AST) -> Tuple[bool, str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in ALLOWED_IMPORTS:
                        return False, f"Unauthorized import: '{alias.name}'"

            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"Unauthorized import from: '{node.module}'"

        return True, "ok"
