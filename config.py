"""
Shared configuration bootstrap.

Loads environment variables from a local .env file when present.
"""

from __future__ import annotations

from dotenv import load_dotenv


load_dotenv()
