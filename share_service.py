"""
share_service.py
────────────────────────────────────────────────────────────────────────────────
Sends a POST request to an n8n webhook when the user shares a filter.

Payload
-------
{
    "filter_name": "...",
    "code":        "...",
    "preview":     "<base64-encoded JPEG or empty string>"
}

Configure the webhook URL via the SHARE_WEBHOOK_URL environment variable,
or pass it directly to ShareService().
"""

import json
import os
import urllib.request
import urllib.error
from typing import Tuple

import config  # noqa: F401


DEFAULT_WEBHOOK_URL = os.environ.get(
    "SHARE_WEBHOOK_URL",
    "http://localhost:5678/webhook/share-filter",  # default n8n local instance
)

TIMEOUT_SEC = 10


class ShareService:
    """Posts filter metadata to an n8n webhook."""

    def __init__(self, webhook_url: str = DEFAULT_WEBHOOK_URL) -> None:
        self.webhook_url = webhook_url

    def share(
        self,
        filter_name: str,
        code: str,
        preview_b64: str = "",
    ) -> Tuple[bool, str]:
        """
        Send share payload to the webhook.

        Returns (True, success_msg) or (False, error_msg).
        """
        payload = json.dumps(
            {
                "filter_name": filter_name,
                "code":        code,
                "preview":     preview_b64,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                status = resp.status
                body   = resp.read().decode("utf-8", errors="replace")[:200]
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", str(e))
            return False, f"❌ Webhook error: {reason}"
        except Exception as e:
            return False, f"❌ Unexpected share error: {e}"

        if 200 <= status < 300:
            return True, f"✅ Filter '{filter_name}' shared successfully! (HTTP {status})"
        else:
            return False, f"❌ Webhook returned HTTP {status}: {body}"
