"""HMAC-SHA256 token authentication for WorldForge inference endpoints.

Per spec §4.7 + Global Constraint G12:
- HMAC over (method + path + body + timestamp) with shared secret
- Reject stale timestamps (> max_skew_seconds)
- Phase-2: upgrade to mTLS for production
"""
from __future__ import annotations

import hashlib
import hmac
import time


class HmacTokenAuth:
    def __init__(self, secret: str, max_skew_seconds: int = 300) -> None:
        self.secret = secret.encode()
        self.max_skew_seconds = max_skew_seconds

    def _canonical(self, method: str, path: str, body: bytes, timestamp: str) -> bytes:
        return f"{method}\n{path}\n{len(body)}\n{timestamp}".encode() + body

    def sign(self, method: str, path: str, body: bytes, timestamp: str) -> str:
        canonical = self._canonical(method, path, body, timestamp)
        return hmac.new(self.secret, canonical, hashlib.sha256).hexdigest()

    def verify(self, method: str, path: str, body: bytes, timestamp: str, signature: str) -> bool:
        try:
            ts_int = int(timestamp)
        except ValueError:
            return False
        now = int(time.time())
        if abs(now - ts_int) > self.max_skew_seconds:
            return False
        expected = self.sign(method, path, body, timestamp)
        return hmac.compare_digest(expected, signature)
