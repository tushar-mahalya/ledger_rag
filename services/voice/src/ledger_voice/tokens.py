"""Short-lived, single-use tokens that bind a media stream to a verified call.

The signed Twilio webhook (``/twilio/voice``) mints a token for that call's SID and
passes it as a ``<Stream>`` custom parameter; the media WebSocket accepts the
stream only if the token verifies for the same SID, has not expired and has not
been used before in this process.
"""

import base64
import hashlib
import hmac
import time
from collections import OrderedDict

MAX_TOKEN_CHARS = 128


def _signature(secret: bytes, call_sid: str, expires: int) -> str:
    digest = hmac.new(secret, f"{call_sid}.{expires}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def mint(secret: str, call_sid: str, ttl_seconds: int, now: float | None = None) -> str:
    expires = int((time.time() if now is None else now) + ttl_seconds)
    return f"{expires}.{_signature(secret.encode(), call_sid, expires)}"


class TokenVerifier:
    def __init__(self, secret: str, ttl_seconds: int, remember: int = 4096) -> None:
        self._secret = secret.encode()
        self._ttl = ttl_seconds
        self._used: OrderedDict[str, float] = OrderedDict()
        self._remember = remember

    def verify(self, token: object, call_sid: str, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        if not isinstance(token, str) or len(token) > MAX_TOKEN_CHARS or not call_sid:
            return False
        expires_text, _, signature = token.partition(".")
        if not expires_text.isdigit() or not signature:
            return False
        expires = int(expires_text)
        if not current <= expires <= current + self._ttl:
            return False
        if not hmac.compare_digest(signature, _signature(self._secret, call_sid, expires)):
            return False
        if call_sid in self._used:
            return False  # single use per call within this process
        self._used[call_sid] = expires
        while len(self._used) > self._remember:
            self._used.popitem(last=False)
        return True
