import asyncio
import hmac
import time
from collections import OrderedDict

import jwt
from fastapi import HTTPException
from jwt import PyJWKClient

from ledger_rag.config import Settings
from ledger_rag.models import Principal


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jwks = (
            PyJWKClient(settings.oidc_jwks_url, cache_jwk_set=True, lifespan=300, timeout=5)
            if settings.auth_mode == "oidc"
            else None
        )

    async def authenticate(self, authorization: str | None) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                401, "Bearer authentication required", headers={"WWW-Authenticate": "Bearer"}
            )
        token = authorization[7:]
        if not token or len(token) > 16384:
            raise HTTPException(401, "Invalid credentials")
        if self.settings.auth_mode == "demo":
            secret = self.settings.demo_token
            if secret is None or not hmac.compare_digest(
                token.encode(), secret.get_secret_value().encode()
            ):
                raise HTTPException(401, "Invalid credentials")
            return Principal(
                subject="demo-user", tenant_id="demo", roles=["reader", "admin"], groups=["analyst"]
            )
        try:
            assert self.jwks is not None
            key = await asyncio.to_thread(self.jwks.get_signing_key_from_jwt, token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256", "ES256"],
                issuer=self.settings.oidc_issuer,
                audience=self.settings.oidc_audience,
                options={"require": ["exp", "iat", "sub", "iss", "aud", "tenant_id"]},
                leeway=10,
            )
        except jwt.PyJWKClientConnectionError as exc:
            raise HTTPException(503, "Identity provider unavailable") from exc
        except jwt.PyJWTError as exc:
            raise HTTPException(401, "Invalid credentials") from exc
        subject = claims.get("sub")
        tenant = claims.get("tenant_id")
        roles = claims.get("roles", [])
        groups = claims.get("groups", [])
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(tenant, str)
            or not tenant
            or not isinstance(roles, list)
            or not all(isinstance(role, str) for role in roles)
            or not isinstance(groups, list)
            or not all(isinstance(group, str) for group in groups)
        ):
            raise HTTPException(401, "Invalid identity claims")
        return Principal(subject=subject, tenant_id=tenant, roles=roles, groups=groups)


class RateLimiter:
    """Per-process limiter; replace with a shared gateway quota before horizontal scaling."""

    def __init__(self, limit: int, capacity: int = 10000) -> None:
        self.limit = limit
        self.capacity = capacity
        self.windows: OrderedDict[tuple[str, str], tuple[float, int]] = OrderedDict()

    def check(self, principal: Principal) -> None:
        key = (principal.tenant_id, principal.subject)
        now = time.monotonic()
        start, count = self.windows.get(key, (now, 0))
        if now - start >= 60:
            start, count = now, 0
        if count >= self.limit:
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
        if key not in self.windows and len(self.windows) >= self.capacity:
            oldest_start = next(iter(self.windows.values()))[0]
            if now - oldest_start < 60:
                raise HTTPException(503, "Rate limiter capacity exhausted")
            self.windows.popitem(last=False)
        self.windows[key] = (start, count + 1)
        if count == 0:
            self.windows.move_to_end(key)
