import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

from adaptive.contracts import AuthorizationScope


class AuthFailure(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _canonical_ids(allowed_page_ids: Optional[list]) -> str:
    if allowed_page_ids is None:
        return "*"
    return ",".join(str(page_id) for page_id in sorted(set(int(x) for x in allowed_page_ids)))


def fingerprint_for(principal: str, allowed_page_ids: Optional[list], acl_version: str) -> str:
    raw = f"{principal}|{_canonical_ids(allowed_page_ids)}|{acl_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def acl_version_for(allowed_page_ids: Optional[list]) -> str:
    return hashlib.sha256(_canonical_ids(allowed_page_ids).encode("utf-8")).hexdigest()[:16]


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise AuthFailure(403, f"Invalid security token field: {field}")
    return value


def _parse_page_ids(value: Any, is_admin: bool) -> Optional[list]:
    if value is None:
        if is_admin:
            return None
        raise AuthFailure(403, "Missing page scope")
    if not isinstance(value, list):
        raise AuthFailure(403, "Invalid page scope")
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise AuthFailure(403, "Invalid page scope")
        parsed.append(int(item))
    return parsed


def scope_from_payload(data: Dict[str, Any], ttl_seconds: int) -> AuthorizationScope:
    if not isinstance(data, dict):
        raise AuthFailure(403, "Invalid security token")
    user_id = data.get("user_id")
    if user_id is None or isinstance(user_id, bool) or not isinstance(user_id, (int, str)):
        raise AuthFailure(403, "Invalid security token")
    if isinstance(user_id, str) and not user_id.strip():
        raise AuthFailure(403, "Invalid security token")
    is_admin = _as_bool(data.get("is_admin"), "is_admin")
    can_use_ai = _as_bool(data.get("can_use_ai"), "can_use_ai")
    if can_use_ai is False:
        raise AuthFailure(403, "AI Assistant is not enabled for your user role.")
    roles = data.get("roles", [])
    if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
        raise AuthFailure(403, "Invalid security token")
    ts = data.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise AuthFailure(403, "Invalid security token")
    issued_at = int(ts)
    allowed = _parse_page_ids(data.get("allowed_page_ids"), is_admin)
    version = acl_version_for(allowed)
    principal = str(user_id)
    return AuthorizationScope(
        principal=principal,
        is_admin=is_admin,
        can_use_ai=can_use_ai,
        allowed_page_ids=allowed,
        fingerprint=fingerprint_for(principal, allowed, version),
        acl_version=version,
        issued_at=issued_at,
        expires_at=issued_at + int(ttl_seconds),
        roles=roles,
    )


def assert_fresh(scope: AuthorizationScope, now: Optional[int] = None, skew_seconds: int = 60) -> None:
    current = int(time.time() if now is None else now)
    if scope.issued_at > current + skew_seconds:
        raise AuthFailure(403, "Security token is not yet valid")
    if current >= scope.expires_at:
        raise AuthFailure(403, "Security token expired")


def verify_signed_payload(
    user_token: Dict[str, Any],
    secret: str,
    ttl_seconds: int,
    now: Optional[int] = None,
    skew_seconds: int = 60,
) -> AuthorizationScope:
    if not secret:
        raise AuthFailure(401, "Service secret is not configured")
    raw_payload = user_token.get("payload")
    signature = user_token.get("sig")
    if not isinstance(raw_payload, str) or not isinstance(signature, str):
        raise AuthFailure(403, "Invalid security token")
    if not signature or len(signature) != 64:
        raise AuthFailure(403, "Invalid security token signature")
    expected = hmac.new(secret.encode("utf-8"), raw_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise AuthFailure(403, "Invalid security token signature")
    try:
        data = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise AuthFailure(403, "Invalid security token") from exc
    scope = scope_from_payload(data, ttl_seconds)
    assert_fresh(scope, now=now, skew_seconds=skew_seconds)
    return scope


def sign_payload(payload: Dict[str, Any], secret: str) -> Dict[str, str]:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    signature = hmac.new(secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()
    return {"payload": raw, "sig": signature}


def service_token_ok(presented: Optional[str], secret: str) -> bool:
    if not secret or not presented:
        return False
    if len(presented) != len(secret):
        return False
    return hmac.compare_digest(presented, secret)
