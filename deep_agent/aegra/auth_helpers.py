"""Shared authentication helpers for Aegra route handlers."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def check_group_access(permissions: list[str], *, developer_only: bool = False) -> None:
    """Enforce group-based access restriction when RESTRICT_TO_GROUPS is enabled.

    No-op when RESTRICT_TO_GROUPS=false or ENABLE_AUTH=false (caller handles bypass).
    Raises HTTPException(403) when the user lacks required group membership.

    Args:
        permissions: Keycloak roles from realm_access.roles.
        developer_only: When True, only DEVELOPER_GROUP passes (eval endpoints).
    """
    if not settings.RESTRICT_TO_GROUPS:
        return

    dev_group = settings.DEVELOPER_GROUP
    user_group = settings.USER_GROUP

    if developer_only:
        if dev_group and dev_group in permissions:
            return
        detail = (
            f"Access denied: '{dev_group}' group membership required."
            if dev_group
            else "Access denied: DEVELOPER_GROUP is not configured."
        )
        raise HTTPException(status_code=403, detail=detail)

    if dev_group and dev_group in permissions:
        return
    if user_group and user_group in permissions:
        return
    allowed = " or ".join(g for g in [dev_group, user_group] if g)
    detail = (
        f"Access denied: '{allowed}' group membership required."
        if allowed
        else "Access denied: no access groups are configured."
    )
    raise HTTPException(status_code=403, detail=detail)


async def authenticated_user_id(
    request: Request, *, reject_anonymous: bool = False, developer_only: bool = False
) -> str:
    """Extract and return the authenticated user's identity from the JWT.

    Args:
        request: The incoming FastAPI request.
        reject_anonymous: If True, raise 401 when credentials are missing
            instead of returning ``"anonymous"``.
        developer_only: If True, only DEVELOPER_GROUP members pass group check.

    Returns:
        The ``sub`` claim from the JWT, ``DEV_USER_ID`` when auth is
        disabled, or ``"anonymous"`` if credentials are absent and
        *reject_anonymous* is False.
    """
    from deep_agent.aegra.auth import DEV_USER_ID, ENABLE_AUTH, _decode_token

    if not ENABLE_AUTH:
        return DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        if reject_anonymous:
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )
        return "anonymous"

    payload = await asyncio.to_thread(_decode_token, auth_header[7:])
    permissions = payload.get("realm_access", {}).get("roles", [])
    check_group_access(permissions, developer_only=developer_only)
    return str(payload["sub"])
