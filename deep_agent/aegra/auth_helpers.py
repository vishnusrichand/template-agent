"""Shared authentication helpers for Aegra route handlers."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def check_group_access(permissions: list[str], *, developer_only: bool = False) -> None:
    """Enforce group-based access from DEVELOPER_GROUP / USER_GROUP.

    Both empty: unrestricted (caller still enforces ENABLE_AUTH).
    Only one group set: that group is an allow-list.
    Both set: DEVELOPER_GROUP has full access; USER_GROUP has non-eval access.
    developer_only=True requires DEVELOPER_GROUP (eval endpoints).
    """
    dev_group = (settings.DEVELOPER_GROUP or "").strip()
    user_group = (settings.USER_GROUP or "").strip()
    if not dev_group and not user_group:
        return

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
    raise HTTPException(
        status_code=403,
        detail=f"Access denied: '{allowed}' group membership required.",
    )


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
