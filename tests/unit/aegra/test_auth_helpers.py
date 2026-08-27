"""Unit tests for shared authentication helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from deep_agent.aegra.auth_helpers import authenticated_user_id, check_group_access


class TestCheckGroupAccess:
    def test_no_check_when_restrict_disabled(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = False
            # Should not raise regardless of permissions
            check_group_access([])
            check_group_access(["some-role"])

    def test_developer_in_developer_group_passes(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "lightspeed-developer"
            mock_settings.USER_GROUP = "lightspeed-user"
            check_group_access(["lightspeed-developer"])  # should not raise

    def test_user_in_user_group_passes_non_eval(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "lightspeed-developer"
            mock_settings.USER_GROUP = "lightspeed-user"
            check_group_access(["lightspeed-user"])  # should not raise

    def test_user_in_neither_group_denied(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "lightspeed-developer"
            mock_settings.USER_GROUP = "lightspeed-user"
            with pytest.raises(HTTPException) as exc:
                check_group_access(["other-role"])
            assert exc.value.status_code == 403

    def test_developer_only_blocks_user_group(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "lightspeed-developer"
            mock_settings.USER_GROUP = "lightspeed-user"
            with pytest.raises(HTTPException) as exc:
                check_group_access(["lightspeed-user"], developer_only=True)
            assert exc.value.status_code == 403

    def test_developer_only_passes_for_developer(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "lightspeed-developer"
            mock_settings.USER_GROUP = "lightspeed-user"
            check_group_access(
                ["lightspeed-developer"], developer_only=True
            )  # should not raise

    def test_empty_groups_denies_everyone(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = ""
            mock_settings.USER_GROUP = ""
            with pytest.raises(HTTPException) as exc:
                check_group_access(["lightspeed-developer"])
            assert exc.value.status_code == 403

    def test_only_developer_group_configured_denies_others(self):
        with patch("deep_agent.aegra.auth_helpers.settings") as mock_settings:
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "lightspeed-developer"
            mock_settings.USER_GROUP = ""
            with pytest.raises(HTTPException) as exc:
                check_group_access(["some-other-role"])
            assert exc.value.status_code == 403


class TestAuthenticatedUserIdGroupCheck:
    @pytest.mark.asyncio
    async def test_group_check_skipped_when_auth_disabled(self):
        """Dev bypass skips group check entirely."""
        request = MagicMock()
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.auth_helpers.settings") as mock_settings,
        ):
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "devs"
            mock_settings.USER_GROUP = ""
            result = await authenticated_user_id(request)
        assert result == "dev-user"

    @pytest.mark.asyncio
    async def test_group_check_called_with_token_roles(self):
        """Group check uses roles from the decoded JWT payload."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer tok"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "u1", "realm_access": {"roles": ["devs"]}},
            ),
            patch("deep_agent.aegra.auth_helpers.settings") as mock_settings,
        ):
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "devs"
            mock_settings.USER_GROUP = ""
            result = await authenticated_user_id(request)
        assert result == "u1"

    @pytest.mark.asyncio
    async def test_developer_only_param_raises_403_for_user_group(self):
        """developer_only=True blocks USER_GROUP members."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer tok"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "u1", "realm_access": {"roles": ["users"]}},
            ),
            patch("deep_agent.aegra.auth_helpers.settings") as mock_settings,
        ):
            mock_settings.RESTRICT_TO_GROUPS = True
            mock_settings.DEVELOPER_GROUP = "devs"
            mock_settings.USER_GROUP = "users"
            with pytest.raises(HTTPException) as exc:
                await authenticated_user_id(request, developer_only=True)
        assert exc.value.status_code == 403


class TestAuthenticatedUserId:
    @pytest.mark.asyncio
    async def test_returns_dev_user_when_auth_disabled(self):
        request = MagicMock()
        with patch("deep_agent.aegra.auth.ENABLE_AUTH", False):
            result = await authenticated_user_id(request)
        assert result == "dev-user"

    @pytest.mark.asyncio
    async def test_returns_anonymous_when_no_bearer_token(self):
        request = MagicMock()
        request.headers = {"authorization": ""}
        with patch("deep_agent.aegra.auth.ENABLE_AUTH", True):
            result = await authenticated_user_id(request)
        assert result == "anonymous"

    @pytest.mark.asyncio
    async def test_raises_401_when_reject_anonymous_and_no_token(self):
        request = MagicMock()
        request.headers = {"authorization": ""}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await authenticated_user_id(request, reject_anonymous=True)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_extracts_sub_from_valid_jwt(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "user-123"},
            ),
        ):
            result = await authenticated_user_id(request)
        assert result == "user-123"

    @pytest.mark.asyncio
    async def test_invalid_token_propagates_exception(self):
        """An expired or malformed JWT causes _decode_token to raise; verify it propagates."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer expired-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=Exception("Token expired"),
            ),
            pytest.raises(Exception, match="Token expired"),
        ):
            await authenticated_user_id(request)
