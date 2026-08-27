"""Unit tests for aegra auth module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.aegra.auth import (
    _build_dev_user,
    _check_group_for_langgraph,
    _resolve_jwks_uri,
    encrypt_user_id,
)


class TestCheckGroupForLanggraph:
    def test_open_when_both_groups_empty(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", ""),
            patch("deep_agent.aegra.auth.USER_GROUP", ""),
        ):
            _check_group_for_langgraph(["any-role"])

    def test_only_developer_group_allows_members(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", "devs"),
            patch("deep_agent.aegra.auth.USER_GROUP", ""),
        ):
            _check_group_for_langgraph(["devs"])

    def test_only_developer_group_denies_others(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", "devs"),
            patch("deep_agent.aegra.auth.USER_GROUP", ""),
        ):
            with pytest.raises(PermissionError):
                _check_group_for_langgraph(["any-role"])

    def test_only_user_group_allows_members(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", ""),
            patch("deep_agent.aegra.auth.USER_GROUP", "users"),
        ):
            _check_group_for_langgraph(["users"])

    def test_only_user_group_denies_others(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", ""),
            patch("deep_agent.aegra.auth.USER_GROUP", "users"),
        ):
            with pytest.raises(PermissionError):
                _check_group_for_langgraph(["any-role"])

    def test_developer_group_passes(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", "devs"),
            patch("deep_agent.aegra.auth.USER_GROUP", "users"),
        ):
            _check_group_for_langgraph(["devs"])

    def test_user_group_passes(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", "devs"),
            patch("deep_agent.aegra.auth.USER_GROUP", "users"),
        ):
            _check_group_for_langgraph(["users"])

    def test_neither_group_raises_permission_error(self):
        with (
            patch("deep_agent.aegra.auth.DEVELOPER_GROUP", "devs"),
            patch("deep_agent.aegra.auth.USER_GROUP", "users"),
        ):
            with pytest.raises(PermissionError):
                _check_group_for_langgraph(["other-role"])


class TestEncryptUserId:
    def test_passthrough_when_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", False):
                assert encrypt_user_id("user123") == "user123"

    def test_passthrough_when_no_key(self):
        with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", True):
            with patch("deep_agent.aegra.auth.USER_ID_ENCRYPTION_KEY", ""):
                assert encrypt_user_id("user123") == "user123"

    def test_deterministic_encryption(self):
        with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", True):
            with patch(
                "deep_agent.aegra.auth.USER_ID_ENCRYPTION_KEY",
                "secret_key_32_bytes_hex",
            ):
                result1 = encrypt_user_id("user123")
                result2 = encrypt_user_id("user123")
                assert result1 == result2
                assert result1 != "user123"
                assert len(result1) == 16

    def test_different_users_different_hashes(self):
        with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", True):
            with patch(
                "deep_agent.aegra.auth.USER_ID_ENCRYPTION_KEY",
                "secret_key_32_bytes_hex",
            ):
                r1 = encrypt_user_id("alice")
                r2 = encrypt_user_id("bob")
                assert r1 != r2


class TestBuildDevUser:
    def test_dev_user_structure(self):
        user = _build_dev_user()
        assert user["is_authenticated"] is True
        assert "identity" in user
        assert "display_name" in user
        assert "permissions" in user
        assert "admin" in user["permissions"]
        assert "email" in user

    def test_dev_user_identity(self):
        with patch("deep_agent.aegra.auth.DEV_USER_ID", "custom-dev"):
            user = _build_dev_user()
            assert user["identity"] == "custom-dev"


class TestResolveJwksUri:
    def test_explicit_jwks_uri(self):
        with patch(
            "deep_agent.aegra.auth.SSO_JWKS_URI", "https://sso.example.com/jwks"
        ):
            result = _resolve_jwks_uri()
            assert result == "https://sso.example.com/jwks"

    def test_cached_uri(self):
        with patch("deep_agent.aegra.auth.SSO_JWKS_URI", ""):
            with patch.dict(
                os.environ, {"_RESOLVED_JWKS_URI": "https://cached.example.com/jwks"}
            ):
                result = _resolve_jwks_uri()
                assert result == "https://cached.example.com/jwks"

    def test_missing_issuer_raises(self):
        with patch("deep_agent.aegra.auth.SSO_JWKS_URI", ""):
            with patch("deep_agent.aegra.auth.SSO_ISSUER_URL", ""):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("_RESOLVED_JWKS_URI", None)
                    with pytest.raises(RuntimeError, match="SSO_ISSUER_URL"):
                        _resolve_jwks_uri()
