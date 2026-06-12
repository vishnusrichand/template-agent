"""Unit tests for aegra auth module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.aegra.auth import (
    _build_dev_user,
    _resolve_jwks_uri,
    _thread_scope_metadata,
    encrypt_user_id,
    on_thread_create,
    on_thread_search,
    on_thread_update,
)


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


class TestThreadScopeMetadata:
    def test_empty_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("THREAD_SCOPE_METADATA", None)
            assert _thread_scope_metadata() == {}

    def test_empty_when_blank(self):
        with patch.dict(os.environ, {"THREAD_SCOPE_METADATA": "   "}):
            assert _thread_scope_metadata() == {}

    def test_parses_json_object(self):
        with patch.dict(
            os.environ,
            {"THREAD_SCOPE_METADATA": '{"deployment_id":"org/agent"}'},
        ):
            assert _thread_scope_metadata() == {"deployment_id": "org/agent"}

    def test_invalid_json_returns_empty(self):
        with patch.dict(os.environ, {"THREAD_SCOPE_METADATA": "not-json"}):
            assert _thread_scope_metadata() == {}

    def test_non_object_json_returns_empty(self):
        with patch.dict(os.environ, {"THREAD_SCOPE_METADATA": '["a"]'}):
            assert _thread_scope_metadata() == {}


class TestThreadAuthHandlers:
    @pytest.mark.asyncio
    async def test_create_merges_scope_into_metadata(self):
        with patch.dict(
            os.environ,
            {"THREAD_SCOPE_METADATA": '{"deployment_id":"org/agent"}'},
        ):
            value = {"metadata": {"user_identity": "user-1"}}
            result = await on_thread_create(MagicMock(), value)
            assert result == {"metadata": {"deployment_id": "org/agent"}}
            assert value["metadata"] == {
                "user_identity": "user-1",
                "deployment_id": "org/agent",
            }

    @pytest.mark.asyncio
    async def test_create_noop_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("THREAD_SCOPE_METADATA", None)
            value = {"metadata": {"user_identity": "user-1"}}
            result = await on_thread_create(MagicMock(), value)
            assert result == {}
            assert value["metadata"] == {"user_identity": "user-1"}

    @pytest.mark.asyncio
    async def test_update_merges_scope_into_metadata(self):
        with patch.dict(
            os.environ,
            {"THREAD_SCOPE_METADATA": '{"deployment_id":"org/agent"}'},
        ):
            value = {"metadata": {"thread_name": "hello"}}
            result = await on_thread_update(MagicMock(), value)
            assert result == {"metadata": {"deployment_id": "org/agent"}}
            assert value["metadata"] == {
                "thread_name": "hello",
                "deployment_id": "org/agent",
            }

    @pytest.mark.asyncio
    async def test_search_returns_scope_filter(self):
        with patch.dict(
            os.environ,
            {"THREAD_SCOPE_METADATA": '{"deployment_id":"org/agent"}'},
        ):
            result = await on_thread_search(MagicMock(), {})
            assert result == {"metadata": {"deployment_id": "org/agent"}}

    @pytest.mark.asyncio
    async def test_search_empty_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("THREAD_SCOPE_METADATA", None)
            result = await on_thread_search(MagicMock(), {})
            assert result == {}


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
