"""Unit tests for Postgres token usage repository."""
from __future__ import annotations
from unittest.mock import AsyncMock, patch
import pytest
from deep_agent.src.token_budget.postgres_repository import TokenUsagePostgresRepository

TEST_URI = "postgresql://postgres:postgres@localhost:5432/test"


def _make_conn_mock(fetchone_return=None):
    """Return (mock_conn, mock_connect) ready to patch psycopg.AsyncConnection.connect.

    psycopg.AsyncConnection.connect is a coroutine, so the patch uses AsyncMock.
    Awaiting mock_connect(...) returns mock_conn; `async with mock_conn` returns mock_conn.
    """
    mock_cur = AsyncMock()
    mock_cur.fetchone = AsyncMock(return_value=fetchone_return)

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cur)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    mock_connect = AsyncMock(return_value=mock_conn)
    return mock_conn, mock_connect


@pytest.mark.asyncio
async def test_increment_usage_returns_dict() -> None:
    repo = TokenUsagePostgresRepository(TEST_URI)
    repo._tables_ensured = True

    mock_row = {
        "thread_id": "t1",
        "total_tokens": 150,
        "input_tokens": 100,
        "output_tokens": 50,
        "agent_name": "health-assistant",
        "updated_at": None,
    }
    _, mock_connect = _make_conn_mock(fetchone_return=mock_row)

    with patch("psycopg.AsyncConnection.connect", new=mock_connect):
        result = await repo.increment_usage("t1", 100, 50, agent_name="health-assistant")

    assert result["thread_id"] == "t1"
    assert result["total_tokens"] == 150


@pytest.mark.asyncio
async def test_increment_daily_usage_returns_dict() -> None:
    repo = TokenUsagePostgresRepository(TEST_URI)
    repo._tables_ensured = True

    mock_row = {
        "user_id": "u1",
        "org_id": "org-a",
        "agent_name": "health-assistant",
        "date": "2026-07-25",
        "total_tokens": 200,
        "updated_at": None,
    }
    _, mock_connect = _make_conn_mock(fetchone_return=mock_row)

    with patch("psycopg.AsyncConnection.connect", new=mock_connect):
        result = await repo.increment_daily_usage(
            "u1", 200, org_id="org-a", agent_name="health-assistant", date="2026-07-25"
        )

    assert result["org_id"] == "org-a"
    assert result["agent_name"] == "health-assistant"
    assert result["total_tokens"] == 200


@pytest.mark.asyncio
async def test_get_thread_usage_returns_none_when_missing() -> None:
    repo = TokenUsagePostgresRepository(TEST_URI)
    repo._tables_ensured = True

    _, mock_connect = _make_conn_mock(fetchone_return=None)

    with patch("psycopg.AsyncConnection.connect", new=mock_connect):
        result = await repo.get_thread_usage("missing")

    assert result is None


@pytest.mark.asyncio
async def test_ensure_tables_runs_once() -> None:
    repo = TokenUsagePostgresRepository(TEST_URI)
    repo._tables_ensured = False

    _, mock_connect = _make_conn_mock()

    with patch("psycopg.AsyncConnection.connect", new=mock_connect):
        await repo.ensure_tables()
        await repo.ensure_tables()

    assert mock_connect.call_count == 1
