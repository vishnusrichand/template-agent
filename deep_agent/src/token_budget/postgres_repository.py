"""PostgreSQL store for per-thread and per-agent daily token usage."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TABLES_ENSURED = False


def _validated_date(date: str | None) -> str:
    if date is None:
        return datetime.now(UTC).strftime("%Y-%m-%d")
    if not _DATE_RE.match(date):
        raise ValueError(f"Invalid date format {date!r}, expected YYYY-MM-DD")
    return date


CREATE_THREAD_TABLE = """
CREATE TABLE IF NOT EXISTS thread_token_usage (
    thread_id       TEXT PRIMARY KEY,
    total_tokens    BIGINT NOT NULL DEFAULT 0,
    input_tokens    BIGINT NOT NULL DEFAULT 0,
    output_tokens   BIGINT NOT NULL DEFAULT 0,
    agent_name      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_thread_token_usage_updated_at
    ON thread_token_usage (updated_at);
"""

CREATE_AGENT_DAILY_TABLE = """
CREATE TABLE IF NOT EXISTS agent_daily_token_usage (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    org_id          TEXT NOT NULL DEFAULT 'default',
    agent_name      TEXT NOT NULL,
    date            DATE NOT NULL,
    total_tokens    BIGINT NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, org_id, agent_name, date)
);
CREATE INDEX IF NOT EXISTS idx_agent_daily_token_usage_date
    ON agent_daily_token_usage (date);
CREATE INDEX IF NOT EXISTS idx_agent_daily_token_usage_user
    ON agent_daily_token_usage (user_id, org_id, agent_name);
"""

_UPSERT_THREAD = """
INSERT INTO thread_token_usage
    (thread_id, total_tokens, input_tokens, output_tokens, agent_name, updated_at)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (thread_id) DO UPDATE SET
    total_tokens  = thread_token_usage.total_tokens  + EXCLUDED.total_tokens,
    input_tokens  = thread_token_usage.input_tokens  + EXCLUDED.input_tokens,
    output_tokens = thread_token_usage.output_tokens + EXCLUDED.output_tokens,
    agent_name    = EXCLUDED.agent_name,
    updated_at    = now()
RETURNING thread_id, total_tokens, input_tokens, output_tokens, agent_name, updated_at;
"""

_UPSERT_AGENT_DAILY = """
INSERT INTO agent_daily_token_usage
    (user_id, org_id, agent_name, date, total_tokens, updated_at)
VALUES (%s, %s, %s, %s::date, %s, now())
ON CONFLICT (user_id, org_id, agent_name, date) DO UPDATE SET
    total_tokens = agent_daily_token_usage.total_tokens + EXCLUDED.total_tokens,
    updated_at   = now()
RETURNING user_id, org_id, agent_name, date::text, total_tokens, updated_at;
"""

_GET_THREAD = """
SELECT thread_id, total_tokens, input_tokens, output_tokens, agent_name, updated_at
FROM thread_token_usage WHERE thread_id = %s;
"""

_GET_AGENT_DAILY = """
SELECT user_id, org_id, agent_name, date::text, total_tokens, updated_at
FROM agent_daily_token_usage
WHERE user_id = %s AND org_id = %s AND agent_name = %s AND date = %s::date;
"""


class TokenUsagePostgresRepository:
    """PostgreSQL token usage: per-thread counts and per-(user, org, agent) daily rollup."""

    def __init__(self, database_uri: str) -> None:
        self._uri = database_uri

    async def ensure_tables(self) -> None:
        """Create token usage tables if they do not exist (idempotent, once per process)."""
        global _TABLES_ENSURED  # noqa: PLW0603
        if _TABLES_ENSURED:
            return
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(CREATE_THREAD_TABLE)
            await conn.execute(CREATE_AGENT_DAILY_TABLE)
            await conn.commit()
        _TABLES_ENSURED = True
        logger.info("token_usage tables ensured in postgres")

    async def increment_usage(
        self,
        thread_id: str,
        input_tokens: int,
        output_tokens: int,
        *,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Atomically add tokens for a thread and return the updated row."""
        input_tokens = max(input_tokens, 0)
        output_tokens = max(output_tokens, 0)
        total_delta = input_tokens + output_tokens
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                _UPSERT_THREAD,
                (thread_id, total_delta, input_tokens, output_tokens, agent_name),
            )
            row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise RuntimeError("Failed to increment postgres thread token usage")
        return dict(row)

    async def increment_agent_daily_usage(
        self,
        user_id: str,
        tokens: int,
        *,
        org_id: str = "default",
        agent_name: str,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Increment daily token count for (user, org, agent) on the given UTC date."""
        tokens = max(tokens, 0)
        day = _validated_date(date)
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                _UPSERT_AGENT_DAILY,
                (user_id, org_id, agent_name, day, tokens),
            )
            row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise RuntimeError("Failed to increment postgres agent daily token usage")
        return dict(row)

    async def get_thread_usage(self, thread_id: str) -> dict[str, Any] | None:
        """Return the token usage row for thread_id, or None."""
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(_GET_THREAD, (thread_id,))
            row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def get_agent_daily_usage(
        self,
        user_id: str,
        *,
        org_id: str = "default",
        agent_name: str,
        date: str | None = None,
    ) -> dict[str, Any] | None:
        """Return daily usage row for (user, org, agent, date), or None."""
        day = _validated_date(date)
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(_GET_AGENT_DAILY, (user_id, org_id, agent_name, day))
            row = await cur.fetchone()
        return dict(row) if row is not None else None
