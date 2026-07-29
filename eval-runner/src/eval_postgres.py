"""PostgreSQL helpers for the eval pod.

Handles all database access: table setup, writing completed eval results to
the `evals` table, and reading back per-run results from `evaluation_results`.

Uses synchronous psycopg2 — called from a thread pool inside the asyncio
event loop managed by eval_api.py.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "template_agent")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
if not POSTGRES_PASSWORD:
    log.warning("POSTGRES_PASSWORD not set — connection may fail")

AGENT_ORG = os.environ.get("AI_PLATFORM_AGENT_ORG", "default")
AGENT_NAME = os.environ.get("AI_PLATFORM_AGENT_NAME", "agent")


_HASH_EXTENSIONS = {".md", ".yaml", ".json"}
_HASH_EXCLUDE_DIRS = {"evals", "deployment"}


def _compute_config_hash(config_dir: str) -> str:
    """SHA256 of behavior-relevant config files (prompts, skills, runtime, tools)."""
    import hashlib
    from pathlib import Path

    h = hashlib.sha256()
    base = Path(config_dir)
    if base.exists():
        for fpath in sorted(base.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix not in _HASH_EXTENSIONS:
                continue
            if any(
                part in _HASH_EXCLUDE_DIRS for part in fpath.relative_to(base).parts
            ):
                continue
            h.update(str(fpath.relative_to(base)).encode())
            h.update(fpath.read_bytes())
    return h.hexdigest()[:16]  # 16-char prefix is enough


_config_dir = os.environ.get("AGENT_CONFIG_DIR", "config/agent")
_env_hash = os.environ.get("AGENT_CONFIG_HASH", "")


def _get_config_hash() -> str:
    """Return config hash — env var first, compute from files as fallback (CLI path only)."""
    return _env_hash or _compute_config_hash(_config_dir)


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS evals (
    id              SERIAL PRIMARY KEY,
    org             TEXT NOT NULL,
    name            TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    eval_status     TEXT NOT NULL DEFAULT 'in_progress',
    eval_score      FLOAT,
    ls_run_ids      TEXT[],
    pass            INTEGER DEFAULT 0,
    fail            INTEGER DEFAULT 0,
    error           INTEGER DEFAULT 0,
    judge_model     TEXT,
    results_detail  JSONB,
    force_reeval    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS evals_org_name_hash ON evals (org, name, config_hash);
CREATE INDEX IF NOT EXISTS evals_status ON evals (eval_status);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id                      SERIAL PRIMARY KEY,
    run_id                  VARCHAR(36) NOT NULL,
    timestamp               TIMESTAMP NOT NULL,
    conversation_group_id   VARCHAR(255) NOT NULL,
    tag                     VARCHAR(100),
    turn_id                 VARCHAR(100),
    metric_identifier       VARCHAR(255) NOT NULL,
    metric_metadata         TEXT,
    result                  VARCHAR(20) NOT NULL,
    score                   FLOAT,
    threshold               FLOAT,
    reason                  TEXT,
    query                   TEXT,
    response                TEXT,
    execution_time          FLOAT,
    evaluation_latency      FLOAT,
    api_input_tokens        INTEGER,
    api_output_tokens       INTEGER,
    judge_llm_input_tokens  INTEGER,
    judge_llm_output_tokens INTEGER,
    embedding_tokens        INTEGER,
    judge_scores            TEXT,
    time_to_first_token     FLOAT,
    streaming_duration      FLOAT,
    agent_latency           FLOAT,
    tokens_per_second       FLOAT,
    tool_calls              TEXT,
    contexts                TEXT,
    expected_response       TEXT,
    expected_intent         TEXT,
    expected_keywords       TEXT,
    expected_tool_calls     TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run_id ON evaluation_results (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_group_id ON evaluation_results (conversation_group_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_metric ON evaluation_results (metric_identifier);
CREATE INDEX IF NOT EXISTS idx_eval_results_timestamp ON evaluation_results (timestamp);
"""


def _get_conn() -> Any:
    import psycopg2

    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        connect_timeout=5,
    )


_table_ensured = False


_MIGRATIONS = [
    "ALTER TABLE evals ADD COLUMN IF NOT EXISTS ls_run_ids TEXT[]",
    "ALTER TABLE evals DROP COLUMN IF EXISTS ls_run_id",
    "DROP INDEX IF EXISTS evals_ls_run_id",
]


def ensure_table() -> None:
    """Create evals table if it doesn't exist, then apply additive migrations."""
    global _table_ensured
    if _table_ensured:
        return
    try:
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE)
                    for migration in _MIGRATIONS:
                        cur.execute(migration)
        finally:
            conn.close()
        _table_ensured = True
        log.info("eval_postgres_tables_ready: evals + evaluation_results verified")
    except Exception as exc:
        log.warning(
            "eval_postgres_ensure_table_failed (%s): %s", type(exc).__name__, exc
        )


def write_eval_result(
    passed: int,
    failed: int,
    errors: int,
    eval_score: float,
    ls_run_ids: list[str] | None = None,
    judge_model: str = "",
    results_detail: dict[str, Any] | None = None,
    config_hash: str | None = None,
    org: str | None = None,
    name: str | None = None,
) -> None:
    """Persist completed eval results to PostgreSQL.

    Uses (config_hash, org, name) passed from the agentpod trigger response
    via the UI — no local hash computation needed. Falls back to module-level
    defaults if not provided (local dev without UI).
    """
    effective_org = org or AGENT_ORG
    effective_name = name or AGENT_NAME
    effective_hash = config_hash or _get_config_hash()
    try:
        conn = _get_conn()
        now = datetime.now(UTC)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE evals
                        SET eval_status   = 'completed',
                            ls_run_ids    = %s,
                            eval_score    = %s,
                            pass          = %s,
                            fail          = %s,
                            error         = %s,
                            judge_model   = %s,
                            results_detail = %s,
                            updated_at    = %s,
                            completed_at  = %s
                        WHERE id = (
                            SELECT id FROM evals
                            WHERE org = %s AND name = %s AND config_hash = %s
                              AND eval_status IN ('in_progress', 'error')
                            ORDER BY created_at DESC
                            LIMIT 1
                        )
                        RETURNING id
                        """,
                        (
                            ls_run_ids,
                            eval_score,
                            passed,
                            failed,
                            errors,
                            judge_model,
                            json.dumps(results_detail) if results_detail else None,
                            now,
                            now,
                            effective_org,
                            effective_name,
                            effective_hash,
                        ),
                    )
                    row = cur.fetchone()
        finally:
            conn.close()

        if row:
            log.info(
                "eval_result_written_to_postgres id=%s score=%.3f pass=%d fail=%d error=%d",
                row[0],
                eval_score,
                passed,
                failed,
                errors,
            )
        else:
            log.warning(
                "eval_postgres_no_matching_record org=%s name=%s config_hash=%s "
                "(check AGENT_CONFIG_DIR env var — CWD may resolve wrong directory)",
                effective_org,
                effective_name,
                effective_hash,
            )
    except Exception as exc:
        log.error("eval_postgres_write_failed (%s): %s", type(exc).__name__, exc)


def load_results_since(run_started_at: datetime) -> dict[str, Any]:
    """Load eval results from evaluation_results written since run_started_at.

    Finds all run_ids produced after run_started_at (parallel tag runs each
    write their own run_id), fetches every row for those runs, and returns a
    dict with 'turns', 'summary', and 'ls_run_ids'. Returns {} on any failure.
    """
    try:
        import psycopg2.extras

        conn = _get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM evaluation_results WHERE timestamp >= %s",
                    (run_started_at,),
                )
                raw_rows = cur.fetchall()
        finally:
            conn.close()

        if not raw_rows:
            log.warning("No evaluation_results rows found since %s", run_started_at)
            return {}

        run_ids = list(dict.fromkeys(r["run_id"] for r in raw_rows))
        log.info("Loading DB results for run_ids=%s", run_ids)
    except Exception as exc:
        log.warning("load_results_since_failed (%s): %s", type(exc).__name__, exc)
        return {}

    overall_counts: dict[str, int] = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    by_metric: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"pass": 0, "fail": 0, "scores": []}
    )
    by_conversation: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"pass": 0, "fail": 0}
    )

    turns: list[dict[str, Any]] = []
    for r in raw_rows:
        result_val = str(r.get("result") or "").upper()
        if result_val not in overall_counts:
            result_val = "ERROR"
        overall_counts[result_val] += 1

        metric = str(r.get("metric_identifier") or "unknown")
        if result_val == "PASS":
            by_metric[metric]["pass"] += 1
        elif result_val == "FAIL":
            by_metric[metric]["fail"] += 1
        try:
            by_metric[metric]["scores"].append(float(r.get("score") or 0))
        except (TypeError, ValueError):
            pass

        conv_id = str(r.get("conversation_group_id") or "unknown")
        if result_val == "PASS":
            by_conversation[conv_id]["pass"] += 1
        elif result_val == "FAIL":
            by_conversation[conv_id]["fail"] += 1

        turns.append(
            {
                k: (
                    v.isoformat()
                    if isinstance(v, datetime)
                    else str(v)
                    if isinstance(v, float)
                    else v
                )
                for k, v in r.items()
            }
        )

    total = len(raw_rows)
    overall_with_rates: dict[str, Any] = dict(overall_counts)
    overall_with_rates["pass_rate"] = (
        round(overall_counts["PASS"] / total, 3) if total else 0.0
    )
    overall_with_rates["fail_rate"] = (
        round(overall_counts["FAIL"] / total, 3) if total else 0.0
    )
    overall_with_rates["error_rate"] = (
        round(overall_counts["ERROR"] / total, 3) if total else 0.0
    )

    by_metric_out: dict[str, Any] = {}
    for metric, data in by_metric.items():
        m_total = data["pass"] + data["fail"]
        by_metric_out[metric] = {
            "pass": data["pass"],
            "fail": data["fail"],
            "pass_rate": round(data["pass"] / m_total, 3) if m_total else 0.0,
        }

    summary: dict[str, Any] = {
        "total_evaluations": total,
        "summary_stats": {
            "overall": overall_with_rates,
            "by_metric": by_metric_out,
            "by_conversation": dict(by_conversation),
        },
    }

    return {"turns": turns, "summary": summary, "ls_run_ids": run_ids}


def get_results_by_run_id(run_id: str) -> list[dict[str, Any]]:
    """Return per-turn rows for a specific run_id. Returns [] if not found."""
    log.info("get_results_by_run_id: run_id=%s", run_id)
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT conversation_group_id, turn_id, metric_identifier, result, score, reason "
                    "FROM evaluation_results WHERE run_id = %s ORDER BY id",
                    (run_id,),
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        log.error("get_results_by_run_id failed (%s): %s", type(exc).__name__, exc)
        raise
    log.info("get_results_by_run_id: run_id=%s returned %d rows", run_id, len(rows))
    return rows
