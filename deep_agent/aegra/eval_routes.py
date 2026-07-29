"""Eval endpoint for the template agent.

POST /v1/eval/run
    Runs the agent via the LangGraph Platform's internal thread API (self-call).
    Returns the full final state — tool calls at every depth including MCP tools
    called inside subagents (calculate_bmi, validate_email, send_email, etc.).
    HITL interrupts are auto-approved when auto_approve=True.

Uses the Platform's own /threads + /runs endpoints so auth context, graph cache,
and checkpointer all work correctly without side effects on other requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel

_AGENT_ORG = os.environ.get("AI_PLATFORM_AGENT_ORG", "default")
_AGENT_NAME = os.environ.get("AI_PLATFORM_AGENT_NAME", "agent")
_EVAL_RUNNER_URL = os.environ.get("EVAL_RUNNER_URL", "")


_HASH_EXTENSIONS = {".md", ".yaml", ".json"}
_HASH_EXCLUDE_DIRS = {"evals", "deployment"}


def _compute_config_hash() -> str:
    """SHA256 of behavior-relevant config files (prompts, skills, runtime, tools)."""
    config_dir = Path(os.environ.get("AGENT_CONFIG_DIR", "config/agent"))
    h = hashlib.sha256()
    if config_dir.exists():
        for fpath in sorted(config_dir.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix not in _HASH_EXTENSIONS:
                continue
            if any(
                part in _HASH_EXCLUDE_DIRS
                for part in fpath.relative_to(config_dir).parts
            ):
                continue
            h.update(str(fpath.relative_to(config_dir)).encode())
            h.update(fpath.read_bytes())
    return h.hexdigest()[:16]


log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/eval", tags=["eval"])

_INTERNAL_TOOLS = {
    "write_todos",
    "task",  # framework internals
    "read_file",
    "write_file",  # filesystem
    "ls",
    "grep",
    "glob",  # filesystem
    "execute_command",  # shell
    "compact_conversation",  # memory management
}
MAX_HITL_APPROVALS = 10
AGENT_BASE_URL = "http://localhost:5002"
ASSISTANT_ID = "agent"


class EvalRunRequest(BaseModel):
    """Request body for a single eval run turn."""

    query: str
    conversation_id: str | None = None
    auto_approve: bool = True


class EvalRunResponse(BaseModel):
    """Response from a single eval run turn."""

    response: str
    pre_approval_response: str | None
    was_interrupted: bool
    tool_calls: list[dict[str, Any]]
    contexts: list[str]
    conversation_id: str


def _extract_from_messages(messages: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Extract response, MCP tool calls, and contexts from state messages (plain dicts)."""
    final_response = ""
    tool_calls: list[dict[str, Any]] = []
    contexts: list[str] = []

    for msg in messages:
        msg_type = msg.get("type", "")

        if msg_type == "ai":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                final_response = content.strip()
            elif isinstance(content, list):
                text = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
                if text:
                    final_response = text

            for tc in msg.get("tool_calls", []):
                name = tc.get("name", "")
                if name and name not in _INTERNAL_TOOLS:
                    tool_calls.append(
                        {
                            "tool_name": name,
                            "arguments": tc.get("args", {}),
                        }
                    )

        elif msg_type == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                contexts.append(content.strip())

    return final_response, tool_calls, contexts


async def _create_thread(client: httpx.AsyncClient) -> str:
    resp = await client.post(f"{AGENT_BASE_URL}/threads", json={})
    resp.raise_for_status()
    return resp.json()["thread_id"]  # type: ignore[no-any-return]


async def _run_and_get_state(
    client: httpx.AsyncClient,
    thread_id: str,
    body: dict,
    timeout: float = 120.0,
) -> dict:
    """/runs/wait — returns state with messages at top level + __interrupt__ at top level."""
    resp = await client.post(
        f"{AGENT_BASE_URL}/threads/{thread_id}/runs/wait",
        json={**body, "stream_mode": "values"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _extract_tool_calls_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract non-internal tool calls from a list of LangChain messages."""
    import json as _json

    tool_calls: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = (
            msg
            if isinstance(msg, dict)
            else (msg.__dict__ if hasattr(msg, "__dict__") else {})
        )
        if msg_dict.get("type") != "ai":
            continue
        # Format 1: tool_calls list (newer LangChain)
        for tc in msg_dict.get("tool_calls", []):
            name = (
                tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            )
            args = (
                tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            )
            if name and name not in _INTERNAL_TOOLS:
                tool_calls.append({"tool_name": name, "arguments": args})
        # Format 2: additional_kwargs.function_call (Gemini format)
        if not tool_calls or not any(
            tc["tool_name"] not in _INTERNAL_TOOLS for tc in tool_calls
        ):
            fc = msg_dict.get("additional_kwargs", {}).get("function_call")
            if fc:
                name = fc.get("name", "")
                if name and name not in _INTERNAL_TOOLS:
                    args_raw = fc.get("arguments", "{}")
                    try:
                        args = (
                            _json.loads(args_raw)
                            if isinstance(args_raw, str)
                            else args_raw
                        )
                    except Exception:
                        args = {}
                    tool_calls.append({"tool_name": name, "arguments": args})
    return tool_calls


async def _collect_subagent_tool_calls_via_remote_graph(
    thread_id: str,
) -> list[dict[str, Any]]:
    """Fetch subagent tool calls via RemoteGraph.get_state(subgraphs=True).

    This uses the LangGraph SDK's official API to retrieve all subgraph states,
    recursively collecting tool calls from every subagent namespace.
    """
    from langgraph.pregel.remote import RemoteGraph

    tool_calls: list[dict[str, Any]] = []

    try:
        graph = RemoteGraph(ASSISTANT_ID, url=AGENT_BASE_URL)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config, subgraphs=True)

        def _collect(state_snapshot: Any) -> None:
            tasks = getattr(state_snapshot, "tasks", [])
            for task in tasks:
                substate = getattr(task, "state", None)
                if substate is None:
                    continue
                msgs = getattr(substate, "values", {})
                if isinstance(msgs, dict):
                    msgs = msgs.get("messages", [])
                tool_calls.extend(_extract_tool_calls_from_messages(msgs))
                _collect(substate)

        _collect(snapshot)
    except Exception as exc:
        log.warning(
            "RemoteGraph subgraph fetch failed: %s — falling back to Postgres", exc
        )
        return await _collect_subagent_tool_calls_from_postgres(thread_id)

    return tool_calls


async def _collect_subagent_tool_calls_from_postgres(
    thread_id: str,
) -> list[dict[str, Any]]:
    """Fallback: read subagent tool calls directly from Postgres checkpoint store.

    Blob format (LangGraph checkpoint serde):
        list of [module_path, class_name, message_dict] per message.

    The Gemini thought-signature fields use msgpack ext types — the ext_hook
    returns raw bytes for unknown types instead of raising, so those fields are
    ignored without aborting the entire blob parse.
    """
    import msgpack
    import psycopg

    from deep_agent.src.settings import settings

    def _ext_hook(code: int, data: bytes) -> Any:
        # Return raw bytes for unknown ext types instead of raising —
        # prevents Gemini thought-signature binary payloads from aborting parse.
        try:
            return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)
        except Exception:
            return data

    tool_calls: list[dict[str, Any]] = []

    try:
        async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
            rows = await conn.execute(
                """
                SELECT DISTINCT ON (checkpoint_ns) blob
                FROM checkpoint_blobs
                WHERE thread_id = %s
                  AND checkpoint_ns != ''
                  AND channel = 'messages'
                ORDER BY checkpoint_ns, version DESC
                """,
                (thread_id,),
            )
            async for (blob,) in rows:
                if blob is None:
                    continue
                try:
                    items = msgpack.unpackb(bytes(blob), raw=False, ext_hook=_ext_hook)
                    if not isinstance(items, (list, tuple)):
                        continue
                    for item in items:
                        # Each item: [module_path, class_name, message_dict]
                        if not isinstance(item, (list, tuple)) or len(item) < 3:
                            continue
                        msg = item[2]
                        if not isinstance(msg, dict) or msg.get("type") != "ai":
                            continue
                        tool_calls.extend(_extract_tool_calls_from_messages([msg]))
                except Exception as exc:
                    log.debug("Skipping blob for %s: %s", thread_id, exc)
                    continue
    except Exception as exc:
        log.warning("Postgres subagent tool call extraction failed: %s", exc)

    return tool_calls


def _messages_from_run(run_state: dict) -> list[dict]:
    """Extract messages from /runs/wait response (top-level) or thread state (under values)."""
    if "messages" in run_state:
        return run_state["messages"]  # type: ignore[no-any-return]
    return run_state.get("values", {}).get("messages", [])  # type: ignore[no-any-return]


def _detect_interrupt(run_state: dict) -> bool:
    """Return True if the run state has a pending HITL interrupt."""
    if run_state.get("__interrupt__"):
        return True
    next_nodes = run_state.get("next", [])
    values = run_state.get("values", {})
    return bool(values.get("__interrupt__")) or any(
        "__interrupt__" in str(n) for n in next_nodes
    )


@router.get("/thread-tool-calls/{thread_id}")
async def get_thread_tool_calls(thread_id: str) -> dict[str, Any]:
    """Return all subagent tool calls for a completed thread.

    Reads directly from Postgres checkpoint_blobs across all subagent namespaces
    since LangGraph's HTTP API only exposes subgraph state during interrupts.
    """
    tool_calls = await _collect_subagent_tool_calls_from_postgres(thread_id)
    return {"thread_id": thread_id, "tool_calls": tool_calls}


@router.post("/run", response_model=EvalRunResponse)
async def eval_run(body: EvalRunRequest) -> EvalRunResponse:
    """Run one conversation turn for eval purposes.

    Returns ALL tool calls from the full state including MCP tools called
    inside subagents (calculate_bmi, validate_email, send_email, etc.).
    HITL interrupts are auto-approved when auto_approve=True.
    """
    conversation_id = body.conversation_id or str(uuid.uuid4())
    pre_approval_response: str | None = None
    was_interrupted = False

    async with httpx.AsyncClient(timeout=120) as client:
        thread_id = await _create_thread(client)

        run_body = {
            "assistant_id": ASSISTANT_ID,
            "input": {"messages": [{"role": "human", "content": body.query}]},
        }

        # Initial run — wait for completion or interrupt
        run_state = await _run_and_get_state(client, thread_id, run_body)
        messages = _messages_from_run(run_state)

        if _detect_interrupt(run_state):
            was_interrupted = True
            # Capture what the agent said when it paused for approval
            for msg in reversed(messages):
                if msg.get("type") == "ai":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        pre_approval_response = content.strip()
                        break

            if body.auto_approve:
                for _ in range(MAX_HITL_APPROVALS):
                    log.info("Auto-approving HITL interrupt for thread %s", thread_id)
                    approve_body = {
                        "assistant_id": ASSISTANT_ID,
                        "command": {"resume": {"decisions": [{"type": "approve"}]}},
                    }
                    run_state = await _run_and_get_state(
                        client, thread_id, approve_body
                    )
                    if not _detect_interrupt(run_state):
                        break
                else:
                    log.warning(
                        "HITL approval limit reached for thread %s — run may be incomplete",
                        thread_id,
                    )

                messages = _messages_from_run(run_state)

        # Collect subagent tool calls via RemoteGraph (falls back to Postgres on failure)
        subagent_tool_calls = await _collect_subagent_tool_calls_via_remote_graph(
            thread_id
        )

    final_response, orchestrator_tool_calls, contexts = _extract_from_messages(messages)
    # Merge: orchestrator-level tool calls first, then all subagent tool calls
    tool_calls = orchestrator_tool_calls + subagent_tool_calls

    log.info(
        "Eval run complete: thread=%s tools=%d response_len=%d interrupted=%s",
        thread_id,
        len(tool_calls),
        len(final_response),
        was_interrupted,
    )

    return EvalRunResponse(
        response=final_response,
        pre_approval_response=pre_approval_response,
        was_interrupted=was_interrupted,
        tool_calls=tool_calls,
        contexts=contexts,
        conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# Eval management routes — trigger / status / results
# Called by Template UI via agentpod (cache check + atomic in_progress guard)
# ---------------------------------------------------------------------------

eval_mgmt_router = APIRouter(prefix="/evals", tags=["eval-management"])

_EVALS_DDL_STATEMENTS = [
    """
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
    )
    """,
    "CREATE INDEX IF NOT EXISTS evals_org_name_hash ON evals (org, name, config_hash)",
    "CREATE INDEX IF NOT EXISTS evals_status ON evals (eval_status)",
    "CREATE INDEX IF NOT EXISTS evals_history ON evals (org, name, eval_status, completed_at DESC)",
    "ALTER TABLE evals ADD COLUMN IF NOT EXISTS ls_run_ids TEXT[]",
    "ALTER TABLE evals DROP COLUMN IF EXISTS ls_run_id",
    "DROP INDEX IF EXISTS evals_ls_run_id",
]


async def _pg_conn() -> Any:
    import psycopg

    from deep_agent.src.settings import settings

    return await psycopg.AsyncConnection.connect(settings.database_uri)


async def _ensure_evals_table() -> None:
    conn = await _pg_conn()
    await conn.set_autocommit(True)
    try:
        for stmt in _EVALS_DDL_STATEMENTS:
            try:
                await conn.execute(stmt)
            except Exception as exc:
                log.warning(
                    "evals DDL statement failed: %s | stmt: %.120s", exc, stmt.strip()
                )
                raise
    finally:
        await conn.close()


_table_ensured = False


async def _ensure_evals_table_once() -> None:
    global _table_ensured
    if _table_ensured:
        return
    await _ensure_evals_table()
    _table_ensured = True


def _pg_row_to_dict(row: Any, cursor: Any) -> dict[str, Any]:
    return dict(zip([d.name for d in cursor.description], row))


async def _atomic_set_in_progress(
    config_hash: str, force: bool
) -> tuple[dict | None, bool]:
    """Set eval to in_progress in PostgreSQL.

    Returns (doc, is_new):
      - (None, False)         → in_progress exists on force; don't interrupt
      - (existing_doc, False) → already in_progress on normal trigger
      - (new_doc, True)       → fresh row inserted; start eval pod
    """
    await _ensure_evals_table_once()
    now = datetime.now(UTC)

    async with await _pg_conn() as conn:
        row = await conn.execute(
            "SELECT * FROM evals WHERE org=%s AND name=%s AND config_hash=%s "
            "AND eval_status='in_progress' ORDER BY created_at DESC LIMIT 1",
            (_AGENT_ORG, _AGENT_NAME, config_hash),
        )
        existing = await row.fetchone()
        if existing is not None:
            if force:
                return None, False
            return _pg_row_to_dict(existing, row), False

        cur = await conn.execute(
            """INSERT INTO evals
               (org, name, config_hash, eval_status, eval_score, pass, fail, error,
                results_detail, force_reeval, created_at, updated_at, completed_at)
               VALUES (%s,%s,%s,'in_progress',NULL,0,0,0,NULL,%s,%s,%s,NULL)
               RETURNING *""",
            (_AGENT_ORG, _AGENT_NAME, config_hash, force, now, now),
        )
        new_row = await cur.fetchone()
        doc = _pg_row_to_dict(new_row, cur)
        doc.pop("id", None)
        return doc, True


def _get_config_hash() -> str:
    """Get config hash from env var or compute from config dir."""
    env_hash = os.environ.get("AGENT_CONFIG_HASH")
    if env_hash:
        return env_hash
    return _compute_config_hash()


async def _fire_eval_run(config_hash: str, auth_token: str = "") -> None:
    """Fire-and-forget call to eval runner. Errors are logged, never raised."""
    if not _EVAL_RUNNER_URL:
        log.warning("EVAL_RUNNER_URL not set — eval pod not started")
        return
    try:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = (
                auth_token
                if auth_token.startswith("Bearer ")
                else f"Bearer {auth_token}"
            )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_EVAL_RUNNER_URL}/evals/run",
                json={
                    "config_hash": config_hash,
                    "org": _AGENT_ORG,
                    "name": _AGENT_NAME,
                },
                headers=headers,
            )
            log.info("eval_runner_called status=%s", resp.status_code)
    except Exception as exc:
        log.warning("eval_runner_call_failed: %s", exc)


def _require_eval_cases() -> None:
    """Raise HTTPException(400) if eval_cases.yaml is missing."""
    from fastapi import HTTPException

    config_dir = os.environ.get("CONFIG_PATH", "config/agent")
    eval_cases = Path(f"{config_dir}/evals/lightspeed-agent/eval_cases.yaml")
    if not eval_cases.exists():
        raise HTTPException(
            status_code=400,
            detail="No eval dataset found. Add eval cases before running evaluation.",
        )


@eval_mgmt_router.post("/trigger")
async def trigger_eval(request: Request) -> dict[str, Any]:
    """Cache-first eval trigger. Returns cached result or sets in_progress."""
    _require_eval_cases()

    config_hash = _get_config_hash()

    await _ensure_evals_table_once()
    async with await _pg_conn() as conn:
        row = await conn.execute(
            "SELECT * FROM evals WHERE org=%s AND name=%s AND config_hash=%s "
            "AND eval_status='completed' AND completed_at > NOW() - INTERVAL '24 hours' "
            "ORDER BY completed_at DESC LIMIT 1",
            (_AGENT_ORG, _AGENT_NAME, config_hash),
        )
        existing = await row.fetchone()
        if existing:
            doc = _pg_row_to_dict(existing, row)
            doc.pop("id", None)
            return {"cached": True, **doc}

    record, is_new = await _atomic_set_in_progress(config_hash, force=False)
    if not is_new:
        return {"eval_status": "in_progress", "message": "evaluation already running"}

    auth_token = request.headers.get("authorization", "")
    asyncio.create_task(_fire_eval_run(config_hash, auth_token))

    return {
        "eval_status": "in_progress",
        "queued": True,  # UI should call eval pod only when queued=True
        "config_hash": config_hash,
        "org": _AGENT_ORG,
        "name": _AGENT_NAME,
    }


@eval_mgmt_router.post("/force-trigger")
async def force_trigger_eval(request: Request) -> dict[str, Any]:
    """Force a fresh eval run, bypassing cache."""
    _require_eval_cases()

    config_hash = _get_config_hash()

    record, is_new = await _atomic_set_in_progress(config_hash, force=True)
    if not is_new:
        return {"eval_status": "in_progress", "message": "evaluation already running"}

    auth_token = request.headers.get("authorization", "")
    asyncio.create_task(_fire_eval_run(config_hash, auth_token))

    return {
        "eval_status": "in_progress",
        "queued": True,  # UI should call eval pod only when queued=True
        "config_hash": config_hash,
        "org": _AGENT_ORG,
        "name": _AGENT_NAME,
        "forced": True,
    }


_EVAL_STALE_TIMEOUT_MINUTES = int(os.environ.get("EVAL_STALE_TIMEOUT_MINUTES", "300"))


@eval_mgmt_router.get("/status")
async def eval_status() -> dict[str, Any]:
    """Return the latest eval record for this agent.

    Auto-expires in_progress rows older than EVAL_STALE_TIMEOUT_MINUTES (default 30)
    so a crashed eval run does not leave the UI stuck in Evaluating forever.
    """
    await _ensure_evals_table_once()
    async with await _pg_conn() as conn:
        await conn.execute(
            "UPDATE evals SET eval_status='error', completed_at=NOW(), updated_at=NOW() "
            "WHERE org=%s AND name=%s AND eval_status='in_progress' "
            "AND created_at < NOW() - INTERVAL '%s minutes'",
            (_AGENT_ORG, _AGENT_NAME, _EVAL_STALE_TIMEOUT_MINUTES),
        )
        row = await conn.execute(
            "SELECT * FROM evals WHERE org=%s AND name=%s "
            "ORDER BY created_at DESC LIMIT 1",
            (_AGENT_ORG, _AGENT_NAME),
        )
        result = await row.fetchone()
        if not result:
            return {"eval_status": "not_started", "message": "no eval runs yet"}
        doc = _pg_row_to_dict(result, row)
        doc.pop("id", None)
        return doc


@eval_mgmt_router.get("/results")
async def eval_results(request: Request) -> dict[str, Any]:
    """Return a completed eval report.

    Optional query param ``completed_at`` fetches a specific run by its
    completion timestamp.  Without it the most recent run is returned.
    """
    from fastapi import HTTPException

    completed_at = request.query_params.get("completed_at")
    await _ensure_evals_table_once()
    async with await _pg_conn() as conn:
        if completed_at:
            row = await conn.execute(
                "SELECT * FROM evals WHERE org=%s AND name=%s "
                "AND eval_status='completed' AND completed_at=%s "
                "LIMIT 1",
                (_AGENT_ORG, _AGENT_NAME, completed_at),
            )
        else:
            row = await conn.execute(
                "SELECT * FROM evals WHERE org=%s AND name=%s "
                "AND eval_status='completed' "
                "ORDER BY completed_at DESC LIMIT 1",
                (_AGENT_ORG, _AGENT_NAME),
            )
        result = await row.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="no completed eval results")
        doc = _pg_row_to_dict(result, row)
        doc.pop("id", None)
        return doc


@eval_mgmt_router.get("/history")
async def eval_history(request: Request) -> dict[str, Any]:
    """Return historical completed eval runs (scalars only, no results_detail)."""
    limit = min(int(request.query_params.get("limit", "20")), 100)
    await _ensure_evals_table_once()

    async with await _pg_conn() as conn:
        cur = await conn.execute(
            "SELECT eval_score, pass, fail, error, config_hash, "
            "       created_at, completed_at, "
            "       COUNT(*) OVER() AS total_count "
            "FROM evals "
            "WHERE org=%s AND name=%s AND eval_status='completed' "
            "ORDER BY completed_at DESC LIMIT %s",
            (_AGENT_ORG, _AGENT_NAME, limit),
        )
        cols = [d.name for d in cur.description]
        runs = []
        total = 0
        async for row in cur:
            d = dict(zip(cols, row))
            total = d.pop("total_count", 0)
            for k in ("created_at", "completed_at"):
                if d.get(k) is not None:
                    d[k] = d[k].isoformat() if hasattr(d[k], "isoformat") else str(d[k])
            runs.append(d)

    return {"runs": runs, "total": total}


@eval_mgmt_router.get("/trends")
async def eval_trends(request: Request) -> dict[str, Any]:
    """Return per-metric score trends across historical eval runs."""
    limit = min(int(request.query_params.get("limit", "20")), 100)
    await _ensure_evals_table_once()

    async with await _pg_conn() as conn:
        cur = await conn.execute(
            "SELECT results_detail->'summary'->'summary_stats'->'by_metric' as by_metric, "
            "       eval_score, completed_at "
            "FROM evals "
            "WHERE org=%s AND name=%s AND eval_status='completed' "
            "  AND results_detail IS NOT NULL "
            "ORDER BY completed_at DESC LIMIT %s",
            (_AGENT_ORG, _AGENT_NAME, limit),
        )

        metrics: dict[str, list[dict]] = {}
        overall: list[dict] = []

        async for row in cur:
            by_metric_raw, eval_score, completed_at = row
            ts = (
                completed_at.isoformat()
                if hasattr(completed_at, "isoformat")
                else str(completed_at)
            )
            overall.append({"completed_at": ts, "eval_score": eval_score})

            by_metric = by_metric_raw if isinstance(by_metric_raw, dict) else {}
            for metric_name, stats in by_metric.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(
                    {
                        "completed_at": ts,
                        "pass_rate": stats.get("pass_rate")
                        if isinstance(stats, dict)
                        else None,
                    }
                )

    return {"metrics": metrics, "overall": overall}
