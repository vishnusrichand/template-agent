"""Eval runner REST API.

Runs as a separate Deployment (KEDA HTTP-scaled, min=0 max=2) in the same
namespace as the agentpod. Reads eval_cases.yaml and system.yaml from the
agent config PVC (written by agent-engine at deploy time). Results are
written to Postgres so the agentpod /evals/results route can serve them.

Endpoints:
    POST /evals/run                Run all patterns
    POST /evals/run/{pattern}      Run one pattern (tool_use / hitl / structured_output / multi_agent)
    GET  /evals/status             Current run status
    GET  /evals/results            Latest run summary
    GET  /evals/results/{run_id}   Specific run summary
    GET  /health                   Liveness check
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from eval_cases import filter_cases_by_tag, load_cases
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

AGENT_URL = os.environ.get("AGENT_HOST", "http://localhost:5002")


# Derive eval file paths from AGENT_CONFIG_DIR so no separate env vars needed.
# Can still be overridden explicitly via EVAL_CASES_PATH / EVAL_SYSTEM_CONFIG.
def _resolve_config_dir() -> str:
    """Return AGENT_CONFIG_DIR from env, or auto-detect the config directory.

    Walks up from this file looking for config/agent
    (local dev layout: eval-runner/../config/agent).
    """
    if val := os.environ.get("AGENT_CONFIG_DIR"):
        return val
    here = Path(__file__).resolve().parent
    for candidate in [
        here.parent / "config" / "agent",
        here / ".." / "config" / "agent",
    ]:
        resolved = candidate.resolve()
        if resolved.is_dir():
            log.info("AGENT_CONFIG_DIR not set — using auto-detected: %s", resolved)
            return str(resolved)
    log.warning(
        "AGENT_CONFIG_DIR not set and config/agent not found; falling back to /agent-config"
    )
    return "/agent-config"


_agent_config_dir = _resolve_config_dir()
EVAL_CASES_PATH = Path(
    os.environ.get(
        "EVAL_CASES_PATH", f"{_agent_config_dir}/evals/lightspeed-agent/eval_cases.yaml"
    )
)
EVAL_SYSTEM_CONFIG = Path(
    os.environ.get(
        "EVAL_SYSTEM_CONFIG", f"{_agent_config_dir}/evals/lightspeed-agent/system.yaml"
    )
)
EVAL_OUTPUT_DIR = Path(
    os.environ.get("EVAL_OUTPUT_DIR", tempfile.gettempdir() + "/eval_output")
)

AGENT_ORG = os.environ.get("AI_PLATFORM_AGENT_ORG", "default")
AGENT_NAME = os.environ.get("AI_PLATFORM_AGENT_NAME", "agent")


_HASH_EXTENSIONS = {".md", ".yaml", ".json"}
_HASH_EXCLUDE_DIRS = {"evals", "deployment"}


def _compute_config_hash(config_dir: str) -> str:
    """SHA256 of behavior-relevant config files (prompts, skills, runtime, tools)."""
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


AGENT_CONFIG_HASH = os.environ.get("AGENT_CONFIG_HASH") or _compute_config_hash(
    _agent_config_dir
)

AGENT_AUTH_TOKEN = os.environ.get("AGENT_AUTH_TOKEN", "")
EVAL_MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "3"))
ALL_PATTERNS = ["tool_use", "structured_output", "hitl", "multi_agent"]

log.info(
    "eval_api config: agent_url=%s org=%s name=%s config_hash=%s "
    "eval_cases=%s eval_system=%s max_concurrency=%d",
    AGENT_URL,
    AGENT_ORG,
    AGENT_NAME,
    AGENT_CONFIG_HASH,
    EVAL_CASES_PATH,
    EVAL_SYSTEM_CONFIG,
    EVAL_MAX_CONCURRENCY,
)

# ── State ─────────────────────────────────────────────────────────────────────

_status: dict[str, Any] = {"state": "idle", "run_id": None}
_latest_result: dict[str, Any] | None = None


# ── Eval runner ───────────────────────────────────────────────────────────────


def _find_eval_files(pattern: str | None) -> list[Path]:
    """Return one temp file per tag (parallel all) or one file for a specific pattern.

    When pattern is None every tag gets its own temp file so _run_eval can run
    them concurrently via asyncio.gather while keeping conversations sequential
    within each tag subprocess.
    """
    if not EVAL_CASES_PATH.exists():
        raise FileNotFoundError(
            "No eval dataset found. Add eval cases before running evaluation."
        )

    cases = load_cases(EVAL_CASES_PATH)
    log.info("Loaded %d eval cases from %s", len(cases), EVAL_CASES_PATH)

    if pattern is None:
        # One temp file per tag — preserves ordering within each tag
        tags: list[str] = list(
            dict.fromkeys(str(c["tag"]) for c in cases if c.get("tag"))
        )
        if not tags:
            log.warning(
                "No tags found in eval cases — running full file as single batch"
            )
            return [EVAL_CASES_PATH]
        log.info("Splitting cases into %d tag batches: %s", len(tags), tags)
        files: list[Path] = []
        for tag in tags:
            filtered = filter_cases_by_tag(EVAL_CASES_PATH, tag)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix=f"eval_{tag}_", delete=False
            )
            yaml.dump(filtered, tmp, default_flow_style=False, allow_unicode=True)
            tmp.close()
            files.append(Path(tmp.name))
            log.debug("Tag '%s': %d case(s) → %s", tag, len(filtered), tmp.name)
        return files

    # Specific pattern — single filtered temp file
    filtered = filter_cases_by_tag(EVAL_CASES_PATH, pattern)
    if not filtered:
        raise FileNotFoundError(
            f"No eval cases found for pattern '{pattern}'. "
            "Add cases with this tag before running evaluation."
        )
    log.info("Pattern '%s': %d matching case(s)", pattern, len(filtered))
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"eval_{pattern}_", delete=False
    )
    yaml.dump(filtered, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return [Path(tmp.name)]


def _get_system_yaml_content() -> str:
    """Return system.yaml content with postgres credentials filled from env vars."""
    if not EVAL_SYSTEM_CONFIG.exists():
        raise FileNotFoundError(
            "Eval system configuration not found. "
            "Ensure the agent config is mounted and system.yaml is present."
        )

    config = yaml.safe_load(EVAL_SYSTEM_CONFIG.read_text())

    for backend in config.get("storage", []):
        if backend.get("type") == "postgres":
            backend["host"] = os.environ.get(
                "POSTGRES_HOST", backend.get("host", "localhost")
            )
            backend["port"] = int(
                os.environ.get("POSTGRES_PORT", str(backend.get("port", 5432)))
            )
            backend["database"] = os.environ.get(
                "POSTGRES_DB", backend.get("database", "template_agent")
            )
            backend["user"] = os.environ.get(
                "POSTGRES_USER", backend.get("user", "postgres")
            )
            backend["password"] = os.environ.get(
                "POSTGRES_PASSWORD", backend.get("password", "")
            )

    return yaml.dump(config, default_flow_style=False, allow_unicode=True)


def _system_yaml_path() -> Path:
    """Write system.yaml with env-injected credentials to a temp file and return its path."""
    content = _get_system_yaml_content()
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="eval_system_", delete=False
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _run_eval_pattern_sync(
    eval_file: Path, system_yaml: Path, output_dir: Path, auth_token: str = ""
) -> int:
    """Blocking subprocess call — must be run in a thread pool, not the event loop."""
    runner = Path(__file__).parent / "run_eval.py"
    env = dict(os.environ)
    if auth_token:
        env["AGENT_AUTH_TOKEN"] = auth_token  # user session token for MCP tool calls
    cmd = [
        sys.executable,
        str(runner),
        "--agent-url",
        AGENT_URL,
        "--eval-data",
        str(eval_file),
        "--system",
        str(system_yaml),
        "--output-dir",
        str(output_dir),
    ]
    log.info("Spawning subprocess: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, env=env)
    log.info(
        "Subprocess exited: file=%s exit_code=%d", eval_file.name, result.returncode
    )
    if result.returncode != 0:
        log.warning(
            "Non-zero exit for %s — check run_eval.py output above", eval_file.name
        )
    return result.returncode


async def _run_eval_pattern(
    eval_file: Path, system_yaml: Path, output_dir: Path, auth_token: str = ""
) -> int:
    """Run one eval pattern in a thread pool so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _run_eval_pattern_sync, eval_file, system_yaml, output_dir, auth_token
    )


def _score_from_counts(passed: int, failed: int, errors: int) -> tuple[str, float]:
    total = passed + failed + errors
    if total == 0:
        return "error", 0.0
    score = round(passed / total, 3)
    if errors > 0 and passed == 0:
        status = "error"
    elif passed == total:
        status = "passed"
    else:
        status = "failed"
    return status, score


async def _run_eval(
    pattern: str | None,
    config_hash: str | None = None,
    org: str | None = None,
    name: str | None = None,
    auth_token: str = "",
    run_id: str = "",
) -> None:
    """Core eval runner — invoked in background."""
    global _latest_result

    log.info(
        "Eval run started: run_id=%s pattern=%s auth_token_present=%s",
        run_id,
        pattern or "all",
        bool(auth_token),
    )

    tmp_files: list[Path] = []
    try:
        system_yaml = _system_yaml_path()
        eval_files = _find_eval_files(pattern)
        # Track temp files created by _find_eval_files (tag-filtered) for cleanup
        tmp_files = [f for f in eval_files if f.parent != EVAL_CASES_PATH.parent]
    except FileNotFoundError as exc:
        log.error("eval_setup_failed: %s", exc)
        _status.update({"state": "error", "run_id": run_id})
        return

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_output = EVAL_OUTPUT_DIR / run_id
    run_output.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(EVAL_MAX_CONCURRENCY)

    async def _run_one(eval_file: Path) -> int:
        async with sem:
            log.info("Running %s", eval_file.name)
            rc = await _run_eval_pattern(eval_file, system_yaml, run_output, auth_token)
            log.info("%s → exit code %d", eval_file.name, rc)
            return rc

    run_started_at = datetime.now(timezone.utc)
    try:
        await asyncio.gather(*[_run_one(f) for f in eval_files])
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)
        system_yaml.unlink(missing_ok=True)

    total_pass, total_fail, total_error = 0, 0, 0
    eval_status, eval_score = _score_from_counts(total_pass, total_fail, total_error)

    result: dict[str, Any] = {
        "run_id": run_id,
        "org": AGENT_ORG,
        "name": AGENT_NAME,
        "config_hash": AGENT_CONFIG_HASH,
        "eval_status": eval_status,
        "eval_score": eval_score,
        "pass": total_pass,
        "fail": total_fail,
        "error": total_error,
        "output_dir": str(run_output),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _latest_result = result
    _status.update({"state": "completed", "run_id": run_id})

    # Build rich results_detail from the evaluation_results PostgreSQL table
    results_detail = dict(result)
    db_data = await asyncio.get_running_loop().run_in_executor(
        None, load_results_since, run_started_at
    )
    results_detail.update(db_data)
    if not db_data:
        log.warning(
            "No DB results found for run_id=%s — summary will show zeros", run_id
        )

    # Recompute scalars from DB summary — file storage was removed so
    # _aggregate_summaries() returned zeros; DB summary has the real counts.
    overall = (
        results_detail.get("summary", {}).get("summary_stats", {}).get("overall", {})
    )
    if overall:
        total_pass = int(overall.get("PASS", total_pass))
        total_fail = int(overall.get("FAIL", total_fail))
        total_error = int(overall.get("ERROR", total_error))
        eval_status, eval_score = _score_from_counts(
            total_pass, total_fail, total_error
        )
        result.update(
            {
                "eval_status": eval_status,
                "eval_score": eval_score,
                "pass": total_pass,
                "fail": total_fail,
                "error": total_error,
            }
        )
        results_detail.update(result)

    # Log after DB recompute so values are accurate
    log.info(
        "Eval complete: status=%s score=%.3f pass=%d fail=%d error=%d",
        eval_status,
        eval_score,
        total_pass,
        total_fail,
        total_error,
    )

    # Write results to Postgres so agentpod /evals/results can serve them
    try:
        write_eval_result(
            passed=total_pass,
            failed=total_fail,
            errors=total_error,
            eval_score=eval_score,
            ls_run_ids=results_detail.get("ls_run_ids"),
            results_detail=results_detail,
            config_hash=config_hash,
            org=org,
            name=name,
        )
    except Exception as exc:
        log.error(
            "postgres_write_failed (%s) — results still available locally",
            type(exc).__name__,
        )


# ── FastAPI app ────────────────────────────────────────────────────────────────
# eval_cases.yaml and system.yaml are pre-written to the PVC by agent-engine
# at deploy time — no startup auto-run or case management needed here.


from eval_postgres import (  # noqa: E402
    ensure_table,
    get_results_by_run_id,
    load_results_since,
    write_eval_result,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    await asyncio.get_running_loop().run_in_executor(None, ensure_table)
    yield


app = FastAPI(title="eval-runner", version="2.0.0", lifespan=_lifespan)


class EvalRunBody(BaseModel):
    """Request body for POST /evals/run — carries agent identity from the trigger response."""

    config_hash: str | None = None
    org: str | None = None
    name: str | None = None


def _extract_token(request: Request) -> str:
    """Extract user auth token from Authorization header to forward to the agent.

    The eval runner is an internal service (NetworkPolicy/ClusterIP) — no auth
    enforcement on its own endpoints. Token is only used for agent subprocess calls.
    """
    auth_header = request.headers.get("authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ")
        else ""
    )
    return token or AGENT_AUTH_TOKEN


async def _trigger(
    pattern: str | None,
    background: BackgroundTasks,
    body: EvalRunBody | None = None,
    auth_token: str = "",
) -> dict[str, Any]:
    if _status["state"] == "running":
        log.warning(
            "Eval trigger rejected — run %s already in progress", _status.get("run_id")
        )
        raise HTTPException(
            status_code=409, detail="An eval run is already in progress"
        )
    if not EVAL_CASES_PATH.exists():
        log.error("Eval trigger rejected — eval_cases not found: %s", EVAL_CASES_PATH)
        raise HTTPException(
            status_code=400,
            detail="No eval dataset found. Add eval cases before running evaluation.",
        )
    if not EVAL_SYSTEM_CONFIG.exists():
        log.error(
            "Eval trigger rejected — system config not found: %s", EVAL_SYSTEM_CONFIG
        )
        raise HTTPException(
            status_code=400,
            detail="Eval system configuration not found. Ensure the agent config is mounted.",
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    config_hash = body.config_hash if body else None
    org = body.org if body else None
    name = body.name if body else None
    log.info(
        "Eval triggered: run_id=%s pattern=%s org=%s name=%s config_hash=%s auth_token_present=%s",
        run_id,
        pattern or "all",
        org,
        name,
        config_hash,
        bool(auth_token),
    )
    _status.update({"state": "running", "run_id": run_id})
    background.add_task(_run_eval, pattern, config_hash, org, name, auth_token, run_id)
    return {
        "run_id": run_id,
        "status": "started",
        "pattern": pattern or "all",
    }


@app.post("/evals/run", status_code=202)
async def run_all(
    request: Request, background: BackgroundTasks, body: EvalRunBody = EvalRunBody()
) -> dict[str, Any]:
    """Run all eval patterns against the agent."""
    return await _trigger(None, background, body, _extract_token(request))


@app.post("/evals/run/{pattern}", status_code=202)
async def run_pattern(
    pattern: str, request: Request, background: BackgroundTasks
) -> dict[str, Any]:
    """Run one eval pattern (tool_use / hitl / structured_output / multi_agent)."""
    if pattern not in ALL_PATTERNS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pattern '{pattern}'. Valid: {ALL_PATTERNS}",
        )
    return await _trigger(pattern, background, auth_token=_extract_token(request))


@app.get("/evals/status")
async def get_status() -> dict[str, Any]:
    """Return current run state: idle | running | completed | error."""
    return _status


@app.get("/evals/results")
async def get_latest_results() -> JSONResponse:
    """Return the latest run summary (in-memory)."""
    if _latest_result is None:
        raise HTTPException(status_code=404, detail="No eval results available yet")
    return JSONResponse(_latest_result)


@app.get("/evals/results/{run_id}")
async def get_run_results(run_id: str) -> JSONResponse:
    """Return results for a specific run ID from the evaluation_results Postgres table."""
    try:
        rows = await asyncio.get_running_loop().run_in_executor(
            None, get_results_by_run_id, run_id
        )
    except Exception as exc:
        log.error("get_run_results failed for run_id=%s: %s", run_id, exc)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve eval results"
        ) from exc
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"No results found for run '{run_id}'"
        )
    return JSONResponse({"run_id": run_id, "results": rows})


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — always returns ok."""
    return {"status": "ok"}


@app.post("/mcp")
async def mcp_stub() -> dict[str, str]:
    """Stub to suppress 404 noise from lightspeed-eval MCP probe on startup."""
    return {"error": "MCP not supported on eval runner"}
