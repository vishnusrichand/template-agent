"""Eval case management — CRUD for eval_cases.yaml.

All test cases live in a single eval_cases.yaml file with a tag per conversation.
Cases are indexed by case_id in .cases_index.json for fast lookup.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

CASES_FILE = "eval_cases.yaml"
INDEX_FILE = ".cases_index.json"

# ── Supported metrics registry ────────────────────────────────────────────────

SUPPORTED_METRICS: dict[str, Any] = {
    "turn_level": {
        "ragas": [
            "faithfulness",
            "response_relevancy",
            "context_recall",
            "context_relevance",
            "context_utilization",
            "context_precision",
        ],
        "custom": [
            "answer_correctness",
            "intent_eval",
            "keywords_eval",
            "tool_eval",
            "proposal_status",
            "proposal_evaluation_correctness",
        ],
        "nlp": ["bleu", "rouge", "semantic_similarity_distance"],
        "geval": ["<user-defined — add criteria in system.yaml metrics_metadata>"],
        "script": ["action_eval"],
    },
    "conversation_level": {
        "deepeval": [
            "conversation_completeness",
            "conversation_relevancy",
            "knowledge_retention",
        ],
        "geval": ["<user-defined>"],
    },
}

# ── Default metrics per tag ───────────────────────────────────────────────────

_DEFAULT_METRICS: dict[str, dict[str, Any]] = {
    "tool_use": {
        "turn_metrics": ["custom:answer_correctness"],
        # keywords_eval added only if expected_keywords provided
        # tool_eval added only if expected_tool_calls provided
    },
    "hitl": {
        "turn_metrics": ["custom:intent_eval"],
        "hitl": True,
        "expected_intent": "request approval before taking action",  # fixed default
    },
    "structured_output": {
        "turn_metrics": ["geval:tone_safety"],
        # keywords_eval added only if expected_keywords provided
    },
    "multi_agent": {
        "turn_metrics": ["custom:answer_correctness", "geval:delegation_compliance"],
        "turn_metrics_metadata": {
            "custom:tool_eval": {"ordered": False, "full_match": False},
        },
        # keywords_eval added only if expected_keywords provided
        # tool_eval added only if expected_tool_calls provided
    },
}
_DEFAULT_FALLBACK: dict[str, Any] = {
    "turn_metrics": ["custom:answer_correctness"],
}


def get_defaults_for_tag(tag: str) -> dict[str, Any]:
    """Return a copy of the default metric config for the given tag."""
    import copy

    return copy.deepcopy(_DEFAULT_METRICS.get(tag, _DEFAULT_FALLBACK))


# ── File helpers ──────────────────────────────────────────────────────────────


def _cases_path(eval_data_dir: str) -> Path:
    return Path(eval_data_dir) / CASES_FILE


def _index_path(eval_data_dir: str) -> Path:
    return Path(eval_data_dir) / INDEX_FILE


def _load_cases(eval_data_dir: str) -> list[dict[str, Any]]:
    p = _cases_path(eval_data_dir)
    if not p.exists():
        log.warning("eval_cases file not found: %s", p)
        return []
    return yaml.safe_load(p.read_text()) or []


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load cases from eval_cases.yaml, normalizing keyword fields at read time.

    Accepts a file path or a directory (in which case eval_cases.yaml is appended).
    """
    p = Path(path)
    if p.is_dir():
        p = p / CASES_FILE
    if not p.exists():
        log.warning("eval_cases file not found: %s", p)
        return []
    cases = yaml.safe_load(p.read_text()) or []
    for conv in cases:
        for turn in conv.get("turns", []):
            if "expected_keywords" in turn:
                turn["expected_keywords"] = _normalize_keywords(
                    turn["expected_keywords"]
                )
    return cases


def filter_cases_by_tag(path: str | Path, tag: str) -> list[dict[str, Any]]:
    """Return all conversations matching the given tag."""
    return [c for c in load_cases(path) if c.get("tag") == tag]


def _save_cases(eval_data_dir: str, cases: list[dict[str, Any]]) -> None:
    # Dump each conversation group separately with a blank line between them
    header = "# Managed by eval sidecar — use POST /evals/cases to add cases.\n\n"
    body = "\n\n".join(
        yaml.dump(
            [conv], default_flow_style=False, allow_unicode=True, sort_keys=False
        ).rstrip()
        for conv in cases
    )
    _cases_path(eval_data_dir).write_text(header + body + "\n")


def _load_index(eval_data_dir: str) -> dict[str, str]:
    p = _index_path(eval_data_dir)
    if not p.exists():
        return {}
    return json.loads(p.read_text())  # type: ignore[no-any-return]


def _save_index(eval_data_dir: str, index: dict[str, str]) -> None:
    _index_path(eval_data_dir).write_text(json.dumps(index, indent=2))


# ── Case CRUD ─────────────────────────────────────────────────────────────────


def _normalize_keywords(keywords: list | str) -> list[list[str]]:
    """Normalize keywords to list[list[str]] (the format lightspeed-eval expects).

    Accepts:
        "22.9, Normal"              → [["22.9"], ["Normal"]]
        ["22.9", "Normal"]          → [["22.9"], ["Normal"]]
        [["22.9","22.8"], ["Normal"]] → used as-is (full control)
    """
    if isinstance(keywords, str):
        return [[kw.strip()] for kw in keywords.split(",") if kw.strip()]
    if isinstance(keywords, list) and keywords:
        # Already nested list-of-lists?
        if isinstance(keywords[0], list):
            return keywords
        # Flat list of strings → each becomes its own mandatory group
        return [[str(kw)] for kw in keywords]
    return keywords


def create_case(
    eval_data_dir: str,
    *,
    query: str,
    expected_response: str,
    tag: str,
    expected_intent: str | None = None,
    expected_keywords: list | str | None = None,
    expected_tool_calls: list | None = None,
    metrics: list[str] | None = None,
    metrics_metadata: dict[str, Any] | None = None,
    conversation_id: str | None = None,
) -> dict[str, str]:
    """Create a new eval case and append it to eval_cases.yaml.

    Args:
        eval_data_dir: Path to the /evals PVC directory.
        query: The user query to send to the agent.
        expected_response: Reference answer for scoring.
        tag: Eval tag (tool_use / hitl / structured_output / multi_agent / ...).
        expected_intent: Optional expected intent string.
        expected_keywords: Optional list of keyword alternation groups.
        expected_tool_calls: Optional expected tool call sequences.
        metrics: Override default turn_metrics for this tag.
        metrics_metadata: Override per-metric config.
        conversation_id: Optional stable ID (generated if omitted).

    Returns:
        dict with case_id, conversation_id, tag, status.
    """
    case_id = str(uuid.uuid4()).replace("-", "")[:12]
    conv_id = conversation_id or f"{tag}_{case_id}"

    # Build turn with tag defaults merged with user overrides
    defaults = get_defaults_for_tag(tag)
    turn: dict[str, Any] = {
        "turn_id": "turn_1",
        "query": query,
        "expected_response": expected_response,
    }

    # Apply hitl flag from defaults if set
    if defaults.get("hitl"):
        turn["hitl"] = True

    # expected_intent: use explicit value, fall back to tag default, then omit
    resolved_intent = expected_intent or defaults.get("expected_intent")
    if resolved_intent:
        turn["expected_intent"] = resolved_intent
    if expected_keywords is not None:
        turn["expected_keywords"] = _normalize_keywords(expected_keywords)
    if expected_tool_calls:
        turn["expected_tool_calls"] = expected_tool_calls

    # Metrics: user override wins over tag defaults
    resolved_metrics = list(
        metrics if metrics is not None else defaults.get("turn_metrics", [])
    )

    # Auto-add keywords_eval only when expected_keywords are provided
    if expected_keywords is not None and "custom:keywords_eval" not in resolved_metrics:
        resolved_metrics.append("custom:keywords_eval")

    # Auto-add tool_eval only when expected_tool_calls are provided
    if expected_tool_calls and "custom:tool_eval" not in resolved_metrics:
        resolved_metrics.append("custom:tool_eval")

    turn["turn_metrics"] = resolved_metrics

    # Metrics metadata: merge defaults + user overrides
    meta = dict(defaults.get("turn_metrics_metadata") or {})
    if metrics_metadata:
        meta.update(metrics_metadata)
    # Auto-add tool_eval metadata if tool_eval was auto-added
    if (
        expected_tool_calls
        and "custom:tool_eval" in resolved_metrics
        and "custom:tool_eval" not in meta
    ):
        meta["custom:tool_eval"] = {"ordered": False, "full_match": False}
    if meta:
        turn["turn_metrics_metadata"] = meta

    conversation: dict[str, Any] = {
        "conversation_group_id": conv_id,
        "description": f"Auto-created — case_id={case_id}",
        "tag": tag,
        "turns": [turn],
    }

    # Append to cases file
    cases = _load_cases(eval_data_dir)
    cases.append(conversation)
    _save_cases(eval_data_dir, cases)

    # Update index
    index = _load_index(eval_data_dir)
    index[case_id] = conv_id
    _save_index(eval_data_dir, index)

    log.info("Created case %s (conv=%s, tag=%s)", case_id, conv_id, tag)
    return {
        "case_id": case_id,
        "conversation_id": conv_id,
        "tag": tag,
        "status": "created",
    }


def list_cases(eval_data_dir: str) -> list[dict[str, Any]]:
    """Return all cases with their conversation_group_id, tag, and metrics."""
    cases = _load_cases(eval_data_dir)
    index = _load_index(eval_data_dir)
    # Reverse index: conv_id → case_id
    conv_to_case = {v: k for k, v in index.items()}

    result = []
    for conv in cases:
        conv_id = conv.get("conversation_group_id", "")
        turns = conv.get("turns", [])
        result.append(
            {
                "case_id": conv_to_case.get(conv_id, "unknown"),
                "conversation_id": conv_id,
                "tag": conv.get("tag", ""),
                "description": conv.get("description", ""),
                "turn_count": len(turns),
                "metrics": turns[0].get("turn_metrics", []) if turns else [],
            }
        )
    log.info("list_cases: returned %d cases from %s", len(result), eval_data_dir)
    return result


def delete_case(eval_data_dir: str, case_id: str) -> bool:
    """Remove a case by case_id. Returns True if found and removed."""
    index = _load_index(eval_data_dir)
    conv_id = index.get(case_id)
    if not conv_id:
        log.warning("delete_case: case_id=%s not found in index", case_id)
        return False

    cases = _load_cases(eval_data_dir)
    new_cases = [c for c in cases if c.get("conversation_group_id") != conv_id]
    if len(new_cases) == len(cases):
        return False

    _save_cases(eval_data_dir, new_cases)
    del index[case_id]
    _save_index(eval_data_dir, index)
    log.info("Deleted case %s (conv=%s)", case_id, conv_id)
    return True
