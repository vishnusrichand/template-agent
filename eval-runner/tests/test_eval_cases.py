"""Tests for eval_cases.py — CRUD for eval_cases.yaml."""
from __future__ import annotations

import pytest
import eval_cases


# ── _normalize_keywords ────────────────────────────────────────────────────────

def test_normalize_keywords_comma_string():
    result = eval_cases._normalize_keywords("22.9, Normal")
    assert result == [["22.9"], ["Normal"]]


def test_normalize_keywords_flat_list():
    result = eval_cases._normalize_keywords(["22.9", "Normal"])
    assert result == [["22.9"], ["Normal"]]


def test_normalize_keywords_already_nested():
    nested = [["22.9", "22.8"], ["Normal"]]
    result = eval_cases._normalize_keywords(nested)
    assert result == nested


def test_normalize_keywords_empty_list():
    assert eval_cases._normalize_keywords([]) == []


def test_normalize_keywords_string_strips_whitespace():
    result = eval_cases._normalize_keywords("  a ,  b  ")
    assert result == [["a"], ["b"]]


# ── get_defaults_for_tag ───────────────────────────────────────────────────────

def test_get_defaults_tool_use():
    d = eval_cases.get_defaults_for_tag("tool_use")
    assert "custom:answer_correctness" in d["turn_metrics"]


def test_get_defaults_hitl():
    d = eval_cases.get_defaults_for_tag("hitl")
    assert d.get("hitl") is True
    assert "custom:intent_eval" in d["turn_metrics"]


def test_get_defaults_structured_output():
    d = eval_cases.get_defaults_for_tag("structured_output")
    assert "geval:tone_safety" in d["turn_metrics"]


def test_get_defaults_unknown_tag():
    d = eval_cases.get_defaults_for_tag("unknown_tag")
    assert d == {"turn_metrics": ["custom:answer_correctness"]}


def test_get_defaults_returns_deep_copy():
    """Mutations to the returned dict must not affect subsequent calls."""
    d1 = eval_cases.get_defaults_for_tag("tool_use")
    d1["turn_metrics"].append("injected")
    d2 = eval_cases.get_defaults_for_tag("tool_use")
    assert "injected" not in d2["turn_metrics"]


# ── _load_cases: missing file warning ─────────────────────────────────────────

def test_load_cases_logs_warning_when_missing(eval_dir, caplog):
    cases = eval_cases._load_cases(eval_dir)
    assert cases == []
    assert "not found" in caplog.text


# ── create_case ───────────────────────────────────────────────────────────────

def test_create_case_basic(eval_dir):
    result = eval_cases.create_case(
        eval_dir,
        query="What is my BMI?",
        expected_response="Your BMI is 22.9",
        tag="tool_use",
    )
    assert result["status"] == "created"
    assert result["tag"] == "tool_use"
    assert len(result["case_id"]) == 12


def test_create_case_auto_adds_keywords_eval(eval_dir):
    eval_cases.create_case(
        eval_dir,
        query="Q", expected_response="A", tag="tool_use",
        expected_keywords=["22.9", "Normal"],
    )
    cases = eval_cases._load_cases(eval_dir)
    turn = cases[0]["turns"][0]
    assert "custom:keywords_eval" in turn["turn_metrics"]
    assert turn["expected_keywords"] == [["22.9"], ["Normal"]]


def test_create_case_auto_adds_tool_eval(eval_dir):
    eval_cases.create_case(
        eval_dir,
        query="Q", expected_response="A", tag="tool_use",
        expected_tool_calls=["calculate_bmi"],
    )
    cases = eval_cases._load_cases(eval_dir)
    turn = cases[0]["turns"][0]
    assert "custom:tool_eval" in turn["turn_metrics"]
    assert turn["expected_tool_calls"] == ["calculate_bmi"]


def test_create_case_hitl_sets_flag_and_intent(eval_dir):
    eval_cases.create_case(
        eval_dir,
        query="Send email", expected_response="Awaiting approval", tag="hitl",
    )
    cases = eval_cases._load_cases(eval_dir)
    turn = cases[0]["turns"][0]
    assert turn.get("hitl") is True
    assert "expected_intent" in turn


def test_create_case_custom_metrics_override(eval_dir):
    eval_cases.create_case(
        eval_dir,
        query="Q", expected_response="A", tag="tool_use",
        metrics=["ragas:faithfulness"],
    )
    cases = eval_cases._load_cases(eval_dir)
    turn = cases[0]["turns"][0]
    assert turn["turn_metrics"] == ["ragas:faithfulness"]


def test_create_case_custom_conversation_id(eval_dir):
    result = eval_cases.create_case(
        eval_dir,
        query="Q", expected_response="A", tag="tool_use",
        conversation_id="my_fixed_id",
    )
    assert result["conversation_id"] == "my_fixed_id"


def test_create_case_appends_multiple(eval_dir):
    eval_cases.create_case(eval_dir, query="Q1", expected_response="A1", tag="tool_use")
    eval_cases.create_case(eval_dir, query="Q2", expected_response="A2", tag="hitl")
    listed = eval_cases.list_cases(eval_dir)
    assert len(listed) == 2


# ── list_cases ────────────────────────────────────────────────────────────────

def test_list_cases_empty(eval_dir):
    assert eval_cases.list_cases(eval_dir) == []


def test_list_cases_returns_metadata(eval_dir):
    eval_cases.create_case(
        eval_dir, query="Q", expected_response="A", tag="tool_use"
    )
    listed = eval_cases.list_cases(eval_dir)
    assert listed[0]["tag"] == "tool_use"
    assert listed[0]["turn_count"] == 1


def test_list_cases_unknown_case_id_for_unindexed(eval_dir):
    """Cases added manually (no index entry) show case_id='unknown'."""
    import yaml
    (eval_cases._cases_path(eval_dir)).write_text(
        yaml.dump([{
            "conversation_group_id": "manual",
            "tag": "tool_use",
            "turns": [{"query": "Q", "turn_id": "t1"}],
        }])
    )
    listed = eval_cases.list_cases(eval_dir)
    assert listed[0]["case_id"] == "unknown"


# ── delete_case ───────────────────────────────────────────────────────────────

def test_delete_case_success(eval_dir):
    result = eval_cases.create_case(
        eval_dir, query="Q", expected_response="A", tag="tool_use"
    )
    deleted = eval_cases.delete_case(eval_dir, result["case_id"])
    assert deleted is True
    assert eval_cases.list_cases(eval_dir) == []


def test_delete_case_not_found_returns_false(eval_dir, caplog):
    deleted = eval_cases.delete_case(eval_dir, "nonexistent")
    assert deleted is False
    assert "not found" in caplog.text


def test_delete_case_removes_from_index(eval_dir):
    result = eval_cases.create_case(
        eval_dir, query="Q", expected_response="A", tag="tool_use"
    )
    eval_cases.delete_case(eval_dir, result["case_id"])
    index = eval_cases._load_index(eval_dir)
    assert result["case_id"] not in index


# ── filter_cases_by_tag ───────────────────────────────────────────────────────

def test_filter_cases_by_tag_match(eval_dir):
    eval_cases.create_case(eval_dir, query="Q1", expected_response="A1", tag="tool_use")
    eval_cases.create_case(eval_dir, query="Q2", expected_response="A2", tag="hitl")
    results = eval_cases.filter_cases_by_tag(eval_dir, "tool_use")
    assert len(results) == 1
    assert results[0]["tag"] == "tool_use"


def test_filter_cases_by_tag_no_match(eval_dir):
    eval_cases.create_case(eval_dir, query="Q", expected_response="A", tag="tool_use")
    assert eval_cases.filter_cases_by_tag(eval_dir, "hitl") == []


def test_filter_cases_by_tag_empty(eval_dir):
    assert eval_cases.filter_cases_by_tag(eval_dir, "tool_use") == []
