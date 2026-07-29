"""Tests for run_eval.py — SSE parsing, agent interaction helpers."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import run_eval


# ── _parse_sse_stream ─────────────────────────────────────────────────────────

def test_parse_sse_stream_single_event():
    lines = ["event: updates", 'data: {"key": "value"}', ""]
    events = run_eval._parse_sse_stream(lines)
    assert len(events) == 1
    assert events[0] == ("updates", {"key": "value"})


def test_parse_sse_stream_multiple_events():
    lines = [
        "event: updates", 'data: {"a": 1}', "",
        "event: events", 'data: {"b": 2}', "",
    ]
    events = run_eval._parse_sse_stream(lines)
    assert len(events) == 2
    assert events[0] == ("updates", {"a": 1})
    assert events[1] == ("events", {"b": 2})


def test_parse_sse_stream_invalid_json_skipped():
    lines = ["event: updates", "data: not-json", ""]
    events = run_eval._parse_sse_stream(lines)
    assert events == []


def test_parse_sse_stream_default_event_type_is_message():
    lines = ['data: {"x": 1}', ""]
    events = run_eval._parse_sse_stream(lines)
    assert events[0][0] == "message"


def test_parse_sse_stream_trailing_data_without_blank_line():
    lines = ['data: {"z": 9}']  # no trailing blank line
    events = run_eval._parse_sse_stream(lines)
    assert len(events) == 1


def test_parse_sse_stream_empty_input():
    assert run_eval._parse_sse_stream([]) == []


def test_parse_sse_stream_trailing_invalid_json():
    lines = ["event: updates", "data: not-json"]  # no blank line, bad JSON
    assert run_eval._parse_sse_stream(lines) == []


def test_parse_sse_stream_blank_line_without_data():
    lines = ["event: updates", ""]
    assert run_eval._parse_sse_stream(lines) == []


# ── _extract_text ─────────────────────────────────────────────────────────────

def test_extract_text_plain_string():
    assert run_eval._extract_text("hello world") == "hello world"


def test_extract_text_list_of_text_blocks():
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert run_eval._extract_text(content) == "first\nsecond"


def test_extract_text_list_skips_non_text_blocks():
    content = [
        {"type": "image", "url": "http://x.com/img.png"},
        {"type": "text", "text": "only this"},
    ]
    result = run_eval._extract_text(content)
    assert "only this" in result


def test_extract_text_empty_list():
    assert run_eval._extract_text([]) == ""


def test_extract_text_unrecognised_type():
    assert run_eval._extract_text(42) == ""


# ── _resolve_tool_name ────────────────────────────────────────────────────────

def test_resolve_tool_name_direct():
    tc = {"name": "calculate_bmi", "args": {}}
    assert run_eval._resolve_tool_name(tc) == "calculate_bmi"


def test_resolve_tool_name_task_subagent():
    tc = {"name": "task", "args": {"subagent_type": "analyst"}}
    assert run_eval._resolve_tool_name(tc) == "analyst"


def test_resolve_tool_name_task_without_subagent_type():
    tc = {"name": "task", "args": {}}
    assert run_eval._resolve_tool_name(tc) == "task"


# ── _unwrap_update_data ───────────────────────────────────────────────────────

def test_unwrap_subgraph_list_format():
    data = [["ns1", "ns2"], {"agent": {"messages": []}}]
    assert run_eval._unwrap_update_data(data) == {"agent": {"messages": []}}


def test_unwrap_dict_passthrough():
    data = {"agent": {"messages": []}}
    assert run_eval._unwrap_update_data(data) == data


def test_unwrap_other_returns_empty():
    assert run_eval._unwrap_update_data("bad_input") == {}
    assert run_eval._unwrap_update_data(None) == {}


# ── _extract_node_updates ─────────────────────────────────────────────────────

def test_extract_node_updates_top_level_dict():
    data = {"agent": {"messages": [{"type": "ai", "content": "hi"}]}}
    updates = run_eval._extract_node_updates(data)
    assert len(updates) == 1
    assert updates[0]["messages"][0]["content"] == "hi"


def test_extract_node_updates_subgraph_list_format():
    data = [["ns1", "ns2"], {"agent": {"messages": []}}]
    updates = run_eval._extract_node_updates(data)
    assert len(updates) == 1
    assert updates[0] == {"messages": []}


def test_extract_node_updates_invalid_returns_empty():
    assert run_eval._extract_node_updates("bad") == []
    assert run_eval._extract_node_updates([]) == []


# ── _has_interrupt ────────────────────────────────────────────────────────────

def test_has_interrupt_true():
    events = [("updates", {"__interrupt__": [{"value": {"action_requests": []}}]})]
    assert run_eval._has_interrupt(events) is True


def test_has_interrupt_false():
    events = [("updates", {"agent": {"messages": []}})]
    assert run_eval._has_interrupt(events) is False


def test_has_interrupt_empty_events():
    assert run_eval._has_interrupt([]) is False


def test_has_interrupt_subgraph_list_format():
    events = [("updates", [["ns"], {"__interrupt__": [{"value": {}}]}])]
    assert run_eval._has_interrupt(events) is True


def test_has_interrupt_skips_non_updates_events():
    events = [("events", {"event": "on_tool_start"}), ("updates", {"agent": {}})]
    assert run_eval._has_interrupt(events) is False


# ── _count_interrupted_tool_calls ─────────────────────────────────────────────

def test_count_interrupted_tool_calls_two():
    events = [("updates", {
        "__interrupt__": [{"value": {"action_requests": ["a", "b"]}}]
    })]
    assert run_eval._count_interrupted_tool_calls(events) == 2


def test_count_interrupted_tool_calls_defaults_to_one():
    events = [("updates", {"__interrupt__": [{"value": {}}]})]
    assert run_eval._count_interrupted_tool_calls(events) == 1


def test_count_interrupted_tool_calls_no_interrupt():
    events = [("updates", {"agent": {}})]
    assert run_eval._count_interrupted_tool_calls(events) == 1


def test_count_interrupted_tool_calls_skips_non_updates():
    events = [
        ("events", {"event": "on_tool_start"}),
        ("updates", {"__interrupt__": [{"value": {"action_requests": ["x"]}}]}),
    ]
    assert run_eval._count_interrupted_tool_calls(events) == 1


# ── _last_nonempty ────────────────────────────────────────────────────────────

def test_last_nonempty_returns_last_non_blank():
    assert run_eval._last_nonempty(["", "first", "last", ""]) == "last"


def test_last_nonempty_all_empty():
    assert run_eval._last_nonempty(["", "   ", ""]) == ""


def test_last_nonempty_empty_list():
    assert run_eval._last_nonempty([]) == ""


def test_last_nonempty_single_entry():
    assert run_eval._last_nonempty(["only"]) == "only"


# ── _headers ──────────────────────────────────────────────────────────────────

def test_headers_with_token():
    h = run_eval._headers("tok123")
    assert h["Authorization"] == "Bearer tok123"
    assert h["Content-Type"] == "application/json"


def test_headers_without_token():
    h = run_eval._headers(None)
    assert "Authorization" not in h
    assert h["Content-Type"] == "application/json"


def test_headers_with_empty_string():
    h = run_eval._headers("")
    assert "Authorization" not in h


# ── _collect_from_events ──────────────────────────────────────────────────────

def test_collect_from_events_ai_message():
    events = [
        ("updates", {"agent": {"messages": [
            {"type": "ai", "content": "BMI is 22.9", "tool_calls": []}
        ]}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert "BMI is 22.9" in texts


def test_collect_from_events_tool_start_event():
    events = [
        ("events", {"event": "on_tool_start", "name": "calculate_bmi",
                    "run_id": "r1", "data": {"input": {"weight": 70}}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert any(tc["tool_name"] == "calculate_bmi" for tc in tcs)


def test_collect_from_events_internal_tools_excluded():
    events = [
        ("events", {"event": "on_tool_start", "name": "write_todos",
                    "run_id": "r2", "data": {"input": {}}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_tool_result_captured_as_context():
    events = [
        ("updates", {"agent": {"messages": [
            {"type": "tool", "content": "BMI result: 22.9"}
        ]}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert "BMI result: 22.9" in ctxs


def test_collect_from_events_tool_message_with_empty_content():
    events = [
        ("updates", {"agent": {"messages": [
            {"type": "tool", "content": ""}
        ]}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert ctxs == []


def test_collect_from_events_deduplicates_tool_calls_by_run_id():
    events = [
        ("events", {"event": "on_tool_start", "name": "calc",
                    "run_id": "dup", "data": {"input": {}}}),
        ("events", {"event": "on_tool_start", "name": "calc",
                    "run_id": "dup", "data": {"input": {}}}),
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert len(tcs) == 1


def test_collect_from_events_empty():
    texts, tcs, ctxs = run_eval._collect_from_events([])
    assert texts == tcs == ctxs == []


def test_collect_from_events_ai_tool_calls_from_updates():
    events = [
        ("updates", {"agent": {"messages": [
            {"type": "ai", "content": "", "tool_calls": [
                {"name": "calculate_bmi", "args": {"weight": 70}}
            ]}
        ]}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert any(tc["tool_name"] == "calculate_bmi" for tc in tcs)


def test_collect_from_events_stream_mode_list_wrapper():
    events = [
        ("updates", ["updates", {"agent": {"messages": [
            {"type": "ai", "content": "wrapped", "tool_calls": []}
        ]}}])
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert "wrapped" in texts


def test_collect_from_events_skips_invalid_update_and_message():
    events = [
        ("updates", {"agent": "not-a-dict", "other": [1, 2]}),
        ("updates", {"node": {"messages": ["not-a-dict", {"type": "tool", "content": ""}]}}),
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert texts == tcs == ctxs == []


def test_collect_from_events_excludes_internal_ai_tool_calls():
    events = [
        ("updates", {"agent": {"messages": [
            {"type": "ai", "content": "", "tool_calls": [
                {"name": "write_todos", "args": {}}
            ]}
        ]}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_skips_duplicate_run_id():
    events = [
        ("events", {"event": "on_tool_start", "name": "calc",
                    "run_id": "seen", "data": {"input": {}}}),
        ("events", {"event": "on_tool_start", "name": "other",
                    "run_id": "seen", "data": {"input": {}}}),
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert len(tcs) == 1


def test_collect_from_events_non_tool_start_event_ignored():
    events = [("events", {"event": "on_chain_start", "name": "agent"})]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_tool_start_without_run_id():
    events = [
        ("events", {"event": "on_tool_start", "name": "calc",
                    "run_id": "", "data": {"input": {"x": 1}}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert len(tcs) == 1


def test_collect_from_events_tool_start_non_dict_args():
    events = [
        ("events", {"event": "on_tool_start", "name": "calc",
                    "run_id": "r3", "data": {"input": "not-a-dict"}})
    ]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_ignores_unknown_event_type():
    events = [("message", {"data": "ignored"})]
    texts, tcs, ctxs = run_eval._collect_from_events(events)
    assert texts == tcs == ctxs == []


# ── _subprocess_env ───────────────────────────────────────────────────────────

def test_subprocess_env_no_gcp_creds(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    assert tmp_files == []


def test_subprocess_env_with_valid_gcp_json(monkeypatch):
    sa = json.dumps({
        "project_id": "my-proj",
        "type": "service_account",
        "client_email": "sa@my-proj.iam.gserviceaccount.com",
    })
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", sa)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    try:
        assert extra_env.get("GOOGLE_CLOUD_PROJECT") == "my-proj"
        assert extra_env.get("VERTEXAI_PROJECT") == "my-proj"
        assert extra_env.get("VERTEXAI_LOCATION") == "us-central1"
        assert "GOOGLE_APPLICATION_CREDENTIALS" in extra_env
        assert len(tmp_files) == 1
        assert tmp_files[0].exists()
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)


def test_subprocess_env_skips_write_when_creds_already_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", '{"project_id":"x"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/already/set.json")
    extra_env, tmp_files = run_eval._subprocess_env()
    assert tmp_files == []


def test_subprocess_env_handles_invalid_json_gracefully(monkeypatch, caplog):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", "not-json")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    try:
        assert "could not parse GCP" in caplog.text
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)


def test_subprocess_env_valid_json_without_project_id(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", '{"type":"service_account"}')
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    try:
        assert "GOOGLE_APPLICATION_CREDENTIALS" in extra_env
        assert "GOOGLE_CLOUD_PROJECT" not in extra_env
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)


def test_subprocess_env_write_failure(monkeypatch, caplog):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", '{"project_id":"x"}')
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with patch("run_eval.tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
        extra_env, tmp_files = run_eval._subprocess_env()
    assert extra_env == {}
    assert tmp_files == []
    assert "could not write GCP credentials" in caplog.text


# ── _extract_interrupt_response ───────────────────────────────────────────────

def test_extract_interrupt_response_with_description():
    events = [("updates", {
        "__interrupt__": [{"value": {"action_requests": [
            {"description": "Approve sending email?", "name": "send_email", "args": {}}
        ]}}]
    })]
    response = run_eval._extract_interrupt_response(events)
    assert "Approve sending email?" in response


def test_extract_interrupt_response_falls_back_to_name_args():
    events = [("updates", {
        "__interrupt__": [{"value": {"action_requests": [
            {"name": "send_email", "args": {"to": "user@example.com"}}
        ]}}]
    })]
    response = run_eval._extract_interrupt_response(events)
    assert "send_email" in response


def test_extract_interrupt_response_empty_when_no_interrupt():
    events = [("updates", {"agent": {}})]
    assert run_eval._extract_interrupt_response(events) == ""


def test_extract_interrupt_response_skips_non_updates():
    events = [
        ("events", {"event": "on_tool_start"}),
        ("updates", {"agent": {}}),
    ]
    assert run_eval._extract_interrupt_response(events) == ""


def test_extract_interrupt_response_empty_action_requests():
    events = [("updates", {
        "__interrupt__": [{"value": {"action_requests": []}}]
    })]
    assert run_eval._extract_interrupt_response(events) == ""


def test_extract_interrupt_response_multiple_requests():
    events = [("updates", {
        "__interrupt__": [{"value": {"action_requests": [
            {"description": "First approval"},
            {"description": "Second approval"},
        ]}}]
    })]
    response = run_eval._extract_interrupt_response(events)
    assert "First approval" in response
    assert "Second approval" in response


# ── _find_lightspeed_cmd ────────────────────────────────────────────────────────

def test_find_lightspeed_cmd_from_venv_bin(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "lightspeed-eval").write_text("#!/bin/sh\n")
    monkeypatch.setattr(run_eval.sys, "executable", str(fake_bin / "python"))
    result = run_eval._find_lightspeed_cmd()
    assert result == [str(fake_bin / "lightspeed-eval")]


def test_find_lightspeed_cmd_via_which(monkeypatch):
    monkeypatch.setattr(run_eval.sys, "executable", "/usr/bin/python3")
    with patch.object(Path, "exists", return_value=False):
        which_result = MagicMock(returncode=0, stdout="/usr/local/bin/lightspeed-eval\n")
        with patch("run_eval.subprocess.run", return_value=which_result):
            result = run_eval._find_lightspeed_cmd()
    assert result == ["/usr/local/bin/lightspeed-eval"]


def test_find_lightspeed_cmd_not_found(monkeypatch):
    monkeypatch.setattr(run_eval.sys, "executable", "/usr/bin/python3")
    with patch.object(Path, "exists", return_value=False):
        which_result = MagicMock(returncode=1, stdout="")
        with patch("run_eval.subprocess.run", return_value=which_result):
            result = run_eval._find_lightspeed_cmd()
    assert result is None


# ── _parse_args ─────────────────────────────────────────────────────────────────

def test_parse_args_defaults(monkeypatch):
    monkeypatch.delenv("AGENT_HOST", raising=False)
    monkeypatch.delenv("AGENT_AUTH_TOKEN", raising=False)
    with patch.object(run_eval.sys, "argv", ["run_eval.py"]):
        args = run_eval._parse_args()
    assert args.agent_url == run_eval.DEFAULT_AGENT_URL
    assert args.eval_data == [run_eval.DEFAULT_EVAL_DATA]
    assert args.system == run_eval.DEFAULT_SYSTEM
    assert args.output_dir == run_eval.DEFAULT_OUTPUT_DIR
    assert args.auth_token is None
    assert args.timeout == 300


def test_parse_args_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_HOST", "http://agent.test")
    monkeypatch.setenv("AGENT_AUTH_TOKEN", "secret")
    with patch.object(run_eval.sys, "argv", ["run_eval.py"]):
        args = run_eval._parse_args()
    assert args.agent_url == "http://agent.test"
    assert args.auth_token == "secret"
