"""Unit tests for the OpenAI-compatible extra_content pass-through."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from deep_agent.src.agent.llm import _create_vllm_model
from deep_agent.src.agent.openai_compat import (
    EXTRA_CONTENT_KEY,
    ExtraContentChatOpenAI,
    _extra_content_by_tool_call_id,
)

SIG = {"google": {"thought_signature": "abc123"}}


def _response(tool_calls):
    return {
        "id": "resp-1",
        "model": "gemini-3.5-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
            }
        ],
    }


def _tool_call(id_, name="get_weather", extra=None):
    tc = {
        "id": id_,
        "type": "function",
        "function": {"name": name, "arguments": '{"city":"Paris"}'},
    }
    if extra is not None:
        tc["extra_content"] = extra
    return tc


class TestExtraContentByToolCallId:
    def test_maps_id_to_extra_content(self):
        assert _extra_content_by_tool_call_id([_tool_call("call_1", extra=SIG)]) == {
            "call_1": SIG
        }

    def test_skips_entry_without_id(self):
        tc = _tool_call("call_1", extra=SIG)
        del tc["id"]
        assert _extra_content_by_tool_call_id([tc]) == {}

    def test_skips_entry_without_extra_content(self):
        assert _extra_content_by_tool_call_id([_tool_call("call_1")]) == {}

    @pytest.mark.parametrize("bad", ["a string", None, 42, []])
    def test_skips_non_dict_extra_content(self, bad):
        assert _extra_content_by_tool_call_id([_tool_call("call_1", extra=bad)]) == {}

    @pytest.mark.parametrize("bad", [None, {}, "nope", 7])
    def test_non_list_input_returns_empty(self, bad):
        assert _extra_content_by_tool_call_id(bad) == {}


class TestCreateChatResultCapture:
    def _llm(self):
        return ExtraContentChatOpenAI(
            model="gemini-3.5-flash", api_key="test", base_url="http://x/v1"
        )

    def test_captures_signature_into_additional_kwargs(self):
        result = self._llm()._create_chat_result(
            _response([_tool_call("call_1", extra=SIG)])
        )
        message = result.generations[0].message
        assert message.additional_kwargs[EXTRA_CONTENT_KEY] == {"call_1": SIG}

    def test_captures_each_of_multiple_tool_calls(self):
        other = {"google": {"thought_signature": "def456"}}
        result = self._llm()._create_chat_result(
            _response(
                [
                    _tool_call("call_1", extra=SIG),
                    _tool_call("call_2", name="get_time", extra=other),
                ]
            )
        )
        assert result.generations[0].message.additional_kwargs[EXTRA_CONTENT_KEY] == {
            "call_1": SIG,
            "call_2": other,
        }

    def test_no_key_added_when_absent(self):
        result = self._llm()._create_chat_result(_response([_tool_call("call_1")]))
        assert EXTRA_CONTENT_KEY not in result.generations[0].message.additional_kwargs

    def test_plain_text_response_unaffected(self):
        response = {
            "id": "resp-2",
            "model": "gemini-3.5-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
        }
        result = self._llm()._create_chat_result(response)
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
        assert message.content == "hello"
        assert EXTRA_CONTENT_KEY not in message.additional_kwargs


def _ai_with_signature(ids_to_sig):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"city": "Paris"},
                "id": id_,
                "type": "tool_call",
            }
            for id_ in ids_to_sig
        ],
        additional_kwargs={EXTRA_CONTENT_KEY: dict(ids_to_sig)},
    )


class TestGetRequestPayloadReplay:
    def _llm(self):
        return ExtraContentChatOpenAI(
            model="gemini-3.5-flash", api_key="test", base_url="http://x/v1"
        )

    def _conversation(self, ai):
        return [
            HumanMessage(content="weather in Paris?"),
            ai,
            ToolMessage(content="18C", tool_call_id="call_1"),
        ]

    def test_reattaches_signature_to_matching_tool_call(self):
        payload = self._llm()._get_request_payload(
            self._conversation(_ai_with_signature({"call_1": SIG}))
        )
        assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["id"] == "call_1"
        assert assistant["tool_calls"][0]["extra_content"] == SIG

    def test_reattaches_each_of_multiple_tool_calls(self):
        other = {"google": {"thought_signature": "def456"}}
        payload = self._llm()._get_request_payload(
            self._conversation(_ai_with_signature({"call_1": SIG, "call_2": other}))
        )
        assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
        by_id = {tc["id"]: tc.get("extra_content") for tc in assistant["tool_calls"]}
        assert by_id == {"call_1": SIG, "call_2": other}

    def test_stashed_id_absent_from_payload_is_ignored(self):
        ai = _ai_with_signature({"call_1": SIG})
        ai.additional_kwargs[EXTRA_CONTENT_KEY]["call_stale"] = SIG
        payload = self._llm()._get_request_payload(self._conversation(ai))
        assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
        assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_1"]

    def test_payload_unchanged_when_no_signature_stashed(self):
        messages = [HumanMessage(content="hi")]
        llm = self._llm()
        assert llm._get_request_payload(messages) == super(
            ExtraContentChatOpenAI, llm
        )._get_request_payload(messages)

    def test_no_extra_content_key_on_tool_calls_without_signature(self):
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "Paris"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        payload = self._llm()._get_request_payload(self._conversation(ai))
        assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
        assert "extra_content" not in assistant["tool_calls"][0]


def _delta_chunk(tool_calls):
    return {
        "id": "chunk-1",
        "model": "gemini-3.5-flash",
        "choices": [
            {"index": 0, "finish_reason": None, "delta": {"tool_calls": tool_calls}}
        ],
    }


def _delta_tool_call(id_, extra=None, index=None):
    tc = {
        "index": index,
        "id": id_,
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
    }
    if extra is not None:
        tc["extra_content"] = extra
    return tc


class TestStreamingCapture:
    def _llm(self):
        return ExtraContentChatOpenAI(
            model="gemini-3.5-flash", api_key="test", base_url="http://x/v1"
        )

    def test_captures_signature_from_delta_with_null_index(self):
        gen = self._llm()._convert_chunk_to_generation_chunk(
            _delta_chunk([_delta_tool_call("call_1", extra=SIG, index=None)]),
            AIMessageChunk,
            None,
        )
        assert gen.message.additional_kwargs[EXTRA_CONTENT_KEY] == {"call_1": SIG}

    def test_no_key_added_when_absent(self):
        gen = self._llm()._convert_chunk_to_generation_chunk(
            _delta_chunk([_delta_tool_call("call_1", index=0)]), AIMessageChunk, None
        )
        assert EXTRA_CONTENT_KEY not in gen.message.additional_kwargs

    def test_signatures_survive_chunk_merge(self):
        other = {"google": {"thought_signature": "def456"}}
        llm = self._llm()
        first = llm._convert_chunk_to_generation_chunk(
            _delta_chunk([_delta_tool_call("call_1", extra=SIG)]), AIMessageChunk, None
        )
        second = llm._convert_chunk_to_generation_chunk(
            _delta_chunk([_delta_tool_call("call_2", extra=other)]),
            AIMessageChunk,
            None,
        )
        merged = first + second
        assert merged.message.additional_kwargs[EXTRA_CONTENT_KEY] == {
            "call_1": SIG,
            "call_2": other,
        }

    def test_chunk_without_choices_returns_super_result(self):
        gen = self._llm()._convert_chunk_to_generation_chunk(
            {"id": "c", "model": "m", "choices": [], "usage": None},
            AIMessageChunk,
            None,
        )
        assert gen is None or EXTRA_CONTENT_KEY not in gen.message.additional_kwargs


class TestVllmRouteUsesSubclass:
    def test_create_vllm_model_returns_subclass(self):
        # Patch the whole settings object, matching tests/unit/agent/test_llm.py.
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://x/v1"
        mock_settings.VLLM_API_KEY = "test"
        with patch("deep_agent.src.agent.llm.settings", mock_settings):
            model = _create_vllm_model("gemini-3.5-flash", 0.0, 8192)
        assert isinstance(model, ExtraContentChatOpenAI)
        assert model.model_name == "gemini-3.5-flash"


class TestLangchainOpenAIUpgradeTripwire:
    """Fails loudly if langchain-openai renames or reshapes what we override."""

    @pytest.mark.parametrize(
        ("method", "params"),
        [
            ("_create_chat_result", ["self", "response", "generation_info"]),
            (
                "_convert_chunk_to_generation_chunk",
                ["self", "chunk", "default_chunk_class", "base_generation_info"],
            ),
            ("_get_request_payload", ["self", "input_", "stop"]),
        ],
    )
    def test_overridden_method_signature_is_unchanged(self, method, params):
        assert hasattr(ChatOpenAI, method)
        actual = list(inspect.signature(getattr(ChatOpenAI, method)).parameters)
        assert actual[: len(params)] == params

    def test_use_responses_api_helper_still_exists(self):
        assert hasattr(ChatOpenAI, "_use_responses_api")
