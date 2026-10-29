"""Unit tests for the OpenAI-compatible extra_content pass-through."""

import pytest
from langchain_core.messages import AIMessage

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
