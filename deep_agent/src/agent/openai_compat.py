"""OpenAI-compatible chat client that preserves per-tool-call ``extra_content``.

Some OpenAI-compatible gateways attach vendor metadata to each tool call under
an ``extra_content`` key. Gemini 3.x carries its ``thought_signature`` there and
rejects any follow-up request whose function calls arrive without it:

    400 Function call is missing a thought_signature in functionCall parts.

``langchain_openai`` discards the field while parsing a response, so the
signature is gone before the agent loop can echo it back, and every multi-turn
tool call fails on its second turn. This module captures the field on the way
in, stashes it on the message keyed by tool-call id, and re-attaches it on the
way out.

The relay is provider-agnostic: it keys off the presence of ``extra_content``
and never inspects the contents, so endpoints that do not send the field are
unaffected.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

# Key used both for the raw field on a tool call and for the stash on
# ``AIMessage.additional_kwargs`` (where it holds {tool_call_id: extra_content}).
EXTRA_CONTENT_KEY = "extra_content"


def _extra_content_by_tool_call_id(tool_calls: Any) -> dict[str, Any]:
    """Map tool-call ``id`` to its ``extra_content``, for well-formed entries only.

    Anything malformed is skipped rather than raised on: this runs inside a
    response-parsing path, where a vendor quirk must never become a crash.

    Args:
        tool_calls: Raw ``tool_calls`` value from a response message or delta.

    Returns:
        Mapping of tool-call id to its extra content. Empty when the input is
        not a list or no entry carries usable extra content.
    """
    if not isinstance(tool_calls, list):
        return {}

    found: dict[str, Any] = {}
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        extra = tool_call.get(EXTRA_CONTENT_KEY)
        if isinstance(call_id, str) and call_id and isinstance(extra, dict) and extra:
            found[call_id] = extra
    return found


class ExtraContentChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that round-trips per-tool-call ``extra_content``.

    Overrides the three points that see the raw wire payload. Each is a no-op
    when no tool call carries ``extra_content``, so models served by endpoints
    that never send it behave exactly as they do under plain ``ChatOpenAI``.
    """

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Capture ``extra_content`` from a non-streaming response."""
        result = super()._create_chat_result(response, generation_info)

        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        choices = response_dict.get("choices") or []

        # super() builds one generation per choice, in order.
        for generation, choice in zip(result.generations, choices, strict=False):
            if not isinstance(generation.message, AIMessage) or not isinstance(
                choice, dict
            ):
                continue
            raw_message = choice.get("message")
            if not isinstance(raw_message, dict):
                continue
            captured = _extra_content_by_tool_call_id(raw_message.get("tool_calls"))
            if captured:
                generation.message.additional_kwargs[EXTRA_CONTENT_KEY] = captured

        return result
