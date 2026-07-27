"""LangChain callback handler for per-thread token budget tracking."""

from __future__ import annotations

import collections
import threading
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

from deep_agent.src.token_budget.identity import resolve_thread_id, resolve_user_id
from deep_agent.src.token_budget.service import (
    check_and_record,
    extract_tokens_from_llm_result,
    extract_tokens_from_message,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

THREAD_ID_METADATA_KEY = "token_budget_thread_id"
USER_ID_METADATA_KEY = "token_budget_user_id"
TRACE_ID_METADATA_KEY = "token_budget_trace_id"
ORG_ID_METADATA_KEY = "token_budget_org_id"

# Emit an ERROR-level alert after this many consecutive failures so ops
# teams can detect a persistently degraded token-tracking feature.
_CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 5

_counter_lock = threading.Lock()
_consecutive_failures = 0
_total_failures = 0

# register_configure_hook (inheritable=True) creates a new handler instance at each
# nested RunnableConfig merge in LangGraph, so on_llm_end can fire multiple times
# for the same LLM call with the same run_id. Deduplicate by run_id to prevent
# the same call from being counted more than once.
_DEDUP_MAX = 1024
_seen_run_ids: set[str] = set()
_seen_run_ids_order: collections.deque[str] = collections.deque()
_dedup_lock = threading.Lock()


def _is_duplicate_run(run_id: Any) -> bool:
    """Return True if this run_id has already been recorded; register it otherwise."""
    rid = str(run_id)
    with _dedup_lock:
        if rid in _seen_run_ids:
            return True
        _seen_run_ids.add(rid)
        _seen_run_ids_order.append(rid)
        if len(_seen_run_ids_order) > _DEDUP_MAX:
            _seen_run_ids.discard(_seen_run_ids_order.popleft())
    return False


def _on_tracking_success() -> None:
    global _consecutive_failures
    with _counter_lock:
        _consecutive_failures = 0


def _on_tracking_failure() -> None:
    global _consecutive_failures, _total_failures
    with _counter_lock:
        _consecutive_failures += 1
        _total_failures += 1
        consecutive = _consecutive_failures
        total = _total_failures
    if consecutive >= _CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
        logger.error(
            "token_budget_tracking_degraded",
            consecutive_failures=consecutive,
            total_failures=total,
        )


def _extract_from_metadata(
    metadata: dict[str, Any] | None,
    key: str,
    *,
    fallback_keys: tuple[str, ...] = (),
) -> str | None:
    """Return the first non-empty metadata value for key or fallback keys."""
    if not metadata:
        return None
    for metadata_key in (key, *fallback_keys):
        value = metadata.get(metadata_key)
        if value:
            return str(value)
    return None


def thread_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Resolve thread_id from RunnableConfig metadata."""
    return _extract_from_metadata(
        metadata,
        THREAD_ID_METADATA_KEY,
        fallback_keys=("langfuse_session_id",),
    )


def user_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Resolve the chatting user's id from RunnableConfig metadata."""
    return _extract_from_metadata(metadata, USER_ID_METADATA_KEY)


def trace_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Resolve trace_id from RunnableConfig metadata."""
    return _extract_from_metadata(metadata, TRACE_ID_METADATA_KEY)


def org_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Resolve org_id from RunnableConfig metadata."""
    return _extract_from_metadata(metadata, ORG_ID_METADATA_KEY)


class TokenBudgetCallbackHandler(AsyncCallbackHandler):
    """Increment per-thread token usage after each LLM call."""

    async def _record_tokens(
        self,
        response: LLMResult,
        metadata: dict[str, Any] | None,
        extraction_fn: Any,
    ) -> None:
        """Shared logic for recording token usage from LLM responses."""
        thread_id = thread_id_from_metadata(metadata) or resolve_thread_id()
        if not thread_id:
            logger.debug("token_budget_callback_no_thread_id")
            return

        user_id = user_id_from_metadata(metadata) or resolve_user_id()
        trace_id = trace_id_from_metadata(metadata)
        org_id = org_id_from_metadata(metadata)

        input_tokens, output_tokens = extraction_fn(response)
        if input_tokens <= 0 and output_tokens <= 0:
            input_tokens, output_tokens = _tokens_from_generations(response)

        try:
            await check_and_record(
                thread_id,
                input_tokens,
                output_tokens,
                user_id=user_id,
                org_id=org_id,
                trace_id=trace_id,
            )
            _on_tracking_success()
        except Exception:
            logger.warning(
                "token_budget_callback_failed",
                exc_info=True,
            )
            _on_tracking_failure()

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record token usage when an LLM call completes."""
        if _is_duplicate_run(run_id):
            logger.debug("token_budget_callback_duplicate_run_id", run_id=str(run_id))
            return
        await self._record_tokens(response, metadata, extract_tokens_from_llm_result)


def _tokens_from_generations(response: LLMResult) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for generation_list in response.generations:
        for generation in generation_list:
            if not isinstance(generation, ChatGeneration):
                continue
            message = generation.message
            in_t, out_t = extract_tokens_from_message(message)
            input_tokens += in_t
            output_tokens += out_t
    return input_tokens, output_tokens
