# Preserving `extra_content` across OpenAI-compatible tool-call turns

Status: proposed
Date: 2026-10-29

## Intent and why

Gemini 3.x models served through an OpenAI-compatible gateway reject multi-turn
tool use unless the `thought_signature` the model emitted alongside a function
call is echoed back on the following request. Our agent loses that signature, so
any Gemini 3.x model fails on the second turn of every tool call — which is to
say, it fails at being an agent at all.

Observed with `model: {provider: openai-compatibility, name: gemini-3.5-flash}`
in `config/agent/PROMPT.md` against `VLLM_BASE_URL=https://models.corp.redhat.com/oai/v1`:

```
400 - Function call is missing a thought_signature in functionCall parts.
This is required for tools to work correctly, and missing thought_signature
may lead to degraded [results]
```

### Root cause (reproduced, not inferred)

The gateway *does* return the signature, on both streaming and non-streaming
responses, at:

```
choices[].message.tool_calls[].extra_content.google.thought_signature
```

`langchain-openai` 1.7.2 discards it at parse time. After
`ChatOpenAI.bind_tools(...).invoke(...)`, the resulting `AIMessage` has
`additional_kwargs == {"refusal": None}` — the signature is gone before any of
our code can see it, so it cannot be recovered later.

Confirmed in both directions with a raw `openai` client against the live
gateway: replaying the assistant tool call **with** `extra_content` succeeds,
and the identical request **without** it returns the 400 above.

## Goals

- Gemini 3.x models work for multi-turn tool calling over the
  `openai-compatibility` provider route.
- Existing config keeps working unchanged: `VLLM_BASE_URL`, `VLLM_API_KEY`, the
  `Provider.OPENAI_COMPATIBILITY` route, model fallbacks, and the model cache.
- Streaming and non-streaming both covered — `_create_vllm_model` sets
  `streaming=True`, so a non-streaming-only fix would be dead code in practice.

## Non-goals

- Using `ChatGoogleGenerativeAI`. See Approaches.
- Reading or interpreting the signature. It is an opaque blob; we relay bytes.
- Changing how models are selected, named, or configured. No new env vars, no
  new YAML keys, no change to `config/agent/models/*.yaml`.

## Anti-goals

- Anything that makes non-Gemini models on the same gateway (Granite, Nemotron)
  behave differently than they do today.
- Forking or vendoring `langchain-openai`.

## Constraints

- The gateway speaks OpenAI chat-completions only. Probed `/v1beta/models`,
  `/gemini/v1beta/models`, `/v1beta/openai/models` — all 404. Only `/oai/v1`
  responds.
- `langchain-openai` is pinned at 1.7.2. The fix touches three private methods
  on `ChatOpenAI`, so it is coupled to internals and must be pinned by tests
  that fail loudly on upgrade.

## Approaches

**1. `ChatGoogleGenerativeAI` with `base_url` + API key** — the original
framing. Ruled out by the gateway, not by preference. In
`langchain-google-genai` 4.4.2 `base_url` is an alias for `client_options`,
which repoints the *native* Gemini `generateContent` API at another host. The
gateway serves no such endpoint, so this client would have nothing to talk to.

**2. Middleware that repairs messages before send** — fits the repo's existing
middleware layer, but cannot work: the signature is destroyed during response
parsing, so it never reaches a message for middleware to repair.

**3. A `ChatOpenAI` subclass that round-trips `extra_content`** — **chosen.**
The only seam that sees the raw payload on both legs.

Within (3), two scopes were considered: a Gemini-gated version keyed off model
name, and a generic pass-through. **Generic pass-through chosen** — the field is
simply absent for the gateway's other models, so the practical blast radius is
identical, and it needs no revisiting when the next model starts using it.

## Design

A new module holding one class, `ExtraContentChatOpenAI`, subclassing
`ChatOpenAI`. `_create_vllm_model` in `deep_agent/src/agent/llm.py` returns it
in place of `ChatOpenAI`. Nothing else in the call chain changes.

Three overrides, all no-ops when no `extra_content` is present:

**Capture, non-streaming** — `_create_chat_result(response, generation_info)`.
Call `super()`, then read the raw response's `tool_calls[]`, and for each one
carrying `extra_content`, record it under the tool call's `id` in the resulting
`AIMessage.additional_kwargs["extra_content"]` as `{tool_call_id: extra_content}`.

**Capture, streaming** — `_convert_chunk_to_generation_chunk(chunk,
default_chunk_class, base_generation_info)`. Same idea against the delta's
`tool_calls[]`. Keying by tool-call `id` rather than `index` is deliberate: the
gateway sends `index: null` on these deltas but always sends `id`.

LangChain merges `additional_kwargs` across chunks with `merge_dicts`, which
recurses into nested dicts and **concatenates strings** on key collision. A
bare `{id: blob}` stash is therefore unsafe: if a gateway repeats a tool call's
envelope on a later delta, the signature merges into `"abc" + "abc"` and the
replay sends a doubled blob — producing the exact 400 this design prevents,
silently. Each stash value is a single-element **list** instead, because
`merge_dicts` hands lists to `merge_lists`, which appends entries carrying no
integer `index` key. Duplicates then accumulate harmlessly and the replay takes
the first.

**Replay** — `_get_request_payload(input_, stop=..., **kwargs)`. Call `super()`,
then build a `{tool_call_id: extra_content}` map from the `AIMessage`s in the
converted input and re-attach each entry onto the matching tool call in
`payload["messages"]`. Matching is by tool-call `id`, so message order and any
history trimming are irrelevant.

Guarded to the chat-completions shape: when `_use_responses_api(payload)` is
true the payload has a different structure, and the override returns `super()`'s
result untouched.

### What is left to the builder

- Module placement and name (`deep_agent/src/agent/` alongside `llm.py` is the
  obvious home).
- Whether capture writes through `additional_kwargs` directly or via a small
  helper shared by the streaming and non-streaming paths.
- Defensive handling of malformed `tool_calls` entries (missing `id`, non-dict
  `extra_content`) — relay only well-formed entries, never raise from a parse
  path.

### Testing

- Unit, offline: a canned response dict with `extra_content` survives
  `_create_chat_result`; streaming chunks accumulate correctly across a merge;
  `_get_request_payload` re-attaches to the right tool call by id; all three are
  exact no-ops when `extra_content` is absent.
- Upgrade tripwire: assert the three overridden methods still exist on
  `ChatOpenAI` with the expected signatures, so a `langchain-openai` bump fails
  a test rather than silently reverting to the 400.
- Live round-trip against the gateway is how the bug was diagnosed and is the
  acceptance check, but it needs credentials, so it stays out of the default
  unit run.

Existing tests in `tests/unit/agent/test_llm.py` and
`test_provider_factory.py` assert on the vLLM path and must keep passing.
