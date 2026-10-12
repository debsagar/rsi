"""One model call over litellm, with retries, usage and cost.

`complete(messages, tools=None, model=None, on_delta=None)` returns
`{"message": dict, "usage": {"in": int, "out": int}, "cost": float}`.

Invariants:
- No default model. `RSI_MODEL` (and `RSI_IMPROVER_MODEL` for evolve) must be
  set; a missing id or provider key raises naming the variable.
- Retries only what can succeed on a second try: 429, 5xx and timeouts, with
  delays of 1, 3 and 9 seconds. Anything else, and the last retry, raises.
- The returned message is a plain dict holding only role, content and
  tool_calls, so it can be sent to the harness and back unchanged.
"""
import os
import time

# litellm reads the nearest .env on import unless told it is not in dev mode. That would put the real
# keys into every process that imports this module, tests included; `.env` is rsi.load_env's job alone.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import litellm  # noqa: E402

from . import env

RETRY_DELAYS = (1, 3, 9)
TIMEOUT_S = 120


def check(*names: str) -> None:
    """Raise unless each model-id variable is set and its provider key is present."""
    for name in names:
        model = env(name)
        found = litellm.validate_environment(model)
        if not found["keys_in_environment"]:
            raise RuntimeError(f"{name}={model} needs {', '.join(found['missing_keys']) or 'a provider key'} "
                               "in the environment or .env")


def _retryable(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    return isinstance(e, litellm.Timeout) or code == 429 or (isinstance(code, int) and code >= 500)


def _plain(message) -> dict:
    out = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [{"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                             for tc in message.tool_calls]
    return out


def _once(messages: list, tools, model: str, on_delta):
    kwargs = {"model": model, "messages": messages, "timeout": TIMEOUT_S, "num_retries": 0}
    if tools:
        kwargs["tools"] = tools
    if model.startswith("openrouter/"):
        kwargs["extra_body"] = {"usage": {"include": True}}  # OpenRouter then reports the real charge
    if on_delta is None:
        return litellm.completion(**kwargs)
    chunks = []
    for chunk in litellm.completion(**kwargs, stream=True, stream_options={"include_usage": True}):
        chunks.append(chunk)
        text = chunk.choices[0].delta.content if chunk.choices else None
        if text:
            on_delta(text)
    built = litellm.stream_chunk_builder(chunks, messages=messages)
    for chunk in chunks:  # the provider's own charge rides on the final usage chunk; the builder drops it
        charged = getattr(getattr(chunk, "usage", None), "cost", None)
        if charged is not None:
            built.usage.cost = charged
    return built


def complete(messages: list, tools: list | None = None, model: str | None = None, on_delta=None) -> dict:
    """One completion. `on_delta(text)` receives answer text as it arrives.

    A retry after a partial stream sends its text again; the caller's final
    `message` is always the complete one."""
    model = model or env("RSI_MODEL")
    for delay in (0,) + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            response = _once(messages, tools, model, on_delta)
            break
        except Exception as e:
            if not _retryable(e):
                raise
            last = e
    else:
        raise RuntimeError(f"model call failed after {len(RETRY_DELAYS)} retries: {last}") from last
    usage = response.usage
    cost = getattr(usage, "cost", None)  # what the provider says it charged, when it says
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            # ponytail: a model with no reported charge and no litellm price is recorded
            # as 0.0, so cost totals are a floor for such models. Token counts stay exact.
            cost = 0.0
    cost = float(cost)
    return {"message": _plain(response.choices[0].message),
            "usage": {"in": usage.prompt_tokens, "out": usage.completion_tokens}, "cost": cost}
