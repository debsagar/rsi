"""Scripted stand-ins for the model and Jev, so no test ever calls a paid API.

`scripted_llm` and `fake_jev` return functions with the same signatures as
`rsi.llm.complete` and `rsi.jev.ask`.
"""
import json


def tool_call(name: str, args, call_id: str = "call_1") -> dict:
    """An assistant message asking for one tool. `args` may be a raw string to
    script malformed JSON."""
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]}


def scripted_llm(replies):
    """Pops one canned reply per call. A str is an assistant text message, a
    dict is used as the message, an Exception is raised. `.calls` records what
    the broker sent."""
    replies = list(replies)
    calls = []

    def complete(messages, tools=None, model=None, on_delta=None):
        calls.append({"messages": messages, "tools": tools})
        if not replies:
            raise RuntimeError("scripted_llm ran out of replies")
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = {"role": "assistant", "content": reply} if isinstance(reply, str) else reply
        if on_delta and message.get("content"):
            on_delta(message["content"])
        return {"message": message, "usage": {"in": 10, "out": 5}, "cost": 0.001}

    complete.calls = calls
    return complete


def fake_jev(p):
    """Answers every Noul with probability `p`; `fake_jev(None)` is an outage."""
    def ask(state, questions, timeout=10):
        ask.calls.append({"state": state, "questions": questions})
        if p is None:
            return None
        return {qid: {"noul": p} for qid in questions}

    ask.calls = []
    return ask
