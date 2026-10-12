"""Runs inside the sandbox container: loads a harness file and calls run(task, api).

    python runner.py /harness.py        (in the container; this file is mounted alone)
    python -m rsi.runner <harness>      (plain subprocess, tests only)

The harness never touches a model, a tool or a key. Every `api` call is one
JSON line to the broker on stdout and one JSON line back on stdin.

Invariants:
- Stdlib only, no `rsi` imports: this file is mounted into the container by
  itself.
- Only this module writes to the protocol stream. File descriptor 1 is
  pointed at stderr before the harness loads, so a `print`, or a child
  process the harness starts, cannot corrupt the protocol.
"""
import importlib.util
import json
import os
import sys


class Api:
    """What a harness may do. `tools` is the OpenAI-style schema list."""

    def __init__(self, out, tools: list):
        self._out = out
        self.tools = tools

    def _call(self, request: dict) -> dict:
        self._out.write(json.dumps(request, default=str) + "\n")
        self._out.flush()
        line = sys.stdin.readline()
        if not line:
            raise SystemExit("broker closed the connection")  # the broker ended the run
        return json.loads(line)

    def generate(self, messages: list, use_tools: bool = True) -> dict:
        """One model call. Returns the assistant message; raises RuntimeError
        when the broker refuses (call limit) or the model call failed."""
        reply = self._call({"op": "generate", "messages": messages, "use_tools": use_tools})
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["message"]

    def execute(self, name: str, args) -> dict:
        """One tool call. `args` is a dict, or the model's raw JSON string, which
        the broker parses. Returns {"result": str, "clipped": bool, "event_id": int}."""
        return self._call({"op": "execute", "name": name, "args": args})

    def judge(self, state: dict, questions: dict):
        """Ask Jev. Returns {question_id: answer}, or None when Jev is unavailable."""
        return self._call({"op": "judge", "state": state, "questions": questions})["answers"]

    def submit(self, text: str, claims: list) -> dict:
        """Offer the final answer. Returns {"accepted": bool, "reason": str}."""
        return self._call({"op": "submit", "text": text, "claims": claims})


def main() -> None:
    harness_path = sys.argv[1]
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    first = json.loads(sys.stdin.readline())
    spec = importlib.util.spec_from_file_location("harness", harness_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(first["task"], Api(out, first["tools"]))


if __name__ == "__main__":
    main()
