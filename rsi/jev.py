"""Jev client: TypeSafe System One (jev-latest), stdlib urllib only.

Jev answers typed questions about a `state` dict. A Noul question comes back
as a probability, so callers threshold numbers; there is no text to parse.

Invariants:
- A missing `TYPESAFE_API_KEY` raises naming the variable.
- Every other failure (network, HTTP status, bad body) returns None. The
  caller records that and treats it as "no flag", so a Jev outage never ends
  a run.
"""
import json
import sys
import urllib.request

from . import env

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


def ask(state: dict, questions: dict, timeout: float = 10) -> dict | None:
    """-> {question_id: {"noul": float} | {"choice": str, ...}}, or None on failure."""
    key = env("TYPESAFE_API_KEY")
    body = json.dumps({"model": MODEL, "questions": questions, "state": state}, default=str).encode()
    request = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["answers"]
    except Exception as e:  # urllib raises several unrelated types; all mean "no answer"
        # No retry: the gate and the watcher are advisory and the run is on a clock.
        print(f"jev: no answer ({type(e).__name__}: {e})", file=sys.stderr)
        return None
