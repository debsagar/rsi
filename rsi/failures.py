"""Failure classification: why did a submitted answer score 0?

Heuristic, from the trajectory alone: "use" means the reference value was in
some tool result and the answer was still wrong; "gather" means it never was;
"unchecked" means the reference cannot be searched for at all.

What is searched: number tokens of every tool result that is not an ERROR, read
with the shared parser (rsi.grading) under both comma readings (1,234.5 as one
number and as CSV cells), so thousands separators, scientific notation and
leading-dot decimals are found. A reference written as a percentage ("12.5%")
is looked for as 12.5 and as 0.125.

What is not counted as evidence, to keep "use" conservative:
- a number inside a date (2021-05-14, 14/05/2021): a year is not a result;
- the integer that starts a table row and is followed by two or more spaces or
  a tab (the index column of a printed frame);
- integer references below MIN_INT, references with under two significant
  digits, labels and option letters: they occur anywhere, so they are never
  searched for. An episode none of whose reference parts is searchable is
  "unchecked", whether or not a tool ran.

Chance control: `classify(..., decoy=other_task)` searches the same trajectory
for another task's reference, exactly as hard as for the real one. The rate at
which that succeeds is the rate at which "use" appears by accident. (The earlier
control, reference x 1.37, could never match a raw data cell, so it understated
chance whenever the reference was itself a round data value.)

Known false positive: a reference equal to an unrelated number (a row count, a
p-value). Known false negative: evidence the model would still have to combine
or convert (a sum of printed parts, a different unit) counts as "gather"; so
does a real result printed in the index column or as a bare year in a date.
"""
import json
import re
from pathlib import Path

import numpy as np

from . import tasks
from .grading import DA_TOL, NUMBER_RE, PLAIN_RE, QR_TOL, clean, numbers, whole

KINDS = ["no_answer", "use", "gather", "unchecked"]
MIN_INT = 100                # evidence search skips integer references below this
DATE_RE = re.compile(r"(?<![\w.])(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})(?![\w])")
INDEX_RE = re.compile(r"^[ \t]*\d+(?=[ \t]{2,}\S|\t\S)", re.M)


def _jsonl(path) -> list:
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _events(run_dir) -> list:
    return _jsonl(Path(run_dir) / "events.jsonl")


def _targets(task: dict) -> list:
    """Per sub-answer: ([(value, decimals), ...] alternatives, relative tolerance), or None when it cannot be searched for."""
    ref = task["reference"]
    tol = {"dabench": 0.0, "qrdata": QR_TOL}.get(task["family"], tasks.REL_TOL)
    out = []
    for text in ([v for _, v in ref] if task["family"] == "dabench" else [str(ref)]):
        parsed, digits = whole(text), text.strip().rstrip("% ")
        decimals = len(digits.partition(".")[2])
        if parsed is None or len(re.sub(r"\D", "", digits).lstrip("0")) < 2 or (decimals == 0 and abs(parsed[0]) < MIN_INT):
            out.append(None)
            continue
        value, percent = parsed
        out.append(([(value, decimals)] + ([(value / 100, decimals + 2)] if percent else []), tol))
    return out


def _seen(text: str) -> np.ndarray:
    text = INDEX_RE.sub(" ", DATE_RE.sub(" ", clean(text)))
    return np.array([value for pattern in (NUMBER_RE, PLAIN_RE) for value, _ in numbers(text, pattern)])


def classify(task: dict, episode: dict, run_dir, decoy: dict | None = None) -> tuple:
    """-> (failure kind, id of the tool_result that first shows the first searchable reference value, or None).
    With `decoy`, that task's reference is searched for instead: the chance control."""
    if not episode.get("answer") or episode.get("stop_reason") != "submitted":
        return "no_answer", None
    targets = _targets(decoy or task)
    if not any(targets):
        return "unchecked", None
    seen = []
    for event in _events(run_dir):
        if event["kind"] == "tool_result" and not str(event.get("result", "")).startswith("ERROR"):
            full = Path(run_dir) / str(event.get("file", ""))  # a clipped result keeps its full text on disk
            text = full.read_text(errors="replace") if event.get("clipped") and full.is_file() else str(event["result"])
            seen.append((event["id"], _seen(text)))
    first = None
    for alternatives, tol in filter(None, targets):
        hit = next((eid for eid, xs in seen if xs.size and any(
            ((np.abs(xs - value) <= max(tol * abs(value), DA_TOL)) | (np.abs(np.round(xs, decimals) - value) < 1e-9)).any()
            for value, decimals in alternatives)), None)
        if hit is None:
            return "gather", None
        first = first or hit
    return ("unchecked" if None in targets else "use"), first


def decoy_for(task: dict, batch: list) -> dict | None:
    """The chance control for one task: the next task of the same eval (in tasks.jsonl order, wrapping) whose
    reference is fully searchable and differs. None when the task's own reference is not fully searchable
    (it can never be "use", so it has no chance level) or no such other task exists."""
    if not all(_targets(task)):
        return None
    at = next((i for i, other in enumerate(batch) if other["id"] == task["id"]), 0)
    return next((other for other in batch[at + 1:] + batch[:at]
                 if other["reference"] != task["reference"] and all(_targets(other))), None)
