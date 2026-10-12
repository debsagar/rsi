"""The free replay check: does a candidate harness hand on an answer the parent already got right?

For each train task the parent passed in this generation, the candidate is run
with a scripted model whose every reply is the parent's stored final answer,
with no tool calls. What the candidate submits is graded. A candidate that
submits such an answer changed into a wrong one is dropped before any paid
episode. It exists because the first real generation lost all three candidates
to an unrelated "cleanup" that stripped `[1]` from `@outliers[1]`.

Invariants:
- No model provider, no Jev, no tools and no cost: the model is `_scripted`,
  Jev answers None (unavailable), the toolset is empty, the gate is off.
- The candidate runs the way every episode runs, through the caller's
  `run_episode`, which means in the harness container. A candidate is code a
  model wrote that nothing has run yet, and this is its first execution: the
  plain-subprocess mode would give it this user's files and environment for a
  saving of about a second per episode, on a check that runs at most once per
  train pass and candidate.
- Replay episodes live in <root>/replay-<sha12>/<task id>/ and are never read
  as scored episodes. A finished one (replay.json) is not run again.
- Nothing stored in the lineage row holds an answer value: the before and after
  lines are `grading.shape` forms, because the parent's line is the reference.

Limits, by design:
- It sees only the path from the model's final reply to `api.submit`. A
  candidate that changes what the model is told, or how tools are used, is not
  tested by it at all.
- The scripted model repeats the same text on every call, so a harness that
  sends a first reply back (asks the model to check, or to add an ANSWER line)
  gets it again and can submit it. A harness that never submits a reply made
  without tool calls ends with nothing submitted: that is counted as
  `no_submit` and does not drop the candidate, since nothing was mangled.
- The parent's stored answer is what the parent submitted. For a parent that
  itself rewrites replies, that is not the model's raw reply.
- mathbench is skipped: it scores a tool call, not the reply.
"""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import grading
from .episodes import WORKERS, grade, sha

NO_TOOLS = ([], lambda name, args: "ERROR this is a replay: no tools are available; give your final answer")
LINE_CHARS = 200


def _scripted(text: str):
    def llm(messages, tools=None, model=None, on_delta=None):
        return {"message": {"role": "assistant", "content": text}, "usage": {"in": 0, "out": 0}, "cost": 0.0}
    return llm


def _form(answer) -> str:
    line = grading.answer_line(answer) or (str(answer).strip().splitlines() or ["(empty)"])[-1]
    return grading.shape(line)[:LINE_CHARS].replace("`", "'")


def check(ctx: dict, path: Path, passed: list) -> dict | None:
    """Replay [(task, the parent's passing episode)] through the harness at `path`.
    -> {"n", "broken", "no_submit", "examples": [{"task", "before", "after"}]}, or None when the family has no replay."""
    if ctx["family"] == "mathbench":
        return None
    folder = ctx["root"] / f"replay-{sha(path)[:12]}"

    def one(pair):
        task, theirs = pair
        run_dir = folder / task["id"]
        done = run_dir / "replay.json"
        if done.is_file():
            return json.loads(done.read_text())
        shutil.rmtree(run_dir, ignore_errors=True)  # a half-written folder: the broker appends to events.jsonl
        episode = ctx["run"](task, path, run_dir, ctx["files"][task["id"]], toolset=NO_TOOLS,
                             llm=_scripted(theirs["answer"]), jev=lambda *args, **kwargs: None, gate=False)
        out = {"answer": episode.get("answer"), "stop_reason": episode.get("stop_reason")}
        run_dir.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(out))
        return out

    with ThreadPoolExecutor(WORKERS) as pool:
        outs = list(pool.map(one, passed))
    out = {"n": len(passed), "broken": 0, "no_submit": 0, "examples": []}
    for (task, theirs), mine in zip(passed, outs):
        if mine["answer"] is None:
            out["no_submit"] += 1
        elif grade(task, mine["answer"]) < theirs["reward"]:
            out["broken"] += 1
            out["examples"].append({"task": task["id"], "before": _form(theirs["answer"]), "after": _form(mine["answer"])})
    return out
