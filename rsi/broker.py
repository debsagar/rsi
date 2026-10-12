"""The Execution Broker: owns one run of one harness on one task.

It starts the harness process (through rsi/transport.py), answers its JSON-line
requests (generate, execute, judge, submit), enforces the limits, and writes the
trajectory to `<run_dir>/events.jsonl` and the summary to `<run_dir>/episode.json`.

Invariants:
- The harness gets no network, no keys and no files. Every model, tool and
  Jev call is made here, on the host, and logged before the reply is sent.
- `task["reference"]` and any other host-side key never reach the harness.
- A tool result is never shortened silently: a clipped event carries
  `clipped`, `chars_full`, `chars_shown` and the name of the file that holds
  the full text. `chars_shown` is the length of what was sent, the clip marker
  included, and is never more than CLIP_CHARS.
- The pre-submit gate (Jev rates whether the answer is supported and can send it
  back once) is a setting, off by default: `gate=True`, or RSI_GATE=1 when the
  caller passes nothing. When on it runs here, so a rewritten harness cannot
  remove it; when off Jev is not asked and every submit is accepted.
  episode.json records which (`gate`).
- Whatever ends the run, an exception in here included, the harness process
  and its container are killed, the python tool's container is removed, and
  episode.json is written. An exception is re-raised after that, with
  `stop_reason` "broker_error" unless the run had already ended.
- A run always ends: at an accepted submit, when the harness exits, by the
  over-limit rule, or at RUN_LIMIT_S. `stop_reason` says which. RUN_LIMIT_S
  also bounds whatever is in flight: a model call, a tool call and a write to
  a harness that is not reading are each given only the time that is left.
- The over-limit rule, kept separately for each kind of call (model, tool,
  judge). A request past that kind's limit is not carried out; it is answered
  with an error and logged. The run ends instead, with no reply and
  `stop_reason` "<kind>_call_limit", when such a request is not the first of
  its kind and either (a) the harness has had its chance to react to the
  first: for model calls that is at once, because the error reaches the
  harness itself; for tool and judge calls it is once a model call has
  succeeded since, because only the model can read an error result; or (b) it
  is more than the OVER_LIMIT_BURST-th of its kind, which stops a harness that
  spins without ever calling the model. So one model message asking for many
  tools past the limit gets an error for each, and the model can still answer.
  A harness that exits without an accepted answer after running into a limit
  gets the `stop_reason` of the last limit it ran into.
"""
import base64
import contextlib
import hashlib
import json
import os
import queue
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import home, jev as _jev, llm as _llm, tools as _tools, transport
from .transport import OutOfTime

MODEL_CALL_LIMIT = 12
TOOL_CALL_LIMIT = 12
JUDGE_CALL_LIMIT = 12  # the harness's own judge requests; the gate's Jev call is not counted
OVER_LIMIT_BURST = 20  # requests past one limit that are answered with an error, whatever else; see the module docstring
TOOL_LIMIT_S = 60
# The python tool kills its own container at 60 s and reports it. The broker's
# own cut-off waits a little longer so that clearer message wins the race.
TOOL_GRACE_S = 20
RUN_LIMIT_S = 600
CLIP_CHARS = 4000
GATE_THRESHOLD = 0.5
GATE_EVENTS = 15
GATE_CLAIMS_CHARS = 3000
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
SEED_HARNESS = Path(__file__).with_name("harness.py")
GATE_QUESTIONS = {
    "supported": {"type": "noul",
                  "instructions": "Is `answer` to `question` supported by the tool results in `trajectory`? Check each "
                                  "number and finding in `answer` and `claims` against what the tools returned.",
                  "criteria": {"true": "every number and finding in the answer appears in or follows from the tool results, "
                                       "or the answer states nothing that would need tool evidence",
                               "false": "the answer states a number or finding that the tool results do not show or that they contradict"}},
}


@dataclass
class Run:
    """Mutable state of one episode."""
    task: dict
    run_dir: Path
    files_dir: Path
    llm: Callable
    jev: Callable
    schemas: list
    call: Callable[[str, dict], str]
    on_event: Callable | None
    gate: bool
    deadline: float  # time.monotonic() at which the run limit is reached
    events: list = field(default_factory=list)
    calls: int = 0
    tool_calls: int = 0
    judge_calls: int = 0
    submits: int = 0
    usage: dict = field(default_factory=lambda: {"in": 0, "out": 0})
    cost: float = 0.0
    over: dict = field(default_factory=dict)  # kind -> [requests past that limit, run.calls at the first of them]
    limit_hit: str | None = None    # last limit the harness ran into; it may still submit after that
    stop_reason: str | None = None  # set once, by whatever ended the run
    answer: str | None = None
    unsupported: bool = False

    def left(self) -> float:
        return self.deadline - time.monotonic()


def current_harness(home_dir: Path | None = None) -> Path:
    """The harness every run uses: the highest-numbered file the lineage records as adopted, else 000.py.

    Copies the seed to 000.py on first use. A numbered file that no lineage row
    adopted is ignored here (and refused by `rsi evolve`): a harness only becomes
    current by winning the adoption test. Files are ordered by number, so 1000.py
    is after 999.py.
    """
    folder = (home_dir or home()) / "harnesses"
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "000.py").is_file():
        shutil.copyfile(SEED_HARNESS, folder / "000.py")
    adopted = {"000.py"}
    lineage = folder / "lineage.jsonl"
    for line in lineage.read_text().splitlines() if lineage.is_file() else []:
        try:
            adopted.add(json.loads(line).get("adopted") or "000.py")
        except (ValueError, AttributeError):
            continue  # a damaged row cannot vouch for a harness
    usable = [folder / name for name in adopted if Path(name).stem.isdigit() and (folder / name).is_file()]
    return max(usable, key=lambda p: int(p.stem))


def _emit(run: Run, kind: str, **fields) -> dict:
    event = {"id": len(run.events) + 1, "ts": datetime.now(timezone.utc).isoformat(), "step": run.calls, "kind": kind, **fields}
    run.events.append(event)
    with open(run.run_dir / "events.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    if run.on_event:
        run.on_event(event)
    return event


def _over_limit(run: Run, kind: str, message: str) -> dict | None:
    """A request past the `kind` limit. -> the error event to answer it with, or
    None when the run ends here (the over-limit rule in the module docstring)."""
    seen = run.over.setdefault(kind, [0, run.calls])
    seen[0] += 1
    run.limit_hit = f"{kind}_call_limit"
    could_react = kind == "model" or run.calls > seen[1]
    if seen[0] > 1 and (could_react or seen[0] > OVER_LIMIT_BURST):
        run.stop_reason = run.limit_hit
        return None
    return _emit(run, "error", message=message)


def _with_images(run: Run, messages: list) -> list:
    """Attach the task's image files to the first user message. Done here on
    every call because the harness has no files and must not carry megabytes of
    base64 through the protocol."""
    images = [run.files_dir / name for name in run.task.get("files", [])
              if Path(name).suffix.lower() in IMAGE_TYPES and (run.files_dir / name).is_file()]
    if not images:
        return messages
    out = list(messages)
    for i, message in enumerate(out):
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
            parts = [{"type": "text", "text": message["content"]}]
            for path in images:
                data = base64.b64encode(path.read_bytes()).decode()
                parts.append({"type": "image_url", "image_url": {"url": f"data:{IMAGE_TYPES[path.suffix.lower()]};base64,{data}"}})
            out[i] = {**message, "content": parts}
            break
    return out


def _generate(run: Run, req: dict) -> dict | None:
    if run.calls >= MODEL_CALL_LIMIT:
        message = f"model call limit ({MODEL_CALL_LIMIT}) reached; submit your answer now"
        return {"error": message} if _over_limit(run, "model", message) else None
    run.calls += 1

    def on_delta(text):  # a call abandoned at the run limit must not keep streaming into a finished run
        if run.stop_reason is None:
            run.on_event({"kind": "delta", "text": text})

    started = time.monotonic()
    messages, tools = _with_images(run, req["messages"]), run.schemas if req.get("use_tools", True) else None
    try:
        out = transport.bounded(lambda: run.llm(messages, tools=tools, on_delta=on_delta if run.on_event else None), run.left())
    except OutOfTime:
        raise
    except Exception as e:  # retries are already spent inside llm.complete
        message = f"model call failed: {type(e).__name__}: {e}"
        _emit(run, "error", message=message)
        run.stop_reason = "llm_error"
        return {"error": message}
    message, usage = out["message"], out["usage"]
    run.usage["in"] += usage["in"]
    run.usage["out"] += usage["out"]
    run.cost += out["cost"]
    _emit(run, "model", content=message.get("content"), tool_calls=message.get("tool_calls") or [],
          tokens_in=usage["in"], tokens_out=usage["out"], cost=out["cost"], elapsed=round(time.monotonic() - started, 3))
    return {"message": message}


def _call_tool(run: Run, name: str, args: dict) -> str:
    """Run one tool so that a hung one cannot hold the run: it gets its own
    allowance or what is left of the run, whichever is shorter."""
    try:
        return transport.bounded(lambda: str(run.call(name, args)), min(TOOL_LIMIT_S + TOOL_GRACE_S, run.left()))
    except OutOfTime:
        if run.left() <= 0:
            raise
        return f"ERROR {name} timed out after {TOOL_LIMIT_S}s"
    except Exception as e:  # rsi.tools never raises; a task family's own toolset might
        return f"ERROR {name} failed: {type(e).__name__}: {e}"


def _execute(run: Run, req: dict) -> dict | None:
    name, args = req["name"], req.get("args")
    if run.tool_calls >= TOOL_CALL_LIMIT:
        message = f"tool call limit ({TOOL_CALL_LIMIT}) reached; submit your answer now"
        event = _over_limit(run, "tool", message)
        return {"result": f"ERROR {message}", "clipped": False, "event_id": event["id"]} if event else None
    run.tool_calls += 1
    call_event = _emit(run, "tool_call", tool=name, args=args)
    started = time.monotonic()
    error = None
    if isinstance(args, str):  # the model's raw argument string, passed through by the harness
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            error = f"ERROR malformed tool call arguments: {e}"
    if error is None and not isinstance(args, dict):
        error = f"ERROR tool call arguments must be a JSON object, got {type(args).__name__}"
    result = error or _call_tool(run, name, args)
    full = len(result)
    fields = {"chars_full": full, "chars_shown": full, "clipped": full > CLIP_CHARS}
    if fields["clipped"]:
        fields["file"] = f"tool-{run.tool_calls:03d}.txt"
        (run.run_dir / fields["file"]).write_text(result, errors="replace")  # a lone surrogate must not end the run

        def marker(cut):
            return f"\n[... {cut} of {full} characters clipped here; the full text is saved as {fields['file']} ...]\n"

        keep = CLIP_CHARS - len(marker(full))  # the real count has no more digits, so the marker fits
        result = result[:keep - keep // 2] + marker(full - keep) + result[-(keep // 2):]
        fields["chars_shown"] = len(result)
    event = _emit(run, "tool_result", tool=name, call_id=call_event["id"], result=result, **fields,
                  elapsed=round(time.monotonic() - started, 3))
    return {"result": result, "clipped": fields["clipped"], "event_id": event["id"]}


def _ask_jev(run: Run, state: dict, questions: dict, source: str) -> dict | None:
    """Ask Jev and log it. None (an outage) is logged with an error and means "no flag"."""
    error = None
    try:
        answers = run.jev(state, questions)
    except Exception as e:
        answers, error = None, f"{type(e).__name__}: {e}"
    if answers is None:
        _emit(run, "judge", source=source, questions=list(questions), answers=None, error=error or "Jev returned no answer")
    else:
        _emit(run, "judge", source=source, questions=list(questions), answers=answers)
    return answers


def _judge(run: Run, req: dict) -> dict | None:
    if run.judge_calls >= JUDGE_CALL_LIMIT:
        # Past the limit the harness gets None, which it already has to treat
        # as "Jev is unavailable"; Jev itself is not called.
        message = f"judge call limit ({JUDGE_CALL_LIMIT}) reached; carry on without Jev"
        return {"answers": None} if _over_limit(run, "judge", message) else None
    run.judge_calls += 1
    return {"answers": _ask_jev(run, req["state"], req["questions"], "harness")}


def _submit(run: Run, req: dict) -> dict:
    text, claims = str(req["text"]), req.get("claims") or []
    run.submits += 1
    p = None  # gate off, or no usable answer from Jev: no flag
    if run.gate:
        trajectory = [{"id": e["id"], "tool": e["tool"], "args": json.dumps(e.get("args"), default=str)[:400]} if e["kind"] == "tool_call"
                      else {"id": e["id"], "tool": e["tool"], "result": e["result"][:600]}
                      for e in run.events if e["kind"] in ("tool_call", "tool_result")][-GATE_EVENTS:]
        state = {"question": run.task["question"][:1500], "answer": text[:3000],
                 "claims": json.dumps(claims, default=str)[:GATE_CLAIMS_CHARS], "trajectory": trajectory}
        answers = _ask_jev(run, state, GATE_QUESTIONS, "gate")
        try:
            p = float(answers["supported"]["noul"])
        except (TypeError, KeyError, ValueError):
            pass
    flagged = p is not None and p < GATE_THRESHOLD
    accepted = not flagged or run.submits > 1  # the gate can send an answer back once, never twice
    reason = (f"Jev rates the chance that this answer is supported by the tool results at {p:.2f}, "
              f"below {GATE_THRESHOLD}.") if flagged else ""
    _emit(run, "submit", text=text, claims=claims, accepted=accepted, reason=reason, p_supported=p)
    if accepted:
        run.answer, run.unsupported, run.stop_reason = text, flagged, "submitted"
        _emit(run, "answer", text=text, claims=claims, unsupported=flagged)
    return {"accepted": accepted, "reason": reason}


HANDLERS = {"generate": _generate, "execute": _execute, "judge": _judge, "submit": _submit}


def _serve(run: Run, proc, first: dict) -> None:
    """Answer the harness's requests until something ends the run or the harness leaves."""
    lines = transport.lines(proc)
    try:
        alive = transport.send(proc, first, run.left())
        while alive and run.stop_reason is None:
            try:
                line = lines.get(timeout=max(0.0, run.left()))
            except queue.Empty:
                raise OutOfTime from None
            if line is None:
                # The harness left by itself. Let it finish exiting so the exit
                # code the caller reads is its own and not the kill's.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=max(0.0, min(10, run.left())))
                return
            try:
                req = json.loads(line.decode())
                reply = HANDLERS[req["op"]](run, req)
            except (KeyError, TypeError, ValueError, RecursionError) as e:  # RecursionError: JSON nested too deep
                run.stop_reason = "harness_error"
                _emit(run, "error", message=f"bad request from the harness ({type(e).__name__}: {e}): {line[:300]!r}")
                return
            if reply is None:
                return  # ended by the over-limit rule
            alive = transport.send(proc, reply, run.left())
    except OutOfTime:
        run.stop_reason = "run_limit"
        _emit(run, "error", message=f"run limit ({RUN_LIMIT_S}s) reached; the harness was stopped")


def run_episode(task: dict, harness_path: Path, run_dir: Path, files_dir: Path | None = None, *,
                on_event=None, llm=_llm.complete, jev=_jev.ask, toolset=None, in_docker: bool = True,
                gate: bool | None = None) -> dict:
    """Run one harness on one task and return the episode.json dict.

    gate: the pre-submit gate on or off; None reads RSI_GATE ("1" is on, anything else or unset is off).
    toolset: None for rsi.tools, or (schemas, call(name, args) -> str).
    on_event(event) fires for every event written, plus {"kind": "delta", "text": str}, which is not written.
    in_docker=False runs `python -m rsi.runner <harness>` as a plain subprocess (tests only).
    """
    started = time.monotonic()
    harness_path, run_dir = Path(harness_path).resolve(), Path(run_dir).resolve()
    harness_sha256 = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    run_dir.mkdir(parents=True, exist_ok=True)
    files_dir = Path(files_dir).resolve() if files_dir else run_dir / "files"
    python = _tools.PythonSession(files_dir)  # starts no container until the python tool is first called
    if toolset is None:
        ctx = {"files_dir": files_dir, "python": python}
        schemas, call = _tools.tool_schemas(), lambda name, args: _tools.call_tool(ctx, name, args)
    else:
        schemas, call = toolset
    if gate is None:
        gate = os.environ.get("RSI_GATE") == "1"
    run = Run(task, run_dir, files_dir, llm, jev, schemas, call, on_event, gate, started + RUN_LIMIT_S)
    public_task = {"id": task["id"], "question": task["question"], "files": task.get("files", []), "history": task.get("history", [])}
    try:
        _emit(run, "user", text=task["question"], files=task.get("files", []))
        with open(run_dir / "harness-stderr.txt", "w") as stderr, contextlib.closing(python):
            proc, container = transport.start(harness_path, in_docker, stderr)
            try:
                _serve(run, proc, {"task": public_task, "tools": schemas})
            finally:  # however _serve ended, the harness does not outlive it
                code = transport.stop(proc, container)
        if run.stop_reason is None:
            crashed = code != 0 and not run.limit_hit
            run.stop_reason = run.limit_hit or ("harness_error" if crashed else "no_submit")
            if crashed:
                tail = (run_dir / "harness-stderr.txt").read_text(errors="replace")[-2000:]
                _emit(run, "error", message=f"the harness exited with code {code} before submitting", stderr=tail)
            elif not run.limit_hit:
                _emit(run, "error", message="the harness exited without an accepted answer")
    except BaseException as e:
        run.stop_reason = run.stop_reason or "broker_error"
        with contextlib.suppress(Exception):  # the failure may be the event log itself
            _emit(run, "error", message=f"the broker failed: {type(e).__name__}: {e}")
        raise
    finally:
        episode = {
            "run_id": run_dir.name, "session_id": task.get("session_id"), "task_id": task["id"],
            "harness_sha256": harness_sha256,
            "model": os.environ.get("RSI_MODEL"), "calls": run.calls, "tool_calls": run.tool_calls,
            "usage": run.usage, "cost": round(run.cost, 6), "stop_reason": run.stop_reason,
            "unsupported": run.unsupported, "gate": gate, "answer": run.answer,
            "reward": None,  # filled in by the caller for graded tasks; the broker never reads task["reference"]
            "elapsed": round(time.monotonic() - started, 3),
        }
        (run_dir / "episode.json").write_text(json.dumps(episode, indent=1))
    return episode
