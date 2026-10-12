"""What past runs show, as text for the improver's prompt. Used only by rsi.improver.

`failures` and `controls` render the parent's train runs of this generation;
`history` renders every earlier candidate of the family with a per-task ledger.

Invariants:
- Nothing here reads `task["reference"]`. The one thing derived from it is the
  count inside `grading.feedback`. The improver learns pass or fail per task
  and nothing finer.
- A run that passed holds the reference: in its tool results and in its
  answer. So for a passing run (a control, or the parent's side of a task a
  candidate broke) no result text and no answer value is shown: results are
  given by length, answers by `grading.shape`. The candidate's side of a broken
  task is shown as a shape too, because it is the parent's right answer with
  something removed.
- The result of a tool that returns the grader's verdicts (GRADED_TOOLS) is
  replaced by one neutral line, and for mathbench the final reply, written
  after those verdicts, is not shown.
- Trajectory text is data. Every line of it after the first of a field is
  indented four spaces, so it cannot open a heading, close a fence or pass for
  a field label. Harness source and diffs sit in a fence longer than any run of
  backticks inside them.
- Long results, answers and questions are cut in the middle, the cut marked,
  because the informative part (a traceback's last line, the ANSWER line, the
  answer format) is at the end. The last ANSWER line is shown whatever was cut.
- Flags are computed by code from the events and the episode, and their counts
  cover every failed train run of the parent, not only the ones shown.
- History is capped at HISTORY_CHARS, each diff inside it at DIFF_CHARS. The
  ledger compares a candidate and its parent task by task on the rewards stored
  in their episode files of that generation.
"""
import difflib
import json
import re
from itertools import zip_longest
from pathlib import Path

from . import broker, grading
from .episodes import margin, sha

MAX_FAILURES = 8             # failed Train trajectories shown to the improver
MAX_CONTROLS = 3             # passing Train trajectories shown next to them
QUESTION_CHARS, ARGS_CHARS, RESULT_CHARS, ANSWER_CHARS = 1500, 300, 500, 800
HISTORY_CHARS = 16_000       # all past candidates together; the newest that fit are shown
DIFF_CHARS = 2_500           # one past candidate's diff
GIST_CHARS = 200             # a question, in a ledger line
LEDGER_TASKS = 5             # tasks listed per stage and direction in a ledger
LINEAGE = "lineage.jsonl"
KINDS = ("no answer submitted", "wrong answer")
GRADED_TOOLS = ("submit_laws",)  # their result is the grader's verdicts and score
# Measured on 96 seed-harness episodes (24 failures, 72 passes), hits on failures / false alarms on passes:
# 10/0, 10/0, 13/1, 13/0, 8/0, 1/0. ASKS_RE is the pattern those counts were made with. (The first flag then
# also counted three table tools, since removed, as python.)
FLAGS = ("the python tool was never called", "four or more read_file calls and no python call", "the model-call limit was hit",
         "no ANSWER line in the final reply", "the reply asks the user something or says it cannot run code",
         "the last tool result was an error")
ASKS_RE = re.compile(r"\bshall i\b|\bmay i\b|please (choose|tell|confirm)|do you want|would you like|can ?not (execute|run)"
                     r"|no (code )?execution tool|I cannot", re.I)


def _clip(text, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]} [... {len(text) - limit} more characters cut]"


def _ends(text, limit: int) -> str:
    """The first and last halves of `limit`, with the cut marked between them."""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit // 2]} [... {len(text) - limit} characters cut here ...] {text[limit // 2 - limit:]}"


def _data(text) -> str:
    """Recorded text as the prompt shows it: every line after the first indented four spaces."""
    return "\n    ".join(str(text).splitlines())


def fenced(text: str, tag: str) -> str:
    """`text` in a fence one backtick longer than any run inside it, so a line of backticks in it closes nothing."""
    fence = "`" * max(3, 1 + max(map(len, re.findall("`+", text)), default=0))
    return f"{fence}{tag}\n{text}\n{fence}"


def kind(task: dict, episode: dict) -> str:
    """Which of KINDS a failed run is, from the question and the submitted text alone."""
    answer = episode.get("answer")
    lines = grading.ANSWER_RE.findall(answer or "")
    # The grader reads the last ANSWER line, so an empty one is no answer.
    if not answer or ("ANSWER:" in task["question"] and not (lines and lines[-1].strip(" \t*_`"))):
        return KINDS[0]
    return KINDS[1]


def _events(path: Path) -> list:
    out = []
    for line in path.read_text(errors="replace").splitlines() if path.is_file() else []:
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a line torn by an interrupted write: the rest of the trajectory still reads
        if isinstance(event, dict):
            out.append(event)
    return out


def _is_error(result) -> bool:
    return str(result).startswith("ERROR") or "Traceback" in str(result)


def flags(episode: dict, events: list) -> list:
    """Which of FLAGS hold for one run."""
    called = [e.get("tool") for e in events if e.get("kind") == "tool_call"]
    results = [e.get("result") for e in events if e.get("kind") == "tool_result"]
    answer = episode.get("answer") or ""
    hold = ("python" not in called,
            called.count("read_file") >= 4 and "python" not in called,
            (episode.get("calls") or 0) >= broker.MODEL_CALL_LIMIT,
            grading.answer_line(answer) is None,
            bool(ASKS_RE.search(answer)),
            bool(results) and _is_error(results[-1]))
    return [flag for flag, on in zip(FLAGS, hold) if on]


def record(task: dict, episode: dict, run_dir: Path, passed: bool = False) -> str:
    """One run as compact text, built only from the question and the trajectory on disk."""
    # ponytail: the answer format of a DABench question is in its last 400 characters or so, which
    # the tail half keeps. Ceiling: split the question at "Answer format:" if formats get longer.
    events = _events(run_dir / "events.jsonl")
    lines = [f"QUESTION: {_data(_ends(task['question'], QUESTION_CHARS))}"]
    for event in events:
        if event.get("kind") == "tool_call":
            lines.append(f"CALL {event.get('tool')} {_data(_clip(json.dumps(event.get('args'), default=str), ARGS_CHARS))}")
        elif event.get("kind") == "tool_result" and event.get("tool") in GRADED_TOOLS:
            how = "refused the call with an error" if str(event.get("result")).startswith("ERROR") else "took the submission"
            lines.append(f"RESULT (not shown: {event['tool']} {how}; what it returns is the grader's output)")
        elif event.get("kind") == "tool_result" and passed:
            lines.append(f"RESULT ({len(str(event.get('result')))} characters{', an error' if _is_error(event.get('result')) else ''})")
        elif event.get("kind") == "tool_result":
            lines.append(f"RESULT {_data(_ends(event.get('result'), RESULT_CHARS))}")
        elif event.get("kind") == "submit" and not event.get("accepted"):
            lines.append(f"SUBMIT NOT ACCEPTED: {_data(_ends(event.get('reason'), ARGS_CHARS))}")
        elif event.get("kind") == "error":
            lines.append(f"ERROR {_data(_ends(event.get('message'), ARGS_CHARS))}")
    answer = episode.get("answer") or "(none)"
    if task.get("family") == "mathbench":
        # Written after submit_laws returned the verdicts, which it may repeat, and not what is scored.
        lines.append("FINAL ANSWER: (not shown: this family scores the submit_laws call, not the reply)")
    elif passed:
        lines.append(f"FINAL ANSWER: ({len(answer)} characters, values not shown) its last ANSWER line has the form "
                     f"{_data(_clip(grading.shape(grading.answer_line(answer) or '(no ANSWER line)'), ARGS_CHARS))}")
    else:
        shown = _ends(answer, ANSWER_CHARS)
        lines.append(f"FINAL ANSWER: {_data(shown)}")
        last = grading.answer_line(answer)
        if last and last not in shown:
            lines.append(f"ITS LAST ANSWER LINE: {_data(_clip(last, ANSWER_CHARS))}")
    lines.append(f"STOP REASON: {episode['stop_reason']}. OUTCOME: {'pass' if passed else 'fail'}")
    lines.append("FLAGS: " + ("; ".join(flags(episode, events)) or "none"))
    if not passed and task.get("family") != "mathbench":
        lines.append("GRADER FEEDBACK: " + ("; ".join(grading.feedback(task, episode.get("answer"))) or
                                            "an ANSWER line in the asked form, judged wrong"))
    return "\n".join(lines)


def failures(failed: list, parent_dir: Path) -> tuple:
    """(text, number shown): the flag counts over every failed run, then failed runs under one
    heading per kind, so a pattern is visible as a group. Each heading counts every failure of its
    kind; the runs shown are taken from the kinds in turn so neither crowds the other out."""
    if not failed:
        return "(none)", 0
    held = [flags(episode, _events(parent_dir / task["id"] / "events.jsonl")) for task, episode in failed]
    parts = [f"Flags, counted by code over all {len(failed)} failed runs (shown or not): "
             + "; ".join(f"{flag}: {sum(flag in mine for mine in held)}" for flag in FLAGS) + "."]
    groups = {k: [pair for pair in failed if kind(*pair) == k] for k in KINDS}
    turns = [pair for row in zip_longest(*groups.values()) for pair in row if pair][:MAX_FAILURES]
    i = 0
    for k, pairs in groups.items():
        shown = [pair for pair in pairs if pair in turns]
        if pairs:
            parts.append(f"### {k.capitalize()}: {len(pairs)} of {len(failed)} failed runs ({len(shown)} shown)")
        for task, episode in shown:
            i += 1
            parts.append(f"#### Failure {i} ({k})\n{record(task, episode, parent_dir / task['id'])}")
    return "\n\n".join(parts), len(turns)


def controls(passed: list, parent_dir: Path) -> str:
    """The first MAX_CONTROLS passing runs in draw order: what the current harness already does right."""
    return "\n\n".join(f"#### Control {i} (passed)\n{record(task, episode, parent_dir / task['id'], passed=True)}"
                       for i, (task, episode) in enumerate(passed[:MAX_CONTROLS], 1)) or "(none)"


# --- earlier attempts -------------------------------------------------------------------

def _share(score) -> str:
    return "not scored" if score is None else f"{score:.0%}"


def _stored(folder: Path) -> dict:
    """{task id: its episode.json} for the finished episodes of one harness folder."""
    out = {}
    for path in sorted(folder.glob("*/episode.json")):
        try:
            episode = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(episode, dict) and isinstance(episode.get("reward"), (int, float)):
            out[path.parent.name] = episode
    return out


def _line(episode: dict) -> str:
    answer = episode.get("answer")
    if not answer:
        return "(nothing submitted)"
    return "`" + _clip(grading.shape(grading.answer_line(answer) or "(no ANSWER line)"), ARGS_CHARS).replace("`", "'") + "`"


def _ledger(gen: Path, row: dict, cand: dict) -> tuple:
    """(lines, tasks lost net on val or else on train, how many were compared there) for one past candidate."""
    val = {str(seed) for seed in (row.get("val_seeds") or []) + (row.get("confirm_seeds") or [])}
    stages = {"train screen": [], "val": [], "decision (fresh runs of both)": []}
    for prefix, fresh in (("", False), ("fresh-", True)):
        theirs = _stored(gen / (prefix + row["parent_sha"][:12]))
        mine = _stored(gen / (prefix + (cand.get("sha") or "-")[:12]))
        for task_id in sorted(set(theirs) & set(mine)):
            on_val = task_id in val or task_id.rsplit("-", 1)[-1] in val
            stages["decision (fresh runs of both)" if fresh else "val" if on_val else "train screen"].append(
                (task_id, theirs[task_id], mine[task_id], gen / (prefix + row["parent_sha"][:12]) / task_id))
    lines, lost = [], {}
    for stage, pairs in stages.items():
        fixed = [p for p in pairs if p[2]["reward"] > p[1]["reward"]]
        broken = [p for p in pairs if p[2]["reward"] < p[1]["reward"]]
        lost[stage] = (len(broken) - len(fixed), len(pairs))
        if not pairs:
            continue
        lines.append(f"- {stage}, {len(pairs)} tasks run by both: fixed {len(fixed)}, broke {len(broken)}")
        for word, group in (("fixed", fixed), ("broke", broken)):
            for task_id, parent, candidate, run_dir in group[:LEDGER_TASKS]:
                asked = next((e.get("text") for e in _events(run_dir / "events.jsonl") if e.get("kind") == "user"), "")
                gist = " ".join(str(asked).split())
                line = f"  - {word} {task_id}: {gist[:GIST_CHARS]}{'...' if len(gist) > GIST_CHARS else ''}"
                if word == "broke":
                    line += f"\n    form of the parent's ANSWER line {_line(parent)}, of the candidate's {_line(candidate)}"
                lines.append(line)
            if len(group) > LEDGER_TASKS:
                lines.append(f"  - ({len(group) - LEDGER_TASKS} more {word} tasks not listed)")
    where = next((s for s in reversed(list(stages)) if lost[s][1]), "val")
    return lines, *lost[where]


def verdict(row: dict, cand: dict, lost: int, n: int) -> str:
    """How much a past result says about the idea: a loss inside run-to-run noise says nothing."""
    if cand.get("reason") == "adopted":
        return "adopted"
    if (cand.get("replay") or {}).get("broken"):
        return (f"VERIFIED REGRESSION: replayed on {cand['replay']['n']} answers the parent had right, it turned "
                f"{cand['replay']['broken']} into wrong ones")
    if not n:
        return "not scored, so nothing is known about the idea"
    if lost >= margin(n):
        return f"VERIFIED REGRESSION: lost {lost} tasks net over {n}, at least the margin of {margin(n)}"
    return (f"WITHIN NOISE: net {-lost:+d} over {n} tasks, inside the margin of {margin(n)}. Not evidence against the "
            "idea, and not evidence for it")


def history(home_dir: Path, family: str) -> str:
    """Every candidate this family has scored, most recent first, within HISTORY_CHARS: what it
    changed, the tasks it fixed and broke against its parent, and what that result is worth.
    Sources are read from runs/evolve/gen-<n>/candidates/ by sha."""
    lineage = home_dir / "harnesses" / LINEAGE
    rows = [json.loads(line) for line in lineage.read_text().splitlines() if line.strip()] if lineage.is_file() else []
    entries = []
    for row in reversed(rows):
        if row.get("family") != family:
            continue
        gen = home_dir / "runs" / f"evolve/gen-{row['generation']}"
        by_sha = {sha(path): path for path in (gen / "candidates").glob("*.py")}
        # parent.py is this generation's own copy; older generations only have harnesses/<parent>.
        parent = next((p for p in (gen / "candidates" / "parent.py", lineage.with_name(row["parent"]))
                       if p.is_file() and sha(p) == row["parent_sha"]), None)
        for i, cand in enumerate(row["candidates"], 1):
            ledger, lost, n = _ledger(gen, row, cand)
            lines = [f"### Generation {row['generation']}, candidate {i}: {cand['reason']}",
                     f"Verdict: {verdict(row, cand, lost, n)}."]
            if cand.get("score") is not None:
                lines.append(f"Val score {_share(cand['score'])} against the parent's {_share(row['parent_score'])} "
                             f"on {row['n_val']} tasks ({cand['wins']} tasks gained, {cand['losses']} lost)")
            if cand.get("rationale"):
                lines.append(f"Stated change: {cand['rationale']}")
            for example in (cand.get("replay") or {}).get("examples", [])[:LEDGER_TASKS]:
                lines.append(f"- replay broke {example['task']}: form of the parent's ANSWER line `{example['before']}`, "
                             f"of the candidate's `{example['after']}`")
            lines += ledger
            if parent and cand.get("sha") in by_sha:
                diff = "\n".join(difflib.unified_diff(parent.read_text().splitlines(),
                                                      by_sha[cand["sha"]].read_text(errors="replace").splitlines(),
                                                      "parent", "candidate", n=1, lineterm=""))
                # ponytail: a clipped diff of a wholesale rewrite is mostly the removed lines. Ceiling:
                # clip per hunk if rewrites keep arriving despite the cap on changed lines.
                lines.append(fenced(_clip(diff, DIFF_CHARS), "diff") if diff else "(identical to its parent)")
            elif cand.get("sha"):
                lines.append("(source no longer on disk: no diff)")
            entries.append("\n".join(lines))
    shown, used = [], 0
    for entry in entries:
        if used + len(entry) > HISTORY_CHARS:
            break
        shown.append(entry)
        used += len(entry)
    if len(shown) < len(entries):
        shown.append(f"({len(entries) - len(shown)} older candidates not shown)")
    return "\n\n".join(shown) or "(none yet)"
