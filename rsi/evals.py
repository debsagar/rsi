"""Evaluation suite: does the harness really work, on ground truth graded by code?

`run_eval` runs sampled tasks under named conditions and writes every episode
under <RSI_HOME>/runs/evals/<eval_id>/; `summarize` and `report` rebuild every
number from those files. Families: "generated" (rsi.tasks, Test seeds) and two
external sets copied by `fetch`: "dabench" (InfiAgent-DABench, Apache-2.0) and
"qrdata" (QRData, CC BY-NC 4.0: research use only, never redistribute).
`pool` splits DABench into the train, val and test pools `rsi evolve` and
`rsi compare` draw from; `mcnemar` and `interval` are the one copy of the
paired test and the interval. Reading an answer lives in rsi.grading.

Where grading departs from the benchmark authors' code (GRADER_VERSION in rsi.grading names
this rule set; bump it with any change):

  benchmark  authors' rule                      here                                  why
  QRData     choice: reply starts with gold     gold as a whole token, then nothing   "Cannot tell" scored on all 75
             (bare prefix)                      or punctuation; no other option named golds C, "nope" on "no"
  QRData     first number by eval.py's regex    shared parser: .5, 4e-5, 2,457.89     ".4" and "4e-5" were read as 4
  QRData     gold "8.53%" needs 0.0853 or       "8.53" accepted too                   the question asks for a percentage
             "8.53%"
  QRData     strict +-3% of gold                gold 0: |x| <= 1e-6                   strict bounds around 0 are empty
  DABench    items read from the whole reply    from the last line-anchored ANSWER    examples in the reasoning counted
             (after a GPT-3.5 reformat)         line; whole reply when there is none
  DABench    value equal as a string, or as     also equal ignoring quotes, letter    'Kuwait,Saudi Arabia' and
             floats within 1e-6                 case, spacing around , and :, and     '"Kuwait"' are the right answer;
                                                thousands separators; nan/inf never   float("nan") is not a number
                                                compare as numbers

Invariants:
- Grading is code only; no model judges. A reply with an ANSWER line is graded on
  its last ANSWER line alone. Every eval folder's meta.json and every results
  row carry GRADER_VERSION.
- Benchmark files are data: read and copied, never imported or executed.
- A benchmark row with a broken file or label is excluded by `load` with a
  reason (`excluded`), and the reason is written to the eval's meta.json.
- A DABench table belongs to exactly one pool. Files with the same normalised
  content (titanic.csv, titanic_train.csv) are one table; tables are dealt in
  order of the hash of their first file name: nothing random, the same split on
  every call, and no table seen in training is ever scored in val or test.
- The free null baseline always runs, and runs first. A 2-task timing run and a
  printed cost estimate precede the paid batch, which stops at `budget`.
- A condition fixes the pre-submit gate explicitly (`full` and `no_tools` on,
  `no_gate` off), so RSI_GATE never changes what an eval folder means.
- results.jsonl is appended one row per task x condition as it finishes; a
  finished episode is never re-run. An episode that ended with `llm_error` (the
  provider failed) is not finished: it gets no row, any row it has is ignored,
  and the next run does it again. After more than LLM_ERRORS of them in a row
  the run starts no more episodes and stops, saying so. A folder made with
  another harness,
  model or task list is refused rather than mixed. A folder holding only free
  `null` rows and no episode file was made by no model, so the first paid run
  adopts it.
- `summarize` rebuilds every row from tasks.jsonl and the run folders. A row of
  the current grader version must match its raw files exactly, or there is no
  report. A row of an older version is re-graded: its ungraded fields must
  still match, the re-graded rows go to results.regraded-v<N>.jsonl beside
  results.jsonl (never over it), and the report says how many rewards moved.
  results.jsonl is deduplicated by (task, condition), last row wins.
  Accuracies and paired tests use only tasks finished under every condition
  present, so no number compares different task sets.
- Intervals: Wilson score for 0/1 rewards, percentile bootstrap otherwise.
  Paired test: exact binomial on the discordant pairs, which is McNemar's test
  for 0/1 rewards and the exact sign test on paired differences otherwise; the
  result names which.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import broker, home as _home, tasks
from .grading import GRADER_VERSION, grade_external, whole

# Benchmark copies live in the repo's git-ignored evals/src/ (QRData is CC BY-NC: do not redistribute).
SRC = Path(__file__).resolve().parent.parent / "evals" / "src"
SOURCES = {"dabench": ("RSI_DABENCH_SRC", str(SRC / "dabench")), "qrdata": ("RSI_QRDATA_SRC", str(SRC / "qrdata"))}
FAMILIES = ["generated", *SOURCES]
CONDITIONS = ["null", "no_tools", "full", "no_gate"]
PAIRS = [("full", "no_gate"), ("full", "no_tools"), ("full", "null")]
POOLS = {"train": 0.25, "val": 0.60, "test": 1.0}  # cumulative share of questions: 25%, 35%, 40%
WORKERS = 4                  # episodes mostly wait on the model
TIMING_TASKS = 2
LLM_ERRORS = 3               # more episodes in a row than this ending with llm_error stop the run
BOOTSTRAPS, BOOT_SEED = 10_000, 0
Z95 = 1.959964               # normal quantile for a 95% Wilson interval
# Fields of a results row that no grader decides: they must match the raw files whatever the grader version.
FACTS = ["task_id", "family", "level", "condition", "stop_reason", "calls", "tool_calls", "cost", "elapsed", "unsupported"]
DA_QUESTION = ("The data is in the attached file `{file}`.\n\n{question}\n\nConstraints: {constraints}\n\n"
               "Answer format: {format}\n\nEnd your reply with one final line that starts with `ANSWER:` and holds "
               "every item of the answer format as @name[value], for example `ANSWER: @first_name[1.23] "
               "@second_name[yes]`. Names and values are compared exactly (numbers within 0.000001), so round "
               "as the constraints say.")
QR_QUESTION = "{description}\n\n{question}\n\nEnd your reply with one final line `ANSWER: <value>`, where <value> is {form}."
QR_FORMS = {"numerical": "one plain decimal number with no units, no thousands separators and no scientific "
                         "notation (write a percentage with a % sign)",
            "multiple_choice": "exactly one of: {options}"}


OUTAGE = (f"more than {LLM_ERRORS} episodes in a row ended with llm_error: the model provider is failing, so the run "
          "stopped. Those episodes are not finished; the same command runs them again")


def _jsonl(path) -> list:
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _events(run_dir) -> list:
    return _jsonl(Path(run_dir) / "events.jsonl")


def _rows(eval_dir: Path) -> list:
    """results.jsonl without the rows of episodes the provider failed on: those are not finished."""
    return [row for row in _jsonl(eval_dir / "results.jsonl") if row["stop_reason"] != "llm_error"]


# --- benchmark data -----------------------------------------------------------

def _home_dir(home) -> Path:
    return Path(home).resolve() if home else _home()


def fetch(name: str, home=None) -> Path:
    """Copy one benchmark from its source folder into <home>/evals/data/<name>/."""
    var, default = SOURCES[name]
    src, dest = Path(os.environ.get(var) or default), _home_dir(home) / "evals" / "data" / name
    if not src.is_dir():
        raise RuntimeError(f"{var}={src} is not a folder; point it at the downloaded {name} data")
    dest.mkdir(parents=True, exist_ok=True)
    if name == "dabench":
        for file in ("da-dev-questions.jsonl", "da-dev-labels.jsonl"):
            shutil.copyfile(src / file, dest / file)
        shutil.copytree(src / "da-dev-tables", dest / "da-dev-tables", dirs_exist_ok=True)
    else:
        shutil.copyfile(src / "QRData.json", dest / "QRData.json")
        (dest / "data").mkdir(exist_ok=True)
        with zipfile.ZipFile(src / "data.zip") as archive:
            for info in archive.infolist():
                if not info.is_dir():  # written flat by base name, so no member can land outside data/
                    (dest / "data" / Path(info.filename).name).write_bytes(archive.read(info))
    return dest


@lru_cache(maxsize=None)
def _file_problem(path: Path, as_table: bool) -> str | None:
    if path.name in ("", ".", "..") or not path.is_file():
        return f"data file {path.name!r} is missing"
    if as_table:
        try:
            pd.read_csv(path)
        except Exception as e:  # whatever pandas raises, the agent's tools would hit it too
            return f"pandas cannot read {path.name!r}: {type(e).__name__}"
    return None


def _records(name: str, home=None) -> list:
    """Every benchmark row as (task, source folder, reason); reason is None for a usable row."""
    root = _home_dir(home) / "evals" / "data" / name
    if not root.is_dir():
        fetch(name, home)
    out = []
    if name == "dabench":
        labels = {row["id"]: row["common_answers"] for row in _jsonl(root / "da-dev-labels.jsonl")}
        for q in _jsonl(root / "da-dev-questions.jsonl"):
            ref = [[str(k), str(v)] for k, v in labels.get(q["id"]) or []]
            names = [k for k, _ in ref]
            reason = _file_problem(root / "da-dev-tables" / Path(q["file_name"]).name, True) or (
                "no label" if not ref else
                "label repeats an answer name with different values" if len(set(names)) < len(names) else
                "label has an empty value or a cut-off bracket" if not all(
                    re.fullmatch(r"\w+", k) and re.fullmatch(r"[^\[\]]*[^\[\]\s][^\[\]]*", v) for k, v in ref) else None)
            question = DA_QUESTION.format(file=q["file_name"], question=q["question"], constraints=q["constraints"],
                                          format=q["format"])
            out.append(({"id": f"dabench-{q['id']}", "family": name, "question": question, "files": [Path(q["file_name"]).name],
                         "history": [], "reference": ref, "level": q["level"]}, root / "da-dev-tables", reason))
        return out
    for i, row in enumerate(json.loads((root / "QRData.json").read_text())):
        kind, options = row["meta_data"]["question_type"], row["meta_data"].get("multiple_choices") or []
        files = [Path(f).name for f in row["data_files"]]
        # ponytail: QRData files are checked for existence only; reading all 312 MB on every load is not worth it.
        reason = next(filter(None, (_file_problem(root / "data" / f, False) for f in files)), None)
        if kind == "numerical":
            reason = reason or (None if whole(row["answer"]) else "gold answer is not a number")
        else:
            reason = reason or (None if row["answer"].lower() in [o.lower() for o in options] else "gold is not one of the options")
        form = QR_FORMS.get(kind, "").format(options=", ".join(options))
        question = QR_QUESTION.format(description=row["data_description"], question=row["question"], form=form)
        out.append(({"id": f"qrdata-{i}", "family": name, "question": question, "files": files, "history": [],
                     "reference": row["answer"], "level": kind, "options": options}, root / "data", reason))
    return out


def excluded(name: str, home=None) -> dict:
    """{task id: why `load` leaves it out}."""
    return {task["id"]: reason for task, _, reason in _records(name, home) if reason}


def load(name: str, home=None, n=None, seed=0) -> list:
    """Usable tasks in the project's contract shape, each with its own `files_dir`.
    n samples deterministically per seed, in proportion to level / question type."""
    good = [(task, src) for task, src, reason in _records(name, home) if reason is None]
    if n is not None and not 0 <= n <= len(good):
        raise ValueError(f"{name} has {len(good)} usable tasks; n must be 0..{len(good)}")
    ranked = []
    for level in sorted({task["level"] for task, _ in good}):
        group = [pair for pair in good if pair[0]["level"] == level]
        random.Random(f"{seed}/{level}").shuffle(group)
        # Rank by quantile within the level: the first n of the merged order is
        # proportional to level sizes, and a larger n extends a smaller one.
        ranked += [((i + 0.5) / len(group), level, pair) for i, pair in enumerate(group)]
    return [_with_files(task, src) for _, _, (task, src) in sorted(ranked, key=lambda item: item[:2])[:n]]


def _with_files(task: dict, src: Path) -> dict:
    files_dir = src.parent / "task-files" / task["id"]  # only this task's files, so /data holds nothing else
    files_dir.mkdir(parents=True, exist_ok=True)
    for file in task["files"]:
        if not (files_dir / file).exists():
            try:
                os.link(src / file, files_dir / file)
            except OSError:
                shutil.copyfile(src / file, files_dir / file)
    return {**task, "files_dir": str(files_dir)}


def _rank(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=None)
def _content(path: Path) -> str:
    """Hash of a table's text without byte-order mark, line endings and trailing blanks."""
    lines = path.read_bytes().removeprefix(b"\xef\xbb\xbf").splitlines()
    return hashlib.sha256(b"\n".join(line.rstrip() for line in lines).rstrip()).hexdigest()


def pool(name: str, split: str, home=None) -> list:
    """The usable tasks of one pool ("train", "val" or "test"), in the shape `load` returns.
    Tables (files grouped by normalised content) are taken in order of the hash of their
    first file name and dealt to train, then val, then test, each until it holds its share
    of the questions (POOLS). Tasks come back in order of the hash of their id, so the
    first n are a fixed sample that a larger n extends."""
    if name != "dabench" or split not in POOLS:  # ponytail: DABench only; a QRData task can span several files
        raise ValueError(f"pools exist for dabench and the splits {list(POOLS)}, not {name!r} / {split!r}")
    good = [(task, src) for task, src, reason in _records(name, home) if reason is None]
    table = {task["id"]: _content(src / task["files"][0]) for task, src in good}
    owner, seen = {}, 0
    for task, _ in sorted(good, key=lambda pair: _rank(pair[0]["files"][0])):
        if table[task["id"]] not in owner:
            owner[table[task["id"]]] = next(s for s, share in POOLS.items() if seen < share * len(good))
            seen += sum(key == table[task["id"]] for key in table.values())
    mine = [pair for pair in good if owner[table[pair[0]["id"]]] == split]
    return [_with_files(task, src) for task, src in sorted(mine, key=lambda pair: _rank(pair[0]["id"]))]


# --- grading (rsi.grading and rsi.tasks) -------------------------------------------

def grade(task: dict, text) -> float:
    return tasks.grade(task, text) if task["family"] == "generated" else grade_external(task, text)


def null_guess(task: dict) -> str:
    """The constant answer available without the data: 0 for a number, the first
    option for a choice, 0 in every DABench item."""
    if task["family"] == "generated":
        return tasks.guess(task)
    if task["family"] == "dabench":
        return "ANSWER: " + " ".join(f"@{name}[0]" for name, _ in task["reference"])
    return f"ANSWER: {task['options'][0] if task['level'] == 'multiple_choice' else '0'}"


# --- running ----------------------------------------------------------------------

NO_TOOLS = ([], lambda name, args: "ERROR no tools are available in this run; answer from what you know")
# The gate is named in every condition, so RSI_GATE cannot change what a condition means.
OVERRIDES = {"full": {"gate": True}, "no_gate": {"gate": False}, "no_tools": {"gate": True, "toolset": NO_TOOLS}}


def _row(task: dict, condition: str, episode: dict) -> dict:
    return {"task_id": task["id"], "family": task["family"], "level": task["level"], "condition": condition,
            "reward": episode["reward"], "stop_reason": episode["stop_reason"], "calls": episode.get("calls", 0),
            "tool_calls": episode.get("tool_calls", 0), "cost": episode.get("cost") or 0.0,
            "elapsed": episode.get("elapsed") or 0.0, "unsupported": bool(episode.get("unsupported")),
            "grader_version": GRADER_VERSION}

def _episode(task: dict, condition: str, harness, eval_dir: Path, run) -> dict:
    """One results row, always re-graded from the files on disk.
    A finished episode is read, never re-run; run=None (the audit) only reads."""
    if condition == "null":
        return _row(task, condition, {"reward": grade(task, null_guess(task)), "stop_reason": "null"})
    run_dir = eval_dir / condition / task["id"]
    path = run_dir / "episode.json"
    episode = json.loads(path.read_text()) if path.is_file() else {}
    if episode.get("reward") is None or episode.get("stop_reason") == "llm_error":
        if run is None:
            raise FileNotFoundError(path)
        shutil.rmtree(run_dir, ignore_errors=True)  # half-written: the broker appends to events.jsonl
        episode = run(task, harness, run_dir, Path(task["files_dir"]), **OVERRIDES[condition])
        episode["reward"] = grade(task, episode.get("answer") or "")
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(episode, indent=1))
    episode["reward"] = grade(task, episode.get("answer") or "")
    if condition == "no_gate" and any(e["kind"] == "submit" and not e.get("accepted", True) for e in _events(run_dir)):
        raise RuntimeError(f"{run_dir}: the gate bounced a submission under no_gate; the broker did not honour "
                           "gate=False, so this condition is not what its name says")
    return _row(task, condition, episode)


def _tasks(family: str, n: int, seed: int, home_dir: Path, files_root: Path) -> list:
    if family != "generated":
        return load(family, home_dir, n, seed)
    out = []
    for task_seed in tasks.split_seeds("test", n):  # owned ground truth: `seed` does not move the Test split
        task = tasks.make_task(task_seed, files_root / str(task_seed))
        out.append({**task, "level": task["id"].split("-")[1], "files_dir": str(files_root / str(task_seed))})
    return out


def run_eval(family, n, conditions, home=None, seed=0, run=broker.run_episode, harness=None, budget=None, say=print) -> dict:
    """Run n tasks under each condition and return summarize()'s dict (plus `stopped` on a budget stop)."""
    conditions = ["null"] + [c for c in dict.fromkeys(conditions) if c != "null"]
    if family not in FAMILIES or set(conditions) - set(CONDITIONS):
        raise ValueError(f"family must be one of {FAMILIES} and conditions among {CONDITIONS}")
    home_dir = _home_dir(home)
    harness = Path(harness).resolve() if harness else broker.current_harness(home_dir)
    sha = hashlib.sha256(harness.read_bytes()).hexdigest()
    eval_dir = home_dir / "runs" / "evals" / f"{family}-n{n}-s{seed}-{sha[:8]}"
    batch = _tasks(family, n, seed, home_dir, eval_dir / "files")
    meta = {"eval_id": eval_dir.name, "family": family, "n": n, "seed": seed, "harness": str(harness), "harness_sha": sha,
            "model": os.environ.get("RSI_MODEL"), "grader_version": GRADER_VERSION, "task_ids": [t["id"] for t in batch],
            "excluded": excluded(family, home_dir) if family != "generated" else {}}
    old = json.loads((eval_dir / "meta.json").read_text()) if (eval_dir / "meta.json").is_file() else meta
    rows = {(r["task_id"], r["condition"]): r for r in _rows(eval_dir)}
    paid = any(True for c in CONDITIONS[1:] for _ in (eval_dir / c).glob("*/episode.json"))  # a crash can leave these without a row
    if all(condition == "null" for _, condition in rows) and not paid:
        old = {**old, "model": meta["model"]}  # nothing paid so far: no model made the folder, so this run's model adopts it
    changed = [k for k in ("harness_sha", "model", "task_ids") if old[k] != meta[k]]
    if changed:
        raise ValueError(f"{eval_dir} was made with a different {', '.join(changed)}; use another seed or remove the folder")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (eval_dir / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in batch))
    spent, outage = [sum(r["cost"] for r in rows.values())], [0]

    def work(job):
        condition, task = job
        if (task["id"], condition) in rows:
            return None
        if outage[0] > LLM_ERRORS or (budget is not None and spent[0] > budget):
            return "stop"
        row = _episode(task, condition, harness, eval_dir, run)
        outage[0] = outage[0] + 1 if row["stop_reason"] == "llm_error" or outage[0] > LLM_ERRORS else 0  # past the limit it stays
        return row

    def go(jobs) -> bool:
        """Run jobs, appending each row as it arrives. False when the budget stopped any."""
        with ThreadPoolExecutor(WORKERS) as pool, open(eval_dir / "results.jsonl", "a") as f:
            done = []
            for row in pool.map(work, jobs):
                done.append(row)
                if isinstance(row, dict) and row["stop_reason"] != "llm_error":
                    rows[(row["task_id"], row["condition"])] = row
                    spent[0] += row["cost"]
                    f.write(json.dumps(row) + "\n")
                    f.flush()
        if outage[0] > LLM_ERRORS:
            raise RuntimeError(OUTAGE)
        return "stop" not in done

    # ponytail: one episode per task x condition, so model sampling noise is inside every paired difference.
    # McNemar accounts for it across tasks; telling it apart per task needs repeats (a `rep` in the run path).
    go([("null", task) for task in batch])
    say(f"Null baseline on {len(batch)} {family} tasks: {np.mean([rows[(t['id'], 'null')]['reward'] for t in batch] or [0]):.1%} (free).")
    jobs = [(c, task) for c in conditions[1:] for task in batch]
    stopped = None
    if jobs:
        first = "full" if "full" in conditions else conditions[1]
        timing = [(first, task) for task in batch[:TIMING_TASKS]]
        finished = go(timing)
        timed = [rows[(t["id"], c)] for c, t in timing if (t["id"], c) in rows]
        left = sum((task["id"], c) not in rows for c, task in jobs)
        estimate = float(np.mean([r["cost"] for r in timed] or [0.0])) * left
        minutes = float(np.mean([r["elapsed"] for r in timed] or [0.0])) * left / WORKERS / 60
        say(f"Estimate from {len(timed)} `{first}` episodes: ${estimate:.2f} more and about {minutes:.0f} min for the {left} "
            f"episodes left of {len(jobs)} ({len(conditions) - 1} conditions x {len(batch)} tasks); ${spent[0]:.2f} spent so far"
            + ("." if budget is None else f", budget ${budget:.2f}."))
        if budget is not None and spent[0] + estimate > budget:
            stopped = (f"${spent[0]:.2f} spent plus the estimate ${estimate:.2f} exceeds the budget ${budget:.2f}; "
                       "nothing more was run")
        elif not (finished and go(jobs)):
            stopped = f"spend ${spent[0]:.2f} passed the budget ${budget:.2f}; the same command with a larger --budget resumes"
    summary = summarize(eval_dir)
    report(summary, say)
    if stopped:
        summary["stopped"] = stopped
        say(f"Stopped: {stopped}.")
    return summary


# --- report -----------------------------------------------------------------------

def _rebuild(eval_dir: Path, rows: list, task_by: dict) -> tuple:
    """-> (every row as the current graders give it, regrade note or None). Every row must be rebuilt from its
    raw files or there is no report: field for field when it was written by this grader version; in every
    field no grader decides (FACTS) when it was written by an older one, whose rewards are then replaced."""
    rebuilt, bad, old = [], [], []
    for row in rows:
        try:
            new = _episode(task_by[row["task_id"]], row["condition"], None, eval_dir, None)
        except (KeyError, OSError, ValueError):
            new = None
        current = row.get("grader_version") == GRADER_VERSION
        # Compared on the fields a row has now: rows written before the failure classifier was removed carry more.
        if new is None or any(new[k] != row.get(k) for k in (new if current else FACTS)):
            bad.append(f"{row['condition']}/{row['task_id']}")
            continue
        rebuilt.append(new)
        if not current:
            old.append((row, new))
    if bad:
        raise ValueError(f"{len(bad)} row(s) of {eval_dir / 'results.jsonl'} do not match their raw files: {', '.join(bad[:10])}")
    if not old:
        return rebuilt, None
    path = eval_dir / f"results.regraded-v{GRADER_VERSION}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rebuilt))
    changes = {c: {"up": sum(n["reward"] > o["reward"] for o, n in old if o["condition"] == c),
                   "down": sum(n["reward"] < o["reward"] for o, n in old if o["condition"] == c)}
               for c in CONDITIONS if any(o["condition"] == c for o, _ in old)}
    return rebuilt, {"from": sorted({str(o.get("grader_version", "1")) for o, _ in old}), "rows": len(old),
                     "file": str(path), "changes": changes}


def interval(rewards) -> list:
    """95% interval of the mean reward over tasks: Wilson score when every reward is 0 or 1 (never zero-width,
    so 40 of 40 is 91.2%-100%), percentile bootstrap otherwise. No rewards: [0, 1], nothing is known."""
    rewards = np.asarray(rewards, dtype=float)
    n = len(rewards)
    if not n:
        return [0.0, 1.0]
    if _binary(rewards):
        p, z2 = float(rewards.mean()), Z95 ** 2 / n
        half = Z95 * float(np.sqrt(p * (1 - p) / n + z2 / (4 * n)))
        low, high = (p + z2 / 2 - half) / (1 + z2), (p + z2 / 2 + half) / (1 + z2)
    else:
        picks = np.random.default_rng(BOOT_SEED).integers(0, n, (BOOTSTRAPS, n))
        low, high = np.percentile(rewards[picks].mean(axis=1), [2.5, 97.5])
    return [round(min(1.0, max(0.0, float(x))), 6) for x in (low, high)]  # 6 digits: the report prints 0.1% steps of this


def _binary(rewards) -> bool:
    return all(x in (0, 1) for x in rewards)


def mcnemar(a, b) -> dict:
    """Paired rewards of a against b on the same tasks: an exact two-sided binomial test on the pairs where
    one side did better. For 0/1 rewards that is McNemar's test. For partial credit it is the exact sign
    test on the paired differences (it uses the direction of each difference, not its size), and `test`
    says so. The p-value is the exact float; printing formats it."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError(f"paired test needs one reward per task on both sides, got {len(a)} and {len(b)}")
    wins, losses = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    return {"wins": wins, "losses": losses, "ties": len(a) - wins - losses, "p_value": p,
            "test": "McNemar" if _binary(a + b) else "sign"}


def show_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def summarize(eval_dir) -> dict:
    """Everything the report prints, recomputed from results.jsonl, tasks.jsonl and the run folders."""
    eval_dir = Path(eval_dir).resolve()
    written, task_by = _rows(eval_dir), {t["id"]: t for t in _jsonl(eval_dir / "tasks.jsonl")}
    rows = list({(r["task_id"], r["condition"]): r for r in written}.values())  # a re-run after a crash appends again: last wins
    rows, regraded = _rebuild(eval_dir, rows, task_by)
    by = {c: {r["task_id"]: r for r in rows if r["condition"] == c} for c in CONDITIONS if any(r["condition"] == c for r in rows)}
    common = sorted(set.intersection(*map(set, by.values()))) if by else []
    meta = json.loads((eval_dir / "meta.json").read_text()) if (eval_dir / "meta.json").is_file() else {}
    out = {"eval_dir": str(eval_dir), "meta": meta, "grader_version": GRADER_VERSION, "regraded": regraded,
           "n_tasks": len(common), "unpaired_rows": len(rows) - len(common) * len(by),
           "duplicate_rows": len(written) - len(rows), "conditions": {}, "pairs": {}, "gate": None}
    for c, got in by.items():
        rewards = np.array([got[t]["reward"] for t in common], dtype=float)
        out["conditions"][c] = {
            "n": len(common), "accuracy": round(float(rewards.mean()), 4) if common else None,
            "ci95": interval(rewards) if common else None, "ci_method": "Wilson" if _binary(rewards) else "bootstrap",
            "cost_per_task": round(float(np.mean([got[t]["cost"] for t in common])), 6) if common else None,
            "cost_total": round(sum(r["cost"] for r in got.values()), 6),  # includes unpaired rows: money spent is money spent
            "no_answer": sum(got[t]["stop_reason"] not in ("submitted", "null") for t in common)}
    for a, b in PAIRS:
        if a in by and b in by:
            out["pairs"][f"{a} vs {b}"] = {"n": len(common), **mcnemar(*([by[c][t]["reward"] for t in common] for c in (a, b)))}
    if "full" in by:
        bounced = {}
        for t in common:
            submits = [e for e in _events(eval_dir / "full" / t) if e["kind"] == "submit"]
            if submits and not submits[0]["accepted"]:
                bounced[t] = (grade(task_by[t], submits[0]["text"]), len(submits) > 1)
        ended = sum(by["full"][t]["reward"] >= 1.0 for t in bounced)
        again = sum(resubmitted for _, resubmitted in bounced.values())
        out["gate"] = {"episodes": len(common), "bounced": len(bounced),
                       "first_submission_correct": int(sum(first for first, _ in bounced.values())),
                       "ended_correct": ended, "resubmitted_wrong": again - ended, "never_resubmitted": len(bounced) - again,
                       "same_tasks_no_gate_correct": sum(by["no_gate"][t]["reward"] >= 1.0 for t in bounced) if "no_gate" in by else None}
    return out


def report(summary: dict, say=print) -> None:
    meta, regraded = summary["meta"], summary["regraded"]
    say(f"\nEval {meta.get('eval_id')}: {summary['n_tasks']} {meta.get('family')} tasks finished under every condition "
        f"({summary['unpaired_rows']} rows left out because their task is not finished everywhere"
        + (f"; {summary['duplicate_rows']} duplicate rows in results.jsonl ignored, last one kept" if summary["duplicate_rows"] else "")
        + ").")
    say(f"Folder: {summary['eval_dir']} (results.jsonl, tasks.jsonl, <condition>/<task_id>/episode.json and events.jsonl)")
    say(f"Harness {meta.get('harness')} sha {str(meta.get('harness_sha'))[:12]}, model {meta.get('model')}, grader version "
        f"{summary['grader_version']}; {len(meta.get('excluded') or {})} benchmark rows excluded as invalid (reasons in meta.json).")
    if regraded:
        moved = ", ".join(f"{c} {n['up']} newly correct and {n['down']} newly wrong" for c, n in regraded["changes"].items())
        say(f"RE-GRADED: {regraded['rows']} rows were written under grader version {'/'.join(regraded['from'])}. Every stored "
            f"answer was graded again with version {summary['grader_version']} and every number below uses the new rewards. "
            f"Rewards changed: {moved}. Re-graded rows: {regraded['file']} (results.jsonl is untouched).")
    if not summary["n_tasks"]:
        return
    say(f"\n{'condition':<10}{'n':>4}{'accuracy':>10}{'95% interval':>16}{'$/task':>9}{'no answer':>11}")
    for name, c in summary["conditions"].items():
        say(f"{name:<10}{c['n']:>4}{c['accuracy']:>10.1%}{c['ci95'][0]:>8.1%}-{c['ci95'][1]:<7.1%}{c['cost_per_task']:>9.4f}{c['no_answer']:>11}")
    methods = sorted({c["ci_method"] for c in summary["conditions"].values()})
    say(f"Intervals: {' and '.join(methods)} (Wilson score for 0/1 rewards; otherwise a percentile bootstrap over tasks, "
        f"{BOOTSTRAPS:,} resamples, seed {BOOT_SEED}). `no answer` counts episodes that ended without an accepted submission.")
    if summary["pairs"]:
        say(f"\nPaired comparisons on the same {summary['n_tasks']} tasks (exact two-sided test on the discordant pairs):")
    for name, p in summary["pairs"].items():
        first, second = name.split(" vs ")
        test = "McNemar" if p["test"] == "McNemar" else "sign test on paired differences, rewards are not 0/1"
        say(f"  {name}: {first} right where {second} was wrong on {p['wins']}, the reverse on {p['losses']}, "
            f"same on {p['ties']}; {show_p(p['p_value'])} ({test})")
    gate = summary["gate"]
    if gate:
        say(f"\nGate: in {gate['episodes']} `full` episodes the first submission was bounced {gate['bounced']} times; "
            f"{gate['first_submission_correct']} of those first submissions were already correct. After the bounce "
            f"{gate['ended_correct']} ended correct, {gate['resubmitted_wrong']} resubmitted and ended wrong, and "
            f"{gate['never_resubmitted']} never submitted again"
            + ("." if gate["same_tasks_no_gate_correct"] is None else
               f"; the same {gate['bounced']} tasks under no_gate: {gate['same_tasks_no_gate_correct']} correct."))


def main(argv=None) -> dict:
    """`rsi eval --family dabench --n 40 --conditions null,no_tools,full,no_gate --budget 5` or `rsi eval --report <dir>`."""
    parser = argparse.ArgumentParser(prog="rsi eval", description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, default="generated")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated, from: " + ", ".join(CONDITIONS))
    parser.add_argument("--budget", type=float, default=5.0, help="US dollars")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--harness", help="harness file; default: the current one in harnesses/")
    parser.add_argument("--report", metavar="EVAL_DIR", help="print the report for a finished or interrupted eval and exit")
    args = parser.parse_args(argv)
    if args.report:
        summary = summarize(args.report)
        report(summary)
        return summary
    return run_eval(args.family, args.n, [c.strip() for c in args.conditions.split(",") if c.strip()],
                    seed=args.seed, harness=args.harness, budget=args.budget)
