"""Scored episodes on disk: the machinery `rsi evolve`, `rsi test` and `rsi compare` share.

`draw` names the tasks of a split, `batch` runs one harness on them and returns
the graded episodes, reusing what is on disk. Used by rsi.evolve and rsi.replay.

Invariants:
- An episode is stored under <root>/<harness_sha12>/<task id>/, or under
  <root>/<kind>-<harness_sha12>/<task id>/ when the caller names a kind: a
  second, independent run of the same harness on the same task (the fresh
  runs of the adoption decision) never shares a folder with the first.
- An episode whose episode.json holds a reward is not re-run. Its stored
  answer is graded again with the current grader and that reward is used
  (mathbench excepted: its grade lives in the Lab of the live episode).
- An episode made under another model or gate setting is refused, not mixed in.
- Nothing starts once the spend has passed the budget: `batch` raises Stop.
- Rewards may be fractional. Every comparison of two scores uses `mean`
  (rounded to 4 places), so totals that differ only by float error are a tie.
- For dabench, train and val draws are a per-generation sample of a finite
  pool, so generations overlap. Only split "test" reads the test pool.
"""
import fcntl
import hashlib
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from . import evals, tasks

WORKERS = 4                  # generated and dabench episodes mostly wait on the model
ERRORED = ("harness_error", "llm_error", "run_limit")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def margin(n: int) -> int:
    """Net wins a candidate needs over n paired tasks; the rule and its numbers are in rsi.evolve._decide."""
    return max(3, round(n / 15))

class Stop(Exception):
    """Ends a generation early. `record` says whether it still earns a lineage row."""

    def __init__(self, reason: str, record: bool):
        super().__init__(reason)
        self.record = record


def mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def spent(ctx: dict) -> float:
    return sum(ctx["costs"].values())


def context(home_dir: Path, folder: str, family: str, budget: float, run) -> dict:
    # costs is keyed by run folder, so an episode met twice (the timing run,
    # then the full batch) is counted once.
    return {"home": home_dir, "root": home_dir / "runs" / folder, "family": family, "budget": budget, "run": run,
            "costs": {}, "files": {}, "regraded": set()}


@contextmanager
def lock(path: Path, what: str):
    """Hold an exclusive lock on `path` or refuse at once. The kernel drops it when the process ends, however it ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"{what} is already running here ({path} is locked); wait for it to finish") from None
        yield


def draw(ctx: dict, split: str, n: int, generation: int = 0, confirm: bool = False) -> tuple:
    """n tasks of a split -> (what names them in a row: seeds, or task ids for dabench; the tasks).
    confirm=True is the second Val draw of a generation (draw B): same size, no task in common with the first."""
    if ctx["family"] == "dabench":
        pool = evals.pool("dabench", split, ctx["home"])
        if split == "test":
            batch = pool[:n]  # a fixed order, so every harness meets the same tasks; n is capped at the pool
        elif n * (2 if split == "val" else 1) > len(pool):
            most = f"{len(pool) // 2}: a second draw of the same size comes from the rest" if split == "val" else len(pool)
            raise ValueError(f"the dabench {split} pool has {len(pool)} tasks; --{split} must be at most {most}")
        else:
            # Deterministic per generation. The pool is finite, so unlike fresh
            # seeds the samples of consecutive generations overlap.
            batch = random.Random(f"{split}/{generation}").sample(pool, n)
            if confirm:
                first = {task["id"] for task in batch}
                batch = random.Random(f"confirm/{generation}").sample([t for t in pool if t["id"] not in first], n)
        ctx["files"].update({task["id"]: Path(task["files_dir"]) for task in batch})
        return [task["id"] for task in batch], batch
    # Draw B is the next n seeds of the generation's block.
    seeds, batch = tasks.split_seeds(split, n * (2 if confirm else 1), generation)[-n:] if n else [], []
    for seed in seeds:
        files_dir = ctx["root"] / "files" / f"{split}-{seed}"
        task = tasks.make_task(seed, files_dir, family=ctx["family"])
        ctx["files"][task["id"]] = files_dir
        batch.append(task)
    return seeds, batch


def grade(task: dict, text) -> float:
    return evals.grade_external(task, text) if task["family"] == "dabench" else tasks.grade(task, text)


def null(batch: list) -> float:
    return mean(grade(t, evals.null_guess(t) if t["family"] == "dabench" else tasks.guess(t)) for t in batch)


def batch(ctx: dict, harness: Path, todo: list, kind: str = "") -> list:
    """Run `harness` on each task and return the graded episodes in order.
    Finished episodes are read from disk; raises Stop once spend passes the budget.
    `kind` names a second, separate set of runs of the same harness (see the module docstring)."""
    folder = ctx["root"] / ((f"{kind}-" if kind else "") + sha(harness)[:12])

    def one(task):
        run_dir = folder / task["id"]
        path = run_dir / "episode.json"
        episode = json.loads(path.read_text()) if path.is_file() else {}
        # Episodes are reused by harness and task alone, so one made under another
        # model or gate setting would silently mix into this score. Refuse instead.
        now = {"model": os.environ.get("RSI_MODEL"), "gate": os.environ.get("RSI_GATE") == "1"}
        for key, value in now.items():
            if episode.get("reward") is not None and key in episode and episode[key] != value:
                raise ValueError(f"{run_dir} was made with {key}={episode[key]!r}, now {value!r}; "
                                 f"remove that folder or restore the setting")
        # The stored reward is what some earlier grader said. Grade the stored answer again.
        # mathbench keeps the stored reward: its grade is read from the Lab of the live
        # episode (module state in rsi.tasks), which is gone once the episode has ended.
        if episode.get("reward") is not None and task["family"] != "mathbench":
            reward = grade(task, episode.get("answer") or "")
            if reward != episode["reward"]:
                episode["reward"] = reward  # episode.json is left as the record of what was written then
                ctx["regraded"].add(str(run_dir))
        if episode.get("reward") is None:
            if spent(ctx) > ctx["budget"]:
                return None
            # A half-written folder is from an interrupted episode; the broker
            # appends to events.jsonl, so it must start from nothing.
            shutil.rmtree(run_dir, ignore_errors=True)
            # toolset then grade, back to back in this thread: mathbench keeps
            # the episode's Lab in module state keyed by task id.
            episode = ctx["run"](task, harness, run_dir, ctx["files"][task["id"]], toolset=tasks.toolset(task))
            episode["reward"] = grade(task, episode.get("answer") or "")
            run_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(episode, indent=1))
        ctx["costs"][str(run_dir)] = episode.get("cost") or 0.0
        return episode

    with ThreadPoolExecutor(1 if ctx["family"] == "mathbench" else WORKERS) as pool:
        episodes = list(pool.map(one, todo))
    if None in episodes:
        raise Stop(f"spend ${spent(ctx):.2f} passed the budget ${ctx['budget']:.2f}", record=False)
    return episodes
