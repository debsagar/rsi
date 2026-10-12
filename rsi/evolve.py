"""The evolution loop: ask a model to edit the harness, keep an edit only if it clearly scores better.

`evolve()` runs one generation and returns its lineage row; `score_test()`
scores a harness on the Test split (`rsi test`) and `compare()` scores two on
the same Test tasks (`rsi compare`). Families: "generated" and "mathbench"
(rsi.tasks, by seed) and "dabench" (rsi.evals.pool, by task id). Running and
reusing episodes is rsi.episodes; what the improver is shown is rsi.evidence,
how its reply becomes files is rsi.improver, the free replay check is rsi.replay.

Invariants:
- Spending is ordered cheapest first: the null baseline (free), a 2-task
  timing run, a printed worst-case cost estimate, and only then the batches.
  Nothing starts once the generation's spend has passed the budget.
- The improver never sees `task["reference"]` or a reward beyond pass/fail, and
  nothing a passing run computed or answered (rsi.evidence).
- A candidate passes these stages in order, each against the parent's own runs:
  the free replay (rsi.replay), a one-task preflight, a screen on Train tasks,
  Val draw A, and (the best one only) the decision. Hand-written candidates
  (`candidate_files`) go through the same stages. A candidate whose file is
  identical to the parent's or to an earlier candidate's is not scored, and a
  slot the improver skipped or whose edit did not apply has no file at all.
- Nobody is adopted on draw A. The decision (`_decide`) uses only fresh
  episodes of parent and candidate on draw A again and on a disjoint draw B,
  paired by task: zero errored candidate episodes and net wins of at least
  `episodes.margin`. The exact paired p-value is reported, never the gate.
- Rewards may be fractional. The parent "failed" a task when it got less than
  the full reward 1.0. A candidate fixes or wins a task by getting more reward
  than the parent on that same task and breaks or loses one by getting less.
- The screen and draw A stop early once the tasks still to run cannot change
  what happens to the candidate (`_screen_dead`, `_val_dead`). The rule reads
  only the outcomes so far, so a resumed run stops at the same task, and the
  lineage entry says how many episodes were not run (`skipped`).
- Train and Val seeds are fresh each generation; for dabench they are a
  per-generation sample of a finite pool, so generations overlap. Test tasks
  are read only by `score_test`, and `evolve` never asks for the test pool.
- Everything is on disk under runs/evolve/gen-<n>/ and is reused: a finished
  episode is not re-run (first-draw runs in <sha12>/, the decision's in
  fresh-<sha12>/, replays in replay-<sha12>/), and a saved proposal is not
  bought again. The generation number is the count of lineage rows, so an
  interrupted or budget-stopped generation (no row) resumes in place.
- A row is appended to harnesses/lineage.jsonl when a generation finishes or
  stops for lack of headroom. A budget stop appends nothing.
- Adoption is the row first, the file second (written whole, then renamed).
  The next start repairs a row whose file is missing from the candidate saved
  under runs/, and refuses to run while harnesses/ holds a numbered file that
  no row adopted.
- One generation at a time per home: `evolve` holds harnesses/evolve.lock for
  its whole run and a second one refuses to start. `rsi test` and `rsi compare`
  never write to harnesses/ or runs/evolve and read a harness file that is
  only ever renamed into place, so they do not take that lock; each takes a
  lock on its own runs/test/<harness_sha12>/ folder instead, so two tests of
  one harness cannot clear each other's half-written episodes.
"""
import ast
import json
import math
import os
import shutil
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from . import broker, env, evals, home as _home, improver, llm as _llm, replay
from .broker import run_episode
from .episodes import ERRORED, WORKERS, Stop, batch, context, draw, lock, margin, mean, null, sha, spent
from .improver import LINEAGE, MAX_LINES

TIMING_TASKS = 2
HEADROOM = (0.10, 0.90)      # outside this band Val cannot show an improvement
CHUNK = WORKERS              # tasks a candidate runs between two checks of whether the rest can still matter


def _install(source: Path, target: Path) -> None:
    """Copy so that `target` is either absent or whole: readers glob for NNN.py and never see the .tmp name."""
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def _reconcile(home_dir: Path) -> list:
    """The lineage rows, once harnesses/ agrees with them.

    A row whose adopted file is missing is a crash between the row and the
    copy. It is repaired: the decision is on record, and the candidate with
    the recorded sha is still under runs/, so finishing the copy cannot differ
    from what the interrupted run would have written. A numbered file that no
    row adopted is refused, not repaired: the loop cannot tell an interrupted
    adoption from a file put there by hand, has no scores to write a row from,
    and must not delete a file it did not record."""
    folder = home_dir / "harnesses"
    path = folder / LINEAGE
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    except ValueError as e:
        raise ValueError(f"{path} holds a line that is not JSON ({e}); repair or remove that line") from None
    for row in rows:
        if row.get("adopted") and not (folder / row["adopted"]).is_file():
            want = next((c["sha"] for c in row["candidates"] if c["reason"] == "adopted"), None)
            saved = home_dir / "runs" / f"evolve/gen-{row['generation']}" / "candidates"
            source = next((p for p in sorted(saved.glob("*.py")) if sha(p) == want), None)
            if source is None:
                raise ValueError(f"{path} says generation {row['generation']} adopted {row['adopted']}, but that file is "
                                 f"missing from {folder} and its source is no longer in {saved}; restore the file")
            _install(source, folder / row["adopted"])
    known = {"000.py"} | {row.get("adopted") for row in rows}
    extra = sorted(p.name for p in folder.glob("[0-9][0-9][0-9].py") if p.name not in known)
    if extra:
        raise ValueError(f"{folder} holds {', '.join(extra)}, which no row of {LINEAGE} adopted (an interrupted adoption, or "
                         "a file put there by hand). Move it out: an interrupted generation then resumes from disk at no "
                         "cost, and a hand-written file can be scored with `rsi evolve --candidate PATH`")
    return rows


def _invalid(source: str) -> str | None:
    """Why a candidate cannot be run at all, or None."""
    # Newline characters, as Python and an editor count lines; str.splitlines also splits on
    # form feed and U+2028, which sit inside a line of valid source.
    n = source.count("\n") + (not source.endswith("\n") and bool(source))
    if n > MAX_LINES:
        return f"{n} lines, over the limit of {MAX_LINES}"
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as e:
        return f"does not parse: {e}"
    # rsi.runner executes the module once, then calls `module.run(task, api)` and drops the
    # result. So `run` must be bound by a statement the module body itself runs: a def or an
    # assignment (`run = lambda task, api: ...`, `run = main`). Not accepted: `async def`
    # (the call would only build a coroutine), a class, and a binding nested in `if` or `try`,
    # which this check cannot follow. Whether the bound object works is the preflight's job.
    def binds(node) -> bool:
        if isinstance(node, ast.FunctionDef):
            return node.name == "run"
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) and node.value else []
        return any(isinstance(target, ast.Name) and target.id == "run" for target in targets)

    if not any(binds(node) for node in tree.body):
        return "defines no top-level run(task, api)"
    return None

def _entry(path, **fields) -> dict:
    """A candidate's entry in the lineage row. `path` None is a slot with no file. `confirm` is the
    second-draw record of rows written before the decision replaced it; it stays null on new rows."""
    return {"sha": sha(path) if path else None, "valid": False, "reason": None, "score": None, "errors": None, "wins": None,
            "losses": None, "p_value": None, "rationale": None, "form": None, "replay": None, "screen": None,
            "skipped": None, "confirm": None, "decision": None, **fields}


# ponytail: both rules assume no task pays more than reward 1.0, which holds for every family's
# grader. Ceiling: a family with bonus reward would need its maximum here.
def _screen_dead(rewards: list, theirs: list) -> bool:
    """With `rewards` on the first tasks of a screen whose parent rewards are `theirs`: can the candidate
    no longer end with at least one fix and no more breaks than fixes, whatever the other tasks give?"""
    fixed, broken = sum(r > p for r, p in zip(rewards, theirs)), sum(r < p for r, p in zip(rewards, theirs))
    return fixed + sum(p < 1.0 for p in theirs[len(rewards):]) < max(1, broken)


def _val_dead(rewards: list, errors: int, n: int, parent_score: float) -> bool:
    """With `rewards` on the first tasks of an n-task draw: can the candidate no longer go on to the decision?
    It needs zero errored episodes and a mean strictly above the parent's, even if every other task is right."""
    return errors > 0 or mean(rewards + [1.0] * (n - len(rewards))) <= parent_score


def _until(ctx: dict, path: Path, todo: list, dead) -> list:
    """Episodes of `path` on `todo`, CHUNK tasks at a time, stopping once dead(episodes so far) holds."""
    episodes = []
    for i in range(0, len(todo), CHUNK):
        episodes += batch(ctx, path, todo[i:i + CHUNK])
        if len(episodes) < len(todo) and dead(episodes):
            break
    return episodes


def _score(ctx: dict, path: Path, screen: list, val_tasks: list, parent_rewards: list, passed: list) -> dict:
    """One candidate's entry in the lineage row, up to and including Val draw A.
    `screen` is [(train task, the parent's reward on it)]; `passed` the parent's passing train episodes."""
    cand = _entry(path, reason=_invalid(path.read_text(errors="replace")))
    screen_tasks, theirs = [task for task, _ in screen], [reward for _, reward in screen]
    if cand["reason"] is None:
        cand["replay"] = replay.check(ctx, path, passed)
        if cand["replay"] and cand["replay"]["broken"]:
            cand["reason"] = ("dropped by the free replay: given {n} answers the parent had right, it submitted {broken} "
                              "of them changed into wrong ones".format(**cand["replay"])
                              + f" ({', '.join(e['task'] for e in cand['replay']['examples'][:5])})")
    if cand["reason"] is None:
        # The preflight task is the first task the candidate would meet anyway, so
        # its episode is reused. Any errored ending drops the candidate, not only
        # harness_error: with one errored episode it could not be adopted.
        stop = batch(ctx, path, (screen_tasks or val_tasks)[:1])[0]["stop_reason"]
        if stop in ERRORED:
            cand["reason"] = f"preflight task ended with {stop}"
    if cand["reason"] is not None:
        return cand
    cand.update(valid=True, skipped={"screen": 0, "val": 0})
    if screen_tasks:
        # Train tasks only, so Val is not touched by this filter: the parent's
        # failures first, then as many of its passes. Task by task against the
        # parent, so a fractional gain counts and so does making a partly-right task worse.
        rewards = [e["reward"] for e in _until(ctx, path, screen_tasks, lambda es: _screen_dead([e["reward"] for e in es], theirs))]
        fixed, broken = sum(r > p for r, p in zip(rewards, theirs)), sum(r < p for r, p in zip(rewards, theirs))
        cand["screen"] = {"fixed": fixed, "broken": broken, "n": len(screen_tasks)}
        cand["skipped"]["screen"] = len(screen_tasks) - len(rewards)
        if not fixed or broken > fixed:
            early = f"; stopped after {len(rewards)} tasks, when the rest could not change that" if cand["skipped"]["screen"] else ""
            cand["reason"] = (f"dropped by the screen on {len(screen_tasks)} train tasks: fixed {fixed} of the parent's "
                              f"{sum(p < 1.0 for p in theirs)} failures, broke {broken} tasks{early} (fixed: more reward "
                              "than the parent on that task; broke: less)")
            return cand
    n, errors = len(val_tasks), lambda es: sum(e["stop_reason"] in ERRORED for e in es)
    episodes = _until(ctx, path, val_tasks, lambda es: _val_dead([e["reward"] for e in es], errors(es), n, mean(parent_rewards)))
    rewards = [e["reward"] for e in episodes]
    cand["skipped"]["val"] = n - len(episodes)
    if cand["skipped"]["val"]:
        # Not a score: the draw was not finished, so nothing here is compared with a full draw.
        cand["errors"] = errors(episodes)
        cand["reason"] = (f"{cand['errors']} errored episodes" if cand["errors"] else
                          f"stopped after {len(episodes)} of {n} val tasks ({sum(r >= 1.0 for r in rewards)} right): even with "
                          f"every other task right it could not finish above the parent's {mean(parent_rewards):.0%}")
        return cand
    paired = evals.mcnemar(rewards, parent_rewards)
    cand.update(score=mean(rewards), errors=errors(episodes), wins=paired["wins"], losses=paired["losses"], p_value=paired["p_value"])
    return cand


def _decide(ctx: dict, cand: dict, path: Path, parent: Path, both: list) -> bool:
    """The adoption decision for the draw-A winner; fills cand["decision"] and its reason.

    The winner was picked as the best of several on one noisy draw, so its lead there is biased
    upwards, and the same harness on the same task changes outcome in about 18% of repeated run
    pairs. So the draw-A episodes are not counted at all. Parent and candidate are both run again
    on draw A (new runs, in fresh-<sha12>/) and on the disjoint draw B, and only those 2n paired
    tasks decide: adopt when the candidate has no errored episode and its net wins (tasks where it
    got more reward than the parent, minus tasks where it got less) reach episodes.margin(2n) =
    max(3, round(2n / 15)), which is 6 at the default 45 + 45.

    What that margin buys, from a simulation of this rule at n = 45 + 45, margin 6, three
    candidates per generation, with tasks drawn as 62% always passed, 36% unstable (passed with
    p = 0.5 on any run) and 2% always failed, which reproduces the measured flip rate: a set of
    three candidates with NO effect is adopted 6% of the time (the rule this replaced, "ahead on
    draw A, not behind on draw B, ahead overall", adopted one 39% of the time); a true gain of 10
    points is adopted 81% of the time and of 15 points 99%. A re-run of that simulation with the
    gain in one of the three candidates gave 6%, 73% and 98%, and 54% at 8 points, 26% at 5.
    Plainly: a real gain under about 8 points will usually be REJECTED. With 90 validation tasks
    and this much run-to-run noise a small gain cannot be told from luck, and the rule prefers
    missing it to adopting noise that the next generation then builds on.

    The p-value recorded is the exact paired test (evals.mcnemar) on the same pairs. It is
    reported for the reader; the margin is the gate.
    Does not protect against: drift from reusing the finite dabench Val pool over many
    generations (the Test pool stays the only clean estimate), or a candidate whose gain is real
    on these 2n tasks and absent elsewhere. Only the draw-A best is decided; trying the runner-up
    after a failed decision would turn the decision into a second selection."""
    theirs = [e["reward"] for e in batch(ctx, parent, both, kind="fresh")]
    episodes = batch(ctx, path, both, kind="fresh")
    mine = [e["reward"] for e in episodes]
    paired, need = evals.mcnemar(mine, theirs), margin(len(both))
    cand["decision"] = d = {"n": len(both), "wins": paired["wins"], "losses": paired["losses"],
                            "net": paired["wins"] - paired["losses"], "margin": need,
                            "errors": sum(e["stop_reason"] in ERRORED for e in episodes), "p_value": paired["p_value"],
                            "parent_score": mean(theirs), "score": mean(mine)}
    ok = not d["errors"] and d["net"] >= need
    cand["reason"] = "adopted" if ok else (
        "won draw A but not the decision: on fresh runs of {n} tasks it won {wins} and lost {losses} against the parent "
        "(net {net:+d}, at least {margin} needed), with {errors} errored episodes".format(**d))
    return ok


def _generation(ctx: dict, row: dict, parent: Path, train: int, candidates: int, llm, files, say) -> None:
    generation, val = row["generation"], row["n_val"]
    # All three draws come before any spend: an n over the pool fails for free.
    row["val_seeds"], val_tasks = draw(ctx, "val", val, generation)
    row["confirm_seeds"], second_tasks = draw(ctx, "val", val, generation, confirm=True)  # draw B; the name predates the decision
    train_tasks = draw(ctx, "train", train, generation)[1]
    row["null_score"] = null(val_tasks)
    say(f"null baseline (guessing, no tools, free): {row['null_score']:.0%} on {val} val tasks")

    timing = batch(ctx, parent, val_tasks[:TIMING_TASKS])
    # Worst case: the parent on draw A and train, then again on both draws for the
    # decision; every candidate through a screen as large as train (it is the
    # failures plus at most as many passes, so at most train) and draw A; one
    # candidate through the decision's two draws. Replays cost nothing, and
    # stopping early only lowers the count.
    n_episodes = (val + train + 2 * val) + candidates * (train + val) + 2 * val
    per_cost = sum(e.get("cost") or 0.0 for e in timing) / len(timing)
    per_s = sum(e.get("elapsed") or 0.0 for e in timing) / len(timing)
    # ponytail: the estimate covers episodes only. The single improver call is
    # not in it; the mid-run check in episodes.batch is what bounds actual spend.
    estimate = per_cost * n_episodes
    say(f"estimate from {len(timing)} timed episodes: at most {n_episodes} episodes x ${per_cost:.4f} = ${estimate:.2f} "
        f"(every screen, draw and the decision counted in full, improver call not included), about "
        f"{n_episodes * per_s / 60:.0f} episode-minutes; budget ${ctx['budget']:.2f}")
    if estimate > ctx["budget"]:
        raise Stop(f"cost estimate ${estimate:.2f} exceeds the budget ${ctx['budget']:.2f}", record=False)

    parent_val = batch(ctx, parent, val_tasks)
    parent_rewards = [e["reward"] for e in parent_val]
    row["parent_score"] = mean(parent_rewards)
    say(f"parent {parent.name}: {row['parent_score']:.0%} on val, "
        f"{sum(e['stop_reason'] in ERRORED for e in parent_val)} errored episodes")
    if not HEADROOM[0] <= row["parent_score"] <= HEADROOM[1]:
        side = "above 90%" if row["parent_score"] > HEADROOM[1] else "below 10%"
        raise Stop(f"the parent scores {row['parent_score']:.0%} on val, {side}: these tasks leave no room to "
                   "measure an improvement, so nothing was proposed", record=True)

    graded = list(zip(train_tasks, batch(ctx, parent, train_tasks)))
    failed = [(t, e) for t, e in graded if e["reward"] < 1.0]
    passed = [(t, e) for t, e in graded if e["reward"] >= 1.0]
    # Deterministic: the failures, then the first passes in draw order. Failures come first so a
    # candidate that fixes none is stopped before the passes are paid for. With no parent failure
    # there is nothing to fix, so there is no screen and candidates go straight to Val.
    screen = [(t, e["reward"]) for t, e in failed + passed[:len(failed)]]
    say(f"parent failed {len(failed)} of {train} train tasks; screen set is {len(screen)} train tasks; "
        + (f"scoring {candidates} hand-written candidates" if files else f"asking the improver for {candidates} candidates"))
    (ctx["root"] / "candidates").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent, ctx["root"] / "candidates" / "parent.py")  # what later generations diff against
    proposed = improver.hand(ctx, files) if files else improver.propose(ctx, parent, graded, candidates, 2 * val, llm)
    if ctx.get("note"):
        say(ctx["note"])
    seen = {sha(parent): "the parent"}
    for i, slot in enumerate(proposed, 1):
        path = slot["path"]
        if path is None:  # skipped by the improver, or its edit did not apply: nothing to run
            cand = _entry(None, reason=slot["reason"] if slot["form"] == "skip" else f"no candidate file: {slot['reason']}")
        else:
            # A copy would score exactly what its twin scores; it is recorded and not run.
            same = seen.setdefault(sha(path), f"candidate {i}")
            cand = (_score(ctx, path, screen, val_tasks, parent_rewards, passed) if same == f"candidate {i}"
                    else _entry(path, reason=f"identical to {same}: not scored"))
        cand.update(rationale=slot["rationale"], form=slot["form"])
        row["candidates"].append(cand)
        say(f"candidate {i}: {json.dumps(cand)}")

    best = None
    for cand, slot in zip(row["candidates"], proposed):
        if cand["score"] is None:
            continue
        if cand["errors"]:
            cand["reason"] = f"{cand['errors']} errored episodes"
        elif cand["score"] <= row["parent_score"]:
            cand["reason"] = f"scored {cand['score']:.0%}, not above the parent's {row['parent_score']:.0%}"
        elif best is None or cand["score"] > best[0]["score"]:
            best = (cand, slot["path"])
    for cand in row["candidates"]:
        if cand["reason"] is None and not (best and cand is best[0]):
            cand["reason"] = "beat the parent on draw A, but another candidate scored higher"
    if best and _decide(ctx, *best, parent, val_tasks + second_tasks):
        row["adopted"], ctx["adopt"] = f"{int(parent.stem) + 1:03d}.py", best[1]  # evolve() writes the row, then the file


def evolve(family="generated", train=40, val=45, candidates=3, budget=5.0, home=None, llm=None,
           run=run_episode, say=print, candidate_files=None) -> dict:
    """One generation. Returns the lineage row (appended to lineage.jsonl unless a budget stop).
    With candidate_files, those files are the candidates and no improver is called."""
    files = [Path(f).resolve() for f in candidate_files or []]
    if files:
        candidates = len(files)
    if val < TIMING_TASKS or train < 1 or candidates < 1:
        raise ValueError(f"need --val >= {TIMING_TASKS}, --train >= 1 and --candidates >= 1")
    # NaN makes every `spend > budget` false and a negative one stops after paying for the
    # timing run. inf is refused too: the budget is the only bound on an unattended loop, so
    # "no limit" has to be written as a number someone chose.
    if not (math.isfinite(budget) and budget >= 0):
        raise ValueError(f"--budget must be a finite number of dollars, 0 or more; got {budget}")
    for file in files:
        if not file.is_file():
            raise ValueError(f"--candidate {file} is not a file")
    if not files:
        llm = llm or partial(_llm.complete, model=env("RSI_IMPROVER_MODEL"))  # fails here, before any spend
    home_dir = Path(home).resolve() if home else _home()
    with lock(home_dir / "harnesses" / "evolve.lock", "rsi evolve"):
        generation = len(_reconcile(home_dir))
        parent = broker.current_harness(home_dir)
        if int(parent.stem) >= 999:
            raise ValueError(f"{parent} is the last harness the loop can number: names are three digits")
        lineage = parent.with_name(LINEAGE)
        ctx = context(home_dir, f"evolve/gen-{generation}", family, budget, run)
        row = {"generation": generation, "family": family, "parent": parent.name, "parent_sha": sha(parent),
               "candidates": [], "val_seeds": None, "confirm_seeds": None, "parent_score": None,
               "null_score": None, "n_val": val, "adopted": None, "cost": 0.0, "ts": None}
        record = True
        try:
            _generation(ctx, row, parent, train, candidates, llm, files, say)
        except Stop as stop:
            row["stopped"], record = str(stop), stop.record
        row["cost"], row["ts"] = round(spent(ctx), 6), datetime.now(timezone.utc).isoformat()
        if record:
            # ponytail: no fsync. A killed process is covered (row first, then the file, and
            # _reconcile finishes the copy); a power cut can still lose both. Ceiling: fsync
            # the row and the folder if this ever runs somewhere that loses power.
            with open(lineage, "a") as f:
                f.write(json.dumps(row) + "\n")
            if row["adopted"]:
                _install(ctx["adopt"], lineage.with_name(row["adopted"]))

    say(f"\nGeneration {generation} ({family}), parent {parent.name}: spent ${row['cost']:.2f} of ${budget:.2f}.")
    if ctx["regraded"]:
        say(_regraded(ctx))
    name = lambda c: c["sha"][:8] if c["sha"] else "(no file)"
    if row["parent_score"] is not None:
        say(f"Draw A, {val} val tasks: guessing {row['null_score']:.0%}, parent {row['parent_score']:.0%}"
            + "".join(f", candidate {name(c)} {c['score']:.0%} ({c['wins']} wins, {c['losses']} losses)"
                      for c in row["candidates"] if c["score"] is not None) + ".")
    saved = sum(sum(c["skipped"].values()) for c in row["candidates"] if c["skipped"])
    if saved:
        say(f"{saved} episodes were not run: their candidate could no longer pass the stage it was in.")
    for c in row["candidates"]:
        if c["decision"]:
            say("Decision for candidate {sha8}, fresh runs of both harnesses on {n} tasks paired by task: {score:.0%} against "
                "the parent's {parent_score:.0%}; won {wins}, lost {losses}, net {net:+d} (at least {margin} needed), "
                "{errors} errored episodes, exact paired p = {p_value:.3f} (reported, not the gate)."
                .format(sha8=name(c), **c["decision"]))
        if c["reason"] != "adopted":
            say(f"Candidate {name(c)} not adopted: {c['reason']}.")
    if "stopped" in row:
        say(f"Stopped: {row['stopped']}." + ("" if record else
                                              " Nothing was recorded; the same command with a larger --budget resumes from disk."))
    elif row["adopted"]:
        say(f"Adopted {row['adopted']}: best on draw A, then net wins at or above the margin on fresh paired runs of "
            "draw A and draw B, with no errored episode.")
    else:
        say("Nothing adopted: " + ("no candidate file to score." if not any(c["sha"] for c in row["candidates"])
                                   else "no candidate passed the replay, the screen, draw A and the decision."))
    return row


def _regraded(ctx: dict) -> str:
    return (f"{len(ctx['regraded'])} finished episodes hold a reward in episode.json that differs from what the current "
            "grader gives their stored answer; the current grader's reward was used.")


def score_test(family="generated", n=30, home=None, run=run_episode, say=print, harness=None) -> dict:
    """Score a harness (default: the current one) on the Test split. Episodes are kept under
    runs/test/<harness_sha12>/, so repeating the command costs nothing new; the per-task
    rewards are written next to them."""
    if n < 1:  # 0 would "score" nothing; a negative n would slice the dabench pool from its end
        raise ValueError("--n must be at least 1")
    home_dir = Path(home).resolve() if home else _home()
    harness = Path(harness).resolve() if harness else broker.current_harness(home_dir)
    ctx = context(home_dir, "test", family, float("inf"), run)
    seeds, todo = draw(ctx, "test", n)
    with lock(ctx["root"] / sha(harness)[:12] / "lock", f"a test of {harness.name}"):
        episodes = batch(ctx, harness, todo)
    out = {"family": family, "harness": harness.name, "harness_sha": sha(harness), "n": len(todo),
           "score": mean(e["reward"] for e in episodes), "null_score": null(todo),
           "errors": sum(e["stop_reason"] in ERRORED for e in episodes), "cost": round(spent(ctx), 6), "seeds": seeds,
           "rewards": {t["id"]: e["reward"] for t, e in zip(todo, episodes)}}
    out["results"] = str(ctx["root"] / out["harness_sha"][:12] / f"results-{family}-n{out['n']}.json")
    Path(out["results"]).write_text(json.dumps(out, indent=1))
    say(f"{harness.name} on {out['n']} {family} test tasks: {out['score']:.0%} (guessing scores {out['null_score']:.0%}), "
        f"{out['errors']} errored episodes, ${out['cost']:.2f}. Per-task rewards: {out['results']}")
    if ctx["regraded"]:
        say(_regraded(ctx))
    return out


def compare(family, n, a, b, home=None, run=run_episode, say=print) -> dict:
    """Score harness files a and b on the same Test tasks and report the paired difference."""
    outs = [score_test(family, n, home=home, run=run, say=say, harness=path) for path in (a, b)]
    rewards = [list(out["rewards"].values()) for out in outs]
    if list(outs[0]["rewards"]) != list(outs[1]["rewards"]):
        raise RuntimeError("the two harnesses were scored on different tasks")  # cannot happen: same family, same n
    out = {"family": family, "n": outs[0]["n"], "a": outs[0], "b": outs[1],
           "ci95": [evals.interval(r) for r in rewards], **evals.mcnemar(*rewards)}
    for label, scored, (low, high) in zip("AB", outs, out["ci95"]):
        say(f"{label} {scored['harness']} ({scored['harness_sha'][:8]}): {scored['score']:.1%} on {out['n']} {family} test tasks, "
            f"95% bootstrap interval {low:.1%}-{high:.1%}")
    say(f"Paired on the same tasks: A right where B was wrong on {out['wins']}, the reverse on {out['losses']}, "
        f"same on {out['ties']}; exact two-sided {out.get('test', 'McNemar')} p = {out['p_value']:.3f}")
    return out
