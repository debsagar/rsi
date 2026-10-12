"""The end-to-end test: one scenario through the real system, finishing with an artifact a person can read.

Real: the FastAPI app over HTTP (TestClient), a real multipart upload, the real
broker, the real harness container (the seed harness and a rewritten one), the
real persistent Python container running real code on the uploaded CSV, real
clipping, the event log, the code graders, the eval runner with its report and
resume, and the evolve loop with its lock, edit blocks, free replay (in the
harness container), screen, two Val draws, adoption decision and lineage row.

Scripted, because they cost money: the model (`Agent`), the improver
(`improver`) and Jev (`jev`). Nothing here reads `.env`, and a real call to a
model provider or to any URL trips a wire and fails the test.

All state lives in an RSI_HOME under pytest's tmp_path. The one thing written to
the real repo is the artifact, `runs/e2e/<UTC timestamp>/` (git-ignored):

    report.md     what was run, each step pass/fail, the key numbers
    report.json   the same facts for a program, in three parts:
                    facts       must be identical on every run of the same code
                    provenance  what "the same code" means: file hashes, grader version, image id
                    volatile    what differs on every run: times, durations, session and run ids, the
                                tmp home, how far the interrupted eval got, other people's containers
    home/         a copy of the whole RSI_HOME: every events.jsonl, episode.json, lineage.jsonl, harness file
    eval-report.txt, evolve-gen0.log, evolve-gen1.log   what the commands printed

Repeatability is asserted: when an earlier passing artifact has the same
provenance, this run's `facts` must equal that run's, key for key.

    uv run pytest tests/test_e2e.py -q        # about a minute; prints the artifact path
"""
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rsi import ROOT, api, broker, evals, evolve, grading, tasks, tools

EVAL_N, TRAIN, VAL = 4, 4, 6
# The eval's first two episodes (its timing run) take three model calls each, so the ninth call
# falls in the main batch: at least two episodes are finished when the interrupt arrives.
INTERRUPT_AT_CALL = 9
ARTIFACTS = ROOT / "runs" / "e2e"
STEPS = ["1 ask a question about an uploaded CSV over HTTP",
         "2 follow-up in the same session; a second session sees neither the variable nor the file",
         "3 reload the session over HTTP",
         "4 interrupt an eval part way, resume it, recount it",
         "5 evolve: a better candidate is adopted and answers the next chat message",
         "6 evolve: a worse candidate is not adopted",
         "7 nothing left behind: no container, no reference shown to the improver"]


def _docker_problem() -> str | None:
    if not shutil.which("docker"):
        return "docker is not installed: the e2e test runs the harness and the python tool in real containers"
    if subprocess.run(["docker", "image", "inspect", tools.SANDBOX_IMAGE], capture_output=True).returncode != 0:
        return f"docker is not running or the {tools.SANDBOX_IMAGE} image is missing (docker build -t {tools.SANDBOX_IMAGE} sandbox/)"
    return None


pytestmark = pytest.mark.skipif(bool(_docker_problem()), reason=_docker_problem() or "")


# --- the scripted model ---------------------------------------------------------------
# It plays a competent agent with one fixable habit and one blind spot. It never sees a file:
# every value it reports is the last line of what the sandbox printed for code it asked to run.
# - Habit: it computes Cohen's d as A minus B whatever the question defines, unless the harness's
#   system prompt contains MARKER. That is what makes one harness really score higher than another.
# - Blind spot: it never joins sites.csv, so it gets every filtered_agg task wrong under any
#   harness. That keeps every harness under 90% on Val, where the evolve loop has room to measure.
# With no tools on offer (a harness that cut it off) it says it has no result.
# Task kinds go round by seed, so any four consecutive seeds hold one task of each kind: by
# construction the seed harness gets 2 of every 4 right and a harness with MARKER 3 of every 4.
# Over the 12 tasks of draw A and draw B that is 3 tasks won and none lost: exactly the margin.

MARKER = "keep the sign convention the question defines"
CHAT_STAT = re.compile(r"the (mean|median) of the `(\w+)` column in ([\w.-]+\.csv)")
FOLLOW_UP = re.compile(r"^And the (mean|median)\?")
LOAD = ("import numpy as np, pandas as pd\nruns = pd.read_csv('/data/runs.csv')\n"
        "if 'unit' in runs.columns:\n    runs['duration'] = np.where(runs['unit'] == 'ms', runs['duration'] / 1000, runs['duration'])\n"
        "if 'status' in runs.columns:\n    runs = runs[runs['status'] != 'void'].copy()\nprint(len(runs), 'rows in use')")
ARMS = "a = runs.loc[runs['arm'] == 'A', 'duration'].to_numpy()\nb = runs.loc[runs['arm'] == 'B', 'duration'].to_numpy()\n"
COHEN = ARMS + ("pooled = (((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) ** 0.5\n"
                "print(round(float(({first}.mean() - {second}.mean()) / pooled), 3))")
PERMUTE = ARMS + ("both = np.concatenate([a, b])\nshuffled = np.random.default_rng(0).permuted(np.tile(both, (10000, 1)), axis=1)\n"
                  "diffs = shuffled[:, :len(a)].mean(axis=1) - shuffled[:, len(a):].mean(axis=1)\n"
                  "p = float((np.abs(diffs) >= abs(a.mean() - b.mean()) - 1e-12).mean())\n"
                  "print('no_difference' if p >= 0.05 else 'A_higher' if a.mean() > b.mean() else 'B_higher')")


def plan(question: str, earlier: list, system: str) -> list | None:
    """The tool calls the agent makes for a question, in order; None when it cannot tell what is asked."""
    if "Do not load anything" in question:
        return [("python", {"code": "import os\nprint(sorted(os.listdir('/data')))"}), ("python", {"code": "print(len(table))"})]
    stat, follow = CHAT_STAT.search(question), FOLLOW_UP.search(question)
    if follow:  # "And the median?": the column and the file are only in the conversation so far
        stat = next((found for q in reversed(earlier) if (found := CHAT_STAT.search(q))), None)
        if stat is None:
            return None
    if stat:
        name, (_, column, file) = (follow or stat).group(1), stat.groups()
        return [("read_file", {"name": file}),
                ("python", {"code": f"import pandas as pd\ntable = pd.read_csv('/data/{file}')\nprint(table.to_string())"}),
                ("python", {"code": f"print(round(float(table['{column}'].{name}()), 4))"})]  # `table` is from the call before
    if "runs.csv" not in question:
        return None
    if "Cohen's d" in question:
        first, second = "ba" if MARKER in system else "ab"
        compute = COHEN.format(first=first, second=second)
    elif "correlation" in question:
        method = "spearman" if "Spearman" in question else "pearson"
        compute = f"print(round(float(runs['duration'].corr(runs['load'], method='{method}')), 3))"
    elif "permutation test" in question:
        compute = PERMUTE
    else:  # filtered_agg, without the join it needs
        op, floor = re.search(r"What is the (mean|median|sum) of duration", question).group(1), re.search(r"at least (\d+)", question).group(1)
        compute = f"print(round(float(runs.loc[runs['load'] >= {floor}, 'duration'].{op}()), 3))"
    return [("python", {"code": LOAD}), ("python", {"code": compute})]


class Agent:
    """Same signature as rsi.llm.complete. Stateless about the conversation: what it does next is
    read from the messages, so parallel episodes cannot disturb each other."""

    def __init__(self):
        self.calls, self.interrupt_at, self._lock = 0, None, threading.Lock()

    def __call__(self, messages, tools=None, model=None, on_delta=None):
        with self._lock:
            self.calls += 1
            if self.calls == self.interrupt_at:
                raise KeyboardInterrupt("the e2e scenario interrupts the eval here")
        at = max(i for i, m in enumerate(messages) if m["role"] == "user" and "\n\nAttached files:" in m["content"])
        question = messages[at]["content"].split("\n\nAttached files:")[0]
        earlier = [m["content"] for m in messages[:at] if m["role"] == "user"]
        results = [m["content"] for m in messages[at + 1:] if m["role"] == "tool"]
        steps = plan(question, earlier, messages[0]["content"])
        if steps is None:
            message = self._say("I cannot tell what this refers to.\nANSWER: unknown", on_delta)
        elif len(results) < len(steps) and not tools:
            message = self._say("I was not able to run the computation, so I have no result.\nANSWER: unknown", on_delta)
        elif len(results) < len(steps):
            name, args = steps[len(results)]
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{len(results) + 1}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        else:
            label, _, body = results[-1].partition("\n")  # "[event N]" is how the harness labels a result
            cite, value = label.replace("event ", ""), (body.strip().splitlines() or [""])[-1].strip()
            failed = "Traceback" in body or body.startswith("ERROR")
            message = self._say(f"The sandbox could not compute this: {value} {cite}.\nANSWER: unknown" if failed
                                else f"The sandbox computed {value} {cite}.\nANSWER: {value}", on_delta)
        tokens_in = sum(len(str(m.get("content") or "")) for m in messages) // 4
        tokens_out = len(json.dumps(message)) // 4
        return {"message": message, "usage": {"in": tokens_in, "out": tokens_out},
                "cost": round((tokens_in + 4 * tokens_out) * 1e-6, 6)}

    @staticmethod
    def _say(text, on_delta):
        if on_delta:
            on_delta(text)
        return {"role": "assistant", "content": text}


def jev(state, questions, timeout=10):
    """Fixed Jev: every answer is supported, the agent is never stuck or on a wrong hypothesis."""
    return {name: {"noul": 0.9 if name == "supported" else 0.1} for name in questions}


def candidates() -> tuple[str, str]:
    """(better, worse) harness files, both made from the seed. Better adds MARKER to the system prompt.
    Worse keeps that and gives the model one step with tools before its last step, which is not
    enough to load and compute, so it answers every task with "unknown"."""
    seed = broker.SEED_HARNESS.read_text()
    assert "SYSTEM = " in seed and "MAX_STEPS = " in seed, "the seed harness no longer has the names the e2e candidates edit"
    better = seed + f'\nSYSTEM += " When a question defines a direction (B minus A), {MARKER}."\n'
    return better, better + "\nMAX_STEPS = 2\n"


def improver(messages, tools=None, model=None, on_delta=None):
    """Fixed improver: shown the seed it proposes the better file, shown the better file it proposes the worse one.
    Each as the reply the prompt asks for: a rationale and one SEARCH/REPLACE block that adds lines after the last line."""
    better, worse = candidates()
    current = messages[0]["content"].split("## Failed runs")[0]
    old, new, why = ((better, worse, "Runs take too many steps; allow two.") if MARKER in current
                     else (broker.SEED_HARNESS.read_text(), better, "Wrong effect sizes had the right size and the opposite sign; "
                                                                    "tell the model to keep the question's sign convention."))
    last = old.splitlines()[-1]
    reply = f"## CANDIDATE 1\nRATIONALE: {why}\n\n<<<<<<< SEARCH\n{last}\n=======\n{last}\n{new[len(old):]}>>>>>>> REPLACE\n"
    return {"message": {"role": "assistant", "content": reply}, "usage": {"in": 0, "out": 0}, "cost": 0.01}


# --- small helpers ---------------------------------------------------------------------

def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measurements() -> bytes:
    """The CSV the user uploads: 240 rows, so printing it whole is over the broker's 4,000-character clip."""
    values = np.random.default_rng(20261012).normal(50, 8, 240)
    return "".join(["sample,batch,value\n"] + [f"S{i:03d},{'ABC'[i % 3]},{v:.3f}\n" for i, v in enumerate(values)]).encode()


def answer_value(text: str) -> str:
    return re.findall(r"^ANSWER: (.*)$", text or "", re.M)[-1].strip()


def ask(client, home: Path, sid: str, question: str) -> dict:
    """One message through the API, read to `done`. -> the answered turn; fails on anything a user would see as an error."""
    response = client.post(f"/api/sessions/{sid}/messages", json={"question": question})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    stream = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        stream.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    names = [name for name, _ in stream]
    assert names[-1] == "done" and "error" not in names, [d for n, d in stream if n == "error"]
    episode = stream[-1][1]["episode"]
    events = [data for name, data in stream if name == "event"]
    run_dir = home / "runs" / "sessions" / sid / episode["run_id"]
    assert events == [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]  # stream = the log on disk
    assert episode["stop_reason"] == "submitted" and episode["session_id"] == sid
    assert episode["gate"] is False and not any(e.get("source") == "gate" for e in events)  # chat: the gate's default is off
    # What a user is shown as evidence must exist: every claim cites tool results of this run.
    results = {e["id"]: e for e in events if e["kind"] == "tool_result"}
    answer = next(e for e in events if e["kind"] == "answer")
    assert answer["text"] == episode["answer"] and answer["claims"]
    assert all(claim["evidence"] and set(claim["evidence"]) <= set(results) for claim in answer["claims"])
    assert episode["calls"] > 0 and episode["tool_calls"] > 0 and episode["cost"] > 0 and episode["usage"]["in"] > 0
    return {"episode": episode, "events": events, "results": list(results.values()), "run_dir": run_dir,
            "streamed_text": "".join(d["text"] for n, d in stream if n == "delta"),
            "facts": {"question": question, "answer": episode["answer"], "claims": answer["claims"],
                      "harness_sha256": episode["harness_sha256"], "stop_reason": episode["stop_reason"],
                      "model_calls": episode["calls"], "tool_calls": episode["tool_calls"], "cost": episode["cost"],
                      "tokens": episode["usage"], "events": len(events),
                      "jev_consulted": sum(e["kind"] == "judge" for e in events)}}


def my_grade(answer, reference) -> int:
    """The scenario's own reading of a generated-task answer, written without rsi.grading: the last line
    that starts with ANSWER:, equal to a label or within 1% of a number."""
    lines = [line for line in (answer or "").splitlines() if line.startswith("ANSWER:")]
    said = lines[-1].removeprefix("ANSWER:").strip() if lines else ""
    if isinstance(reference, str):
        return int(said == reference)
    try:
        return int(abs(float(said) - reference) <= 0.01 * abs(reference))
    except ValueError:
        return 0


def our_containers(tmp_path: Path) -> tuple[list, int]:
    """(rsi- containers that mount something under tmp_path, rsi- containers that do not). Another
    `rsi` may be running on this machine with its own containers; those are not this test's to judge."""
    names = subprocess.run(["docker", "ps", "-a", "--filter", "name=rsi-", "--format", "{{.Names}}"],
                           capture_output=True, text=True, check=True).stdout.split()
    ours = []
    for name in names:
        mounts = subprocess.run(["docker", "inspect", "-f", "{{range .Mounts}}{{.Source}}\n{{end}}", name],
                                capture_output=True, text=True).stdout
        if str(tmp_path) in mounts:
            ours.append(name)
    return ours, len(names) - len(ours)


class Record:
    """What the artifact is written from. `facts` must not hold anything that differs between runs."""

    def __init__(self):
        self.steps = {name: {"result": "not run"} for name in STEPS}
        self.facts, self.volatile, self.seconds, self.texts = {}, {}, {}, {}

    @contextmanager
    def step(self, number: int):
        name = next(s for s in STEPS if s.startswith(f"{number} "))
        started = time.monotonic()
        try:
            yield
            self.steps[name]["result"] = "pass"
        except BaseException as e:
            self.steps[name] = {"result": "FAIL", "error": f"{type(e).__name__}: {e}"[:3000]}
            raise
        finally:
            self.seconds[name] = round(time.monotonic() - started, 1)


# --- the scenario ------------------------------------------------------------------------

def scenario(tmp_path: Path, home: Path, record: Record, agent: Agent) -> None:
    facts = record.facts
    csv = measurements()
    table = pd.read_csv(io.BytesIO(csv))
    seed_sha = sha(broker.SEED_HARNESS)
    client = TestClient(api.app)

    with record.step(1):
        sid = client.post("/api/sessions").json()["id"]
        uploaded = client.post(f"/api/sessions/{sid}/files", files={"file": ("measurements.csv", csv, "text/csv")})
        assert uploaded.status_code == 200 and uploaded.json()["size"] == len(csv)
        first = ask(client, home, sid, "What is the mean of the `value` column in measurements.csv?")
        mean = round(float(table["value"].mean()), 4)  # computed here, from the same bytes, by this process
        assert float(answer_value(first["episode"]["answer"])) == mean
        assert first["streamed_text"].endswith(first["episode"]["answer"])  # the answer text arrived as deltas too
        assert first["episode"]["harness_sha256"] == seed_sha == sha(home / "harnesses" / "000.py")
        long = [r for r in first["results"] if r["chars_full"] > broker.CLIP_CHARS]
        assert long and all(r["clipped"] == (r["chars_full"] > broker.CLIP_CHARS) for r in first["results"])
        for result in long:
            full = (first["run_dir"] / result["file"]).read_text()
            assert len(result["result"]) == result["chars_shown"] <= broker.CLIP_CHARS and len(full) == result["chars_full"]
            assert result["file"] in result["result"] and all(sample in full for sample in table["sample"])  # every row is on disk
        facts["chat"] = {"csv_sha256": hashlib.sha256(csv).hexdigest(), "csv_rows": len(table),
                         "mean": {**first["facts"], "value_in_answer": answer_value(first["episode"]["answer"]),
                                  "independent_check": mean,
                                  "clipped": [{k: r[k] for k in ("chars_full", "chars_shown", "file")} for r in long]}}
        record.volatile["sessions"] = {"main": sid}

    with record.step(2):
        second = ask(client, home, sid, "And the median?")
        median = round(float(table["value"].median()), 4)
        assert float(answer_value(second["episode"]["answer"])) == median and median != mean
        defined, used = [e["args"] for e in second["events"] if e["kind"] == "tool_call" and e["tool"] == "python"]
        assert "table = " in defined and "table = " not in used and "table[" in used  # the later call only reads the variable
        other = client.post("/api/sessions").json()["id"]
        probe = ask(client, home, other, "What is len(table)? Do not load anything.")
        listing, attempt = probe["results"]
        assert listing["result"].strip() == "[]" and "NameError: name 'table' is not defined" in attempt["result"]
        assert answer_value(probe["episode"]["answer"]) == "unknown"
        facts["chat"]["median"] = {**second["facts"], "value_in_answer": answer_value(second["episode"]["answer"]),
                                   "independent_check": median}
        facts["chat"]["other_session"] = {**probe["facts"], "files_seen": listing["result"].strip(),
                                          "python_said": attempt["result"].strip().splitlines()[-1]}
        record.volatile["sessions"]["other"] = other

    with record.step(3):
        reloaded = TestClient(api.app).get(f"/api/sessions/{sid}").json()  # a new client: nothing is kept in memory
        assert [f["name"] for f in reloaded["files"]] == ["measurements.csv"] and reloaded["files"][0]["size"] == len(csv)
        assert len(reloaded["runs"]) == 2
        for run, turn in zip(reloaded["runs"], (first, second)):
            assert run["run_id"] == turn["episode"]["run_id"] and run["episode"] == turn["episode"]
            assert run["events"] == turn["events"] and run["answer"] == turn["episode"]["answer"] and not run["log_unreadable"]
        response = client.get("/api/sessions", headers={"Origin": api.WEB_ORIGINS[0]})
        assert response.headers["access-control-allow-origin"] == api.WEB_ORIGINS[0]  # or the browser blocks the web app
        listed = response.json()
        assert {s["id"] for s in listed} == {sid, other}
        facts["reload"] = {"runs": len(reloaded["runs"]), "title": reloaded["title"], "sessions_listed": len(listed)}

    run = partial(broker.run_episode, llm=agent, jev=jev)  # in_docker stays True: real containers

    with record.step(4):
        conditions, log = ["no_tools", "full", "no_gate"], []
        agent.interrupt_at = agent.calls + INTERRUPT_AT_CALL
        with pytest.raises(KeyboardInterrupt):
            evals.run_eval("generated", EVAL_N, conditions, home=home, run=run, say=log.append)
        agent.interrupt_at = None
        (eval_dir,) = (home / "runs" / "evals").iterdir()
        episodes = {p: json.loads(p.read_text()) for p in eval_dir.glob("*/*/episode.json")}
        finished = {p: (sha(p), sha(p.with_name("events.jsonl"))) for p, e in episodes.items() if e["reward"] is not None}
        assert 2 <= len(finished) < len(conditions) * EVAL_N, "the interrupt did not land part way"
        calls_before = agent.calls
        summary = evals.run_eval("generated", EVAL_N, conditions, home=home, run=run, say=log.append)
        assert "stopped" not in summary
        assert all((sha(p), sha(p.with_name("events.jsonl"))) == digest for p, digest in finished.items())  # not one byte rewritten
        episodes = {(p.parts[-3], p.parts[-2]): json.loads(p.read_text()) for p in eval_dir.glob("*/*/episode.json")}
        resumed = [e for (c, t), e in episodes.items() if eval_dir / c / t / "episode.json" not in finished]
        assert agent.calls - calls_before == sum(e["calls"] for e in resumed)  # the model was asked only for what was missing
        printed = io.StringIO()
        with redirect_stdout(printed):
            reported = evals.main(["--report", str(eval_dir)])  # what `rsi eval --report` runs
        assert reported["conditions"] == summary["conditions"] and reported["n_tasks"] == EVAL_N
        batch = [json.loads(line) for line in (eval_dir / "tasks.jsonl").read_text().splitlines()]
        recount = {c: {"n": len(batch),
                       "accuracy": round(sum(my_grade(episodes[(c, t["id"])]["answer"], t["reference"]) for t in batch) / len(batch), 4),
                       "cost_total": round(sum(episodes[(c, t["id"])]["cost"] for t in batch), 6)} for c in conditions}
        recount["null"] = {"n": len(batch), "accuracy": round(sum(t["reference"] == "no_difference" for t in batch) / len(batch), 4),
                           "cost_total": 0.0}
        assert recount == {c: {k: got[k] for k in ("n", "accuracy", "cost_total")} for c, got in reported["conditions"].items()}
        for c in conditions:  # the table a reader sees holds the recounted number
            assert re.search(rf"^{c}\s+{EVAL_N}\s+{recount[c]['accuracy']:.1%}", printed.getvalue(), re.M), printed.getvalue()
        for (c, t), e in episodes.items():  # a condition is what its name says: the gate asks Jev, or nobody does
            asked = sum(json.loads(line).get("source") == "gate" for line in (eval_dir / c / t / "events.jsonl").read_text().splitlines())
            assert e["harness_sha256"] == seed_sha and e["gate"] == (c != "no_gate") == bool(asked), (c, t)
            assert (e["tool_calls"] == 0) == (c == "no_tools")
        facts["eval"] = {"eval_id": eval_dir.name, "harness_sha256": seed_sha, "tasks": EVAL_N, "interrupted_at_model_call": INTERRUPT_AT_CALL,
                         "finished_episodes_rerun": 0, "report": reported["conditions"], "independent_recount": recount,
                         "gate": reported["gate"],
                         "per_task": [{"task": t["id"], "reference": t["reference"],
                                       **{c: answer_value(episodes[(c, t["id"])]["answer"]) for c in conditions}} for t in batch]}
        record.volatile["eval"] = {"episodes_finished_before_the_interrupt": len(finished), "episodes_run_after": len(resumed)}
        record.texts["eval-report.txt"] = printed.getvalue()
        record.texts["eval-run.log"] = "\n".join(map(str, log)) + "\n"

    better, worse = candidates()
    lineage = home / "harnesses" / "lineage.jsonl"

    def generation(name: str) -> dict:
        log = []
        row = evolve.evolve(family="generated", train=TRAIN, val=VAL, candidates=1, budget=5.0, home=home,
                            llm=improver, run=run, say=log.append)
        record.texts[f"evolve-{name}.log"] = "\n".join(map(str, log)) + "\n"
        assert json.loads(lineage.read_text().splitlines()[-1]) == row  # the row on disk is the row returned
        facts.setdefault("evolve", {})[name] = {k: v for k, v in row.items() if k != "ts"}
        return row

    with record.step(5):
        row = generation("gen0")
        (cand,) = row["candidates"]
        adopted = home / "harnesses" / "001.py"
        assert row["parent"] == "000.py" and row["adopted"] == "001.py" and cand["reason"] == "adopted", cand
        assert adopted.read_text() == better and cand["sha"] == sha(adopted) != seed_sha
        assert cand["form"] == "edit" and cand["screen"]["fixed"] >= 1 and cand["screen"]["broken"] == 0
        # The free replay ran the candidate in the real harness container on the parent's two right train answers.
        assert cand["replay"] == {"n": 2, "broken": 0, "no_submit": 0, "examples": []}
        replayed = sorted((home / "runs" / "evolve" / "gen-0" / f"replay-{cand['sha'][:12]}").glob("*/episode.json"))
        assert len(replayed) == 2 and all(json.loads(p.read_text())["cost"] == 0 for p in replayed)
        decision = cand["decision"]
        assert cand["score"] > row["parent_score"] and not cand["errors"] and not decision["errors"]
        assert (decision["n"], decision["margin"]) == (2 * VAL, 3) and decision["net"] >= decision["margin"]
        third = ask(client, home, sid, "What is the mean of the `value` column in measurements.csv?")
        assert third["episode"]["harness_sha256"] == sha(adopted) and float(answer_value(third["episode"]["answer"])) == mean
        served = client.get("/api/harnesses").json()
        assert served["current"] == "001.py" and served["lineage"] == [row]
        facts["after_adoption"] = {**third["facts"], "current_harness": served["current"]}

    with record.step(6):
        with open(home / "harnesses" / "evolve.lock", "w") as held:  # as a second `rsi evolve` on this home would find it
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match="already running"):
                evolve.evolve(family="generated", train=TRAIN, val=VAL, candidates=1, home=home, llm=improver, run=run, say=lambda _: None)
        assert len(lineage.read_text().splitlines()) == 1  # the refused start recorded nothing
        row = generation("gen1")
        (cand,) = row["candidates"]
        assert row["parent"] == "001.py" and row["adopted"] is None and cand["sha"] == hashlib.sha256(worse.encode()).hexdigest()
        assert cand["reason"] and cand["reason"] != "adopted"
        assert sorted(p.name for p in (home / "harnesses").glob("[0-9]*.py")) == ["000.py", "001.py"]
        assert broker.current_harness(home).name == "001.py" and len(lineage.read_text().splitlines()) == 2
        facts["evolve"]["lock"] = "a second evolve on the same home was refused while the first held harnesses/evolve.lock"

    with record.step(7):
        deadline = time.monotonic() + 15  # `docker run --rm` removes a killed container a moment after the kill returns
        while (found := our_containers(tmp_path))[0] and time.monotonic() < deadline:
            time.sleep(0.5)
        assert not found[0], f"containers left behind: {found[0]}"
        record.volatile["rsi_containers_of_other_processes"] = found[1]
        # A numeric reference is searched for as a whole signed number token: the scripted agent's habit makes
        # a failed effect-size answer the reference with the opposite sign, which is a wrong answer, not the
        # reference. A label reference (A_higher, ...) cannot be searched for: every question lists all three
        # labels as its answer format.
        checked, shown = [], []
        for gen in (0, 1):
            folder = home / "runs" / "evolve" / f"gen-{gen}" / "candidates"
            files = [p for p in sorted(folder.iterdir()) if p.is_file()]
            assert "FINAL ANSWER:" in (folder / "prompt.md").read_text()  # the improver was shown failed runs, so the search means something
            drawn = [("train", seed) for seed in tasks.split_seeds("train", TRAIN, gen)] + [("val", seed) for seed in tasks.split_seeds("val", 2 * VAL, gen)]
            written = folder.parent / "files"
            assert sorted(p.name for p in written.iterdir()) == sorted(f"{split}-{seed}" for split, seed in drawn)  # no Test task was drawn
            for split, seed in drawn:
                fresh = tmp_path / "references" / str(seed)
                task = tasks.make_task(seed, fresh)
                # Generated again here, the files are byte for byte what the loop ran on: same seed, same task.
                assert all((written / f"{split}-{seed}" / name).read_bytes() == (fresh / name).read_bytes() for name in task["files"])
                reference = task["reference"]
                if not isinstance(reference, str):
                    token = re.compile(rf"(?<![\d.\-]){re.escape(str(reference))}(?!\.?\d)")
                    checked.append(f"gen {gen} seed {seed}")
                    shown += [f"{reference} in {p}" for p in files if token.search(p.read_text(errors="replace"))]
        assert not shown, shown
        assert not (home / "runs" / "test").exists()
        facts["cleanup"] = {"containers_left_by_this_run": 0, "numeric_references_searched_for": len(checked),
                            "found_in_candidates_folders": 0}


# --- the artifact -------------------------------------------------------------------------

def provenance(home: Path) -> dict:
    def docker(*args):
        return subprocess.run(["docker", *args], capture_output=True, text=True).stdout.strip()

    harnesses = {"rsi/harness.py (seed)": sha(broker.SEED_HARNESS)}
    harnesses.update({f"harnesses/{p.name}": sha(p) for p in sorted((home / "harnesses").glob("[0-9]*.py"))})
    harnesses.update({f"{p.parent.parent.name}/candidates/{p.name}": sha(p) for p in sorted(home.glob("runs/evolve/gen-*/candidates/*.py"))})
    return {"harness_sha256": harnesses, "grader_version": grading.GRADER_VERSION,
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted([*ROOT.glob("rsi/*.py"), Path(__file__).resolve()])},
            "python": sys.version.split()[0], "platform": platform.platform(),
            "sandbox_image": tools.SANDBOX_IMAGE, "sandbox_image_id": docker("image", "inspect", "-f", "{{.Id}}", tools.SANDBOX_IMAGE),
            "docker_server": docker("version", "-f", "{{.Server.Version}}"), "model": "scripted (tests/test_e2e.py Agent)",
            "improver": "scripted (tests/test_e2e.py improver)", "jev": "fixed function (tests/test_e2e.py jev)"}


def repeat_check(facts: dict, proof: dict) -> dict:
    """Compare with the latest earlier passing run of the same code. Same provenance must give the same facts."""
    for folder in sorted((p for p in ARTIFACTS.glob("*") if (p / "report.json").is_file()), reverse=True):
        earlier = json.loads((folder / "report.json").read_text())
        if earlier.get("result") == "pass" and earlier.get("provenance") == proof:
            now = json.loads(json.dumps(facts))
            differing = sorted(k for k in set(now) | set(earlier["facts"]) if now.get(k) != earlier["facts"].get(k))
            return {"compared_with": folder.name, "identical": not differing, "differing_parts": differing}
    return {"compared_with": None, "identical": None,
            "note": "no earlier passing run with the same provenance (first run, or a source file, the image or the grader changed)"}


def markdown(report: dict) -> str:
    facts, proof, vol = report["facts"], report["provenance"], report["volatile"]
    out = [f"# RSI end-to-end run {vol['started_utc']}: {report['result']}", "",
           "One scenario through the real API, broker, harness container, Python sandbox container, graders, eval "
           "runner and evolve loop. The model, the improver and Jev were scripted functions in tests/test_e2e.py; no "
           "paid service was called. Every number below is in report.json and comes from the files under home/.", "",
           "## Steps", "", "| step | result | seconds |", "|---|---|---|"]
    out += [f"| {name} | {step['result']} | {vol['seconds'].get(name, '-')} |" for name, step in report["steps"].items()]
    for name, step in report["steps"].items():
        if "error" in step:
            out += ["", f"FAILED {name}:", "", *("    " + line for line in step["error"].splitlines())]
    chat = facts.get("chat", {})
    if "mean" in chat:
        m = chat["mean"]
        out += ["", "## Chat", "",
                f"- Uploaded measurements.csv ({chat['csv_rows']} rows) and asked: \"{m['question']}\"",
                f"- Answer: `{m['answer'].splitlines()[-1]}`. The value came from the sandbox; the test computed "
                f"{m['independent_check']} from the same bytes with pandas. They agree.",
                f"- {m['model_calls']} model calls, {m['tool_calls']} tool calls, ${m['cost']:.6f}, {m['events']} events; "
                f"claims cite events {sorted({i for c in m['claims'] for i in c['evidence']})}, all tool results of that run.",
                *[f"- A tool result of {c['chars_full']} characters was sent as {c['chars_shown']} with a clip marker; "
                  f"the full text is {c['file']} in the run folder." for c in m["clipped"]]]
    if "median" in chat:
        m, o = chat["median"], chat["other_session"]
        out += [f"- Follow-up \"{m['question']}\": `{m['answer'].splitlines()[-1]}` (independent check {m['independent_check']}). "
                "The column and file came from the session history; the value from a second Python call that read a "
                "variable the first call had defined.",
                f"- A second session asked for that variable: it saw \"{o['files_seen']}\" and the sandbox said "
                f"`{o['python_said']}`."]
    if "reload" in facts:
        out += [f"- Reloading the session over HTTP returned the same {facts['reload']['runs']} runs, event for event."]
    if "eval" in facts:
        e = facts["eval"]
        out += ["", "## Eval, interrupted and resumed", "",
                f"`rsi eval` on {e['tasks']} generated tasks, harness {e['harness_sha256'][:12]}. The model call numbered "
                f"{e['interrupted_at_model_call']} raised KeyboardInterrupt; {vol['eval']['episodes_finished_before_the_interrupt']} "
                f"of {e['tasks'] * (len(e['report']) - 1)} episodes were finished by then (this count varies). The same command again ran the other "
                f"{vol['eval']['episodes_run_after']} and re-ran none: the finished episode files are byte-identical.", "",
                "| condition | n | accuracy (report) | accuracy (recount from episode files) | cost |", "|---|---|---|---|---|"]
        out += [f"| {c} | {got['n']} | {got['accuracy']:.1%} | {e['independent_recount'][c]['accuracy']:.1%} | ${got['cost_total']:.6f} |"
                for c, got in e["report"].items()]
        names = [c for c in e["report"] if c != "null"]
        out += ["", "Answers (the scripted agent never joins sites.csv and, under the seed harness, flips the sign of Cohen's d; "
                    "with no tools it has no result):", "", "| task | reference | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        out += [f"| {t['task']} | {t['reference']} | " + " | ".join(str(t[c]) for c in names) + " |" for t in e["per_task"]]
    for name in ("gen0", "gen1"):
        row = facts.get("evolve", {}).get(name)
        if row:
            out += ["", f"## Evolve generation {row['generation']}: " + (f"adopted {row['adopted']}" if row["adopted"] else "nothing adopted"), "",
                    f"Parent {row['parent']} ({row['parent_sha'][:12]}): {row['parent_score']:.0%} on {row['n_val']} val tasks "
                    f"(guessing: {row['null_score']:.0%}). Spend ${row['cost']:.6f}, the scripted improver's $0.01 included."]
            for c in row["candidates"]:
                out += [f"- Candidate {c['sha'][:12]}: \"{c['rationale']}\"",
                        f"  - screen on train: {c['screen']}" if c["screen"] else "  - no screen",
                        f"  - first val draw: {c['score']:.0%} ({c['wins']} wins, {c['losses']} losses)" if c["score"] is not None
                        else "  - never reached the first val draw",
                        *([f"  - free replay: {c['replay']['broken']} of {c['replay']['n']} right answers changed"] if c["replay"] else []),
                        *([f"  - decision, fresh runs of both on {c['decision']['n']} tasks: won {c['decision']['wins']}, lost "
                           f"{c['decision']['losses']}, net {c['decision']['net']} of the {c['decision']['margin']} needed"]
                          if c["decision"] else []),
                        f"  - outcome: {c['reason']}"]
    if "after_adoption" in facts:
        a = facts["after_adoption"]
        out += ["", f"After the adoption a new chat message was answered by harness {a['harness_sha256'][:12]} "
                    f"({a['current_harness']}): `{a['answer'].splitlines()[-1]}`."]
    if "lock" in facts.get("evolve", {}):
        out += [f"Lock: {facts['evolve']['lock']}."]
    if "cleanup" in facts:
        c = facts["cleanup"]
        out += ["", "## Left behind", "",
                f"- Containers of this run still present: {c['containers_left_by_this_run']} "
                f"(rsi- containers of other processes on this machine at that moment: {vol['rsi_containers_of_other_processes']}).",
                f"- {c['numeric_references_searched_for']} numeric reference answers were searched for in the improver's candidates "
                f"folders (prompt, reply, proposal, files) as signed number tokens: {c['found_in_candidates_folders']} found. Label "
                "references cannot be searched for, because every question lists all three labels."]
    rc = report["repeat_check"]
    out += ["", "## Repeatability", "",
            (f"`facts` in report.json equal those of the earlier run {rc['compared_with']} (same provenance): {rc['identical']}."
             if rc["compared_with"] else f"Not compared: {rc['note']}."),
            "Only `volatile` may differ between runs: times, durations, session and run ids, the tmp home, how many "
            "episodes the interrupt let finish, other processes' containers.", "",
            "## Provenance", "",
            *[f"- {name}: {digest[:16]}" for name, digest in proof["harness_sha256"].items()],
            f"- grader version {proof['grader_version']}; Python {proof['python']}; image {proof['sandbox_image']} "
            f"{proof['sandbox_image_id'][:19]}; Docker {proof['docker_server']}",
            f"- total {vol['total_seconds']} s", ""]
    return "\n".join(out)


def write_artifact(home: Path, record: Record, started: datetime, seconds: float, failure: str | None) -> tuple[Path, dict]:
    proof = provenance(home)
    passed = failure is None and all(step["result"] == "pass" for step in record.steps.values())
    check = repeat_check(record.facts, proof) if passed else {"compared_with": None, "identical": None, "note": "the scenario did not pass"}
    report = {"result": "pass" if passed and check["identical"] is not False else "FAIL", "steps": record.steps,
              "facts": record.facts, "provenance": proof, "repeat_check": check,
              "volatile": {"started_utc": f"{started:%Y-%m-%dT%H:%M:%SZ}", "total_seconds": round(seconds, 1), "seconds": record.seconds,
                           "home": str(home), **record.volatile}}
    if failure:
        report["failure"] = failure
    out = ARTIFACTS / f"{started:%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    shutil.copytree(home, out / "home")
    for name, text in record.texts.items():
        (out / name).write_text(text)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "report.md").write_text(markdown(report))
    return out, check


def test_end_to_end(tmp_path, monkeypatch, capsys):
    started, clock = datetime.now(timezone.utc), time.monotonic()
    home = tmp_path / "home"
    home.mkdir()
    agent, paid = Agent(), []

    def tripwire(what):
        def call(*args, **kwargs):
            paid.append(what)
            raise AssertionError(f"the e2e scenario tried a real {what}")
        return call

    import litellm
    monkeypatch.setattr(litellm, "completion", tripwire("model call"))
    monkeypatch.setattr(urllib.request, "urlopen", tripwire("HTTP request"))
    for name in [k for k in os.environ if k.endswith("_API_KEY") or k == "RSI_IMPROVER_MODEL"]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("RSI_HOME", str(home))
    monkeypatch.setenv("RSI_MODEL", "scripted-e2e")
    monkeypatch.setattr(api, "_LLM", agent)
    monkeypatch.setattr(api, "_JEV", jev)

    record, failure = Record(), None
    try:
        scenario(tmp_path, home, record, agent)
        assert not paid, paid
    except BaseException as e:
        failure = f"{type(e).__name__}: {e}"[:3000]
        raise
    finally:
        out, check = write_artifact(home, record, started, time.monotonic() - clock, failure)
        with capsys.disabled():
            print(f"\ne2e artifact: {out}")
    assert check["identical"] is not False, f"same code, different facts than {check['compared_with']}: {check['differing_parts']}"
