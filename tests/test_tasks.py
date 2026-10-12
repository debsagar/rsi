"""Tests for rsi/tasks.py: task validity, splits, and the generated grader.

The validity tests use `solve()`, an independent reference solver that sees
only what an agent sees (the question text and the written CSVs). If it cannot
reproduce task["reference"], the task is not answerable as posed.

The grader has one table, GRADER_ROWS: each row is a way the grader could
score a reply wrongly, written down before the rule that prevents it.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsi import tasks

N_VALID = 50  # seeds per kind for the validity tests


def solve(task, out_dir):
    """Answer a generated task from the question and CSVs alone (pandas/scipy)."""
    q = task["question"]
    df = pd.read_csv(out_dir / task["files"][0])
    if "status" in df.columns:
        assert "void" in q, "status column present but the question does not explain it"
        df = df[df["status"] != "void"]
    if "unit" in df.columns:
        assert "ms" in q, "unit column present but the question does not explain it"
        df = df.assign(duration=np.where(df["unit"] == "ms", df["duration"] / 1000, df["duration"]))
    if "arm" in df.columns:
        a = df.loc[df["arm"] == "A", "duration"].to_numpy()
        b = df.loc[df["arm"] == "B", "duration"].to_numpy()
        if "Cohen" in q:
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                             / (len(a) + len(b) - 2))
            return round(float((b.mean() - a.mean()) / pooled), 3)
        p = stats.permutation_test((a, b), lambda x, y: x.mean() - y.mean(),
                                   n_resamples=10_000, random_state=7).pvalue
        if p >= 0.05:
            return "no_difference"
        return "A_higher" if a.mean() > b.mean() else "B_higher"
    if "correlation" in q:
        fn = stats.spearmanr if "Spearman" in q else stats.pearsonr
        return round(float(fn(df["duration"], df["load"])[0]), 3)
    region = re.search(r"region is `(\w+)`", q).group(1)
    floor = int(re.search(r"`load` is at least (\d+)", q).group(1))
    op = re.search(r"What is the (mean|median|sum) of", q).group(1)
    sites = pd.read_csv(out_dir / task["files"][1])
    joined = df.merge(sites, on="site_id")
    sub = joined[(joined["region"] == region) & (joined["load"] >= floor)]["duration"]
    return round(float(getattr(sub, op)()), 3)


def arms(task, out_dir):
    df = pd.read_csv(out_dir / task["files"][0])
    if "status" in df.columns:
        df = df[df["status"] != "void"]
    if "unit" in df.columns:
        df = df.assign(duration=np.where(df["unit"] == "ms", df["duration"] / 1000, df["duration"]))
    return (df.loc[df["arm"] == "A", "duration"].to_numpy(),
            df.loc[df["arm"] == "B", "duration"].to_numpy())


NUM, NEG, BIG, LAB = 3.21, -0.412, 1234.5, "B_higher"
GRADER_ROWS = [  # (reference, reply, expected reward, the failure this row would expose)
    # --- junk that must not score ---
    (NUM, "", 0.0, "empty reply"),
    (NUM, None, 0.0, "no reply at all"),
    (NUM, "I could not work it out.", 0.0, "prose without a value"),
    (NUM, "Candidates 7.0 and 3.21 (see [3.21]).\nANSWER:", 0.0, "empty ANSWER line falls back to a number elsewhere"),
    (10.0, "I could not compute it. See reference [10].\nANSWER:", 0.0, "empty ANSWER line scores a citation number"),
    (NUM, "ANSWER:\n3.21", 0.0, "empty ANSWER line reads the next line"),
    (NUM, "The value is 3.21.\nANSWER: 9.9", 0.0, "a number in the reasoning beats the ANSWER line"),
    (NUM, "ANSWER: 3.21\nOn reflection:\nANSWER: 9.9", 0.0, "an earlier ANSWER line beats the last one"),
    (NUM, "The answer is 3.21 (p=0.04)", 0.0, "without an ANSWER line the last number is the answer, not any number"),
    (NUM, "ANSWER: -3.21", 0.0, "sign ignored"),
    (0.412, "ANSWER: \u20130.412", 0.0, "en dash minus read as positive"),
    (0.412, "ANSWER: \u22120.412", 0.0, "unicode minus read as positive"),
    (NUM, "ANSWER: nan", 0.0, "nan matches"),
    (NUM, "ANSWER: inf", 0.0, "inf matches"),
    (NUM, "ANSWER: 1e999", 0.0, "overflowing exponent matches"),
    (NUM, "ANSWER: 3.2421", 0.0, "just outside 1% relative"),
    (NUM, "ANSWER: 3.1778", 0.0, "just outside 1% relative, below"),
    (4.0, "ANSWER: 4e-5", 0.0, "scientific notation cut at the mantissa"),
    (4.0, "ANSWER: .4", 0.0, "leading-dot decimal read without its point"),
    (LAB, "Answer with exactly one of: A_higher, B_higher, no_difference", 0.0, "echoed option list scores"),
    (LAB, "B_higher", 0.0, "a label without the ANSWER line the question demands"),
    (LAB, "It looks like B_higher.\nANSWER: no_difference", 0.0, "label in the reasoning beats the ANSWER line"),
    (LAB, "ANSWER: not B_higher", 0.0, "negated label scores"),
    (LAB, "ANSWER: A_higher or B_higher", 0.0, "two labels on the line"),
    (LAB, "ANSWER: B_higher is wrong, it is A_higher", 0.0, "label followed by its own denial"),
    (LAB, "ANSWER: B_higher (p=0.001)", 0.0, "label plus explanation: refused, the question says nothing else on the line"),
    (LAB, "ANSWER: B is higher", 0.0, "paraphrase accepted as the label"),
    (LAB, "ANSWER: B_higher\nANSWER:", 0.0, "empty last ANSWER line falls back to an earlier one"),
    # --- right answers that must score ---
    (NUM, "ANSWER: 3.21", 1.0, "plain"),
    (NUM, "ANSWER: 3.2419", 1.0, "inside 1% relative"),
    (NUM, "ANSWER: 3.21 (rounded to 3 decimals)", 1.0, "trailing note read as the answer (3)"),
    (NUM, "I first got 9.9.\nANSWER: 3.21\nChecked against 2 tools.", 1.0, "text after the ANSWER line"),
    (NUM, "ANSWER: 9.9\nOn reflection:\nANSWER: 3.21", 1.0, "the last ANSWER line is the answer"),
    (NUM, "**ANSWER:** 3.21 [7]", 1.0, "bold marker, colon inside"),
    (NUM, "**ANSWER**: 3.21\n(rounded from 3.2145, n=50)", 1.0, "bold marker, colon outside"),
    (NUM, "> **Answer:** `3.21`", 1.0, "quote, bold, code, mixed case"),
    (NUM, "ANSWER: +3.21", 1.0, "leading plus"),
    (0.5, "ANSWER: .5", 1.0, "leading-dot decimal"),
    (0.5, "ANSWER: 5e-1", 1.0, "scientific notation"),
    (NEG, "ANSWER: \u22120.412", 1.0, "unicode minus"),
    (NEG, "ANSWER: \u20130.412", 1.0, "en dash minus"),
    (BIG, "ANSWER: 1,234.5 seconds", 1.0, "thousands separator"),
    (NUM, "First I got 9.9. Final answer: **3.21**.", 1.0, "no ANSWER line: the last number of the reply"),
    (3.0, "between 1-3", 1.0, "a range dash is not a minus"),
    (LAB, "ANSWER: B_higher", 1.0, "plain"),
    (LAB, "ANSWER: b_higher", 1.0, "letter case"),
    (LAB, "**ANSWER**: **B\\_higher**.", 1.0, "markdown escape, bold, final full stop"),
    (LAB, 'ANSWER: "B_higher"', 1.0, "quotes"),
    (LAB, "ANSWER: [B_higher]", 1.0, "brackets"),
    (LAB, "no_difference at first.\nANSWER: no_difference\nANSWER: B_higher", 1.0, "the last ANSWER line"),
]


def test_generated_grader_table():
    wrong = [(why, reply) for ref, reply, want, why in GRADER_ROWS
             if tasks.grade({"id": "x", "family": "generated", "reference": ref}, reply) != want]
    assert not wrong, wrong


def test_split_seeds():
    train, val, test = (tasks.split_seeds(s, 40) for s in ("train", "val", "test"))
    assert len(train) == len(val) == len(test) == 40
    assert all(isinstance(s, int) for s in train + val + test)
    everything = set()
    for gen in range(5):
        for split in ("train", "val", "test"):
            everything |= {(split, s) for s in tasks.split_seeds(split, 40, gen)}
    seeds_by_split = {sp: {s for p, s in everything if p == sp} for sp in ("train", "val", "test")}
    assert not seeds_by_split["train"] & seeds_by_split["val"]
    assert not seeds_by_split["train"] & seeds_by_split["test"]
    assert not seeds_by_split["val"] & seeds_by_split["test"]
    assert not set(tasks.split_seeds("val", 40, 0)) & set(tasks.split_seeds("val", 40, 1))
    assert tasks.split_seeds("val", 40, 3) == tasks.split_seeds("val", 40, 3)
    assert tasks.split_seeds("test", 40, 0) == tasks.split_seeds("test", 40, 9)  # Test never moves
    with pytest.raises(ValueError):
        tasks.split_seeds("dev", 4)
    with pytest.raises(ValueError):
        tasks.split_seeds("val", tasks.SPLIT_BLOCK + 1)


@pytest.mark.parametrize("kind", tasks.KINDS)
def test_reference_solver_reproduces_reference(tmp_path, kind):
    """Validity: the question and CSVs alone determine the reference."""
    for seed in range(N_VALID):
        t = tasks.make_task(seed, tmp_path / str(seed), kind)
        got = solve(t, tmp_path / str(seed))
        assert tasks.grade(t, f"ANSWER: {got}") == 1.0, (kind, seed, got, t["reference"], t["question"])
        assert tasks.grade(t, tasks.guess(t)) in (0.0, 1.0)  # the free baseline is gradable on every task


def test_question_states_format_and_does_not_leak(tmp_path):
    for seed in range(40):
        t = tasks.make_task(seed, tmp_path / str(seed))
        q, ref = t["question"], t["reference"]
        if isinstance(ref, str):
            assert "Answer with exactly one of: A_higher, B_higher, no_difference" in q
            # the wording is the same whatever the answer is, so it cannot hint at it
            assert [q.count(x) for x in tasks.LABELS] == [1, 1, 2]
        else:
            assert "Answer with one number rounded to 3 decimals" in q
            assert abs(ref) >= tasks.MIN_ABS_REF        # so 3-decimal rounding sits inside 1%
            assert f"{ref:.3f}" not in q and str(ref) not in q
        for name in t["files"]:
            body = (tmp_path / str(seed) / name).read_text()
            assert "reference" not in body and "higher" not in body and "no_difference" not in body


def test_no_group_task_sits_near_alpha_and_nulls_exist(tmp_path):
    labels = []
    for seed in range(N_VALID):
        t = tasks.make_task(seed, tmp_path / str(seed), "group_diff")
        a, b = arms(t, tmp_path / str(seed))
        p = tasks.perm_pvalue(a, b)                     # the generator's own procedure
        assert not tasks.P_BAND[0] <= p <= tasks.P_BAND[1], (seed, p)
        p_other = tasks.perm_pvalue(a, b, seed=991)     # a tool with different shuffles agrees
        assert (p_other < tasks.ALPHA) == (p < tasks.ALPHA), (seed, p, p_other)
        labels.append(t["reference"])
    assert set(labels) == set(tasks.LABELS)             # true nulls and both directions occur
    assert max(labels.count(x) for x in tasks.LABELS) <= 0.6 * N_VALID


# --- MathBench family -------------------------------------------------------

try:
    tasks._mathbench()
    HAVE_MB = True
except RuntimeError:
    HAVE_MB = False
needs_mb = pytest.mark.skipif(not HAVE_MB, reason="MathBench not importable from MATHBENCH_DIR")


@pytest.fixture(scope="module")
def mb_task(tmp_path_factory):
    return tasks.make_task(1, tmp_path_factory.mktemp("mb"), family="mathbench")


@needs_mb
def test_mathbench_task_hides_the_rule_and_is_deterministic(mb_task, tmp_path):
    t = mb_task
    assert set(t) == {"id", "family", "question", "files", "history", "reference"}
    assert t["family"] == "mathbench"
    rule = t["reference"]["spec"]["rule"]
    assert json.dumps(rule) not in t["question"] and "survive" not in t["question"]
    assert "held_out" not in t["question"] and "9001" not in t["question"]
    assert t == tasks.make_task(1, tmp_path, family="mathbench")
    assert tasks.grade(t, tasks.guess(t)) == 0.0        # no submission, no reward


@needs_mb
def test_mathbench_toolset_and_grade(mb_task):
    t = mb_task
    schemas, call = tasks.toolset(t)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["run_experiment", "define_concept", "submit_laws"]
    assert all(s["type"] == "function" and s["function"]["parameters"]["type"] == "object"
               for s in schemas)
    # bad arguments are data, never exceptions
    for name, args in [("nope", {}), ("run_experiment", {}), ("run_experiment", None),
                       ("run_experiment", {"init": {"class": "random", "p": [0.2, 0.4], "seed": 9005}, "t": 5}),
                       ("run_experiment", {"init": {"class": "pattern", "cells": "x"}, "t": 5, "bogus": 1}),
                       ("define_concept", {"name": "", "cells": []}), ("submit_laws", {"doc": "{not json"})]:
        assert call(name, args).startswith("ERROR"), (name, args)
    assert tasks.grade(t, "anything") == 0.0            # nothing submitted yet
    assert json.loads(call("define_concept", {"name": "dot", "cells": [[0, 0]]})) == {"ok": True}
    obs = json.loads(call("run_experiment", {"init": {"class": "random", "p": [0.3, 0.5], "seed": 1},
                                             "t": 60, "record": [60]}))
    density = obs["snapshots"][0]["live_density"]
    assert obs["budget_left"]["experiments"] == tasks.MB_BUDGET["experiments"] - 1
    law = {"id": "s1", "type": "STAT_LAW", "confidence": 0.5, "quantity": "live_density",
           "init": {"class": "random", "p": [0.3, 0.5]}, "t": 60, "value": density, "tol": 0.02}
    out = json.loads(call("submit_laws", {"doc": {"laws": [law]}}))
    score = out["scorecard"]["score"]
    reward = tasks.grade(t, "ignored")
    assert 0.0 <= reward <= 1.0
    assert reward == pytest.approx(min(1.0, max(0.0, score / tasks.MB_FULL_SCORE)))
    assert call("submit_laws", {"doc": {"laws": []}}).startswith("ERROR")   # one shot
    # a new episode on the same task starts from a fresh lab
    _, call2 = tasks.toolset(t)
    assert tasks.grade(t, "") == 0.0
    assert json.loads(call2("submit_laws", {"doc": {"laws": []}}))["scorecard"]["score"] == 0
    assert tasks.grade(t, "") == 0.0
