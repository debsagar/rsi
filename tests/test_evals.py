"""Evaluation suite: benchmark validity, the external graders, the failure classifier,
and the eval runner end to end.

No Docker, no keys and no paid call: `run` is a scripted stand-in for
broker.run_episode that writes the same files, except two tests that drive the
real broker with in_docker=False and tests/fakes. The benchmark tests read the
downloaded DABench and QRData copies and are skipped when a source is absent;
the report test re-derives three real eval folders under runs/evals and is
skipped where they are absent.

Each grader and the classifier has one table: every row is a way it could
score a reply wrongly, written down before the rule that prevents it. The
junk sweep pins what constant and malformed answers earn on every real row.
"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from conftest import source
from fakes import fake_jev, scripted_llm
from scipy.stats import binom, binomtest

from rsi import broker, evals, tasks

DA_BROKEN = {"dabench-116", "dabench-273", "dabench-352", "dabench-473", "dabench-734"}
REAL = Path(__file__).resolve().parents[1] / "runs" / "evals"
SILENT = lambda *_: None  # noqa: E731
JUDGING_HARNESS = '''def run(task, api):
    seen = api.judge({"task": task["question"]}, {"stuck": {"type": "noul", "instructions": "stuck?"}})
    text = "judge said %s. ANSWER: 1" % seen["stuck"]["noul"]
    if not api.submit(text, [])["accepted"]:
        api.submit(text, [])
'''


def correct(task) -> str:
    if task["family"] == "dabench":
        return "ANSWER: " + " ".join(f"@{name}[{value}]" for name, value in task["reference"])
    return f"ANSWER: {task['reference']}"


def scripted_run(answer=correct, tool_text=lambda task, condition: "rows: 7", bounce=lambda task, condition: False,
                 cost=0.01, fail_on=None):
    """A run_episode stand-in. answer(task) -> text or None; `.log` lists
    (condition, task id, overrides) per finished episode."""
    log = []

    def run(task, harness_path, run_dir, files_dir=None, **overrides):
        condition = Path(run_dir).parent.name
        run_dir.mkdir(parents=True, exist_ok=True)
        text = answer(task) if not callable(getattr(answer, "get", None)) else answer.get(condition, correct)(task)
        events = [{"id": 1, "kind": "user", "text": task["question"]}]
        if "toolset" not in overrides:
            events += [{"id": 2, "kind": "tool_call", "tool": "python", "args": {}},
                       {"id": 3, "kind": "tool_result", "tool": "python", "result": tool_text(task, condition), "clipped": False}]
        if text is not None:
            if bounce(task, condition):
                events.append({"id": len(events) + 1, "kind": "submit", "text": "ANSWER: first try", "accepted": False})
            events += [{"id": len(events) + 1, "kind": "submit", "text": text, "accepted": True},
                       {"id": len(events) + 2, "kind": "answer", "text": text}]
        with open(run_dir / "events.jsonl", "a") as f:
            f.writelines(json.dumps(e) + "\n" for e in events)
        if fail_on and fail_on(condition, task):
            raise RuntimeError("interrupted")
        episode = {"run_id": run_dir.name, "task_id": task["id"], "calls": 2, "tool_calls": 1, "cost": cost,
                   "stop_reason": "submitted" if text is not None else "no_submit", "unsupported": False,
                   "answer": text, "reward": None, "elapsed": 1.5}
        (run_dir / "episode.json").write_text(json.dumps(episode))
        log.append((condition, task["id"], overrides))
        return episode

    run.log = log
    return run


@pytest.fixture
def harness(tmp_path):
    path = tmp_path / "harness.py"
    path.write_text("def run(task, api):\n    pass\n")
    return path


def rows_of(summary) -> list:
    return evals._jsonl(Path(summary["eval_dir"]) / "results.jsonl")


# --- benchmark validity ---------------------------------------------------------

def test_every_dabench_question_has_a_readable_csv_and_a_parsable_label(dabench):
    root = dabench / "evals" / "data" / "dabench"
    questions = evals._jsonl(root / "da-dev-questions.jsonl")
    labels = {row["id"]: row["common_answers"] for row in evals._jsonl(root / "da-dev-labels.jsonl")}
    assert len(questions) == 257
    unreadable, bad_label = set(), set()
    frames = {}
    for q in questions:
        path = root / "da-dev-tables" / q["file_name"]
        try:
            if q["file_name"] not in frames:
                frames[q["file_name"]] = len(pd.read_csv(path).columns)
            assert frames[q["file_name"]] > 0
        except Exception:
            unreadable.add(f"dabench-{q['id']}")
        answers = labels.get(q["id"]) or []
        names = [name for name, _ in answers]
        # A label parses when the benchmark's own @name[value] reading returns it and it has one spelling:
        # an empty value and a value holding a bracket are both an empty list cut by that regex.
        echoed = dict(evals.DA_ITEM_RE.findall(" ".join(f"@{n}[{v}]" for n, v in answers)))
        if (not answers or len(set(names)) < len(names) or echoed != dict(answers)
                or any(not v.strip() or set(v) & set("[]") for _, v in answers)):
            bad_label.add(f"dabench-{q['id']}")
    assert unreadable == set()                 # all 68 CSVs load
    assert bad_label == DA_BROKEN              # 5 of 257 labels cannot be graded
    out = evals.excluded("dabench", dabench)
    assert set(out) == DA_BROKEN and all(out.values())
    assert "empty" in out["dabench-116"] and "repeats" in out["dabench-734"] and "bracket" in out["dabench-273"]
    loaded = evals.load("dabench", dabench)
    assert len(loaded) == 252 and not {t["id"] for t in loaded} & DA_BROKEN


def test_every_qrdata_row_is_usable(qrdata):
    assert evals.excluded("qrdata", qrdata) == {}
    loaded = evals.load("qrdata", qrdata)
    assert len(loaded) == 411
    assert sum(t["level"] == "multiple_choice" for t in loaded) == 248


# --- loading ----------------------------------------------------------------------

def test_dabench_pools_never_share_a_file_are_stable_and_cover_every_usable_question(dabench):
    pools = {split: evals.pool("dabench", split, dabench) for split in ("train", "val", "test")}
    assert pools == {split: evals.pool("dabench", split, dabench) for split in pools}  # same tasks, same order, every call
    files = {split: {f for task in batch for f in task["files"]} for split, batch in pools.items()}
    assert not files["train"] & files["val"] and not files["train"] & files["test"] and not files["val"] & files["test"]
    # The same table under two names must not straddle pools. Read independently of the loader: as parsed frames.
    tables = dabench / "evals" / "data" / "dabench" / "da-dev-tables"
    frame = {f: pd.read_csv(tables / f).to_csv(index=False) for names in files.values() for f in names}
    assert frame["titanic.csv"] == frame["titanic_train.csv"]              # the benchmark really ships such a pair
    content = {split: {frame[f] for f in names} for split, names in files.items()}
    assert not content["train"] & content["val"] and not content["train"] & content["test"] and not content["val"] & content["test"]
    ids = [task["id"] for batch in pools.values() for task in batch]
    assert len(ids) == len(set(ids)) == 252 and set(ids) == {task["id"] for task in evals.load("dabench", dabench)}
    shares = {split: len(batch) / len(ids) for split, batch in pools.items()}
    assert abs(shares["train"] - 0.25) < 0.04 and abs(shares["val"] - 0.35) < 0.04 and abs(shares["test"] - 0.40) < 0.04
    for task in pools["train"][:3]:  # the shape `load` gives: its own folder holding only its files
        assert sorted(p.name for p in Path(task["files_dir"]).iterdir()) == task["files"] and task["family"] == "dabench"
    for name, split in (("qrdata", "train"), ("dabench", "dev")):
        with pytest.raises(ValueError, match="pools exist for dabench"):
            evals.pool(name, split, dabench)


@pytest.mark.parametrize("name", ["dabench", "qrdata"])
def test_load_is_deterministic_stratified_and_in_contract_shape(name, data_home):
    if not source(name).is_dir():
        pytest.skip(f"{name} source folder is not present")
    everything = evals.load(name, data_home)
    a, b, other = evals.load(name, data_home, 40, seed=0), evals.load(name, data_home, 40, seed=0), evals.load(name, data_home, 40, seed=1)
    assert a == b and [t["id"] for t in a] != [t["id"] for t in other]
    assert {t["id"] for t in evals.load(name, data_home, 20, seed=0)} <= {t["id"] for t in a}  # a larger n extends a smaller one
    for level in {t["level"] for t in everything}:
        share = sum(t["level"] == level for t in everything) / len(everything)
        assert abs(sum(t["level"] == level for t in a) - 40 * share) <= 1
    for task in a:
        assert {"id", "family", "question", "files", "history", "reference", "files_dir", "level"} <= set(task)
        assert task["family"] == name and task["history"] == []
        assert sorted(p.name for p in Path(task["files_dir"]).iterdir()) == sorted(task["files"])
        assert "ANSWER:" in task["question"].splitlines()[-1]
    with pytest.raises(ValueError):
        evals.load(name, data_home, 10_000)


def test_questions_carry_the_benchmark_instructions(dabench):
    task = next(t for t in evals.load("dabench", dabench) if t["id"] == "dabench-0")
    assert "Rounding off the answer to two decimal places" in task["question"]        # constraints
    assert "@mean_fare[mean_fare_value]" in task["question"]                          # answer format
    assert task["files"] == ["test_ave.csv"] and task["reference"] == [["mean_fare", "34.65"]]
    assert "34.65" not in task["question"]


# --- graders ----------------------------------------------------------------------

def DA(*values) -> dict:
    return {"id": "dabench-x", "family": "dabench", "level": "easy", "reference": [[f"v{i}", v] for i, v in enumerate(values)]}


def QR(reference, options=()) -> dict:
    return {"id": "qrdata-x", "family": "qrdata", "reference": reference, "options": list(options),
            "level": "multiple_choice" if options else "numerical"}


COUNTRIES = "Kuwait, Saudi Arabia"
DA_ROWS = [  # (reference values v0, v1, ..., reply, expected reward, the failure this row would expose)
    # --- junk that must not score ---
    (["5"], "", 0.0, "empty reply"),
    (["5"], None, 0.0, "no reply"),
    (["5"], "ANSWER:", 0.0, "empty ANSWER line"),
    (["5"], "ANSWER: @v0[]", 0.0, "empty item"),
    (["5"], "ANSWER: 5", 0.0, "value outside the @name[value] format"),
    (["5"], "ANSWER: @other[5]", 0.0, "right value under the wrong name"),
    (["5", "6"], "ANSWER: @v0[5]", 0.0, "one of two items missing"),
    (["34.65"], "ANSWER: @v0[34.66]", 0.0, "number outside 1e-6"),
    (["34.65"], "e.g. @v0[34.65]\nANSWER: @v0[1]", 0.0, "an example in the reasoning counts"),
    (["yes"], "ANSWER: @v0[yes] @v0[no]", 0.0, "two values for one name both count (the last one does)"),
    (["0.5"], "ANSWER: @v0[nan]", 0.0, "nan compares as a number"),
    (["0.5"], "ANSWER: @v0[inf]", 0.0, "inf compares as a number"),
    (["0.5"], "ANSWER: @v0[1e999]", 0.0, "overflowing exponent"),
    (["0.5"], "ANSWER: @v0[-0.5]", 0.0, "sign ignored"),
    (["50"], "ANSWER: @v0[50%]", 0.0, "a percentage read as the plain number"),
    (["1234.5"], "ANSWER: @v0[1,234.6]", 0.0, "thousands separator hides a wrong number"),
    ([COUNTRIES], "ANSWER: @v0[Saudi Arabia, Kuwait]", 0.0, "list order ignored"),
    ([COUNTRIES], "ANSWER: @v0[Kuwait]", 0.0, "partial credit for part of a list"),
    ([COUNTRIES], "ANSWER: @v0[KuwaitSaudi Arabia]", 0.0, "separator dropped"),
    (["Not Normal"], "ANSWER: @v0[Normal]", 0.0, "contained string matches"),
    (["Not Normal"], "ANSWER: @v0[Non Normal]", 0.0, "near spelling matches"),
    # --- right answers that must score ---
    (["5"], "ANSWER: @v0[5]", 1.0, "plain"),
    (["5", "x y"], "ANSWER: @v0[5] @v1[x y]", 1.0, "two items"),
    (["34.65"], "ANSWER: @v0[34.650]", 1.0, "same float, other spelling"),
    (["34.65"], "The mean is @v0[34.65]", 1.0, "no ANSWER line: whole reply, as upstream"),
    (["34.65"], "Work.\n\n**ANSWER:** @v0[34.65]", 1.0, "bold marker"),
    (["34.65"], "**ANSWER**: @v0[34.65]", 1.0, "bold marker, colon outside"),
    (["5"], "ANSWER:\n@v0[5]", 1.0, "items on the line after the marker"),
    (["5"], "ANSWER: @v0[5]\nNote: the ANSWER: line above follows the format.", 1.0, "a later mention of ANSWER: moves the start"),
    (["-0.5"], "ANSWER: @v0[−0.5]", 1.0, "unicode minus"),
    (["1234.5"], "ANSWER: @v0[1,234.5]", 1.0, "thousands separator"),
    (["Not Normal"], "ANSWER: @v0[not normal]", 1.0, "letter case"),
    (["Kuwait"], 'ANSWER: @v0["Kuwait"]', 1.0, "quotes"),
    (["Kuwait"], "ANSWER: @v0[ Kuwait ]", 1.0, "padding inside the brackets"),
    ([COUNTRIES], "ANSWER: @v0[Kuwait,Saudi Arabia]", 1.0, "spacing after a comma"),
    ([COUNTRIES], 'ANSWER: @v0["Kuwait","Saudi Arabia"]', 1.0, "quoted list items (seen in evolve gen 0)"),
    (["{'AT': 590, 'VIS': 8736}"], "ANSWER: @v0[{'AT':590, 'VIS':8736} ]", 1.0, "dict spacing (seen in dabench-451)"),
    (['"Non-Normal"'], "ANSWER: @v0[Non-Normal]", 1.0, "gold itself carries quotes"),
    (["nan"], "ANSWER: @v0[nan]", 1.0, "a gold that is the string nan"),
]
QR_ROWS = [  # (task, reply, expected reward, the failure this row would expose)
    # --- numbers: junk ---
    (QR("0.20"), "The proportion is 0.20", 0.0, "no ANSWER line"),
    (QR("0.20"), "It is 0.20.\nANSWER:", 0.0, "empty ANSWER line falls back to the reply"),
    (QR("0.20"), "ANSWER: none", 0.0, "no number"),
    (QR("0.20"), "ANSWER: 0.2061", 0.0, "just outside 3%"),
    (QR("0.20"), "ANSWER: nan", 0.0, "nan"),
    (QR("0.20"), "ANSWER: 1e999", 0.0, "overflowing exponent"),
    (QR("4.00"), "ANSWER: .4", 0.0, ".4 read as 4"),
    (QR("4.00"), "ANSWER: 4e-5", 0.0, "4e-5 read as 4"),
    (QR("2.00"), "ANSWER: 2,457.89", 0.0, "2,457.89 read as 2"),
    (QR("1.5"), "ANSWER: –1.5", 0.0, "en dash minus read as positive"),
    (QR("8.53%"), "ANSWER: 853", 0.0, "percent gold accepts any scale"),
    (QR("8.53%"), "ANSWER: 0.0853%", 0.0, "percent sign ignored on the reply"),
    (QR("0"), "ANSWER: 0.001", 0.0, "zero gold accepts anything small"),
    # --- numbers: right ---
    (QR("0.20"), "ANSWER: 0.2059", 1.0, "inside 3%"),
    (QR("0.20"), "ANSWER: 20%", 1.0, "eval.py divides a % by 100"),
    (QR("0.20"), "ANSWER: about 0.20 (rounded)", 1.0, "words around the number"),
    (QR("0.20"), "**ANSWER**: 0.20", 1.0, "bold marker, colon outside"),
    (QR("0.7"), "ANSWER: .7", 1.0, "leading-dot decimal"),
    (QR("0.7"), "ANSWER: 7e-1", 1.0, "scientific notation"),
    (QR("2457.89"), "ANSWER: 2,457.89", 1.0, "thousands separator"),
    (QR("-1.5"), "ANSWER: −1.5", 1.0, "unicode minus"),
    (QR("8.53%"), "ANSWER: 8.53%", 1.0, "percent gold, percent reply"),
    (QR("8.53%"), "ANSWER: 8.53", 1.0, "percent gold, the number without its sign"),
    (QR("8.53%"), "ANSWER: 0.0853", 1.0, "percent gold as a fraction (eval.py's reading)"),
    (QR("0"), "ANSWER: 0", 1.0, "a gold of 0 is unsatisfiable under strict relative bounds"),
    # --- choices: junk ---
    (QR("C", "abcd"), "ANSWER:", 0.0, "empty ANSWER line"),
    (QR("C", "abcd"), "ANSWER: A", 0.0, "wrong option"),
    (QR("C", "abcd"), "ANSWER: Cannot tell", 0.0, "a word that starts with the gold letter (75 golds are C)"),
    (QR("B", "abcd"), "ANSWER: Both", 0.0, "a word that starts with the gold letter"),
    (QR("A", "abcd"), "ANSWER: All of the above", 0.0, "a word that starts with the gold letter"),
    (QR("A", "abcd"), "ANSWER: A B C D", 0.0, "every option listed"),
    (QR("C", "abcd"), "ANSWER: a, b, c", 0.0, "every option listed, gold not first"),
    (QR("C", "abcd"), "ANSWER: C, D", 0.0, "two options"),
    (QR("C", "abcd"), "ANSWER: C or D", 0.0, "two options"),
    (QR("C", "abcd"), "ANSWER: C is wrong", 0.0, "gold letter followed by prose"),
    (QR("C", "abcd"), "ANSWER: the answer is C", 0.0, "the line does not start with the option"),
    (QR("C", "abcd"), "ANSWER: C. A larger sample helps", 0.0, "names another option: refused, the price of the rule above"),
    (QR("no", ["yes", "no"]), "ANSWER: nope", 0.0, "a longer word that starts with the gold"),
    (QR("no", ["yes", "no"]), "ANSWER: nobody knows", 0.0, "a longer word that starts with the gold"),
    (QR("no", ["yes", "no"]), "ANSWER: no, yes", 0.0, "both options"),
    (QR("treatment group", ["treatment group", "control group"]), "ANSWER: control group", 0.0, "wrong phrase option"),
    # --- choices: right ---
    (QR("C", "abcd"), "ANSWER: C", 1.0, "plain"),
    (QR("C", "abcd"), "ANSWER: c", 1.0, "letter case"),
    (QR("C", "abcd"), "ANSWER: **C**", 1.0, "bold"),
    (QR("B", "abc"), "**ANSWER**: B", 1.0, "bold marker, colon outside"),
    (QR("B", "abc"), "ANSWER: (B)", 1.0, "brackets"),
    (QR("B", "abc"), "ANSWER: B.", 1.0, "final full stop"),
    (QR("B", "abc"), "ANSWER: B) because the sample is small", 1.0, "option, punctuation, then its reason"),
    (QR("no", ["yes", "no"]), "ANSWER: No.", 1.0, "whole word"),
    (QR("treatment group", ["treatment group", "control group"]), "ANSWER: Treatment group", 1.0, "phrase option, letter case"),
    (QR("C", "abcd"), "ANSWER: A\nOn reflection:\nANSWER: C", 1.0, "the last ANSWER line"),
]


def test_dabench_grader_table():
    wrong = [(why, reply) for values, reply, want, why in DA_ROWS if evals.grade(DA(*values), reply) != want]
    assert not wrong, wrong
    assert evals.grade_parts(DA("5", "6", "7"), "ANSWER: @v0[5] @v2[7]") == {"v0": True, "v1": False, "v2": True}


def test_qrdata_grader_table():
    wrong = [(why, reply) for task, reply, want, why in QR_ROWS if evals.grade(task, reply) != want]
    assert not wrong, wrong


def test_junk_answers_on_every_real_benchmark_row(dabench, qrdata):
    """The exploitable floor: what an answer that never looked at the data earns over all 252 + 411 real rows.
    A grader change that moves one of these numbers changes what the evolve loop can farm."""
    da, qr = evals.load("dabench", dabench), evals.load("qrdata", qrdata)
    assert all(evals.grade(t, correct(t)) == 1.0 for t in da + qr)       # and every reference itself scores
    assert all(evals.grade(t, f"Some working first, 12 rows.\n\n**ANSWER:** {t['reference']}") == 1.0 for t in qr)

    def every(value):
        return lambda t: "ANSWER: " + " ".join(f"@{name}[{value}]" for name, _ in t["reference"])

    def listed(t):
        return "ANSWER: " + " ".join(f"@{name}[{v}]" for name, _ in t["reference"] for v in ("0", "1", "yes", "no", "true", "false"))

    da_junk = {"empty": (lambda t: "", 0), "bare marker": (lambda t: "ANSWER:", 0),
               "template echo": (lambda t: "ANSWER: @first_name[1.23] @second_name[yes]", 0),
               "bare yes": (lambda t: "yes", 0), "null guess": (evals.null_guess, 10),
               "0": (every("0"), 10), "1": (every("1"), 5), "2": (every("2"), 2), "yes": (every("yes"), 2),
               "no": (every("no"), 5), "true": (every("true"), 1), "false": (every("false"), 4),
               "normal": (every("normal"), 1), "not normal": (every("not normal"), 6), "nan": (every("nan"), 1),
               "six values per name": (listed, 4)}          # the last one counts: no more than `false` alone;
    # `nan` earns 1 because one real gold is the string nan
    got = {name: int(sum(evals.grade(t, make(t)) for t in da)) for name, (make, _) in da_junk.items()}
    assert got == {name: want for name, (_, want) in da_junk.items()}

    qr_junk = {"": (0, 0), "ANSWER:": (0, 0), "ANSWER: <value>": (0, 0), "ANSWER: 0": (0, 0), "ANSWER: 1": (0, 1),
               "ANSWER: nan": (0, 0), "ANSWER: 4": (0, 21), "ANSWER: 4.08": (0, 23), "ANSWER: 0.7": (0, 5),
               "ANSWER: A": (35, 0), "ANSWER: B": (60, 0), "ANSWER: C": (75, 0), "ANSWER: D": (32, 0),
               "ANSWER: yes": (5, 0), "ANSWER: no": (14, 0),
               "ANSWER: Cannot tell": (0, 0), "ANSWER: Both": (0, 0), "ANSWER: All of the above": (0, 0),
               "ANSWER: nope": (0, 0), "ANSWER: A B C D": (0, 0), "ANSWER: A, B, C, D, yes, no, true, false": (0, 0)}
    kinds = ("multiple_choice", "numerical")
    got = {reply: tuple(int(sum(evals.grade(t, reply) for t in qr if t["level"] == kind)) for kind in kinds) for reply in qr_junk}
    assert got == qr_junk
    choices = [t for t in qr if t["level"] == "multiple_choice"]
    assert sum(evals.grade(t, "ANSWER: " + " ".join(t["options"])) for t in choices) == 0   # every option listed
    assert sum(evals.grade(t, evals.null_guess(t)) for t in choices) == 51                   # the first-option baseline


# --- failure classification ---------------------------------------------------------

def write_run(run_dir, results, full_text=None) -> Path:
    run_dir.mkdir(parents=True)
    events = [{"id": i + 1, "kind": "tool_result", "tool": "python", "result": text, "clipped": False} for i, text in enumerate(results)]
    if full_text is not None:
        events.append({"id": len(events) + 1, "kind": "tool_result", "tool": "python", "result": "head [... clipped ...] tail",
                       "clipped": True, "file": "tool-009.txt"})
        (run_dir / "tool-009.txt").write_text(full_text)
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return run_dir


WRONG = {"answer": "ANSWER: @v0[1]", "stop_reason": "submitted"}
TWO = DA("34.65", "14.45")
LABEL = {"id": "g", "family": "generated", "reference": "A_higher"}
PERCENT = QR("12.5%")
AGES = "Age: 22.0 38.0 26.0 35.0 36.0 54.0 49.0 28.0"
CLASSIFY_ROWS = [  # (task, episode, tool results, decoy task or None, expected (kind, evidence event), the failure exposed)
    (TWO, {"answer": None, "stop_reason": "no_submit"}, ["34.65 14.45"], None, ("no_answer", None), "nothing submitted"),
    (TWO, {"answer": "x", "stop_reason": "run_limit"}, ["34.65 14.45"], None, ("no_answer", None), "stopped by a limit"),
    (TWO, WRONG, [], None, ("gather", None), "no tool result at all"),
    (TWO, WRONG, ["mean 34.6532", "median=14.4542"], None, ("use", 1), "values that round to the reference"),
    (TWO, WRONG, ["mean 34.6532", "median 14.9"], None, ("gather", None), "one of two values missing"),
    (TWO, WRONG, ["x134.65 col34.65 134.65", "14.45"], None, ("gather", None), "digits inside a longer token"),
    (TWO, WRONG, ["ERROR 34.65 14.45"], None, ("gather", None), "an error message is not evidence"),
    (DA("34.65", "7"), WRONG, ["mean 34.65 over 7 rows"], None, ("unchecked", 1), "a one-digit part is never searched for"),
    (DA("34.65", "7"), WRONG, ["7 rows"], None, ("gather", None), "the searchable part is missing"),
    # 37 real no_tools failures were called `gather` although nothing in their reference can be searched for
    (LABEL, WRONG, [], None, ("unchecked", None), "a label with no tool result"),
    (LABEL, WRONG, ["ERROR no tools are available in this run"], None, ("unchecked", None), "a label under no_tools"),
    (LABEL, WRONG, ["A_higher"], None, ("unchecked", None), "a label is a plain word"),
    (QR("C", "abc"), WRONG, ["C"], None, ("unchecked", None), "an option letter"),
    (QR("1691"), WRONG, ["total 1700.0"], None, ("use", 1), "within the grader's 3%"),
    (QR("1691"), WRONG, ["total 1800"], None, ("gather", None), "outside it"),
    ({"id": "g", "family": "generated", "reference": 0.512}, WRONG, ['"pearson_r": 0.5117'], None, ("use", 1), "generated, 1%"),
    (DA("2021"), WRONG, ["date 2021-05-14 open 3.1"], None, ("gather", None), "a year inside a date"),
    (DA("204"), WRONG, ["203  Smith\n204  Jones\n205  Lee"], None, ("gather", None), "the index column of a printed table"),
    (DA("204"), WRONG, ["0  17  204\n1  18  233"], None, ("use", 1), "the same number in a data column"),
    (DA("1234.5"), WRONG, ["total: 1,234.5"], None, ("use", 1), "thousands separator in tool output"),
    (DA("234.5"), WRONG, ["id,value\n1,234.5"], None, ("use", 1), "CSV cells are not a thousands group"),
    (DA("1500"), WRONG, ["bin 1200-1500 count 7"], None, ("use", 1), "a range dash is not a minus"),
    (DA("5949.08"), WRONG, ["mean-5949.08"], None, ("use", 1), "a hyphen before the value"),
    (PERCENT, WRONG, ["share = 12.5%"], None, ("use", 1), "percent reference printed as a percent"),
    (PERCENT, WRONG, ["share = 0.125"], None, ("use", 1), "percent reference printed as a fraction"),
    (QR("0.00012"), WRONG, ["p = 1.2e-04"], None, ("use", 1), "scientific notation in tool output"),
    # The chance control must be able to match what the reference can: raw cells. (x1.37 gave 49.3, which never can.)
    (DA("36.0"), WRONG, [AGES], None, ("use", 1), "a round reference that is a raw data cell"),
    (DA("36.0"), WRONG, [AGES], DA("49.0"), ("use", 1), "the decoy is searched exactly as hard"),
    (DA("36.0"), WRONG, [AGES], DA("77.7"), ("gather", None), "a decoy that is absent"),
]


def test_failure_classifier_table(tmp_path):
    wrong = []
    for i, (task, episode, results, decoy, want, why) in enumerate(CLASSIFY_ROWS):
        got = evals.classify(task, episode, write_run(tmp_path / str(i), results), decoy)
        if got != want:
            wrong.append((why, got))
    assert not wrong, wrong
    clipped = write_run(tmp_path / "clipped", ["14.45"], full_text="a\n" * 3000 + "34.65")
    assert evals.classify(TWO, WRONG, clipped) == ("use", 2)             # a clipped result is read from its file
    batch = [{**DA("36.0"), "id": "a"}, {**LABEL, "id": "b"}, {**DA("36.0"), "id": "c"}, {**DA("49.0"), "id": "d"}]
    assert evals.decoy_for(batch[0], batch)["id"] == "d"                 # next searchable task with another reference
    assert evals.decoy_for(batch[3], batch)["id"] == "a" and evals.decoy_for(batch[1], batch) is None


# --- the report, re-derived from raw files --------------------------------------------

# Accuracies of the three paid n=40 evals (gpt-5-mini, harness 000) under grader version 2. Version 1 reported
# generated no_tools 0.025, dabench full 0.70 and no_gate 0.675; everything else was the same.
REAL_ACCURACY = {"generated": {"null": 0.1, "no_tools": 0.0, "full": 1.0, "no_gate": 1.0},
                 "dabench": {"null": 0.0, "no_tools": 0.0, "full": 0.725, "no_gate": 0.7},
                 "qrdata": {"null": 0.225, "no_tools": 0.175, "full": 0.575, "no_gate": 0.575}}
REAL_CHANGES = {"generated": {"no_tools": (0, 1)}, "dabench": {"full": (1, 0), "no_gate": (1, 0)}, "qrdata": {}}


@pytest.mark.parametrize("family", list(REAL_ACCURACY))
def test_report_of_a_real_eval_folder_is_recomputed_from_its_raw_files(family, tmp_path, capsys):
    """Every number `rsi eval --report` prints, derived a second way from tasks.jsonl, episode.json and
    events.jsonl. The folder was written under grader version 1, so this also covers the re-grade."""
    source_dir = REAL / f"{family}-n40-s0-fa8cf48b"
    if not source_dir.is_dir():
        pytest.skip(f"{source_dir} is not present")
    folder = tmp_path / source_dir.name
    shutil.copytree(source_dir, folder, ignore=shutil.ignore_patterns("results.regraded-*"))
    before = (folder / "results.jsonl").read_bytes()
    summary = evals.main(["--report", str(folder)])
    printed = capsys.readouterr().out
    assert (folder / "results.jsonl").read_bytes() == before             # never overwritten
    assert evals.summarize(folder) == summary                            # the report needs nothing but the folder

    batch = evals._jsonl(folder / "tasks.jsonl")
    episodes = {(c, t["id"]): json.loads((folder / c / t["id"] / "episode.json").read_text())
                for c in ("no_tools", "full", "no_gate") for t in batch}
    reward = {key: evals.grade(next(t for t in batch if t["id"] == key[1]), ep.get("answer") or "") for key, ep in episodes.items()}
    reward.update({("null", t["id"]): evals.grade(t, evals.null_guess(t)) for t in batch})
    assert summary["n_tasks"] == len(batch) == 40 and summary["unpaired_rows"] == 0 and summary["duplicate_rows"] == 0
    for c, got in summary["conditions"].items():
        rewards = [reward[c, t["id"]] for t in batch]
        k, cost = int(sum(rewards)), sum((episodes[c, t["id"]]["cost"] or 0.0) for t in batch) / 40 if c != "null" else 0.0
        low, high = binomtest(k, 40).proportion_ci(0.95, method="wilson")   # scipy's Wilson, not ours
        assert got["accuracy"] == k / 40 == REAL_ACCURACY[family][c]
        assert got["ci95"] == pytest.approx([low, high], abs=1e-4) and got["ci_method"] == "Wilson" and low < high
        assert got["cost_per_task"] == pytest.approx(cost, abs=1e-6)
        assert f"{c:<10}{40:>4}{k / 40:>10.1%}{low:>8.1%}-{high:<7.1%}{cost:>9.4f}" in printed
        if c != "null":
            failed = [episodes[c, t["id"]] for t in batch if reward[c, t["id"]] < 1.0]
            silent = sum(not ep.get("answer") or ep["stop_reason"] != "submitted" for ep in failed)
            assert sum(got["failure_kinds"].values()) == len(failed) and got["failure_kinds"]["no_answer"] == silent
    # A label or an option letter cannot be searched for: such a failure is never `gather` or `use`.
    words = {t["id"] for t in batch if isinstance(t["reference"], str) and t["family"] != "dabench" and t["level"] != "numerical"}
    regraded = evals._jsonl(folder / f"results.regraded-v{evals.GRADER_VERSION}.jsonl")
    assert len(regraded) == 160 and all(r["grader_version"] == evals.GRADER_VERSION for r in regraded)
    assert all(r["failure_kind"] in (None, "no_answer", "unchecked") for r in regraded if r["task_id"] in words)
    assert {(r["condition"], r["task_id"]): r["reward"] for r in regraded} == reward

    for name, got in summary["pairs"].items():
        a, b = name.split(" vs ")
        wins = sum(reward[a, t["id"]] > reward[b, t["id"]] for t in batch)
        losses = sum(reward[a, t["id"]] < reward[b, t["id"]] for t in batch)
        p = min(1.0, 2 * binom.cdf(min(wins, losses), wins + losses, 0.5)) if wins + losses else 1.0
        assert (got["wins"], got["losses"], got["ties"], got["test"]) == (wins, losses, 40 - wins - losses, "McNemar")
        assert got["p_value"] == pytest.approx(p, rel=1e-9) and got["p_value"] > 0     # 40 wins of 40 is 1.8e-12, not 0
        assert f"on {wins}, the reverse on {losses}, same on {40 - wins - losses}; " + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") in printed
    assert set(summary["pairs"]) == {"full vs no_gate", "full vs no_tools", "full vs null"} and "p = 0.000" not in printed

    gate = {"episodes": 40, "bounced": 0, "first_submission_correct": 0, "ended_correct": 0, "resubmitted_wrong": 0,
            "never_resubmitted": 0, "same_tasks_no_gate_correct": 0}
    for t in batch:
        submits = [e for e in evals._jsonl(folder / "full" / t["id"] / "events.jsonl") if e["kind"] == "submit"]
        if submits and not submits[0]["accepted"]:
            right = reward["full", t["id"]] >= 1.0
            gate["bounced"] += 1
            gate["first_submission_correct"] += int(evals.grade(t, submits[0]["text"]))
            gate["ended_correct"] += right
            gate["resubmitted_wrong"] += len(submits) > 1 and not right
            gate["never_resubmitted"] += len(submits) == 1
            gate["same_tasks_no_gate_correct"] += reward["no_gate", t["id"]] >= 1.0
    assert summary["gate"] == gate
    assert (f"{gate['ended_correct']} ended correct, {gate['resubmitted_wrong']} resubmitted and ended wrong, and "
            f"{gate['never_resubmitted']} never submitted again") in printed

    # The re-grade is announced with what it changed, against the rewards the old rows hold.
    old = {(r["condition"], r["task_id"]): r["reward"] for r in evals._jsonl(folder / "results.jsonl")}
    changes = {c: (sum(reward[c, t] > was for (cc, t), was in old.items() if cc == c),
                   sum(reward[c, t] < was for (cc, t), was in old.items() if cc == c)) for c in evals.CONDITIONS}
    assert {c: n for c, n in changes.items() if n != (0, 0)} == REAL_CHANGES[family]
    assert summary["regraded"]["rows"] == 160 and summary["regraded"]["from"] == ["1"]
    assert {c: (n["up"], n["down"]) for c, n in summary["regraded"]["changes"].items()} == changes
    assert "RE-GRADED: 160 rows" in printed and "results.jsonl is untouched" in printed
    for c, (up, down) in changes.items():
        assert f"{c} {up} newly correct and {down} newly wrong" in printed


# --- the runner, end to end ---------------------------------------------------------

def test_run_eval_resumes_refuses_to_mix_and_audits_its_rows(tmp_path, harness, monkeypatch, capsys):
    """One folder through its life: a free null run, a paid run that dies after an episode is on disk but before
    its row is written, another model trying to take the folder over, the resume, a duplicated row, a tampered row."""
    conditions = ["no_tools", "full", "no_gate"]
    monkeypatch.delenv("RSI_MODEL", raising=False)       # as `rsi eval --conditions null` runs: free, no model set
    free = evals.run_eval("generated", 4, ["null"], home=tmp_path, run=scripted_run(), harness=harness, say=SILENT)
    folder = Path(free["eval_dir"])
    assert json.loads((folder / "meta.json").read_text())["model"] is None and len(rows_of(free)) == 4

    finishing = scripted_run()

    def dies_after_the_episode(task, harness_path, run_dir, files_dir=None, **overrides):
        finishing(task, harness_path, run_dir, files_dir, **overrides)   # episode.json is on disk, as the broker leaves it
        raise RuntimeError("interrupted")

    monkeypatch.setenv("RSI_MODEL", "model-A")           # nothing paid in the folder yet: model-A adopts it
    with pytest.raises(RuntimeError, match="interrupted"):
        evals.run_eval("generated", 4, conditions, home=tmp_path, run=dies_after_the_episode, harness=harness, say=SILENT)
    orphans = sorted(p.parent.name for p in folder.glob("full/*/episode.json"))
    assert orphans and all(r["condition"] == "null" for r in rows_of(free))    # paid episodes, and still only null rows

    monkeypatch.setenv("RSI_MODEL", "model-B")           # null rows only, but model-A's episodes are there: refused
    with pytest.raises(ValueError, match="different model"):
        evals.run_eval("generated", 4, conditions, home=tmp_path, run=scripted_run(), harness=harness, say=SILENT)

    monkeypatch.setenv("RSI_MODEL", "model-A")
    wrong_once = lambda task: "ANSWER: 123456" if task["id"].endswith("3") else correct(task)  # noqa: E731
    resumed = scripted_run(answer={"no_gate": wrong_once})
    summary = evals.run_eval("generated", 4, conditions, home=tmp_path, run=resumed, harness=harness, say=SILENT)
    rows = rows_of(summary)
    assert len(resumed.log) == 12 and len(rows) == 16 and [r["condition"] for r in rows[:4]] == ["null"] * 4
    assert [int(r["task_id"].rsplit("-", 1)[1]) for r in rows[:4]] == tasks.split_seeds("test", 4)   # owned test seeds
    assert all(r["grader_version"] == evals.GRADER_VERSION for r in rows)
    assert json.loads((folder / "meta.json").read_text())["grader_version"] == evals.GRADER_VERSION
    for condition, task_id, overrides in resumed.log:    # every condition names the gate; only no_tools swaps the tools
        assert overrides["gate"] is (condition != "no_gate") and ("toolset" in overrides) == (condition == "no_tools")
    kinds = [e["kind"] for e in evals._events(folder / "full" / orphans[0])]
    assert kinds.count("user") == 1                      # the unfinished folder was cleared, not appended to
    assert summary["n_tasks"] == 4 and summary["regraded"] is None and summary["duplicate_rows"] == 0
    assert summary["conditions"]["full"]["accuracy"] == 1.0 and summary["conditions"]["no_gate"]["accuracy"] == 0.75
    assert summary["pairs"]["full vs no_gate"] == {"n": 4, "wins": 1, "losses": 0, "ties": 3, "p_value": 1.0, "test": "McNemar"}
    assert not list(folder.glob("results.regraded-*"))   # rows of the current grader version are audited, not re-graded

    written = (folder / "results.jsonl").read_bytes()
    again = scripted_run()
    assert evals.run_eval("generated", 4, conditions, home=tmp_path, run=again, harness=harness, say=SILENT) == summary
    assert again.log == [] and (folder / "results.jsonl").read_bytes() == written   # nothing re-run, nothing re-written

    with open(folder / "results.jsonl", "ab") as f:      # a second process appended the same row
        f.write(written.splitlines(keepends=True)[-1])
    doubled = evals.main(["--report", str(folder)])
    assert doubled == {**summary, "duplicate_rows": 1} and "1 duplicate rows" in capsys.readouterr().out

    tampered = [json.loads(line) for line in written.splitlines()]
    victim = next(r for r in tampered if r["condition"] == "no_gate" and r["reward"] == 0.0)
    victim["reward"] = 1.0
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tampered))
    with pytest.raises(ValueError, match=f"do not match their raw files: no_gate/{victim['task_id']}"):
        evals.summarize(folder)


def test_budget_stops_after_the_timing_run(dabench, harness):
    said = []
    run = scripted_run(cost=0.5)
    summary = evals.run_eval("dabench", 6, ["full", "no_tools"], home=dabench, seed=5, run=run, harness=harness, budget=2.0,
                             say=said.append)
    assert len(run.log) == evals.TIMING_TASKS and "$1.00 spent plus the estimate $5.00 exceeds the budget $2.00" in summary["stopped"]
    assert any("Estimate from 2 `full` episodes: $5.00 more" in line for line in said)
    assert said.index(next(s for s in said if s.startswith("Null baseline"))) < said.index(next(s for s in said if s.startswith("Estimate")))
    assert summary["n_tasks"] == 2 and summary["unpaired_rows"] == 4     # only tasks finished everywhere are compared
    bigger = scripted_run(cost=0.5)
    done = evals.run_eval("dabench", 6, ["full", "no_tools"], home=dabench, seed=5, run=bigger, harness=harness, budget=10.0,
                          say=SILENT)
    assert len(bigger.log) == 10 and "stopped" not in done and done["n_tasks"] == 6
    assert json.loads((Path(done["eval_dir"]) / "meta.json").read_text())["excluded"].keys() == DA_BROKEN


def test_conditions_fix_the_gate_whatever_rsi_gate_says_and_leave_harness_judge_calls_alone(tmp_path, monkeypatch):
    """Through the real broker: Jev says 0.1 to everything, which bounces `full` once."""
    path = tmp_path / "judging_harness.py"
    path.write_text(JUDGING_HARNESS)
    jev = fake_jev(0.1)
    task = {"id": "t1", "family": "generated", "question": "What is 1?", "files": [], "history": [], "reference": 1.0}
    out = {}
    for condition, setting in (("full", "0"), ("no_gate", "1")):  # the environment says the opposite each time
        monkeypatch.setenv("RSI_GATE", setting)
        overrides = {"jev": jev, **evals.OVERRIDES[condition]}
        episode = broker.run_episode(task, path, tmp_path / condition, llm=scripted_llm([]), in_docker=False, **overrides)
        events = evals._events(tmp_path / condition)
        out[condition] = (episode, [e for e in events if e["kind"] == "submit"], [e for e in events if e["kind"] == "judge"])
    episode, submits, judges = out["full"]
    assert [s["accepted"] for s in submits] == [False, True] and episode["unsupported"] is True and episode["gate"] is True
    assert [j["source"] for j in judges] == ["harness", "gate", "gate"]
    episode, submits, judges = out["no_gate"]
    assert [s["accepted"] for s in submits] == [True] and episode["unsupported"] is False and episode["gate"] is False
    assert submits[0]["p_supported"] is None
    assert {j["source"]: j["answers"] for j in judges} == {"harness": {"stuck": {"noul": 0.1}}}  # no gate judge event at all
    assert "judge said 0.1" in episode["answer"]                         # the harness still heard the real Jev


def test_a_gate_bounce_under_no_gate_is_an_error_not_a_row(dabench, harness):
    run = scripted_run(bounce=lambda task, condition: condition == "no_gate")
    with pytest.raises(RuntimeError, match="bounced a submission under no_gate"):
        evals.run_eval("dabench", 2, ["no_gate"], home=dabench, seed=7, run=run, harness=harness, say=SILENT)
