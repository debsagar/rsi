"""Graded tasks: what a harness is scored on, and how.

Two families behind one interface (make_task, grade, toolset, guess):

- "generated": a question about one or two CSVs written to out_dir. Four kinds
  (KINDS): a significance label for a two-arm comparison, Cohen's d, a
  correlation, and a filtered aggregate across a join.
- "mathbench": discover laws of a hidden-rule cellular automaton through the
  MathBench Lab, imported read-only by path from MATHBENCH_DIR.

Invariants:
- Deterministic: the same (seed, kind, knobs) gives the same task dict and
  byte-identical files. All randomness comes from numpy default_rng.
- The answer lives only in task["reference"]. The question and the files never
  contain it (for mathbench, reference holds the hidden rule).
- Grading is code only, read through rsi.grading (one ANSWER-line pattern, one
  number parser). If the reply has any ANSWER line, only the last one is
  graded, even when it is empty. Numbers: the first number on that line, or
  without any ANSWER line the last number in the reply, within REL_TOL of the
  reference. Labels: the ANSWER line, stripped of markdown, quotes, brackets
  and trailing punctuation, must BE the label (case-insensitive); a label
  with no ANSWER line, or with other words beside it, scores 0, because
  "not A_higher" and an echoed option list both contain a label.
- A generated question states its answer format, and every instruction needed
  to reach the reference (units, rows to exclude, formula) is in the question.
- Group-difference references never sit near alpha: draws with a permutation
  p-value inside P_BAND are rejected and redrawn, so a tool using different
  shuffles reaches the same label.
- mathbench is stateful: toolset(task) starts a fresh Lab for task["id"] and
  grade(task, text) reads that Lab's submission (text is ignored). Call them in
  that order, in one process, one episode per task id at a time.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from .grading import ANSWER_RE, EDGE, clean, numbers

KINDS = ["group_diff", "effect_size", "correlation", "filtered_agg"]
LABELS = ["A_higher", "B_higher", "no_difference"]
REL_TOL = 0.01
ALPHA = 0.05
N_SHUFFLES = 10_000          # same count as the assistant's compare_groups tool
P_BAND = (0.02, 0.10)        # p-values this close to ALPHA are redrawn
MIN_ABS_REF = 0.1            # keeps 3-decimal rounding (0.0005) inside REL_TOL
MAX_ATTEMPTS = 200
SPLIT_START = {"train": 0, "val": 1_000_000, "test": 2_000_000}
SPLIT_BLOCK = 1000           # seeds per generation; also the largest n
REGIONS = ["north", "south", "east", "west"]
DISTRACTORS = ["duration_planned", "temperature", "queue_depth"]
NUMBER_FORMAT = ("Answer with one number rounded to 3 decimals. "
                 "End your reply with a line 'ANSWER: <number>'; that line is what gets graded.")
LABEL_FORMAT = ("Answer with exactly one of: A_higher, B_higher, no_difference. "
                "End your reply with a line 'ANSWER: <label>' holding the label and nothing else; "
                "that line is what gets graded.")

MATHBENCH_DEFAULT = "/data/MathBench"
MB_BUDGET = {"experiments": 10, "steps": 30_000}  # fits under the broker's 12 tool calls
MB_FULL_SCORE = 4.0          # portfolio score that maps to reward 1.0
MB_TRIES = 200               # rules sampled per seed while looking for a rich world
_MB = None                   # (lab module, generate module) once imported
_SPECS: dict = {}            # seed -> world spec; profiling a rule costs ~0.4 s
_LABS: dict = {}             # task id -> the Lab of its current episode


# --- public interface -------------------------------------------------------

def make_task(seed, out_dir, kind=None, family="generated", *,
              noise_rows=None, distractor_cols=None, mixed_units=None):
    """Build one task. Knobs left as None are drawn from the seed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if family == "mathbench":
        return _mathbench_task(seed)
    if family != "generated":
        raise ValueError(f"unknown family {family!r}; use 'generated' or 'mathbench'")
    kind = KINDS[seed % len(KINDS)] if kind is None else kind
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; use one of {KINDS}")
    for attempt in range(MAX_ATTEMPTS):
        # Reseeding by attempt keeps rejection deterministic per seed.
        rng = np.random.default_rng([seed, KINDS.index(kind), attempt])
        # Knob defaults are always drawn, so overriding one never shifts the data.
        drawn = {"noise_rows": int(rng.choice([0, 6, 15])),
                 "distractor_cols": int(rng.integers(0, len(DISTRACTORS) + 1)),
                 "mixed_units": bool(rng.random() < 0.5)}
        built = _BUILD[kind](rng)
        if built is not None:
            break
    else:
        raise RuntimeError(f"no valid {kind} task for seed {seed} in {MAX_ATTEMPTS} attempts")
    cols, ask, reference, extra = built
    given = {"noise_rows": noise_rows, "distractor_cols": distractor_cols, "mixed_units": mixed_units}
    knobs = {k: drawn[k] if v is None else v for k, v in given.items()}
    header, rows = _mess(rng, cols, **knobs)
    files = {"runs.csv": (header, rows), **extra}
    for name, (head, body) in files.items():
        with open(out_dir / name, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows([head, *body])
    intro = ["`runs.csv` has one row per run."]
    intro.append("`duration` is recorded in the unit named in `unit` (`s` or `ms`); convert every "
                 "value to seconds first." if knobs["mixed_units"] else "`duration` is in seconds.")
    if knobs["noise_rows"]:
        intro.append("Rows whose `status` is `void` are failed recordings; exclude them.")
    return {"id": f"gen-{kind}-{seed}", "family": "generated",
            "question": " ".join(intro + [ask]), "files": list(files), "history": [],
            "reference": reference}


def grade(task, text):
    """Reward for a final answer: 0.0 or 1.0 (generated), 0..1 (mathbench)."""
    if task.get("family") == "mathbench":
        lab = _LABS.get(task["id"])
        if lab is None or lab.result is None:
            return 0.0  # nothing submitted through the tool: prose earns nothing
        score = lab.result["scorecard"]["score"]
        return float(min(1.0, max(0.0, score / MB_FULL_SCORE)))
    ref = task["reference"]
    text = clean(text)
    lines = ANSWER_RE.findall(text)
    if isinstance(ref, str):
        return float(bool(lines) and lines[-1].strip(EDGE).lower() == ref.lower())
    # The first number on the ANSWER line, so "3.21 (rounded to 3 decimals)" is not read as 3.
    found = numbers(lines[-1])[:1] if lines else numbers(text)[-1:]
    return float(bool(found) and abs(found[0][0] - ref) <= REL_TOL * abs(ref))


def split_seeds(split, n, generation=0):
    """Seeds for a split. Ranges are disjoint; train and val move to a fresh
    block each generation, test never moves so `rsi test` stays comparable."""
    if split not in SPLIT_START:
        raise ValueError(f"unknown split {split!r}; use one of {list(SPLIT_START)}")
    if not 0 <= n <= SPLIT_BLOCK or not 0 <= generation < 1000:
        raise ValueError(f"n must be 0..{SPLIT_BLOCK} and generation 0..999")
    start = SPLIT_START[split] + (0 if split == "test" else generation * SPLIT_BLOCK)
    return list(range(start, start + n))


def toolset(task):
    """None for generated tasks (the assistant's own tools apply); for mathbench,
    (schemas, call) over a fresh Lab. call(name, args) -> str and never raises."""
    if task.get("family") != "mathbench":
        return None
    lab = _mathbench()[0].Lab(task["reference"]["spec"], budget=MB_BUDGET)
    _LABS[task["id"]] = lab
    fns = {"run_experiment": lab.run_experiment, "define_concept": lab.define_concept,
           "submit_laws": lab.submit_laws}

    def call(name, args):
        if name not in fns:
            return f"ERROR unknown tool {name!r}; available: {', '.join(fns)}"
        if not isinstance(args, dict):
            return f"ERROR {name}: arguments must be an object"
        try:
            if isinstance(args.get("doc"), str):  # models often send nested JSON as a string
                args = {**args, "doc": json.loads(args["doc"])}
            out = fns[name](**args)
        except Exception as e:  # the Lab is not ours; any failure is data for the model
            return f"ERROR {name}: {type(e).__name__}: {e}"
        if isinstance(out, dict) and "error" in out:
            return f"ERROR {name}: {out['error']}"
        return json.dumps(out)

    return MB_SCHEMAS, call


def guess(task):
    """The no-tool null baseline: the best answer available without the data."""
    if task.get("family") == "mathbench":
        return "No experiments were run, so no laws are submitted."
    return "ANSWER: no_difference" if isinstance(task["reference"], str) else "ANSWER: 0"


def perm_pvalue(a, b, seed=0):
    """Two-sided permutation p-value for the difference in means, N_SHUFFLES
    shuffles: the procedure the group_diff reference is defined by."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    shuffled = np.random.default_rng(seed).permuted(np.tile(pooled, (N_SHUFFLES, 1)), axis=1)
    diffs = shuffled[:, :len(a)].mean(axis=1) - shuffled[:, len(a):].mean(axis=1)
    return float((np.abs(diffs) >= observed - 1e-12).mean())


# --- generated family -------------------------------------------------------
# Each builder returns (clean columns, question sentence, reference, extra
# files) or None to reject the draw. The reference is computed from the clean
# values exactly as written (4 decimals), before _mess adds the difficulty.

def _two_arms(rng, effect):
    na, nb = (int(x) for x in rng.integers(30, 81, 2))
    sd = rng.uniform(1.5, 3.0)
    a = np.round(rng.normal(12, sd, na), 4)
    b = np.round(rng.normal(12 + effect * sd, sd, nb), 4)
    return {"arm": ["A"] * na + ["B"] * nb, "duration": np.concatenate([a, b])}, a, b


def _group_diff(rng):
    # A third of draws are true nulls, so "there is always an effect" fails.
    effect = 0.0 if rng.random() < 0.36 else rng.choice([-1, 1]) * rng.uniform(0.5, 1.1)
    cols, a, b = _two_arms(rng, effect)
    p = perm_pvalue(a, b)
    if P_BAND[0] <= p <= P_BAND[1] or a.mean() == b.mean():
        return None
    label = "no_difference" if p >= ALPHA else ("A_higher" if a.mean() > b.mean() else "B_higher")
    ask = ("Which `arm` (A or B) has the higher mean duration, and is the difference statistically "
           f"significant? Use a two-sided permutation test on the difference in means ({N_SHUFFLES:,} "
           f"shuffles, alpha {ALPHA}). If it is not significant, the answer is no_difference. " + LABEL_FORMAT)
    return cols, ask, label, {}


def _effect_size(rng):
    cols, a, b = _two_arms(rng, rng.choice([-1, 1]) * rng.uniform(0.15, 1.3))
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = round(float((b.mean() - a.mean()) / pooled), 3)
    ask = ("What is Cohen's d for duration between the two values of `arm`, defined as (mean of B minus "
           "mean of A) divided by the pooled standard deviation, where pooled SD = sqrt(((nA-1)*varA + "
           "(nB-1)*varB) / (nA+nB-2)) with sample variances (n-1 denominator)? Keep the sign. " + NUMBER_FORMAT)
    return (cols, ask, d, {}) if abs(d) >= MIN_ABS_REF else None


def _ranks(x):
    """Average ranks (ties share the mean rank), as Spearman's rho requires."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def _correlation(rng):
    n = int(rng.integers(40, 121))
    rho = rng.choice([-1, 1]) * rng.uniform(0.15, 0.9)
    load = np.round(rng.normal(50, 10, n), 2)
    duration = np.round(12 + 2 * (rho * (load - 50) / 10 + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)), 4)
    spearman = bool(rng.random() < 0.5)
    x, y = (_ranks(duration), _ranks(load)) if spearman else (duration, load)
    r = round(float(np.corrcoef(x, y)[0, 1]), 3)
    name = "Spearman rank correlation (average ranks for ties)" if spearman else "Pearson correlation"
    ask = f"What is the {name} between `duration` and `load`? Keep the sign. " + NUMBER_FORMAT
    return ({"load": load, "duration": duration}, ask, r, {}) if abs(r) >= MIN_ABS_REF else None


def _filtered_agg(rng):
    # The filter spans two files, so no single-table stats tool can answer it.
    n_sites, n = int(rng.integers(8, 13)), int(rng.integers(150, 301))
    site_ids = [f"S{i + 1:02d}" for i in range(n_sites)]
    region = rng.choice(REGIONS, n_sites)
    capacity = rng.integers(10, 200, n_sites)
    site = rng.integers(0, n_sites, n)
    load = rng.integers(0, 100, n)
    duration = np.round(np.exp(np.log(20) + rng.normal(0, 0.4, n_sites)[site] + rng.normal(0, 0.5, n)), 4)
    want, floor = str(region[rng.integers(n_sites)]), int(rng.integers(20, 61))
    op = str(rng.choice(["mean", "median", "sum"]))
    chosen = duration[(region[site] == want) & (load >= floor)]
    if len(chosen) < 8:
        return None
    value = round(float({"mean": np.mean, "median": np.median, "sum": np.sum}[op](chosen)), 3)
    ask = (f"`sites.csv` gives each site's region. What is the {op} of duration, in seconds, over the "
           f"runs whose site's region is `{want}` and whose `load` is at least {floor}? "
           + ("(For an even count, the median is the mean of the two middle values.) " if op == "median" else "")
           + NUMBER_FORMAT)
    sites = (["site_id", "region", "capacity"],
             [[s, str(r), str(int(c))] for s, r, c in zip(site_ids, region, capacity)])
    cols = {"site_id": [site_ids[i] for i in site], "load": load, "duration": duration}
    return cols, ask, value, {"sites.csv": sites}


_BUILD = {"group_diff": _group_diff, "effect_size": _effect_size,
          "correlation": _correlation, "filtered_agg": _filtered_agg}


def _mess(rng, cols, noise_rows, distractor_cols, mixed_units):
    """Apply the difficulty knobs to clean columns -> (header, rows of str).
    Clean rows keep their values; only extra rows, columns and units are added."""
    cols = {k: np.asarray(v) for k, v in cols.items()}
    n = len(cols["duration"])
    for i, name in enumerate(DISTRACTORS[:distractor_cols]):
        # The first distractor looks like the target: same scale, unrelated rows.
        cols[name] = np.round(rng.permutation(cols["duration"]) * rng.uniform(0.7, 1.3, n), 4) if i == 0 \
            else np.round(rng.normal(40 * i, 8 * i, n), 2)
    if noise_rows:
        # Void rows reuse real values per column but inflate duration 3-8x, so
        # forgetting to exclude them visibly moves every statistic.
        extra = {k: rng.choice(v, noise_rows) for k, v in cols.items()}
        extra["duration"] = np.round(extra["duration"] * rng.uniform(3, 8, noise_rows), 4)
        cols = {k: np.concatenate([v, extra[k]]) for k, v in cols.items()}
        cols["status"] = np.array(["ok"] * n + ["void"] * noise_rows)
    total = n + noise_rows
    duration = [f"{v:.4f}" for v in cols["duration"]]
    if mixed_units:
        in_ms = rng.random(total) < 0.4
        in_ms[:2] = [True, False]  # both units always occur
        # 4 decimals in seconds is exactly 1 decimal in milliseconds: no precision lost.
        duration = [f"{v * 1000:.1f}" if ms else s for v, s, ms in zip(cols["duration"], duration, in_ms)]
        cols["unit"] = np.where(in_ms, "ms", "s")
    order = rng.permutation(total)  # void rows end up scattered, not at the bottom
    names = [k for k in cols if k != "duration"]
    header = ["run_id"] + names[:1] + ["duration"] + names[1:]
    if "unit" in cols:  # keep the unit next to the value it describes
        header.remove("unit")
        header.insert(header.index("duration") + 1, "unit")
    text = {k: [str(x) for x in v] for k, v in cols.items()}
    text["duration"] = duration
    return header, [[str(i + 1)] + [text[k][j] for k in header[1:]] for i, j in enumerate(order)]


# --- MathBench family -------------------------------------------------------

def _mathbench():
    """Import MathBench's `lab` and `generate` by path, without writing to it."""
    global _MB
    if _MB is None:
        root = str(Path(os.environ.get("MATHBENCH_DIR", MATHBENCH_DEFAULT)).resolve())
        saved = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # the repo is read-only to us: no __pycache__ writes
        sys.path.insert(0, root)
        try:
            # ponytail: MathBench uses top-level module names (lab, grade, world,
            # value, ...), which stay in sys.modules; move to importlib with a
            # private namespace if RSI ever grows a module with one of those names.
            import generate as mb_generate
            import lab as mb_lab
            if str(Path(mb_lab.__file__).resolve().parent) != root:
                raise ImportError(f"`lab` resolved to {mb_lab.__file__}")
        except Exception as e:
            raise RuntimeError(f"cannot import MathBench from MATHBENCH_DIR={root}: {e}") from e
        finally:
            sys.path.remove(root)
            sys.dont_write_bytecode = saved
        _MB = (mb_lab, mb_generate)
    return _MB


def _mathbench_task(seed):
    lab, gen = _mathbench()
    if seed not in _SPECS:
        # MathBench's own procedure: sample rules from the seed, keep the first
        # whose dynamics its profiler calls interesting (about 1 rule in 20).
        for birth, survive in gen.sample_rules(MB_TRIES, seed):
            spec = gen.make_spec(birth, survive)
            if gen.profile(spec)["class"] == "interesting":
                _SPECS[seed] = spec
                break
        else:
            raise RuntimeError(f"no interesting MathBench world for seed {seed} in {MB_TRIES} rules")
    spec = _SPECS[seed]
    public = lab.Lab(spec).public_spec()  # grid geometry only; the rule stays hidden
    question = MB_QUESTION.format(spec=json.dumps(public), budget=json.dumps(MB_BUDGET))
    return {"id": f"mb-{seed}", "family": "mathbench", "question": question, "files": [],
            "history": [], "reference": {"spec": spec}}


MB_QUESTION = """You are studying an unknown universe: a 2-state cellular automaton on a torus grid \
whose update rule is hidden. World: {spec}. Lab budget: {budget}.
Use run_experiment to observe it, then call submit_laws exactly once with {{"laws": [...]}}. Only the \
submit_laws call is scored; text in your final reply is not. Each law needs a unique "id", a "type" and a \
"confidence" c in [0, 1]. Laws are checked on initial conditions you cannot run. A new true fact pays \
2c - c^2, a false law costs c^2, a repeat of a fact already claimed pays 0, and a law too loose to be \
informative pays 0.
Law types:
- INVARIANT: {{"init": {{"class": "pattern", "cells": [[r, c], ...]}}, "quantity": "live_count" or \
"live_density"}}: the quantity never changes from this start.
- PATTERN_PERSISTS: {{"cells": [[r, c], ...], "period": P, "displacement": [dr, dc]}}: the pattern \
reappears every P steps, shifted by the displacement.
- STAT_LAW: {{"init": {{"class": "random", "p": [lo, hi]}}, "quantity": ..., "t": T, "value": V, "tol": TOL}}: \
from random starts the mean quantity at step T is V within TOL.
- COLLISION: {{"a": {{"cells": [...]}}, "b": {{"cells": [...]}}, "geometry": {{"offset": [dr, dc]}}, \
"products": [{{"cells": [...]}}, ...]}}: b placed at that offset from a settles into exactly these \
products (an empty list means both vanish).
A "cells" value may be a concept name you defined with define_concept."""

_CELLS = {"type": "array", "items": {"type": "array", "items": {"type": "integer"},
                                     "minItems": 2, "maxItems": 2}}


def _schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


MB_SCHEMAS = [
    _schema("run_experiment",
            "Step the world forward from a start and return snapshots (live_count, live_density, and "
            "live_cells when few enough). init is {\"class\": \"random\", \"p\": [lo, hi], \"seed\": int} or "
            "{\"class\": \"pattern\", \"cells\": [[r, c], ...] or a concept name, \"at\": [r, c]}. t is 1..5000. "
            "record lists up to 64 times within [0, t] to snapshot (default: 0 and t). Costs one "
            "experiment and t steps of the budget.",
            {"init": {"type": "object"}, "t": {"type": "integer"},
             "record": {"type": "array", "items": {"type": "integer"}}}, ["init", "t"]),
    _schema("define_concept", "Name a cell pattern so inits and laws can refer to it by name.",
            {"name": {"type": "string"}, "cells": _CELLS}, ["name", "cells"]),
    _schema("submit_laws", "Final and one-shot: submit {\"laws\": [...]} for grading. Returns the verdicts.",
            {"doc": {"type": "object"}}, ["doc"]),
]
