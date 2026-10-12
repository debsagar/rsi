"""Reading an answer: the one ANSWER-line pattern, the one number parser, and the
graders for the two external benchmarks. `rsi.tasks` (generated family) and
`rsi.evals` both read replies through this module, so no two graders disagree
about what a number or an answer line is. The table of deliberate departures
from the benchmark authors' rules is in the `rsi.evals` docstring.

Invariants:
- If a reply has any ANSWER line, only the last one is graded.
- A number is finite: "nan", "inf" and overflowing exponents never parse.
- Nothing fuzzy: no partial credit, no reordering, no nearest-string matching.
- `feedback` and `shape` are what may be said about an answer to the model that
  rewrites the harness. Neither states a reference value, how close an answer
  was, or which sub-answer is right: two wrong replies written in the same form
  get the same feedback, and `shape` keeps punctuation and item names only.
"""
import math
import re

GRADER_VERSION = "2"         # bump on any change to what scores; "1" is every folder written before the field existed
DA_TOL = 1e-6                # eval_closed_form.py is_equal
QR_TOL = 0.03                # eval.py error_scale
QR_ZERO_TOL = 1e-6           # a gold of exactly 0 has empty strict relative bounds
EDGE = " \t*`_\"'()[]{}<>.,;:!"  # markdown, quotes, brackets and trailing punctuation around a one-token answer
# Line-anchored; the colon may sit inside or outside markdown bold: `**ANSWER:** x`, `**ANSWER**: x`, `> ANSWER: x`.
ANSWER_RE = re.compile(r"^[ \t>#*_`]*ANSWER[ \t*_`]*:(.*)$", re.I | re.M)
DA_ITEM_RE = re.compile(r"@(\w+)\[(.*?)\]")       # eval_closed_form.py extract_format
SHAPE_RE = re.compile(r"(@\w+|\bANSWER(?=[ \t*_`]*:))|\d+|[^\W\d_]+", re.I)
THOUSANDS_RE = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d)")


def _number_re(integer: str):
    # A sign counts only at the start of a token, so the dash in "1-3" or "mean-5.2" is not a minus.
    return re.compile(r"(?<![\w.])([-+]?(?:%s(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(\s?%%)?" % integer)


NUMBER_RE = _number_re(r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)")  # 1,234.5 is one number
PLAIN_RE = _number_re(r"\d+")                                # 1,234 is two: for CSV text, where a comma separates cells


def clean(text) -> str:
    """Markdown escapes (mean\\_fare) removed; typographic minus and en dash become a hyphen."""
    return str(text or "").replace("\\", "").replace("−", "-").replace("–", "-")


def numbers(text: str, pattern=NUMBER_RE) -> list:
    """Every finite number in the text, in order, as (value, written with a % sign)."""
    found = ((float(m.group(1).replace(",", "")), bool(m.group(2))) for m in pattern.finditer(text))
    return [pair for pair in found if math.isfinite(pair[0])]


def whole(text):
    """(value, has %) when the whole text is one number, else None."""
    text = clean(text).strip()
    found = numbers(text) if NUMBER_RE.fullmatch(text) else []
    return found[0] if found else None


# --- DABench ----------------------------------------------------------------------

def _da_text(value) -> str:
    """Presentation only: quotes around a token, letter case, spacing around commas and colons, runs of spaces."""
    text = re.sub(r"(?<!\w)['\"]|['\"](?!\w)", "", str(value)).casefold()
    return " ".join(re.sub(r"\s*([,:])\s*", r"\1", text).split())


def _da_equal(response, label) -> bool:
    if response is None:
        return False
    got, want = whole(_da_text(response)), whole(label)
    if got and want and not got[1] and not want[1]:
        return abs(got[0] - want[0]) < DA_TOL
    return _da_text(response) == _da_text(label)


# --- QRData -----------------------------------------------------------------------

def _qr_number(line: str, reference: str) -> bool:
    gold, gold_percent = whole(reference)
    found = numbers(line)[:1]
    if not found:
        return False
    value, percent = found[0]
    target = gold / 100 if gold_percent else gold       # eval.py: a % divides by 100, on both sides
    reads = [value / 100 if percent else value]
    if gold_percent and not percent:
        reads.append(value / 100)                       # "8.53" for a gold of "8.53%"
    if target == 0:
        return any(abs(read) <= QR_ZERO_TOL for read in reads)
    low, high = sorted([target * (1 - QR_TOL), target * (1 + QR_TOL)])
    return any(low < read < high for read in reads)     # strict, as eval.py


def _qr_choice(line: str, gold: str, options: list) -> bool:
    """The line starts with the gold option as a whole token, optionally in brackets; anything
    after it must be set off by punctuation and must not name another option."""
    pred = " ".join(line.split()).strip(" *`_\"'")
    match = re.match(r"[(\[]?" + re.escape(gold) + r"(?!\w)", pred, re.I)
    if not match:
        return False
    rest = pred[match.end():].rstrip(" *`_\"'")
    if rest and not re.match(r"\s*[)\].:,;]", rest):
        return False
    for other in options:
        if other.lower() == gold.lower():
            continue
        # ponytail: a one-letter option is looked for in upper case only, so the article "a" is not option A;
        # "B. A larger sample..." is therefore refused. Ceiling: a correct answer that explains itself too freely.
        letter = len(other) == 1
        if re.search(r"(?<!\w)" + re.escape(other.upper() if letter else other) + r"(?!\w)", rest, 0 if letter else re.I):
            return False
    return True


# --- both -------------------------------------------------------------------------

def grade_parts(task: dict, text) -> dict:
    """{sub-answer name: bool}. The reply is read from the start of its last ANSWER line."""
    text = clean(text)
    marks = list(ANSWER_RE.finditer(text))
    if task["family"] == "dabench":
        # The benchmark scans the whole reply (after a GPT-3.5 reformat step that is not reproduced: no model
        # touches grading). Reading from the last ANSWER line keeps example items in the reasoning from
        # counting; without such a line the whole reply is scanned, as upstream does.
        got = dict(DA_ITEM_RE.findall(text[marks[-1].start():] if marks else text))
        return {name: _da_equal(got.get(name), value) for name, value in task["reference"]}
    if not marks:
        return {"answer": False}
    line = marks[-1].group(1)
    if task["level"] == "numerical":
        return {"answer": _qr_number(line, task["reference"])}
    return {"answer": _qr_choice(line, task["reference"], task.get("options") or [])}


def grade_external(task: dict, text) -> float:
    """1.0 only if every sub-answer matches (DABench's accuracy by question)."""
    parts = grade_parts(task, text)
    return float(bool(parts) and all(parts.values()))


# --- what may be said about an answer -------------------------------------------------

def answer_line(text) -> str | None:
    """The last ANSWER line of a reply, the one that is graded, or None."""
    marks = list(ANSWER_RE.finditer(clean(text)))
    return marks[-1].group(0).strip() if marks else None


def shape(text) -> str:
    """How an answer is written, without what it says: each run of digits becomes `#` and each run of
    letters `w`; the `ANSWER:` label and `@name` (both given by the question) stay, and so does all punctuation.
    `ANSWER: @outliers[1,234.5]` -> `ANSWER: @outliers[#,#.#]`. Enough to see a harness drop a bracket,
    a quote or a percent sign; not enough to read a value, its size or its precision."""
    return SHAPE_RE.sub(lambda m: m.group(1) or ("#" if m.group(0)[0].isdigit() else "w"), str(text or ""))


def feedback(task: dict, text) -> list:
    """Short statements about a failed reply, each derivable from the question and the form of the reply.
    The one use of the reference is a count of the DABench sub-answers judged wrong among those given."""
    text = clean(text)
    marks = list(ANSWER_RE.finditer(text))
    if not marks:
        out, graded = ["the reply has no ANSWER line"], text
    else:
        out, graded = [], text[marks[-1].start():]
        if not marks[-1].group(1).strip(" \t*_`"):
            out.append("the ANSWER line is empty")
    if task.get("family") == "dabench":
        # The names the question asks for are in its answer format; the wrapper's own example
        # (`@first_name[1.23]`) comes after "End your reply" and is not one of them.
        asked = task["question"].split("Answer format:")[-1].split("End your reply")[0]
        wanted = list(dict.fromkeys(re.findall(r"@(\w+)\[", asked)))
        given = [name for name, _ in DA_ITEM_RE.findall(graded)]
        for name in wanted:
            if name in given:
                if given.count(name) > 1:
                    out.append(f"@{name} is given {given.count(name)} times (the last one is graded)")
            elif re.search(rf"@{name}(?!\w)", graded):
                out.append(f"@{name} is written without a closed [value]")
            else:
                out.append(f"@{name}[...] is missing")
        parts = grade_parts(task, text)
        wrong = sum(not ok for name, ok in parts.items() if name in given)
        if wrong:
            out.append(f"{wrong} of the {sum(name in given for name in parts)} sub-answers given were judged wrong "
                       "(which ones is not shown)")
    line = marks[-1].group(1) if marks else ""
    if "%" in line:
        out.append("the ANSWER line holds a percent sign")
    if THOUSANDS_RE.search(line):
        out.append("the ANSWER line holds a number written with a thousands separator")
    return out
