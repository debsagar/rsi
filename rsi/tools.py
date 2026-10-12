"""Tool schemas and dispatch for the assistant.

`TOOLS` is the hand-written catalogue; `call_tool(ctx, name, args)` runs one
already-parsed call and returns a string.

Invariants:
- `call_tool` never raises. Bad arguments, a missing file or column, a
  non-numeric column and a crash inside a tool all come back as "ERROR ..."
  so the model can read the problem and try again.
- A file argument can only name something inside `ctx["files_dir"]`.
- Every simulated result begins with "SIMULATED:" and its catalogue entry has
  `"simulated": True`; the broker copies that flag onto the event.
- Nothing here shortens output. Clipping is the broker's job, because only
  the broker can record that it happened.
- Statistics are seeded, so the same call on the same file gives the same text.
- The python tool's interpreter belongs to one episode: it lives in the `ctx`
  the caller made for that episode (`ctx["python"]`, a PythonSession) and
  nowhere else, so no state crosses episodes. Whoever makes it closes it.
"""
import contextlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats

SANDBOX_IMAGE = "rsi-sandbox"
PYTHON_LIMIT_S = 60
PYLOOP = Path(__file__).with_name("pyloop.py")
# The loop keeps at most 100,000 characters each of stdout, stderr and traceback
# (pyloop.CAP). JSON can spend 12 bytes on one character, so a reply stays under this.
PYTHON_REPLY_BYTES = 4_000_000
READ_CHARS = 3500        # default read_file length; under the broker's 4,000 clip with room for the header
PERMUTATIONS = 10_000
BOOTSTRAPS = 5_000
SEED = 0
TABLE_SEPARATORS = {".csv": ",", ".tsv": "\t"}

_FILE = {"type": "string", "description": "file name as shown by list_files"}

TOOLS = {
    "list_files": {
        "description": "List the files attached to this session: name, size in bytes and type.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "simulated": False,
    },
    "read_file": {
        "description": "Read a slice of a text file. Returns `length` characters starting at `offset`.",
        "parameters": {"type": "object", "properties": {
            "name": _FILE,
            "offset": {"type": "integer", "description": "first character to return; defaults to 0"},
            "length": {"type": "integer", "description": f"characters to return; defaults to {READ_CHARS}"},
        }, "required": ["name"]},
        "simulated": False,
    },
    "describe_table": {
        "description": "Summarise a CSV or TSV: row count, then each column's type, missing values and summary statistics.",
        "parameters": {"type": "object", "properties": {"file": _FILE}, "required": ["file"]},
        "simulated": False,
    },
    "compare_groups": {
        "description": "Compare a numeric column between two groups: mean difference (a minus b), a two-sided "
                       f"permutation p-value ({PERMUTATIONS:,} shuffles) and a bootstrap 95% interval for the difference.",
        "parameters": {"type": "object", "properties": {
            "file": _FILE,
            "value": {"type": "string", "description": "numeric column to compare"},
            "group": {"type": "string", "description": "column holding the group labels"},
            "a": {"type": "string", "description": "first group label; optional when the column has exactly two labels"},
            "b": {"type": "string", "description": "second group label; optional when the column has exactly two labels"},
        }, "required": ["file", "value", "group"]},
        "simulated": False,
    },
    "correlate": {
        "description": "Pearson and Spearman correlation between two numeric columns, with a bootstrap 95% interval for Pearson's r.",
        "parameters": {"type": "object", "properties": {
            "file": _FILE, "x": {"type": "string"}, "y": {"type": "string"},
        }, "required": ["file", "x", "y"]},
        "simulated": False,
    },
    "python": {
        "description": "Run Python (numpy, pandas, scipy) in one interpreter that lasts the whole task, like a notebook: "
                       "variables, imports and files persist between calls, and a last line that is an expression is "
                       "displayed. The session's files are read-only input under /data; /work is the working directory, for "
                       f"scratch files. No network. {PYTHON_LIMIT_S} s limit per call; a call that runs longer resets the "
                       "interpreter, losing all variables and /work files.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        "simulated": False,
    },
    "arxiv_search": {
        "description": "Search arXiv for papers. SIMULATED: returns fixture records, not real search results.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "simulated": True,
    },
    "citation_check": {
        "description": "Check whether a citation exists and says what is claimed. SIMULATED: returns a fixture verdict.",
        "parameters": {"type": "object", "properties": {"citation": {"type": "string"}}, "required": ["citation"]},
        "simulated": True,
    },
}

ARXIV_FIXTURES = [
    {"id": "fixture-0001", "title": "Permutation tests for small samples", "year": 2021},
    {"id": "fixture-0002", "title": "Bootstrap intervals in practice", "year": 2019},
    {"id": "fixture-0003", "title": "Reporting effect sizes with uncertainty", "year": 2023},
]


class ToolError(Exception):
    """A problem the model can fix. `call_tool` turns it into an "ERROR ..." string."""


def tool_schemas(tools: dict = TOOLS) -> list[dict]:
    """OpenAI-compatible chat-completions `tools` list."""
    return [{"type": "function", "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}}
            for name, spec in tools.items()]


def _file(ctx: dict, name) -> Path:
    if not isinstance(name, str) or not name:
        raise ToolError("the file name must be a non-empty string")
    root = Path(ctx["files_dir"]).resolve()
    path = (root / name).resolve()
    # One message for "outside the folder" and "not there": the model needs the
    # same next step either way, and it should not learn what exists elsewhere.
    if not path.is_relative_to(root) or not path.is_file():
        raise ToolError(f"no file named {name!r} in this session; call list_files")
    return path


def _table(ctx: dict, name) -> pd.DataFrame:
    path = _file(ctx, name)
    if path.suffix.lower() not in TABLE_SEPARATORS:
        raise ToolError(f"{name!r} is not a table; expected one of {sorted(TABLE_SEPARATORS)}")
    try:
        return pd.read_csv(path, sep=TABLE_SEPARATORS[path.suffix.lower()])
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ToolError(f"could not parse {name!r} as a table: {e}") from None


def _column(df: pd.DataFrame, col, file: str) -> pd.Series:
    if col not in df.columns:
        raise ToolError(f"no column {col!r} in {file}; columns are {list(df.columns)}")
    missing = int(df[col].isna().sum())
    if missing:
        # Dropping rows here would change the answer without the trajectory
        # showing it, so the model is told and has to decide.
        raise ToolError(f"column {col!r} has {missing} missing value(s); clean it with the python tool, "
                        "then compute the statistic there")
    return df[col]


def _numeric(df: pd.DataFrame, col, file: str) -> np.ndarray:
    series = _column(df, col, file)
    if not pd.api.types.is_numeric_dtype(series):
        raise ToolError(f"column {col!r} is not numeric (dtype {series.dtype}); convert it with the python tool first")
    return series.to_numpy(dtype=float)


def _int(args: dict, key: str, default: int) -> int:
    value = args.get(key, default)
    if type(value) is not int or value < 0:
        raise ToolError(f"{key} must be a non-negative integer")
    return value


def _list_files(ctx: dict, args: dict) -> str:
    root = Path(ctx["files_dir"])
    files = sorted(p for p in root.iterdir() if p.is_file()) if root.is_dir() else []
    if not files:
        return "No files are attached to this session."
    return "\n".join(f"{p.name}\t{p.stat().st_size} bytes\t{p.suffix.lstrip('.').lower() or 'unknown'}" for p in files)


def _read_file(ctx: dict, args: dict) -> str:
    path = _file(ctx, args["name"])
    offset, length = _int(args, "offset", 0), _int(args, "length", READ_CHARS)
    # ponytail: reads the whole file to slice it; fine under the 25 MB upload cap.
    text = path.read_text(errors="replace")
    end = min(offset + length, len(text))
    return f"[{path.name}: characters {offset}-{end} of {len(text)}]\n{text[offset:end]}"


def _describe_table(ctx: dict, args: dict) -> str:
    df = _table(ctx, args["file"])
    lines = [f"{args['file']}: {len(df)} rows, {len(df.columns)} columns"]
    for col in df.columns:
        series = df[col]
        head = f"- {col}: {series.dtype}, {int(series.notna().sum())} values, {int(series.isna().sum())} missing"
        if pd.api.types.is_numeric_dtype(series) and series.notna().any():
            lines.append(f"{head}; mean={series.mean():.6g}, std={series.std():.6g}, min={series.min():.6g}, "
                         f"median={series.median():.6g}, max={series.max():.6g}")
        else:
            top = ", ".join(f"{value!r} ({count})" for value, count in series.value_counts().head(5).items())
            lines.append(f"{head}; {series.nunique()} distinct; most common: {top}")
    return "\n".join(lines)


def _mean_diff(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def _compare_groups(ctx: dict, args: dict) -> str:
    file, value, group = args["file"], args["value"], args["group"]
    df = _table(ctx, file)
    values = _numeric(df, value, file)
    labels = _column(df, group, file).astype(str).to_numpy()
    found = sorted(set(labels))
    a, b = args.get("a"), args.get("b")
    if a is None and b is None:
        if len(found) != 2:
            raise ToolError(f"column {group!r} has {len(found)} distinct labels, not 2; pass a and b. Labels: {found[:20]}")
        a, b = found
    for label in (a, b):
        if str(label) not in found:
            raise ToolError(f"no rows where {group!r} is {label!r}; labels are {found[:20]}")
    xa, xb = values[labels == str(a)], values[labels == str(b)]
    if a == b or min(len(xa), len(xb)) < 2:
        raise ToolError(f"need two different groups with at least 2 rows each; got {a!r} ({len(xa)}) and {b!r} ({len(xb)})")
    # scipy enumerates every relabelling instead of sampling when there are
    # fewer than PERMUTATIONS of them, which is exact and still deterministic.
    perm = stats.permutation_test((xa, xb), _mean_diff, vectorized=True, n_resamples=PERMUTATIONS,
                                  alternative="two-sided", random_state=SEED)
    boot = stats.bootstrap((xa, xb), _mean_diff, vectorized=True, n_resamples=BOOTSTRAPS, batch=100,
                           method="percentile", random_state=SEED)
    return json.dumps({
        "group_a": str(a), "group_b": str(b), "n_a": len(xa), "n_b": len(xb),
        "mean_a": round(float(xa.mean()), 6), "mean_b": round(float(xb.mean()), 6),
        "mean_diff": round(float(perm.statistic), 6), "mean_diff_is": "mean_a - mean_b",
        "p_value": round(float(perm.pvalue), 6), "p_value_is": f"two-sided permutation test, up to {PERMUTATIONS} shuffles",
        "ci95": [round(float(boot.confidence_interval.low), 6), round(float(boot.confidence_interval.high), 6)],
        "ci95_is": f"percentile bootstrap of the mean difference, {BOOTSTRAPS} resamples",
    }, indent=1)


def _pearson(x, y):
    return stats.pearsonr(x, y).statistic


def _correlate(ctx: dict, args: dict) -> str:
    file = args["file"]
    df = _table(ctx, file)
    x, y = _numeric(df, args["x"], file), _numeric(df, args["y"], file)
    if len(x) < 3:
        raise ToolError(f"need at least 3 rows to correlate; {file} has {len(x)}")
    for col, data in ((args["x"], x), (args["y"], y)):
        if np.ptp(data) == 0:
            raise ToolError(f"column {col!r} is constant, so correlation is undefined")
    pearson, spearman = stats.pearsonr(x, y), stats.spearmanr(x, y)
    # ponytail: a resample that draws one row every time has no correlation and
    # would make the interval NaN; that needs a table of a handful of rows.
    boot = stats.bootstrap((x, y), _pearson, paired=True, vectorized=False, n_resamples=BOOTSTRAPS,
                           method="percentile", random_state=SEED)
    return json.dumps({
        "n": len(x),
        "pearson_r": round(float(pearson.statistic), 6), "pearson_p": round(float(pearson.pvalue), 6),
        "spearman_rho": round(float(spearman.statistic), 6), "spearman_p": round(float(spearman.pvalue), 6),
        "pearson_ci95": [round(float(boot.confidence_interval.low), 6), round(float(boot.confidence_interval.high), 6)],
        "pearson_ci95_is": f"percentile bootstrap over rows, {BOOTSTRAPS} resamples",
    }, indent=1)


class PythonSession:
    """One sandbox container holding one Python interpreter, for one episode.

    Started on the first `run`, removed by `close`. After a timeout or a dead
    interpreter the container is removed and the next `run` starts a new one.
    Limits: no network, 1 GB memory (which also bounds /work and /tmp, both
    in-memory), 1 CPU, 128 processes, read-only root filesystem, no capabilities.
    One `run` at a time. `close` may come from another thread while a `run` is
    in flight (the broker does that at the run limit): that `run` ends with an
    error and no container is started afterwards.
    A reply is accepted only if it carries the id sent with the request; see
    rsi/pyloop.py for what the code under test can and cannot do to the channel.
    """

    def __init__(self, files_dir):
        self.files = Path(files_dir).resolve()
        self.name = self.proc = self.stderr = None
        self.closed = False
        self._lock = threading.RLock()

    def _start(self) -> None:
        self.files.mkdir(parents=True, exist_ok=True)
        self.name = f"rsi-py-{uuid4().hex[:12]}"  # random, so parallel episodes and other processes cannot collide
        # No --init: pyloop must be PID 1, the only process holding the stream to the host (see rsi/pyloop.py).
        # ponytail: /work is in-memory and dies with the container, so a timeout
        # loses its files. A host folder would survive but has no size limit.
        cmd = ["docker", "run", "--rm", "-i", "--pull", "never", "--name", self.name, "--network", "none",
               "--memory", "1g", "--cpus", "1", "--pids-limit", "128", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
               "--tmpfs", "/tmp:size=64m", "--tmpfs", "/work:size=512m,mode=1777", "-w", "/work",
               "-v", f"{self.files}:/data:ro", "-v", f"{PYLOOP}:/pyloop.py:ro",
               SANDBOX_IMAGE, "python", "/pyloop.py", str(PYTHON_LIMIT_S + 30)]
        # stderr goes to a file of its own: a docker warning must never be read as a
        # reply, and a docker error ("no such image") is quoted from it when no reply comes.
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr)

    def _stop(self) -> str:
        """Remove the container. -> the end of what docker and the loop wrote to stderr."""
        with self._lock:
            if self.proc is None:
                return ""
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self.proc.kill()
            self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout):
                with contextlib.suppress(OSError):  # closing stdin flushes it into a dead pipe
                    stream.close()
            self.stderr.seek(0)
            said = self.stderr.read().decode(errors="replace").strip()[-300:]
            self.stderr.close()
            self.proc = None
            return said

    def close(self) -> None:
        """Remove the container for good. Safe to call twice, or when it never started."""
        with self._lock:
            self.closed = True
            self._stop()

    def run(self, code: str) -> str:
        with self._lock:
            if self.closed:
                raise ToolError("the python tool is closed: this run has ended")
            if self.proc is None:
                try:
                    self._start()
                except FileNotFoundError:
                    raise ToolError("docker is not installed, so the python tool is unavailable") from None
            proc, name = self.proc, self.name
        timed_out = threading.Event()

        def expire():
            timed_out.set()
            subprocess.run(["docker", "kill", name], capture_output=True)  # ends the readline below

        rid, reply = uuid4().hex, None
        timer = threading.Timer(PYTHON_LIMIT_S, expire)
        timer.start()
        try:
            proc.stdin.write(json.dumps({"id": rid, "code": code}).encode() + b"\n")
            proc.stdin.flush()
            # Bounded reads: a flood cannot fill host memory. A line that is not the
            # reply to this request is skipped, never shown as this call's output.
            while line := proc.stdout.readline(PYTHON_REPLY_BYTES):
                try:
                    got = json.loads(line)
                    if got["id"] == rid:
                        parts = [got["stdout"], got["stderr"], got["error"] or ""]
                        reply = got
                        break
                except (ValueError, TypeError, KeyError, RecursionError):
                    pass
        except (OSError, ValueError):  # the pipe broke, or close() shut it under this call
            pass
        finally:
            timer.cancel()
            timer.join()
        gone = "The interpreter was restarted: all variables, imports and files in /work are gone; /data is unchanged."
        # What was read decides, not the timer: a reply that arrived is a finished call even if the timer fired after it.
        if reply is None:
            said = self._stop()
            if timed_out.is_set():
                raise ToolError(f"timeout ({PYTHON_LIMIT_S}s). {gone} Redo what you need in smaller steps.")
            raise ToolError(f"the interpreter stopped ({said or 'it ran out of memory or the code ended the process'}). {gone}")
        if reply.get("dropped"):
            parts.append(f"[output capped: {reply['dropped']} more characters were printed and not kept]\n")
        if timed_out.is_set():  # the timer killed the container just after this reply was written
            self._stop()
            parts.append(f"[this call ended at the {PYTHON_LIMIT_S}s limit. {gone}]\n")
        return "".join(part if part.endswith("\n") or not part else part + "\n" for part in parts) or "(no output)\n"


def _python(ctx: dict, args: dict) -> str:
    code = args["code"]
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a non-empty string")
    return ctx["python"].run(code)


def _arxiv_search(ctx: dict, args: dict) -> str:
    return (f"SIMULATED: fixture records, not real search results for {args['query']!r}. Do not cite them.\n"
            + json.dumps(ARXIV_FIXTURES, indent=1))


def _citation_check(ctx: dict, args: dict) -> str:
    return (f"SIMULATED: fixture verdict, nothing was looked up for {args['citation']!r}.\n"
            + json.dumps({"verdict": "unverified", "reason": "citation checking is simulated in this build"}))


_DISPATCH = {"list_files": _list_files, "read_file": _read_file, "describe_table": _describe_table,
             "compare_groups": _compare_groups, "correlate": _correlate, "python": _python,
             "arxiv_search": _arxiv_search, "citation_check": _citation_check}


def call_tool(ctx: dict, name: str, args: dict) -> str:
    """Dispatch one already-parsed tool call. Always returns a string."""
    if name not in _DISPATCH:
        return f"ERROR unknown tool {name!r}; expected one of {list(TOOLS)}"
    if not isinstance(args, dict):
        return f"ERROR {name} arguments must be a JSON object, got {type(args).__name__}"
    missing = [key for key in TOOLS[name]["parameters"]["required"] if key not in args]
    if missing:
        return f"ERROR {name} is missing required argument(s): {', '.join(missing)}"
    try:
        return _DISPATCH[name](ctx, args)
    except ToolError as e:
        return f"ERROR {e}"
    except Exception as e:  # the never-raise boundary: a tool bug is data for the model, not a dead run
        return f"ERROR {name} failed: {type(e).__name__}: {e}"
