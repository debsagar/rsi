"""Tool schemas and dispatch for the assistant: `python` and `read_file`.

`TOOLS` is the hand-written catalogue; `call_tool(ctx, name, args)` runs one
already-parsed call and returns a string.

Invariants:
- `call_tool` never raises. Bad arguments, a missing file and a crash inside a
  tool all come back as "ERROR ..." so the model can read the problem and try again.
- A file argument can only name something inside `ctx["files_dir"]`.
- `read_file` returns the head of a file and says so in its last line; anything
  more is the python tool's job.
- Nothing here shortens output silently. Clipping is the broker's job, because
  only the broker can record that it happened.
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

SANDBOX_IMAGE = "rsi-sandbox"
PYTHON_LIMIT_S = 60
PYLOOP = Path(__file__).with_name("pyloop.py")
# The loop keeps at most 100,000 characters each of stdout, stderr and traceback
# (pyloop.CAP). JSON can spend 12 bytes on one character, so a reply stays under this.
PYTHON_REPLY_BYTES = 4_000_000
READ_CHARS = 3500        # under the broker's 4,000 clip with room for the last line

TOOLS = {
    "python": {
        "description": "Run Python (numpy, pandas, scipy) in one interpreter that lasts the whole task, like a notebook: "
                       "variables, imports and files persist between calls, and a last line that is an expression is "
                       "displayed. The session's files are read-only input under /data; /work is the working directory, for "
                       f"scratch files. No network. {PYTHON_LIMIT_S} s limit per call; a call that runs longer resets the "
                       "interpreter, losing all variables and /work files.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
    "read_file": {
        "description": f"Show the first {READ_CHARS:,} characters of an attached text file, for a first look. It cannot "
                       "read further: load the file in python (it is at /data/<name>) for anything more.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "name of an attached file"},
        }, "required": ["name"]},
    },
}


class ToolError(Exception):
    """A problem the model can fix. `call_tool` turns it into an "ERROR ..." string."""


def tool_schemas() -> list[dict]:
    """OpenAI-compatible chat-completions `tools` list."""
    return [{"type": "function", "function": {"name": name, **spec}} for name, spec in TOOLS.items()]


def _read_file(ctx: dict, args: dict) -> str:
    name = args["name"]
    if not isinstance(name, str) or not name:
        raise ToolError("the file name must be a non-empty string")
    root = Path(ctx["files_dir"]).resolve()
    path = (root / name).resolve()
    # One message for "outside the folder" and "not there": the model needs the
    # same next step either way, and it should not learn what exists elsewhere.
    if not path.is_relative_to(root) or not path.is_file():
        raise ToolError(f"no file named {name!r} is attached; os.listdir('/data') in the python tool lists what is")
    with open(path, errors="replace") as f:
        head = f.read(READ_CHARS)
    return (f"{head}\n[{name}: the first {len(head)} characters of a {path.stat().st_size}-byte file. read_file shows "
            f"no more than this; load /data/{name} in the python tool for anything else.]")


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


_DISPATCH = {"python": _python, "read_file": _read_file}


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
