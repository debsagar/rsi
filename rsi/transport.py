"""The harness process as the broker sees it: start it, exchange JSON lines, stop it.

Split from rsi/broker.py, which decides what the lines mean. Nothing here
knows about tasks, limits or events.

Invariants:
- Neither way of starting the harness passes it a key, a network or a file
  other than the harness and the runner.
- Nothing here waits without a bound the caller chose: `send` and `bounded`
  take the seconds they may use and raise OutOfTime past them, and `lines`
  hands lines to a queue the caller reads with its own timeout.
- `lines` always ends with None, however the stream ended, so the caller
  never waits for a harness that is gone.
- `stop` leaves no harness process and no harness container behind.
"""
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable
from uuid import uuid4

from . import ROOT, tools as _tools

RUNNER = Path(__file__).with_name("runner.py")


class OutOfTime(Exception):
    """The run limit passed while something was in flight."""


def bounded(fn: Callable, seconds: float):
    """fn() on a daemon thread, for at most `seconds`: its value, its exception, or OutOfTime."""
    box = []

    def work():
        try:
            box.append((fn(), None))
        except BaseException as e:  # KeyboardInterrupt and SystemExit too: a thread that died is not one that ran out of time
            box.append((None, e))

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(max(0.0, seconds))
    # ponytail: a thread that ran out of time is abandoned, not killed. It ends
    # when what it waits on does: the harness and the python container are
    # killed when the run ends; a model call ends at the provider's own timeout
    # and its cost is then not counted.
    if not box:
        raise OutOfTime
    value, error = box[0]
    if error is not None:
        raise error
    return value


def start(harness_path: Path, in_docker: bool, stderr):
    """-> (process, container name or None). Neither form passes any key."""
    if in_docker:
        name = f"rsi-{uuid4().hex[:12]}"
        cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--memory", "1g", "--cpus", "1", "--name", name,
               "-v", f"{harness_path}:/harness.py:ro", "-v", f"{RUNNER}:/runner.py:ro",
               _tools.SANDBOX_IMAGE, "python", "/runner.py", "/harness.py"]
        env = None  # the docker client needs its own environment; without -e the container gets none of it
    else:
        name = None
        cmd = [sys.executable, "-m", "rsi.runner", str(harness_path)]
        env = {"PATH": os.environ.get("PATH", "")}
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, cwd=ROOT, env=env)
    return proc, name


def stop(proc, container) -> int:
    """Kill the harness and its container. -> the exit code (its own if it had already exited)."""
    if container:
        subprocess.run(["docker", "kill", container], capture_output=True)
    proc.kill()
    return proc.wait()


def lines(proc) -> queue.Queue:
    """A queue of the raw lines the harness writes, then None when it closes stdout."""
    out: queue.Queue = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True).start()
    return out


def _pump(stream, lines: queue.Queue) -> None:
    try:
        for line in stream:  # bytes: decoding is the main loop's job, where a bad line is a bad request
            lines.put(line)
    finally:
        lines.put(None)  # the harness closed stdout: it has exited


def send(proc, reply: dict, seconds: float) -> bool:
    """False when the harness is gone; OutOfTime when it has not read the line within `seconds`."""
    data = (json.dumps(reply, default=str) + "\n").encode()

    def write():
        proc.stdin.write(data)
        proc.stdin.flush()

    try:
        bounded(write, seconds)
        return True
    except OSError:  # BrokenPipeError included
        return False
