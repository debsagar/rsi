"""Runs inside the sandbox container: the `python` tool's interpreter for one episode.

    python pyloop.py [seconds]   (in the container, as PID 1; this file is mounted alone)

Two processes. The parent (`main`) owns the stream to the host and never runs
the model's code. It starts one worker (`python pyloop.py --worker`), which
keeps the namespace and runs every request in it, so variables and imports
persist. Host -> parent -> worker: one JSON line {"id": str, "code": str}.
Worker -> parent: {"id", "stdout", "stderr", "error", "dropped"}. The parent
waits for the worker line that carries the id of the request it just passed
on, drops every other line, rebuilds the reply from that line's fields (so its
size and shape are the parent's doing) and writes it to the host.

What the code under test can and cannot do to the channel:
- It cannot write to the host stream or read the host's requests. The worker
  inherits only its own pipes to the parent. The parent is PID 1 and not
  dumpable, so /proc/1/fd and /proc/1/mem are closed to the worker (same user,
  no capabilities), and a signal from inside the container cannot stop PID 1.
  For this the container must run WITHOUT --init: docker-init would be a
  second, unprotected holder of the host stream.
- It cannot make one call's reply be taken for another's. The id is new per
  request and made on the host, so a line written before a request was sent
  (by an earlier call, a thread or a child it left behind) carries the wrong
  id and is dropped, by the parent and again by the host.
- It CAN decide what its own interpreter says while a call is running: code in
  the worker (a thread left by an earlier call included) can find the worker's
  pipe and the current id and write the reply for the call in flight. That is
  no more than replacing `print` in the shared namespace would do: the worker's
  output is whatever the code in it makes it. The real reply then arrives with
  a stale id and is dropped.
- It CAN break its own session (close or read the worker's pipes, exit, fork
  until the process limit). The host then sees no reply or the end of the
  stream and reports a timeout or a stopped interpreter; never another reply.

Invariants:
- Stdlib only, no `rsi` imports: this file is mounted into the container by itself.
- Nothing the code does ends the worker short of killing the process: every
  exception, SystemExit included, comes back as `error`.
- A reply is bounded: each of stdout, stderr and error holds at most CAP
  characters (error a few more for its marker), and `dropped` counts what was
  printed past that. The parent enforces this again on what it forwards.
- The parent ends itself, and with it the container, when the host closes the
  stream, when the worker dies, or when a call outlives `seconds` (a backstop
  for a host that died mid-call and cannot kill the container).
"""
import ast
import contextlib
import io
import json
import linecache
import os
import select
import subprocess
import sys
import time
import traceback

CAP = 100_000  # characters kept per stream; rsi/tools.py sizes its read limit from this
LINE_BYTES = 4_000_000  # longest worker line the parent keeps; an honest one is under 3 * CAP * 12 bytes


class Capped(io.TextIOBase):
    """Keeps the first `cap` characters written and counts the rest."""

    def __init__(self, cap: int):
        self.cap, self.parts, self.kept, self.dropped = cap, [], 0, 0

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        take = text[:max(0, self.cap - self.kept)]
        self.parts.append(take)
        self.kept += len(take)
        self.dropped += len(text) - len(take)
        return len(text)


def run_cell(code: str, ns: dict, cell: int = 0, cap: int = CAP) -> dict:
    """Run `code` in `ns`. A last line that is an expression is displayed like
    a notebook does (its repr, unless it is None)."""
    out, err, error = Capped(cap), Capped(cap), None
    name = f"<call {cell}>"
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)  # so tracebacks show the source line
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            tree = ast.parse(code, name)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, name, "exec"), ns)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), name, "eval"), ns)
                if value is not None:
                    print(repr(value))
        except BaseException as e:  # SystemExit and KeyboardInterrupt too: the interpreter must outlive the call
            # Drop this function's own frame; a SyntaxError has none to drop.
            tb = e.__traceback__.tb_next if not isinstance(e, SyntaxError) else None
            error = "".join(traceback.format_exception(type(e), e, tb))
            if len(error) > cap:
                error = f"[... {len(error) - cap} characters of traceback dropped ...]\n" + error[-cap:]
    return {"stdout": "".join(out.parts), "stderr": "".join(err.parts), "error": error, "dropped": out.dropped + err.dropped}


def worker() -> None:
    """Run requests from the parent in one namespace, for as long as the parent sends them."""
    requests, reply_to = os.fdopen(os.dup(0)), os.fdopen(os.dup(1), "w")
    # ponytail: output written straight to file descriptors 1 and 2 (a child
    # process that inherits them, a C library) is discarded, not shown. Only
    # Python-level sys.stdout/sys.stderr are captured. Upgrade: point both at a
    # pipe drained by a capped reader thread.
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):  # so input() gets end-of-file instead of eating the next request
        os.dup2(null, fd)
    ns = {"__name__": "__main__"}
    for cell, line in enumerate(requests, 1):
        request = json.loads(line)
        result = run_cell(request["code"], ns, cell)
        # The leading newline ends any unfinished line the code wrote to this pipe, so it cannot swallow the reply.
        reply_to.write("\n" + json.dumps({"id": request["id"], **result}) + "\n")
        reply_to.flush()


def _reply(fd: int, pending: bytes, rid: str, deadline) -> tuple[dict | None, bytes]:
    """Read worker lines until one is a reply to `rid`. -> (reply, unread bytes);
    reply is None when the worker closed its pipe or the deadline passed."""
    while True:
        while b"\n" in pending:
            line, _, pending = pending.partition(b"\n")
            try:
                got = json.loads(line)
                if got["id"] == rid:
                    error = got["error"]
                    return {"id": rid, "stdout": str(got["stdout"])[:CAP], "stderr": str(got["stderr"])[:CAP],
                            "error": None if error is None else str(error)[:CAP + 100], "dropped": int(got["dropped"])}, pending
            except (ValueError, TypeError, KeyError, RecursionError):
                pass  # not a reply to this request: written by the code under test, or left from an earlier call
        if len(pending) > LINE_BYTES:
            pending = b""  # no honest line is this long; what follows it will not parse either
        wait = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not select.select([fd], [], [], wait)[0]:
            return None, pending
        chunk = os.read(fd, 1 << 16)
        if not chunk:
            return None, pending
        pending += chunk


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    import ctypes  # here, not at the top: the worker has no use for it
    if ctypes.CDLL(None).prctl(4, 0) != 0:  # PR_SET_DUMPABLE off: closes /proc/<this pid>/fd and /mem to the worker
        raise SystemExit("pyloop: could not make the parent process undumpable")
    # close_fds (the default) leaves the worker file descriptors 0-2 only, all of them its own pipes or /dev/null.
    child = subprocess.Popen([sys.executable, __file__, "--worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    fd, pending = child.stdout.fileno(), b""
    for line in sys.stdin.buffer:
        rid = json.loads(line)["id"]
        child.stdin.write(line)
        child.stdin.flush()
        reply, pending = _reply(fd, pending, rid, time.monotonic() + limit if limit else None)
        if reply is None:
            break
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
        try:  # as PID 1 this process inherits orphans; collect the ones that have exited
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
    child.kill()  # and when PID 1 returns, the kernel ends everything else in the container


if __name__ == "__main__":
    worker() if sys.argv[1:] == ["--worker"] else main()
