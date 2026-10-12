"""Command line: `rsi serve | run | evolve | compare | eval`.

Invariants:
- `.env` is loaded here and nowhere else.
- A command that needs Docker, a model id or a key checks for it before doing
  any work and exits with one line naming what is missing.
- `rsi.evolve` is imported only inside its subcommands, so `serve` and `run`
  work without it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import broker, env, home, llm, load_env, tools

HOST, PORT = "127.0.0.1", 8000
FAMILIES = ["generated", "mathbench", "dabench"]  # for evolve and compare; `rsi eval` has its own list


def preflight(*model_vars: str) -> None:
    """Raise RuntimeError naming the first thing a real run would be missing."""
    llm.check(*model_vars)
    env("TYPESAFE_API_KEY")
    if not shutil.which("docker"):
        raise RuntimeError("docker is not installed; the harness and the python tool only run in a container")
    if subprocess.run(["docker", "image", "inspect", tools.SANDBOX_IMAGE], capture_output=True).returncode != 0:
        raise RuntimeError(f"docker is not running or the {tools.SANDBOX_IMAGE} image is missing; "
                           f"start docker, then: docker build -t {tools.SANDBOX_IMAGE} sandbox/")


def serve(args) -> None:
    import uvicorn

    preflight("RSI_MODEL")
    uvicorn.run("rsi.api:app", host=HOST, port=int(os.environ.get("RSI_PORT", PORT)))


def run(args) -> None:
    """One question from the terminal, with the trajectory printed as it is written."""
    preflight("RSI_MODEL")
    paths = [Path(f).resolve() for f in args.file]
    folders = {p.parent for p in paths}
    if len(folders) > 1 or any(not p.is_file() for p in paths):
        raise RuntimeError("--file paths must exist and share one folder (it is mounted read-only as /data)")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    task = {"id": run_id, "family": "chat", "question": args.question, "files": [p.name for p in paths], "history": []}

    def show(event: dict) -> None:
        if event["kind"] != "delta":
            print(json.dumps(event, default=str)[:400], file=sys.stderr)

    episode = broker.run_episode(task, broker.current_harness(), home() / "runs" / "cli" / run_id,
                                 folders.pop() if folders else None, on_event=show)
    print(episode["answer"] or f"(no answer: {episode['stop_reason']})")
    print(json.dumps({k: episode[k] for k in ("run_id", "stop_reason", "unsupported", "calls", "tool_calls", "cost")}),
          file=sys.stderr)


def evolve(args) -> None:
    from . import evolve as evolve_mod  # imported here so serve and run do not load scipy

    preflight("RSI_MODEL", *([] if args.candidate else ["RSI_IMPROVER_MODEL"]))  # hand-written candidates need no improver
    # evolve prints its progress and a plain summary; the full row is the last line of harnesses/lineage.jsonl.
    evolve_mod.evolve(family=args.family, train=args.train, val=args.val, candidates=args.candidates, budget=args.budget,
                      candidate_files=args.candidate)


def compare(args) -> None:
    from . import evolve as evolve_mod

    preflight("RSI_MODEL")
    evolve_mod.compare(args.family, args.n, args.a, args.b)


def evaluate(argv: list[str]) -> None:
    """`rsi eval ...`: rsi.evals parses its own flags; only paid conditions need keys and Docker."""
    from . import evals

    free = "--report" in argv or (("--conditions" in argv) and argv[argv.index("--conditions") + 1] == "null")
    if not free:
        preflight("RSI_MODEL")
    evals.main(argv)


def main() -> None:
    if sys.argv[1:2] == ["eval"]:
        load_env()
        try:
            evaluate(sys.argv[2:])
        except (RuntimeError, ValueError) as e:
            sys.exit(f"rsi: {e}")
        except KeyboardInterrupt:
            sys.exit("rsi: interrupted; finished episodes are on disk and the same command resumes from them")
        return
    parser = argparse.ArgumentParser(prog="rsi", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help=f"start the API on {HOST}:{PORT}").set_defaults(func=serve)
    p = sub.add_parser("run", help="ask one question from the terminal")
    p.add_argument("question")
    p.add_argument("--file", action="append", default=[], help="attach a file (repeatable; one folder)")
    p.set_defaults(func=run)
    p = sub.add_parser("evolve", help="one generation of harness evolution")
    p.add_argument("--family", default="generated", choices=FAMILIES)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--val", type=int, default=45)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--budget", type=float, default=5.0, help="US dollars")
    p.add_argument("--candidate", action="append", default=[], metavar="PATH",
                   help="score this harness file instead of asking the improver (repeatable)")
    p.set_defaults(func=evolve)
    p = sub.add_parser("compare", help="score one harness file on the Test split, or two on the same Test tasks")
    p.add_argument("--family", default="generated", choices=FAMILIES)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--a", required=True, help="harness file A")
    p.add_argument("--b", help="harness file B; without it only A is scored")
    p.set_defaults(func=compare)
    args = parser.parse_args()
    load_env()
    try:
        args.func(args)
    except (RuntimeError, ValueError) as e:  # ValueError: an argument out of range, e.g. --val 5000
        sys.exit(f"rsi: {e}")
    except KeyboardInterrupt:
        sys.exit("rsi: interrupted; finished episodes are on disk and the same command resumes from them")


if __name__ == "__main__":
    main()
