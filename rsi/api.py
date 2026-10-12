"""HTTP boundary for the web app: sessions, uploads, and one streamed run per message.

Sessions are folders under `<RSI_HOME>/runs/sessions/<sid>/`:
`meta.json`, `files/`, and one `<run_id>/` per message holding the broker's
`events.jsonl` and `episode.json`. Nothing is kept in memory between
requests, so a restart loses nothing.

Invariants:
- Single user, no auth: `rsi serve` binds 127.0.0.1 only.
- A session id is 12 hex characters or the request is a 404, and an upload is
  stored under a sanitised bare name, so no request can name a path outside
  its session folder.
- The stream sends each trajectory event exactly as the broker wrote it.
- A file on disk that cannot be parsed never fails a request. What could not be
  read is skipped and said so: `log_unreadable` on a run, `meta_unreadable` on
  a session, `lineage_skipped` on the harness lineage.
- An upload keeps its name: the same original name again replaces the stored
  file (`replaced`), a different name that sanitises to a taken one gets a
  numeric suffix. `meta.json` maps stored names to original names (`uploads`).
"""
import json
import mimetypes
import queue
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.formparsers import MultiPartException, MultiPartParser

from . import broker, home, jev, llm

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024
# What a multipart body may add around the file itself: boundaries and part headers.
UPLOAD_ENVELOPE = 64 * 1024
NAME_CHARS = 120  # stored upload names; file systems stop at 255 bytes
TITLE_CHARS = 60
NEW_TITLE = "New session"
SESSION_ID = re.compile(r"[0-9a-f]{12}")
WEB_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
# Test hooks, monkeypatched in tests/test_api.py: a scripted model, a fake Jev,
# and the plain-subprocess harness. Left alone, runs use the real ones in Docker.
_LLM = None
_JEV = None
_IN_DOCKER = True
# meta.json is read, changed and rewritten by uploads and by the first message.
_META_LOCK = threading.Lock()

app = FastAPI(title="RSI")
app.add_middleware(CORSMiddleware, allow_origins=WEB_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RecursionError)
def _too_deep(request: Request, _exc: RecursionError):
    """A JSON body nested thousands of levels deep overflows the parser; that is the client's error."""
    return JSONResponse({"detail": "request body is nested too deeply"}, status_code=400)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions_dir() -> Path:
    return home() / "runs" / "sessions"


def _session_dir(sid: str) -> Path:
    path = _sessions_dir() / sid
    if not SESSION_ID.fullmatch(sid) or not (path / "meta.json").is_file():
        raise HTTPException(404, "session not found")
    return path


def _meta(path: Path) -> dict:
    """The session's meta.json. An unreadable one is rebuilt from the folder and
    flagged `meta_unreadable`, so the session and its runs stay reachable."""
    try:
        meta = json.loads((path / "meta.json").read_text(errors="replace"))
        if isinstance(meta, dict) and isinstance(meta.get("created"), str) and isinstance(meta.get("title"), str):
            return {**meta, "id": path.name}
    except ValueError:
        pass
    # ponytail: `created` becomes the time of the damaged write, the best the
    # folder can say. The next first message rewrites the file and clears the flag.
    created = datetime.fromtimestamp((path / "meta.json").stat().st_mtime, timezone.utc).isoformat()
    return {"id": path.name, "created": created, "title": NEW_TITLE, "meta_unreadable": True}


def _write_meta(path: Path, meta: dict) -> None:
    """Whole file or nothing: a reader never sees a half-written meta.json."""
    partial = path / ".meta.json.part"
    partial.write_text(json.dumps({k: v for k, v in meta.items() if k != "meta_unreadable"}))
    partial.replace(path / "meta.json")


def _base_name(raw: str | None) -> str:
    """The name as uploaded, without any directories."""
    return re.split(r"[/\\]", raw or "")[-1]


def _safe_name(raw: str | None) -> str:
    """A bare file name: no directories, no leading dot, only [A-Za-z0-9._-],
    at most NAME_CHARS long with the extension kept."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", _base_name(raw)).lstrip(".") or "upload"
    ext = Path(name).suffix[:16]
    return name if len(name) <= NAME_CHARS else name[:NAME_CHARS - len(ext)] + ext


def _stored_name(files_dir: Path, raw: str | None, originals: dict) -> tuple[str, bool]:
    """Where an upload goes and whether it replaces a stored file.

    `originals` is meta["uploads"]: stored name -> the name it was uploaded as.
    A file with no entry (stored before the map existed) counts as uploaded
    under its stored name."""
    stored = {p.name for p in files_dir.iterdir() if p.is_file()}
    for name in sorted(stored):
        if originals.get(name, name) == _base_name(raw):
            return name, True
    safe = name = _safe_name(raw)
    n = 1
    while name in stored:
        n += 1
        name = f"{Path(safe).stem}-{n}{Path(safe).suffix}"
    return name, False


def _file_info(path: Path) -> dict:
    return {"name": path.name, "size": path.stat().st_size,
            "type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}


def _files(session: Path) -> list[dict]:
    # Dot files are uploads still being written; a stored name never starts with a dot.
    return [_file_info(p) for p in sorted((session / "files").iterdir()) if p.is_file() and not p.name.startswith(".")]


def _text(events: list[dict], kind: str) -> str | None:
    return next((e["text"] for e in events if e.get("kind") == kind and isinstance(e.get("text"), str)), None)


def _summary(meta: dict) -> dict:
    """What the web app is sent of meta.json; the upload-name map stays on disk."""
    return {k: meta[k] for k in ("id", "created", "title", "meta_unreadable") if k in meta}


def _runs(session: Path) -> list[dict]:
    """Stored runs, oldest first. A run still in flight has events but no episode yet.

    The broker may be mid-write, or a crash may have left half a line: a line
    that is not a JSON object is skipped, an unreadable episode.json reads as
    "not finished", and either sets `log_unreadable` on that run."""
    out = []
    for run_dir in sorted(p for p in session.iterdir() if (p / "events.jsonl").is_file()):
        events, damaged, episode = [], False, None
        for line in (run_dir / "events.jsonl").read_text(errors="replace").splitlines():
            try:
                event = json.loads(line) if line.strip() else None
            except ValueError:
                event = line
            if isinstance(event, dict):
                events.append(event)
            elif event is not None:
                damaged = True
        if (run_dir / "episode.json").is_file():
            try:
                episode = json.loads((run_dir / "episode.json").read_text(errors="replace"))
            except ValueError:
                pass
            if not isinstance(episode, dict):
                episode, damaged = None, True
        out.append({"run_id": run_dir.name, "question": _text(events, "user") or "", "answer": _text(events, "answer"),
                    "episode": episode, "events": events, "log_unreadable": damaged})
    return out


@app.post("/api/sessions")
def create_session():
    meta = {"id": uuid4().hex[:12], "created": _now(), "title": NEW_TITLE}
    path = _sessions_dir() / meta["id"]
    (path / "files").mkdir(parents=True)
    _write_meta(path, meta)
    return meta


@app.get("/api/sessions")
def list_sessions():
    root = _sessions_dir()
    metas = [_meta(p) for p in root.iterdir() if (p / "meta.json").is_file()] if root.is_dir() else []
    return sorted(map(_summary, metas), key=lambda m: m["created"], reverse=True)


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    session = _session_dir(sid)
    return {**_summary(_meta(session)), "files": _files(session), "runs": _runs(session)}


def _too_large() -> HTTPException:
    return HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")


async def _capped(request: Request, cap: int):
    """The request body, refused as soon as it passes `cap`: nothing more is read or stored."""
    seen = 0
    async for chunk in request.stream():
        seen += len(chunk)
        if seen > cap:
            raise _too_large()
        yield chunk


@app.post("/api/sessions/{sid}/files")
async def upload_file(sid: str, request: Request):
    # The form is parsed here, not by a FastAPI parameter, because that would
    # receive the whole body into a temp file before this function ran.
    session = _session_dir(sid)
    cap = MAX_UPLOAD_BYTES + UPLOAD_ENVELOPE
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > cap:
        raise _too_large()
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise HTTPException(400, "send the file as multipart/form-data in a field named file")
    try:
        form = await MultiPartParser(request.headers, _capped(request, cap), max_files=1, max_fields=1).parse()
    except MultiPartException as e:
        raise HTTPException(400, e.message)
    file = form.get("file")
    try:
        if file is None or isinstance(file, str):
            raise HTTPException(400, "send the file as multipart/form-data in a field named file")
        if (file.size or 0) > MAX_UPLOAD_BYTES:
            raise _too_large()
        # ponytail: the copy (at most 25 MB, local disk) runs on the event loop.
        # Single user; move it to a thread if uploads ever stall a stream.
        partial = session / "files" / f".{uuid4().hex}.part"
        try:
            with open(partial, "wb") as out:
                shutil.copyfileobj(file.file, out, UPLOAD_CHUNK)
            with _META_LOCK:
                meta = _meta(session)
                uploads = meta.get("uploads") if isinstance(meta.get("uploads"), dict) else {}
                name, replaced = _stored_name(session / "files", file.filename, uploads)
                partial.replace(session / "files" / name)
                _write_meta(session, {**meta, "uploads": {**uploads, name: _base_name(file.filename)}})
        finally:
            partial.unlink(missing_ok=True)
    finally:
        await form.close()
    return {**_file_info(session / "files" / name), "replaced": replaced}


def _work(task: dict, run_dir: Path, files_dir: Path, events: queue.Queue) -> None:
    """The blocking episode, on its own thread. Everything it has to say goes on the queue."""
    try:
        episode = broker.run_episode(task, broker.current_harness(), run_dir, files_dir, on_event=events.put,
                                     llm=_LLM or llm.complete, jev=_JEV or jev.ask, in_docker=_IN_DOCKER)
        events.put({"kind": "_done", "episode": episode})
    except Exception as e:  # no fallback: surface and stop
        events.put({"kind": "_failed", "message": f"{type(e).__name__}: {e}"})


def _stream(events: queue.Queue):
    last_error = None
    while True:
        event = events.get()
        kind = event["kind"]
        if kind == "delta":
            yield _sse("delta", {"text": event["text"]})
        elif kind == "_failed":
            yield _sse("error", {"message": event["message"]})
            return
        elif kind == "_done":
            episode = event["episode"]
            if episode["stop_reason"] != "submitted":
                yield _sse("error", {"message": last_error or f"the run ended without an answer ({episode['stop_reason']})"})
            yield _sse("done", {"episode": episode})
            return
        else:
            if kind == "error":
                last_error = event["message"]
            yield _sse("event", event)


@app.post("/api/sessions/{sid}/messages")
def post_message(sid: str, payload: Any = Body(None)):
    session = _session_dir(sid)
    question = payload.get("question") if isinstance(payload, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(400, "question must be a non-empty string")
    # A turn that got no answer stays in the history, marked, so the model still
    # sees what was asked. `answer` stays a string: the harness sends it as is.
    history = [{"question": r["question"],
                "answer": r["answer"] if r["answer"] is not None
                else f"(no answer: {(r['episode'] or {}).get('stop_reason') or 'the run did not finish'})"}
               for r in _runs(session) if r["question"]]
    with _META_LOCK:
        meta = _meta(session)
        if meta["title"] == NEW_TITLE:
            _write_meta(session, {**meta, "title": " ".join(question.split())[:TITLE_CHARS]})
    # The timestamp makes run folders sort in the order they were asked.
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}-{uuid4().hex[:4]}"
    task = {"id": run_id, "family": "chat", "question": question, "session_id": sid, "history": history,
            "files": [f["name"] for f in _files(session)]}
    events: queue.Queue = queue.Queue()
    # ponytail: if the browser disconnects, the run still finishes and is on
    # disk for reload; nothing cancels it. Add cancellation when runs get long.
    threading.Thread(target=_work, args=(task, session / run_id, session / "files", events), daemon=True).start()
    return StreamingResponse(_stream(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/harnesses")
def harnesses():
    lineage_path = home() / "harnesses" / "lineage.jsonl"
    lineage, skipped = [], 0
    for line in lineage_path.read_text(errors="replace").splitlines() if lineage_path.is_file() else []:
        try:
            row = json.loads(line) if line.strip() else None
        except ValueError:
            row = line
        if isinstance(row, dict):
            lineage.append(row)
        elif row is not None:
            skipped += 1
    return {"current": broker.current_harness().name, "lineage": lineage, "lineage_skipped": skipped,
            "limits": {"model_calls": broker.MODEL_CALL_LIMIT, "tool_calls": broker.TOOL_CALL_LIMIT,
                       "judge_calls": broker.JUDGE_CALL_LIMIT}}
