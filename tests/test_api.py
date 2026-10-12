"""The HTTP API through its real routes, for what tests/test_e2e.py cannot reach: a failed turn, damaged
files on disk, hostile uploads and bad client input.

The model and Jev are the scripted fakes and the harness runs as a plain
subprocess, so nothing here needs Docker, a key or the network."""
import asyncio
import json

import pytest
from fakes import fake_jev, scripted_llm
from fastapi.testclient import TestClient

from rsi import api, broker


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_HOME", str(tmp_path))
    monkeypatch.setattr(api, "_JEV", fake_jev(0.9))
    monkeypatch.setattr(api, "_IN_DOCKER", False)
    return TestClient(api.app)


def use_llm(monkeypatch, replies):
    llm = scripted_llm(replies)
    monkeypatch.setattr(api, "_LLM", llm)
    return llm


def read_sse(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        assert name.startswith("event: ") and data.startswith("data: ")
        out.append((name[len("event: "):], json.loads(data[len("data: "):])))
    return out


def test_a_turn_that_fails_says_why_and_the_next_turn_is_told_about_it(client, monkeypatch):
    """The answered conversation (upload, turns, reload) is tests/test_e2e.py; this is the turn it cannot reach."""
    sid = client.post("/api/sessions").json()["id"]
    use_llm(monkeypatch, ["First answer."])
    client.post(f"/api/sessions/{sid}/messages", json={"question": "First question?"})

    # A turn that fails says why on the stream and still ends with `done`.
    use_llm(monkeypatch, [RuntimeError("provider is down")])
    failed = read_sse(client.post(f"/api/sessions/{sid}/messages", json={"question": "Second question?"}))
    assert [name for name, _ in failed][-2:] == ["error", "done"] and "provider is down" in failed[-2][1]["message"]
    assert failed[-1][1]["episode"]["stop_reason"] == "llm_error"

    # The next turn is told about both earlier ones, the unanswered one with a marker.
    llm = use_llm(monkeypatch, ["Third answer."])
    client.post(f"/api/sessions/{sid}/messages", json={"question": "Third question?"})
    history = [m["content"] for m in llm.calls[0]["messages"]][1:5]
    assert history == ["First question?", "First answer.", "Second question?", "(no answer: llm_error)"]
    first, second, third = client.get(f"/api/sessions/{sid}").json()["runs"]
    assert second["answer"] is None and third["answer"] == "Third answer."


def add_run(tmp_path, sid, run_id, events_text, episode_text=None):
    run_dir = tmp_path / "runs" / "sessions" / sid / run_id
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(events_text)
    if episode_text is not None:
        (run_dir / "episode.json").write_text(episode_text)


def test_a_session_survives_corrupt_and_half_written_files_and_still_accepts_a_message(client, tmp_path, monkeypatch):
    """Observed: one bad line, episode.json, meta.json or lineage row turned a whole route into a 500."""
    good = client.post("/api/sessions").json()["id"]
    sid = client.post("/api/sessions").json()["id"]
    user = '{"id":1,"kind":"user","text":"q"}\n'
    add_run(tmp_path, sid, "r1-half-line", user + '{"id":2,"ki')         # a crash, or a write in progress
    add_run(tmp_path, sid, "r2-bad-episode", user, '{"run_')
    add_run(tmp_path, sid, "r3-odd-lines", '{"id":1}\n\n7\n')         # no kind; a blank line; not an object
    add_run(tmp_path, sid, "r4-clean", user, '{"stop_reason": "no_submit"}')
    (tmp_path / "runs" / "sessions" / sid / "meta.json").write_text("{")
    (tmp_path / "harnesses").mkdir()
    (tmp_path / "harnesses" / "lineage.jsonl").write_text('{"generation":0}\n\n[1]\n{"gener')

    listed = {s["id"]: s for s in client.get("/api/sessions").json()}
    assert set(listed) == {good, sid} and listed[sid]["meta_unreadable"] is True and "meta_unreadable" not in listed[good]
    session = client.get(f"/api/sessions/{sid}").json()
    runs = {r["run_id"]: r for r in session["runs"]}
    assert [e["id"] for e in runs["r1-half-line"]["events"]] == [1]
    assert runs["r2-bad-episode"]["episode"] is None                     # reads as "not finished"
    assert runs["r3-odd-lines"]["question"] == "" and runs["r3-odd-lines"]["events"] == [{"id": 1}]
    assert {name: r["log_unreadable"] for name, r in runs.items()} == {
        "r1-half-line": True, "r2-bad-episode": True, "r3-odd-lines": True, "r4-clean": False}

    harnesses = client.get("/api/harnesses").json()
    assert harnesses["current"] == "000.py" and harnesses["lineage"] == [{"generation": 0}]
    assert harnesses["lineage_skipped"] == 2                             # the blank line is not counted
    assert harnesses["limits"] == {"model_calls": broker.MODEL_CALL_LIMIT, "tool_calls": broker.TOOL_CALL_LIMIT,
                                   "judge_calls": broker.JUDGE_CALL_LIMIT}

    use_llm(monkeypatch, ["Fine."])
    stream = read_sse(client.post(f"/api/sessions/{sid}/messages", json={"question": "next"}))
    assert stream[-1][0] == "done" and stream[-1][1]["episode"]["stop_reason"] == "submitted"
    healed = client.get(f"/api/sessions/{sid}").json()                   # the first message rewrote meta.json
    assert healed["title"] == "next" and "meta_unreadable" not in healed


def post_raw(path, headers, chunks):
    """One request straight into the ASGI app. Returns the status and how many body chunks the app took."""
    taken, sent = [], []
    chunks = iter(chunks)

    async def receive():
        chunk = next(chunks, None)
        taken.append(chunk)
        return {"type": "http.request", "body": chunk or b"", "more_body": chunk is not None}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()], "http_version": "1.1",
             "scheme": "http", "server": ("testserver", 80), "client": ("testclient", 1)}
    asyncio.run(api.app(scope, receive, send))
    return next(m["status"] for m in sent if m["type"] == "http.response.start"), len(taken)


def test_uploads_names_collisions_replacement_and_the_size_cap(client, monkeypatch, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    files_dir = tmp_path / "runs" / "sessions" / sid / "files"
    url = f"/api/sessions/{sid}/files"

    def up(name, data=b"1"):
        response = client.post(url, files={"file": (name, data)})
        assert response.status_code == 200, (name[:20], response.status_code)
        return response.json()

    # No name can leave the session's files folder, and a 300-character name is cut, not a 500.
    for hostile in ("../evil.csv", "../../x", "a/b\\c.csv", "..", ".env"):
        name = up(hostile)["name"]
        assert "/" not in name and "\\" not in name and not name.startswith(".") and (files_dir / name).is_file()
    long_name = up("x" * 300 + ".csv")["name"]
    assert len(long_name) == api.NAME_CHARS and long_name.endswith(".csv")
    stored = [p for p in (tmp_path / "runs").rglob("*") if p.is_file() and p.name != "meta.json"]
    assert len(stored) == 6 and all(p.parent == files_dir for p in stored)
    assert not (tmp_path / "x").exists() and not (tmp_path / "runs" / "x").exists()

    # The same name again replaces; a different name that sanitises to a taken one is kept beside it.
    assert up("a b.csv", b"AAAA") == {"name": "a_b.csv", "size": 4, "type": "text/csv", "replaced": False}
    assert up("a?b.csv", b"B") == {"name": "a_b-2.csv", "size": 1, "type": "text/csv", "replaced": False}
    assert up("a_b.csv", b"CC")["name"] == "a_b-3.csv"       # the stored name itself, uploaded as a new file
    assert up("a b.csv", b"DDDDD") == {"name": "a_b.csv", "size": 5, "type": "text/csv", "replaced": True}
    assert up("a?b.csv", b"EE") == {"name": "a_b-2.csv", "size": 2, "type": "text/csv", "replaced": True}
    session = client.get(f"/api/sessions/{sid}").json()
    assert [(f["name"], f["size"]) for f in session["files"] if f["name"].startswith("a_b")] == [
        ("a_b-2.csv", 2), ("a_b-3.csv", 2), ("a_b.csv", 5)]
    assert "uploads" not in session  # the name map is server-side state

    # Over the cap: refused, nothing stored, the earlier file of that name kept, and the body not taken in full.
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(api, "UPLOAD_ENVELOPE", 1000)
    before = sorted(p.name for p in files_dir.iterdir())
    assert client.post(url, files={"file": ("a b.csv", b"x" * 1001)}).status_code == 413  # inside the envelope allowance
    kind = "multipart/form-data; boundary=b"
    body = [b'--b\r\nContent-Disposition: form-data; name="file"; filename="big.bin"\r\n\r\n',
            *[b"x" * 1000] * 500, b"\r\n--b--\r\n"]
    # An honest Content-Length is refused without reading the body at all.
    assert post_raw(url, {"content-type": kind, "content-length": "500000"}, body) == (413, 0)
    # No Content-Length, and one that lies: reading stops once the cap is passed.
    for headers in ({"content-type": kind}, {"content-type": kind, "content-length": "10"}):
        status, taken = post_raw(url, headers, body)
        assert status == 413 and taken <= 4
    assert sorted(p.name for p in files_dir.iterdir()) == before and (files_dir / "a_b.csv").stat().st_size == 5


def test_bad_client_input_never_gets_a_500(client):
    client = TestClient(api.app, raise_server_exceptions=False)  # a traceback must show up as a status, not a raise
    sid = client.post("/api/sessions").json()["id"]
    as_json = {"content-type": "application/json"}
    bodies = [dict(json=v) for v in (None, [], [1], 5, "x", True, {}, {"question": 5}, {"question": None},
                                     {"question": ["a"]}, {"question": {"a": 1}}, {"question": ""}, {"question": "  "})]
    bodies += [dict(content=b"not json", headers=as_json), dict(content=b"", headers=as_json),
               dict(content=b"\xff\xfe", headers=as_json), dict(content=b"[" * 200_000, headers=as_json),
               dict(content=b"x=1", headers={"content-type": "application/x-www-form-urlencoded"}),
               dict(content=b"junk", headers={"content-type": "multipart/form-data; boundary=b"}),
               dict(content=b"junk", headers={"content-type": "multipart/form-data"}),
               dict(data={"file": "a text field, not a file"}), dict(files={"other": ("a.csv", b"1")}),
               dict(files=[("file", ("a.csv", b"1")), ("file", ("b.csv", b"2"))]),
               dict(files={"file": ("", b"1")}), dict(files={"file": ("\x00", b"1")}),
               dict(files={"file": ("...", b"1")}), dict(files={"file": ("x" * 5000, b"1")})]
    ids = [sid, "0123456789ab", "not-a-session", "..%2f..%2fetc", "%00", "a" * 5000, "ABCDEF012345", "%F0%9F%98%80",
           sid + "%0a", " "]
    seen = []
    for method, path in [("POST", "/api/sessions"), ("GET", "/api/sessions"), ("GET", "/api/harnesses"),
                         *[(m, f"/api/sessions/{i}{tail}") for i in ids
                           for m, tail in (("GET", ""), ("POST", ""), ("POST", "/files"), ("POST", "/messages"),
                                           ("GET", "/messages"), ("DELETE", "/files"))]]:
        for body in bodies if method == "POST" else [{}]:
            status = client.request(method, path, **body).status_code
            seen.append(status)
            assert status < 500, (method, path[:60], str(body)[:80], status)
    assert {200, 400, 404, 405} <= set(seen)
