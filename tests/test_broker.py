"""Broker: limits, clipping edges, the pre-submit gate and failure endings.

What the end-to-end test (tests/test_e2e.py) cannot reach: a hostile or broken
harness, a model or Jev that fails, the gate sending an answer back, and what
is cleaned up when a run ends badly. The answered run itself is the e2e's.
Everything runs with in_docker=False and scripted fakes except the tests at the
end, which need the real containers and are skipped when Docker is unavailable.
"""
import json
import shutil
import subprocess
import time

import pytest
from fakes import fake_jev, scripted_llm, tool_call

from rsi import broker, harness, tools

SEED = broker.SEED_HARNESS
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def docker_ready() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", tools.SANDBOX_IMAGE], capture_output=True).returncode == 0


def make_task(**extra) -> dict:
    return {"id": "t1", "family": "chat", "question": "What is in the file?", "files": ["data.csv"], "history": [], **extra}


def write_harness(tmp_path, body: str):
    path = tmp_path / "custom_harness.py"
    path.write_text(body)
    return path


def read_events(run_dir) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def kinds(events) -> list[str]:
    return [e["kind"] for e in events]


@pytest.fixture
def dirs(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    (files / "data.csv").write_text("x,y\n1,2\n3,4\n")
    return tmp_path / "run-1", files


def episode(dirs, llm, *, harness_path=SEED, jev=None, task=None, **kwargs):
    """One episode with the gate on unless a test says otherwise (the setting's default is off)."""
    run_dir, files = dirs
    kwargs.setdefault("gate", True)
    return broker.run_episode(task or make_task(), harness_path, run_dir, files, llm=llm,
                              jev=jev or fake_jev(0.9), in_docker=False, **kwargs)


def test_reference_never_reaches_the_harness(dirs, tmp_path):
    path = write_harness(tmp_path, "def run(task, api):\n    api.submit(','.join(sorted(task)), [])\n")
    ep = episode(dirs, scripted_llm([]), harness_path=path, task=make_task(reference="SECRET", session_id="s1"))
    assert ep["answer"] == "files,history,id,question"
    assert ep["session_id"] == "s1"


def test_model_call_limit_errors_then_ends_a_harness_that_never_submits(dirs, tmp_path):
    path = write_harness(tmp_path, (
        "def run(task, api):\n"
        "    while True:\n"
        "        try:\n"
        "            api.generate([{'role': 'user', 'content': 'again'}])\n"
        "        except RuntimeError:\n"
        "            pass\n"))
    llm = scripted_llm(["thinking"] * 20)
    ep = episode(dirs, llm, harness_path=path)
    assert len(llm.calls) == broker.MODEL_CALL_LIMIT == 12
    assert ep["calls"] == 12 and ep["stop_reason"] == "model_call_limit" and ep["answer"] is None
    errors = [e for e in read_events(dirs[0]) if e["kind"] == "error"]
    assert len(errors) == 1 and "model call limit (12)" in errors[0]["message"]


def test_requests_past_a_limit_get_an_error_each_and_the_harness_can_still_answer(dirs, tmp_path):
    # Observed: a judge-limit notice made the first request past the tool limit fatal, and one model
    # message asking for 14 tools ended the run at the 14th, before the model could read the errors.
    path = write_harness(tmp_path, (
        "def run(task, api):\n"
        "    answers = [api.judge({}, {'q': {'type': 'noul'}}) for _ in range(13)]\n"
        "    outs = [api.execute('list_files', {})['result'] for _ in range(14)]\n"
        "    api.generate([{'role': 'user', 'content': outs[-1]}])\n"
        "    api.submit(f'{answers[-2]} {answers[-1]} {outs[-1]}', [])\n"))
    jev, llm = fake_jev(0.9), scripted_llm(["I will answer with what I have."])
    ep = episode(dirs, llm, harness_path=path, jev=jev, gate=False)
    assert ep["stop_reason"] == "submitted" and ep["answer"].startswith("{'q': {'noul': 0.9}} None ERROR tool call limit (12)")
    assert ep["tool_calls"] == broker.TOOL_CALL_LIMIT == 12 and len(jev.calls) == broker.JUDGE_CALL_LIMIT == 12
    events = read_events(dirs[0])
    assert kinds(events).count("tool_result") == 12
    assert [e["message"].split(" (")[0] for e in events if e["kind"] == "error"] == ["judge call limit"] + ["tool call limit"] * 2


@pytest.mark.parametrize("body, stop_reason", [
    ("    while True:\n        api.execute('list_files', {})\n", "tool_call_limit"),             # spins, never asks the model
    ("    while True:\n        api.judge({}, {'q': {'type': 'noul'}})\n", "judge_call_limit"),
    ("    for _ in range(13):\n        api.execute('list_files', {})\n"                          # warned, the model was asked,
     "    api.generate([])\n    api.execute('list_files', {})\n    api.submit('late', [])\n", "tool_call_limit"),  # and it calls again
])
def test_a_harness_that_keeps_going_past_a_limit_is_stopped(dirs, tmp_path, body, stop_reason):
    jev = fake_jev(0.9)
    ep = episode(dirs, scripted_llm(["noted"]), harness_path=write_harness(tmp_path, "def run(task, api):\n" + body), jev=jev)
    assert ep["stop_reason"] == stop_reason and ep["answer"] is None
    assert ep["tool_calls"] <= 12 and len(jev.calls) <= 12  # nothing past a limit was carried out
    assert kinds(read_events(dirs[0])).count("error") <= broker.OVER_LIMIT_BURST


def slow_llm(messages, tools=None, model=None, on_delta=None):
    time.sleep(8)
    return {"message": {"role": "assistant", "content": "late"}, "usage": {"in": 1, "out": 1}, "cost": 0}


@pytest.mark.parametrize("body, kwargs", [
    ("    time.sleep(60)\n", {}),                                                               # the harness hangs
    ("    api.generate([])\n    api.submit('late', [])\n", {"llm": slow_llm}),                   # a model call is in flight
    ("    api.execute('slow', {})\n    api.submit('late', [])\n",                               # a tool call is in flight
     {"toolset": ([], lambda name, args: time.sleep(8) or "late")}),
    ("    api._out.write(json.dumps({'op': 'generate', 'messages': []}) + '\\n')\n"              # the harness never reads
     "    api._out.flush()\n    time.sleep(60)\n", {"llm": scripted_llm(["x" * 300_000])}),      # a reply too big for the pipe
], ids=["harness hangs", "model call", "tool call", "unread reply"])
def test_run_limit_holds_whatever_is_in_flight(dirs, tmp_path, monkeypatch, body, kwargs):
    # Observed: the limit was only checked between requests, so the last three cases overran it (the last one for ever).
    monkeypatch.setattr(broker, "RUN_LIMIT_S", 1)
    path = write_harness(tmp_path, "import json, time\n\ndef run(task, api):\n" + body)
    started = time.monotonic()
    ep = episode(dirs, kwargs.pop("llm", scripted_llm([])), harness_path=path, **kwargs)
    assert time.monotonic() - started < 5
    assert ep["stop_reason"] == "run_limit" and ep["answer"] is None
    assert "run limit" in read_events(dirs[0])[-1]["message"]


def test_tool_time_limit_returns_an_error_result(dirs, monkeypatch):
    monkeypatch.setattr(broker, "TOOL_LIMIT_S", 0.2)
    monkeypatch.setattr(broker, "TOOL_GRACE_S", 0)
    slow = ([], lambda name, args: time.sleep(5) or "late")
    ep = episode(dirs, scripted_llm([tool_call("anything", {}), "gave up"]), toolset=slow)
    result = [e for e in read_events(dirs[0]) if e["kind"] == "tool_result"][0]
    assert result["result"].startswith("ERROR anything timed out")
    assert ep["stop_reason"] == "submitted"


def test_long_tool_result_is_clipped_and_saved_in_full(dirs):
    texts = {"big": "#" * 4500 + "~" * 4500, "edge": "x" * 4000, "just_over": "y" * 4001}
    llm = scripted_llm([tool_call("big", {}), tool_call("edge", {}), tool_call("just_over", {}), "done"])
    episode(dirs, llm, toolset=([], lambda name, args: texts[name]))
    big, edge, just_over = [e for e in read_events(dirs[0]) if e["kind"] == "tool_result"]
    assert (big["chars_full"], big["clipped"], big["chars_shown"]) == (9000, True, len(big["result"]))
    assert 3990 <= big["chars_shown"] <= 4000  # what was sent, marker included, fits the limit
    assert big["result"].startswith("#" * 1900) and big["result"].endswith("~" * 1900)
    cut = int(big["result"].split("[... ")[1].split(" of 9000 characters clipped")[0])
    assert big["result"].count("#") + big["result"].count("~") == 9000 - cut  # the marker's count is the real one
    assert "tool-001.txt" in big["result"] and (dirs[0] / "tool-001.txt").read_text() == texts["big"]
    # The harness, and so the model, is told the same thing the event records.
    assert llm.calls[1]["messages"][-1]["content"].endswith(big["result"])
    assert (edge["chars_full"], edge["chars_shown"], edge["clipped"]) == (4000, 4000, False) and not (dirs[0] / "tool-002.txt").exists()
    assert just_over["clipped"] is True and len(just_over["result"]) <= 4000  # clipping never makes a result longer


def test_gate_rejects_once_then_accepts_and_marks_unsupported(dirs):
    llm = scripted_llm(["The mean is 7.", "The mean is still 7."])
    ep = episode(dirs, llm, jev=fake_jev(0.2))
    events = read_events(dirs[0])
    submits = [e for e in events if e["kind"] == "submit"]
    assert [s["accepted"] for s in submits] == [False, True]
    assert submits[0]["reason"] and submits[0]["p_supported"] == 0.2
    assert "not accepted" in llm.calls[1]["messages"][-1]["content"]
    assert events[-1]["kind"] == "answer" and events[-1]["unsupported"] is True
    assert events[-1]["text"] == "The mean is still 7."
    assert ep["unsupported"] is True and ep["stop_reason"] == "submitted"


def test_seed_harness_keeps_an_answer_the_gate_sends_back_on_the_last_step(dirs):
    # Observed: the loop ended with the answer rejected once and never offered again (stop_reason no_submit).
    llm = scripted_llm([tool_call("list_files", {}, f"c{i}") for i in range(11)] + ["Out of steps; best answer [4]."])
    ep = episode(dirs, llm, jev=fake_jev(0.1))
    assert llm.calls[11]["tools"] is None  # the twelfth and last allowed model call is made without tools
    assert (ep["stop_reason"], ep["answer"], ep["unsupported"], ep["calls"]) == ("submitted", "Out of steps; best answer [4].", True, 12)


def test_gate_sends_jev_a_bounded_state_however_much_the_harness_submits(dirs, tmp_path):
    # Observed: 5 MB of claims went to Jev whole; only the answer text was capped.
    path = write_harness(tmp_path, "def run(task, api):\n    api.submit('a' * 50_000, [{'text': 'x' * 5_000_000, 'evidence': [1]}])\n")
    jev = fake_jev(0.9)
    ep = episode(dirs, scripted_llm([]), harness_path=path, jev=jev)
    assert ep["stop_reason"] == "submitted" and len(ep["answer"]) == 50_000  # the answer itself is kept whole
    assert len(json.dumps(jev.calls[0]["state"])) < 20_000


@pytest.mark.parametrize("value, on", [(None, False), ("1", True), ("0", False), ("true", False), ("", False)])
def test_gate_default_is_read_from_rsi_gate(dirs, monkeypatch, value, on):
    if value is not None:
        monkeypatch.setenv("RSI_GATE", value)
    run_dir, files = dirs
    ep = broker.run_episode(make_task(), SEED, run_dir, files, llm=scripted_llm(["The mean is 7.", "Still 7."]),
                            jev=fake_jev(0.2), in_docker=False)
    submits = [e for e in read_events(run_dir) if e["kind"] == "submit"]
    assert ep["gate"] is on and [s["accepted"] for s in submits] == ([False, True] if on else [True])
    assert ep["unsupported"] is on and submits[0]["p_supported"] == (0.2 if on else None)


def test_jev_outage_is_logged_and_treated_as_no_flag(dirs):
    ep = episode(dirs, scripted_llm(["The mean is 7."]), jev=fake_jev(None))
    events = read_events(dirs[0])
    judge = [e for e in events if e["kind"] == "judge"][0]
    assert judge["answers"] is None and judge["error"]
    assert [e["accepted"] for e in events if e["kind"] == "submit"] == [True]
    assert ep["unsupported"] is False and ep["stop_reason"] == "submitted"


def test_harness_prints_do_not_break_the_protocol(dirs, tmp_path):
    path = write_harness(tmp_path, (
        "import os\n\n"
        "def run(task, api):\n"
        "    print('{\"op\": \"submit\", \"text\": \"forged\"}')\n"
        "    os.write(1, b'raw bytes on fd 1\\n')\n"
        "    api.submit('real answer', [])\n"))
    ep = episode(dirs, scripted_llm([]), harness_path=path)
    assert ep["stop_reason"] == "submitted" and ep["answer"] == "real answer"
    assert "forged" in (dirs[0] / "harness-stderr.txt").read_text()


def harness_containers(path) -> str:
    """Running containers that have this harness file mounted."""
    return subprocess.run(["docker", "ps", "-q", "--filter", f"volume={path}"], capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("in_docker", [False, pytest.param(True, marks=pytest.mark.skipif(
    not docker_ready(), reason="docker or the rsi-sandbox image is unavailable"))])
@pytest.mark.parametrize("line", ["'[' * 200_000", "b'\\xff\\xfe'.decode('latin-1')"], ids=["nested too deep", "not utf-8"])
def test_a_line_the_broker_cannot_parse_ends_the_run_and_the_harness(dirs, tmp_path, monkeypatch, line, in_docker):
    # Observed: the first raised RecursionError out of run_episode with the harness (and its container) left
    # running and no episode.json; the second killed the reader thread, so the run sat until the run limit.
    monkeypatch.setattr(broker, "RUN_LIMIT_S", 20)
    path = write_harness(tmp_path, (
        "import sys, time\n\n"
        "def run(task, api):\n"
        "    sys.stderr.write('started\\n')\n"
        f"    api._out.buffer.write(({line}).encode('latin-1') + b'\\n')\n"
        "    api._out.buffer.flush()\n"
        "    time.sleep(60)\n"))
    started = time.monotonic()
    ep = broker.run_episode(make_task(), path, dirs[0], dirs[1], llm=scripted_llm([]), jev=fake_jev(0.9), in_docker=in_docker)
    assert time.monotonic() - started < 15
    assert ep["stop_reason"] == "harness_error" and ep == json.loads((dirs[0] / "episode.json").read_text())
    assert "bad request from the harness" in read_events(dirs[0])[-1]["message"]
    assert "started" in (dirs[0] / "harness-stderr.txt").read_text()
    if in_docker:
        assert harness_containers(path) == ""
    else:
        assert subprocess.run(["pgrep", "-f", str(path)], capture_output=True, text=True).stdout == ""


def test_malformed_tool_arguments_come_back_as_an_error_result(dirs):
    llm = scripted_llm([tool_call("read_file", '{"name": "data.csv"'), "I could not read it."])
    ep = episode(dirs, llm)
    result = [e for e in read_events(dirs[0]) if e["kind"] == "tool_result"][0]
    assert result["result"].startswith("ERROR malformed tool call arguments")
    assert "ERROR malformed" in llm.calls[1]["messages"][-1]["content"]
    assert ep["stop_reason"] == "submitted" and ep["tool_calls"] == 1


def test_images_are_attached_to_the_first_user_message_host_side(dirs):
    (dirs[1] / "plot.png").write_bytes(PNG)
    llm = scripted_llm([tool_call("list_files", {}), "A plot."])
    episode(dirs, llm, task=make_task(files=["data.csv", "plot.png"]))
    for call in llm.calls:  # every model call, because the harness never holds the image
        users = [m for m in call["messages"] if m["role"] == "user"]
        text, image = users[0]["content"]
        assert text["type"] == "text" and "plot.png" in text["text"]
        assert image["image_url"]["url"].startswith("data:image/png;base64,iVBOR")
    assert "base64" not in (dirs[0] / "events.jsonl").read_text()


def test_model_failure_ends_the_run_as_llm_error(dirs):
    ep = episode(dirs, scripted_llm([RuntimeError("provider is down")]))
    events = read_events(dirs[0])
    assert ep["stop_reason"] == "llm_error" and ep["answer"] is None
    assert events[-1]["kind"] == "error" and "provider is down" in events[-1]["message"]


def test_harness_crash_is_recorded_with_its_stderr(dirs, tmp_path):
    path = write_harness(tmp_path, "def run(task, api):\n    raise ValueError('kaboom')\n")
    ep = episode(dirs, scripted_llm([]), harness_path=path)
    error = read_events(dirs[0])[-1]
    assert ep["stop_reason"] == "harness_error"
    assert error["kind"] == "error" and "kaboom" in error["stderr"]


def test_harness_that_returns_without_submitting(dirs, tmp_path):
    path = write_harness(tmp_path, "def run(task, api):\n    return\n")
    ep = episode(dirs, scripted_llm([]), harness_path=path)
    assert ep["stop_reason"] == "no_submit"
    assert read_events(dirs[0])[-1]["kind"] == "error"


ASKS_FOR_LINE = "What is the mean of y? End your reply with a line 'ANSWER: <number>'."


def test_seed_harness_does_not_submit_a_mid_work_reply_when_an_answer_line_is_asked_for(dirs):
    llm = scripted_llm(["Let me look at the file first. Shall I proceed?", "The mean is 3.\nANSWER: 3"])
    ep = episode(dirs, llm, task=make_task(question=ASKS_FOR_LINE), gate=False)
    assert [e["text"] for e in read_events(dirs[0]) if e["kind"] == "submit"] == ["The mean is 3.\nANSWER: 3"]
    assert ep["answer"].endswith("ANSWER: 3") and ep["calls"] == 2
    assert llm.calls[1]["messages"][-1] == {"role": "user", "content": harness.CONTINUE}
    assert llm.calls[1]["tools"]  # it can still call a tool


def test_current_harness_is_the_highest_one_the_lineage_adopted(tmp_path):
    folder = tmp_path / "harnesses"
    first = broker.current_harness(tmp_path)
    assert first == folder / "000.py" and first.read_text() == SEED.read_text()
    for name in ("001.py", "007.py", "1000.py"):
        (folder / name).write_text("def run(task, api):\n    pass\n")
    (folder / "002.py").mkdir()  # observed: a folder with a harness's name was returned as the harness
    # observed: serve and run used a hand-placed file that evolve itself refuses as unrecorded
    assert broker.current_harness(tmp_path).name == "000.py"
    rows = [{"adopted": "001.py"}, {"adopted": None}, {"adopted": "002.py"}, {"adopted": "1000.py"}]
    (folder / "lineage.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    assert broker.current_harness(tmp_path).name == "1000.py"  # by number, not by name: 1000 is after 999


def container_exists(name: str) -> bool:
    return bool(subprocess.run(["docker", "ps", "-aq", "--filter", f"name={name}"], capture_output=True, text=True).stdout.strip())


@pytest.fixture
def started(monkeypatch):
    """Names of the python-tool containers started during the test."""
    names, start = [], tools.PythonSession._start

    def recording_start(self):
        start(self)
        names.append(self.name)

    monkeypatch.setattr(tools.PythonSession, "_start", recording_start)
    return names


PYTHON_TWICE = ("def run(task, api):\n"
                "    api.execute('python', {'code': 'x = 6 * 7'})\n"
                "    out = api.execute('python', {'code': 'x'})['result']\n")


@pytest.mark.skipif(not docker_ready(), reason="docker or the rsi-sandbox image is unavailable")
@pytest.mark.parametrize("ending, stop_reason", [("    raise ValueError('kaboom ' + out)\n", "harness_error"),
                                                 ("    import time\n    time.sleep(60)\n", "run_limit")])
def test_python_container_lasts_the_episode_and_is_removed_however_it_ends(dirs, tmp_path, started, monkeypatch, ending, stop_reason):
    if stop_reason == "run_limit":
        monkeypatch.setattr(broker, "RUN_LIMIT_S", 8)
    ep = episode(dirs, scripted_llm([]), harness_path=write_harness(tmp_path, PYTHON_TWICE + ending))
    results = [e["result"] for e in read_events(dirs[0]) if e["kind"] == "tool_result"]
    assert results == ["(no output)\n", "42\n"]  # the second call saw the first call's variable
    assert ep["stop_reason"] == stop_reason
    assert len(started) == 1 and not container_exists(started[0])


@pytest.mark.skipif(not docker_ready(), reason="docker or the rsi-sandbox image is unavailable")
def test_python_container_is_removed_when_the_broker_itself_raises(dirs, tmp_path, started):
    def on_event(event):
        if event["kind"] == "tool_result":
            assert container_exists(started[0])  # it is running at this point
            raise RuntimeError("broker bug")

    path = write_harness(tmp_path, PYTHON_TWICE + "    import time\n    time.sleep(60)\n")
    with pytest.raises(RuntimeError, match="broker bug"):
        broker.run_episode(make_task(), path, dirs[0], dirs[1], llm=scripted_llm([]), jev=fake_jev(0.9), on_event=on_event)
    assert len(started) == 1 and not container_exists(started[0])
    assert harness_containers(path) == ""  # the harness's own container went too
    assert json.loads((dirs[0] / "episode.json").read_text())["stop_reason"] == "broker_error"


