"""Tools: ERROR strings for every kind of bad input, read_file's head-only
rule, and what the python sandbox refuses or survives. The answered path (load,
compute, a second session seeing nothing) is in tests/test_e2e.py."""
import shutil
import subprocess

import pytest

from rsi import tools


@pytest.fixture
def ctx(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    (files / "pairs.csv").write_text("x,y\n1,2\n3,4\n")
    (files / "notes.txt").write_text("0123456789" * 1000)
    (tmp_path / "secret.txt").write_text("outside")
    session = tools.PythonSession(files)  # starts no container unless a test calls the python tool
    yield {"files_dir": files, "python": session}
    session.close()


def docker_ready() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", tools.SANDBOX_IMAGE], capture_output=True).returncode == 0


def test_read_file_returns_only_the_head_and_points_at_python(ctx):
    out = tools.call_tool(ctx, "read_file", {"name": "notes.txt", "offset": 8000, "length": 9000})  # paging is not offered
    body, last = out.rsplit("\n", 1)
    assert body == ("0123456789" * 1000)[:tools.READ_CHARS]
    assert "10000-byte" in last and "/data/notes.txt" in last and "python" in last


def test_read_file_cannot_leave_files_dir(ctx):
    out = tools.call_tool(ctx, "read_file", {"name": "../secret.txt"})
    assert out.startswith("ERROR") and "outside" not in out


@pytest.mark.parametrize("name,args,needle", [
    ("no_such_tool", {}, "unknown tool"),
    ("read_file", {}, "name"),                                                      # missing required
    ("read_file", {"name": "nope.txt"}, "nope.txt"),
    ("python", {"code": 3}, "code"),
    ("read_file", ["notes.txt"], "JSON object"),                                    # arguments that are not an object
])
def test_bad_input_is_an_error_string(ctx, name, args, needle):
    out = tools.call_tool(ctx, name, args)
    assert out.startswith("ERROR"), out
    assert needle in out
    assert "Traceback" not in out


def container_exists(name: str) -> bool:
    return bool(subprocess.run(["docker", "ps", "-aq", "--filter", f"name={name}"], capture_output=True, text=True).stdout.strip())


# The python tool, through Docker: the real session, with the limits that only a container can show.

needs_docker = pytest.mark.skipif(not docker_ready(), reason="docker or the rsi-sandbox image is unavailable")


def python(ctx, code: str) -> str:
    return tools.call_tool(ctx, "python", {"code": code})


@needs_docker
def test_python_keeps_state_between_calls_and_survives_an_exception(ctx):
    assert python(ctx, "import pandas as pd\ndf = pd.read_csv('/data/pairs.csv')") == "(no output)\n"
    assert python(ctx, "int(df.y.sum())") == "6\n"
    failed = python(ctx, "print('so far')\ndf.nope")
    assert failed.startswith("so far\n") and "AttributeError" in failed and "df.nope" in failed
    assert python(ctx, "len(df)") == "2\n"                                 # the interpreter and df are still there
    assert python(ctx, "open('note.txt', 'w').write('kept')\nimport os\nos.getcwd()") == "'/work'\n"
    assert python(ctx, "open('/work/note.txt').read()") == "'kept'\n"
    assert container_exists(ctx["python"].name)
    name = ctx["python"].name
    ctx["python"].close()
    assert not container_exists(name)


@needs_docker
def test_python_has_no_network_and_cannot_write_outside_scratch(ctx):
    net = python(ctx, "import urllib.request as u\nu.urlopen('http://93.184.216.34', timeout=3)")
    assert "URLError" in net or "OSError" in net
    assert "Read-only file system" in python(ctx, "open('/data/new.txt', 'w')")
    assert "Read-only file system" in python(ctx, "open('/usr/new.txt', 'w')")
    assert not (ctx["files_dir"] / "new.txt").exists()
    assert python(ctx, "import os\nsorted(os.listdir('/sys/class/net'))") == "['lo']\n"


@needs_docker
def test_python_timeout_restarts_the_interpreter_and_says_so(ctx, monkeypatch):
    monkeypatch.setattr(tools, "PYTHON_LIMIT_S", 2)
    assert python(ctx, "x = 1\nopen('f.txt', 'w').write('1')\nx") == "1\n"
    first = ctx["python"].name
    out = python(ctx, "while True: pass")
    assert out.startswith("ERROR timeout (2s)") and "restarted" in out and "variables" in out and "/work" in out
    assert not container_exists(first)
    after = python(ctx, "import os\nprint(os.path.exists('f.txt'))\nx")
    assert after.startswith("False\n") and "NameError" in after  # a fresh interpreter: x and the file are gone
    assert ctx["python"].name != first


@needs_docker
def test_python_that_kills_its_own_process_restarts_and_says_so(ctx):
    out = python(ctx, "import os\nos._exit(1)")
    assert out.startswith("ERROR the interpreter stopped") and "restarted" in out
    assert python(ctx, "1 + 1") == "2\n"


@needs_docker
def test_python_output_flood_is_capped_before_it_reaches_the_host(ctx):
    out = python(ctx, "for i in range(300_000):\n    print('x' * 999)")  # 300 MB printed
    assert len(out) < 101_000 and "[output capped: 299900000 more characters were printed and not kept]" in out
    assert python(ctx, "1 + 1") == "2\n"


FORGE = """
import json, os, sys, threading, time
line = (json.dumps({"id": None, "stdout": "SPOOF", "stderr": "", "error": None, "dropped": 0}) + chr(10)).encode()
def spray(wait):
    time.sleep(wait)
    for fd in range(32):
        try:
            os.write(fd, line)          # a whole forged reply
            os.write(fd, b"z" * 5000)   # and an unfinished line for the real reply to run into
        except OSError:
            pass
spray(0)
threading.Thread(target=spray, args=(1.5,)).start()   # lands in the middle of the next call
try:
    open("/proc/1/fd/1", "wb")
    host_stream = "reachable"
except OSError:
    host_stream = "closed"
kept = 41
print("first", host_stream, repr(sys.stdin.readline()))
"""


@needs_docker
def test_python_code_cannot_forge_a_reply_or_shift_later_ones(ctx):
    # Observed: the loop's stream to the host was file descriptor 3 of the process running the code, so one
    # os.write there was read as the call's reply and every later call got the reply of the call before it.
    assert python(ctx, FORGE) == "first closed ''\n"
    assert python(ctx, "import time\ntime.sleep(3)\nprint('second')") == "second\n"
    assert python(ctx, "kept + 1") == "42\n"  # same interpreter throughout: nothing was restarted to get here
