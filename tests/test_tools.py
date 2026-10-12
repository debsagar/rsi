"""Tools: real statistics on a small fixture with known answers, ERROR strings
for every kind of bad input, the SIMULATED: prefix, and what the python sandbox
refuses or survives. The answered path (list files, load, compute, a second
session seeing nothing) is in tests/test_e2e.py."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from rsi import tools

GROUP_CSV = "group,score,label\n" + "".join(f"a,{10 + i},x{i}\n" for i in range(1, 9)) + \
    "".join(f"b,{i},y{i}\n" for i in range(1, 9))
XS = list(range(30))
YS = [2 * x + (1 if x % 2 else -1) for x in XS]
PAIR_CSV = "x,y\n" + "".join(f"{x},{y}\n" for x, y in zip(XS, YS))
MISSING_CSV = "group,score\na,1\na,\nb,3\nb,4\n"


@pytest.fixture
def ctx(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    (files / "groups.csv").write_text(GROUP_CSV)
    (files / "pairs.csv").write_text(PAIR_CSV)
    (files / "missing.csv").write_text(MISSING_CSV)
    (files / "notes.txt").write_text("0123456789" * 10)
    (tmp_path / "secret.txt").write_text("outside")
    session = tools.PythonSession(files)  # starts no container unless a test calls the python tool
    yield {"files_dir": files, "run_dir": tmp_path, "python": session}
    session.close()


def docker_ready() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", tools.SANDBOX_IMAGE], capture_output=True).returncode == 0


def test_read_file_offset_and_length(ctx):
    out = tools.call_tool(ctx, "read_file", {"name": "notes.txt", "offset": 8, "length": 4})
    assert out.endswith("8901")
    assert "of 100" in out  # the header says how much of the file this is


def test_read_file_cannot_leave_files_dir(ctx):
    out = tools.call_tool(ctx, "read_file", {"name": "../secret.txt"})
    assert out.startswith("ERROR") and "outside" not in out


def test_describe_table(ctx):
    out = tools.call_tool(ctx, "describe_table", {"file": "missing.csv"})
    assert "4 rows" in out
    assert "score" in out and "1 missing" in out


def test_compare_groups_known_difference(ctx):
    out = json.loads(tools.call_tool(ctx, "compare_groups", {"file": "groups.csv", "value": "score", "group": "group"}))
    assert (out["group_a"], out["group_b"]) == ("a", "b")
    assert out["mean_diff"] == pytest.approx(10.0)
    assert out["p_value"] < 0.01
    lo, hi = out["ci95"]
    assert lo < 10.0 < hi and lo > 5


def test_compare_groups_is_deterministic(ctx):
    args = {"file": "groups.csv", "value": "score", "group": "group", "a": "b", "b": "a"}
    first = tools.call_tool(ctx, "compare_groups", args)
    assert first == tools.call_tool(ctx, "compare_groups", args)
    assert json.loads(first)["mean_diff"] == pytest.approx(-10.0)


def test_correlate_known_correlation(ctx):
    out = json.loads(tools.call_tool(ctx, "correlate", {"file": "pairs.csv", "x": "x", "y": "y"}))
    assert out["pearson_r"] == pytest.approx(np.corrcoef(XS, YS)[0, 1], abs=1e-4)
    assert out["spearman_rho"] > 0.99
    lo, hi = out["pearson_ci95"]
    assert lo <= out["pearson_r"] <= hi <= 1.0


@pytest.mark.parametrize("name,args,needle", [
    ("no_such_tool", {}, "unknown tool"),
    ("read_file", {}, "name"),                                                      # missing required
    ("read_file", {"name": "nope.txt"}, "nope.txt"),
    ("read_file", {"name": "notes.txt", "offset": "x"}, "offset"),
    ("describe_table", {"file": "notes.txt"}, "notes.txt"),                         # not a table
    ("compare_groups", {"file": "groups.csv", "value": "nope", "group": "group"}, "nope"),
    ("compare_groups", {"file": "groups.csv", "value": "label", "group": "group"}, "label"),   # non-numeric
    ("compare_groups", {"file": "missing.csv", "value": "score", "group": "group"}, "score"),  # missing values
    ("compare_groups", {"file": "groups.csv", "value": "score", "group": "label"}, "label"),   # 16 groups, not 2
    ("compare_groups", {"file": "groups.csv", "value": "score", "group": "group", "a": "a", "b": "zzz"}, "zzz"),
    ("correlate", {"file": "groups.csv", "x": "score", "y": "label"}, "label"),
    ("correlate", {"file": "pairs.csv", "x": "x", "y": "nope"}, "nope"),
    ("python", {"code": 3}, "code"),
    ("read_file", ["notes.txt"], "JSON object"),                                    # arguments that are not an object
])
def test_bad_input_is_an_error_string(ctx, name, args, needle):
    out = tools.call_tool(ctx, name, args)
    assert out.startswith("ERROR"), out
    assert needle in out
    assert "Traceback" not in out


@pytest.mark.parametrize("name,args", [("arxiv_search", {"query": "permutation tests"}),
                                       ("citation_check", {"citation": "Fisher 1935"})])
def test_simulated_tools_say_so(ctx, name, args):
    assert tools.TOOLS[name]["simulated"] is True
    assert tools.call_tool(ctx, name, args).startswith("SIMULATED:")


def container_exists(name: str) -> bool:
    return bool(subprocess.run(["docker", "ps", "-aq", "--filter", f"name={name}"], capture_output=True, text=True).stdout.strip())


# The python tool, through Docker: the real session, with the limits that only a container can show.

needs_docker = pytest.mark.skipif(not docker_ready(), reason="docker or the rsi-sandbox image is unavailable")


def python(ctx, code: str) -> str:
    return tools.call_tool(ctx, "python", {"code": code})


@needs_docker
def test_python_keeps_state_between_calls_and_survives_an_exception(ctx):
    assert python(ctx, "import pandas as pd\ndf = pd.read_csv('/data/pairs.csv')") == "(no output)\n"
    assert python(ctx, "int(df.y.sum())") == f"{sum(YS)}\n"
    failed = python(ctx, "print('so far')\ndf.nope")
    assert failed.startswith("so far\n") and "AttributeError" in failed and "df.nope" in failed
    assert python(ctx, "len(df)") == "30\n"                                 # the interpreter and df are still there
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
