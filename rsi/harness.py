"""Seed harness: the loop that drives the agent. `rsi evolve` rewrites this file.

Exports run(task, api). It runs in a no-network container with the standard
library only; every model, tool and Jev call goes through `api` (see
rsi/runner.py). The broker owns the limits and the pre-submit gate, so
nothing here is a safety mechanism.

Invariants:
- A reply with no tool call is the answer and is submitted, except that when
  the question asks for an 'ANSWER:' line and the reply has none, the model is
  told to carry on, at most MAX_CONTINUES times and never on the last step.
- An answer the broker sent back is never dropped: if the steps run out before
  another answer is submitted, the last one sent back is submitted again.
- A claim is one line of the answer plus the event ids it cites as [n].
"""
import re

MAX_STEPS = 12      # the broker allows 12 model calls; the last one is made without tools
MAX_CONTINUES = 2   # replies without the ANSWER line that are sent back instead of submitted
JEV_EVERY = 3
THRESHOLD = 0.65
SYSTEM = ("You are a careful research assistant. Answer the question from evidence, not memory: use the tools to "
          "look at the attached files and compute what you need. Each tool result is labelled [event N]. In your "
          "final answer, end every sentence that states a number or a finding with the label of the result it came "
          "from, like [7]. Say plainly when the evidence is weak or a result is simulated. The python tool is one "
          "interpreter that lasts the whole task: variables, imports and files in /work persist between calls, so work "
          "in small steps, look at each result before the next step, and reuse what you already loaded or computed. "
          "You always have the python tool, and every attached file is at /data/<file name>: load data files there with "
          "pandas (pd.read_csv('/data/<file name>')) and compute the answer in python. read_file is only for a first "
          "look at the top of a file; never page through a data file with it. "
          "Do not ask the user questions; nobody will reply. When you have the answer, "
          "reply with it and call no tool. End your final reply with a line 'ANSWER: <value>' when the question asks for "
          "a specific answer format.")
CONTINUE = ("That reply has no 'ANSWER:' line, so it was not submitted. Either keep working by calling a tool, "
            "or finish now with your answer and the 'ANSWER:' line the question asks for.")
LAST_STEP = "This is your last step and tools are no longer available. Give your final answer now, in the format the question asks for."
# Observed on DABench: 10 of 24 failed runs paged a CSV with read_file, never called python, and ran out of calls.
USE_PYTHON = ("Stop reading the file page by page. You have the python tool: call it now with "
              "`import pandas as pd; df = pd.read_csv('/data/{name}'); print(df.shape); print(df.head())` "
              "and compute the answer there.")
NUDGE = ("Re-evaluate your current hypothesis from the observed evidence and tool results "
         "before taking another action.")
# Wording follows /data/jev_stf/jev.py, with its coding-agent nouns swapped for this agent's.
QUESTIONS = {
    "stuck": {"type": "noul",
              "instructions": "Is the research agent in `recent_events` stuck: repeating the same tool calls or the same "
                              "errors without getting closer to answering `task`?",
              "criteria": {"true": "recent steps repeat calls, arguments or failing results with no new information or progress",
                           "false": "the agent is still exploring, making new calls, or its results are getting closer to an answer"}},
    "wrong_hypothesis": {"type": "noul",
                         "instructions": "Is the research agent in `recent_events` acting on a wrong hypothesis: do the tool "
                                         "results or errors it observed contradict what its latest calls assume about `task`?",
                         "criteria": {"true": "observed results point to a cause or requirement the agent's calls are not addressing",
                                      "false": "the agent's calls are consistent with the evidence it has seen so far"}},
}


def _claims(text, seen):
    claims = []
    for line in text.splitlines():
        ids = [int(n) for n in re.findall(r"\[(\d+)\]", line) if int(n) in seen]
        if ids:
            claims.append({"text": line.strip(), "evidence": ids})
    return claims


def run(task, api):
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in task["history"]:
        messages += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
    files = ", ".join(task["files"]) or "none"
    messages.append({"role": "user", "content": f"{task['question']}\n\nAttached files: {files}"})
    recent, seen, continues, rejected = [], set(), 0, None
    reads, used_python = 0, False
    for step in range(1, MAX_STEPS + 1):
        if step == MAX_STEPS:
            messages.append({"role": "user", "content": LAST_STEP})
        message = api.generate(messages, use_tools=step < MAX_STEPS)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            text = message.get("content") or ""
            if "ANSWER:" in task["question"] and not re.search(r"ANSWER\**\s*:", text) and continues < MAX_CONTINUES and step < MAX_STEPS:
                continues += 1  # mid-work thought, a question back, or a tool call written as prose
                messages.append({"role": "user", "content": CONTINUE})
                continue
            verdict = api.submit(text, _claims(text, seen))
            if verdict["accepted"]:
                return
            rejected = text
            messages.append({"role": "user", "content": f"Your answer was not accepted: {verdict['reason']} Check it "
                                                        "against the tool results, fix what they do not support, and answer again."})
            continue
        for call in calls:
            name, args = call["function"]["name"], call["function"]["arguments"]
            out = api.execute(name, args)  # the raw argument string: the broker parses it and reports bad JSON
            seen.add(out["event_id"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": f"[event {out['event_id']}]\n{out['result']}"})
            recent.append({"step": step, "tool": name, "args": str(args)[:400], "result": out["result"][:600]})
            used_python = used_python or name == "python"
            reads += name == "read_file"
        if not used_python and reads >= 2 and reads % 2 == 0 and task["files"]:
            messages.append({"role": "user", "content": USE_PYTHON.format(name=task["files"][0])})
        if step % JEV_EVERY == 0:
            answers = api.judge({"task": task["question"][:1500], "recent_events": recent[-15:]}, QUESTIONS)
            if answers:  # None means Jev is unavailable; carry on without it
                p = max(answers["stuck"]["noul"], answers["wrong_hypothesis"]["noul"])
                if p > THRESHOLD:  # either fired (> 0.5) and the stronger one is confident
                    messages.append({"role": "user", "content": NUDGE})
    if rejected is not None:  # sent back on the last step: offer it again, the broker accepts a second submit
        api.submit(rejected, _claims(rejected, seen))
