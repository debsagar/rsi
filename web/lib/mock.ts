// Dev mock for NEXT_PUBLIC_MOCK=1. mockFetch() answers the HTTP contract with
// real Response objects (including a streamed text/event-stream body), so the
// UI can be built and screenshotted without the backend.
//
// ?mock=empty  serves no sessions and no lineage (empty states)
// ?mock=error  fails every call (error states)

import type {
  Claim,
  Episode,
  HarnessInfo,
  Run,
  SessionDetail,
  SessionFile,
  TrajectoryEvent,
} from "@/lib/types";

const T0 = Date.parse("2026-10-12T09:14:02Z");
const at = (seconds: number) => new Date(T0 + seconds * 1000).toISOString();

function describeOutput(): { shown: string; full: number } {
  const lines = [
    "runs.csv: 480 rows, 38 columns",
    "",
    "column                 type     count  missing      mean       std       min       max",
  ];
  const names = [
    "run_id", "schedule", "seed", "lr_peak", "warmup_steps", "batch_size",
    "final_val_loss", "best_val_loss", "train_loss", "grad_norm", "wall_minutes",
  ];
  for (let i = 0; lines.join("\n").length < 9120; i++) {
    const name = i < names.length ? names[i] : `layer_${i - names.length}_act_norm`;
    const kind = name === "run_id" || name === "schedule" ? "string " : "float64";
    const mean = (0.8 + ((i * 37) % 100) / 61).toFixed(4);
    const std = (0.05 + ((i * 13) % 40) / 173).toFixed(4);
    lines.push(
      `${name.padEnd(22)} ${kind}    480        ${i % 9 === 4 ? 3 : 0}    ${mean}    ${std}    ${(Number(mean) - 3 * Number(std)).toFixed(4)}    ${(Number(mean) + 3 * Number(std)).toFixed(4)}`,
    );
  }
  const full = lines.join("\n").slice(0, 9120);
  const shown = `${full.slice(0, 2000)}\n[... 5,120 characters clipped; full text in tool-001.txt ...]\n${full.slice(-2000)}`;
  return { shown, full: full.length };
}

const QUESTION =
  "Does the warm-start schedule reach a lower final validation loss than the baseline in runs.csv? Is the difference real?";

const DRAFT =
  "Warm-start reaches a lower final validation loss than baseline (1.412 vs 1.538). This matches prior work showing warm restarts always improve convergence.";

const ANSWER = `**Yes. Warm-start ends lower, and the gap is unlikely to be chance.**

Across 240 runs per schedule, the mean final validation loss is **1.412** for warm-start and **1.538** for baseline, a difference of **-0.126**.

| Schedule | Runs | Mean final val loss |
|---|---|---|
| warm-start | 240 | 1.412 |
| baseline | 240 | 1.538 |

- Permutation test (10,000 shuffles): p = 0.0004
- Bootstrap 95% interval for the difference: -0.171 to -0.082

The literature lookup is simulated in this build, so I am not citing it as support.

\`final_val_loss\` has no missing values; 3 rows are missing \`warmup_steps\`, which this comparison does not use.`;

const CLAIMS: Claim[] = [
  {
    text: "Mean final validation loss is 1.412 for warm-start and 1.538 for baseline (difference -0.126).",
    evidence: [7],
  },
  {
    text: "The difference is significant: permutation p = 0.0004, bootstrap 95% interval -0.171 to -0.082.",
    evidence: [7],
  },
  {
    text: "final_val_loss has no missing values across the 480 rows.",
    evidence: [4],
  },
];

const call = (name: string, args: Record<string, unknown>, n: number) => ({
  id: `call_${n}`, type: "function", function: { name, arguments: JSON.stringify(args) },
});

const GATE_QUESTIONS = ["supported"];

// Shapes follow rsi/broker.py: `step` is the number of model calls so far,
// and the gate logs a judge event before every submit.
function script(question = QUESTION): TrajectoryEvent[] {
  const table = describeOutput();
  const compareArgs = { file: "runs.csv", group: "schedule", value: "final_val_loss" };
  const searchArgs = { query: "warm restarts learning rate schedule validation loss" };
  const draftClaims: Claim[] = [
    { text: "Warm-start reaches a lower final validation loss (1.412 vs 1.538).", evidence: [7] },
    { text: "Prior work shows warm restarts always improve convergence.", evidence: [9] },
  ];
  return [
    { id: 1, ts: at(0), step: 0, kind: "user", text: question, files: ["runs.csv"] },
    {
      id: 2, ts: at(2.1), step: 1, kind: "model", elapsed: 2.08, tokens_in: 1184, tokens_out: 96, cost: 0.0049,
      content: "I'll look at the table's columns first, then compare the two schedules on final_val_loss.",
      tool_calls: [call("describe_table", { file: "runs.csv" }, 1)],
    },
    { id: 3, ts: at(2.2), step: 1, kind: "tool_call", tool: "describe_table", args: { file: "runs.csv" } },
    {
      id: 4, ts: at(2.5), step: 1, kind: "tool_result", tool: "describe_table", call_id: 3,
      result: table.shown, chars_full: table.full, chars_shown: 4000, clipped: true, file: "tool-001.txt",
      simulated: false, elapsed: 0.31,
    },
    {
      id: 5, ts: at(5.4), step: 2, kind: "model", elapsed: 2.87, tokens_in: 2410, tokens_out: 142, cost: 0.0093,
      content: "The grouping column is `schedule` and the outcome is `final_val_loss`. I'll run the group comparison and check whether there is prior work on warm restarts.",
      tool_calls: [call("compare_groups", compareArgs, 2), call("arxiv_search", searchArgs, 3)],
    },
    { id: 6, ts: at(5.5), step: 2, kind: "tool_call", tool: "compare_groups", args: compareArgs },
    {
      id: 7, ts: at(5.9), step: 2, kind: "tool_result", tool: "compare_groups", call_id: 6,
      result: [
        "groups: baseline (n=240), warm-start (n=240)",
        "mean[baseline]   = 1.5380",
        "mean[warm-start] = 1.4120",
        "difference (warm-start - baseline) = -0.1260",
        "permutation p (two-sided, 10000 shuffles, seed 0) = 0.0004",
        "bootstrap 95% interval for the difference = [-0.1710, -0.0820]",
      ].join("\n"),
      chars_full: 268, chars_shown: 268, clipped: false, simulated: false, elapsed: 0.41,
    },
    { id: 8, ts: at(6.0), step: 2, kind: "tool_call", tool: "arxiv_search", args: searchArgs },
    {
      id: 9, ts: at(6.1), step: 2, kind: "tool_result", tool: "arxiv_search", call_id: 8,
      result: [
        "SIMULATED: fixture records, not a live search.",
        "1. SGDR: Stochastic Gradient Descent with Warm Restarts (fixture)",
        "2. A Closer Look at Learning Rate Warmup (fixture)",
      ].join("\n"),
      chars_full: 161, chars_shown: 161, clipped: false, simulated: true, elapsed: 0.02,
    },
    {
      id: 10, ts: at(9.3), step: 3, kind: "model", elapsed: 3.12, tokens_in: 2986, tokens_out: 188, cost: 0.0118,
      content: DRAFT, tool_calls: [],
    },
    { id: 11, ts: at(10.1), step: 3, kind: "judge", source: "gate", questions: GATE_QUESTIONS, answers: { supported: { noul: 0.34 } } },
    {
      id: 12, ts: at(10.2), step: 3, kind: "submit", text: DRAFT, claims: draftClaims,
      accepted: false, p_supported: 0.34,
      reason: "Jev rates the chance that this answer is supported by the tool results at 0.34, below 0.5.",
    },
    {
      id: 13, ts: at(14.0), step: 4, kind: "model", elapsed: 3.74, tokens_in: 3302, tokens_out: 311, cost: 0.0146,
      content: ANSWER, tool_calls: [],
    },
    { id: 14, ts: at(14.7), step: 4, kind: "judge", source: "gate", questions: GATE_QUESTIONS, answers: { supported: { noul: 0.91 } } },
    { id: 15, ts: at(14.8), step: 4, kind: "submit", text: ANSWER, claims: CLAIMS, accepted: true, reason: "", p_supported: 0.91 },
    { id: 16, ts: at(14.9), step: 4, kind: "answer", text: ANSWER, claims: CLAIMS, unsupported: false },
  ];
}

function episode(runId: string, sessionId: string): Episode {
  return {
    run_id: runId, session_id: sessionId, task_id: runId,
    harness_sha256: "9c41e7d20b6a58f3c1d94e0a7b2f6c85d3e19a4470fb26c8e5a1d07b93c4f2e6",
    model: "mock-model", calls: 4, tool_calls: 3,
    usage: { in: 9882, out: 737 }, cost: 0.0406,
    stop_reason: "submitted", unsupported: false, gate: true, answer: ANSWER, reward: null, elapsed: 14.9,
  };
}

const FAIL_QUESTION = "Fit a power law to the loss curve in curve.csv and report the exponent.";
const FAIL_CODE = "import pandas as pd, numpy as np\ndf = pd.read_csv('/data/curve.csv')\nwhile True:\n    pass";

function failedScript(question = FAIL_QUESTION, t = -86400): TrajectoryEvent[] {
  return [
    { id: 1, ts: at(t), step: 0, kind: "user", text: question, files: ["curve.csv"] },
    {
      id: 2, ts: at(t + 2), step: 1, kind: "model", elapsed: 1.9, tokens_in: 1010, tokens_out: 74, cost: 0.0038,
      content: "I'll fit log(loss) against log(step) with the python tool.",
      tool_calls: [call("python", { code: FAIL_CODE }, 1)],
    },
    { id: 3, ts: at(t + 2), step: 1, kind: "tool_call", tool: "python", args: { code: FAIL_CODE } },
    {
      id: 4, ts: at(t + 62), step: 1, kind: "tool_result", tool: "python", call_id: 3,
      result: "ERROR tool call exceeded the 60 second limit", chars_full: 44, chars_shown: 44,
      clipped: false, simulated: false, elapsed: 60.0,
    },
    {
      id: 5, ts: at(t + 63), step: 1, kind: "error",
      message: "the harness exited with code 1 before submitting",
      stderr: "Traceback (most recent call last):\n  File \"/harness/harness.py\", line 41, in run\n    args = json.loads(call[\"arguments\"])\nKeyError: 'arguments'",
    },
  ];
}

function failedEpisode(runId: string, sessionId: string): Episode {
  return {
    run_id: runId, session_id: sessionId, task_id: runId, model: "mock-model",
    harness_sha256: "9c41e7d20b6a58f3c1d94e0a7b2f6c85d3e19a4470fb26c8e5a1d07b93c4f2e6",
    calls: 1, tool_calls: 1, usage: { in: 1010, out: 74 }, cost: 0.0038,
    stop_reason: "harness_error", unsupported: false, gate: false, answer: null, reward: null, elapsed: 63.2,
  };
}

const FAILED_RUN: Run = {
  run_id: "20261011T091402000000-1c9e", question: FAIL_QUESTION, answer: null,
  episode: failedEpisode("20261011T091402000000-1c9e", "5d0e7a44b1c2"), events: failedScript(),
  log_unreadable: true,
};

type Store = { sessions: SessionDetail[]; counter: number };

let store: Store | null = null;

function db(): Store {
  if (store) return store;
  const scenario = scenarioName();
  store = { counter: 8, sessions: [] };
  if (scenario !== "empty") {
    store.sessions = [
      {
        id: "a41f09c27be3", created: at(-40), title: QUESTION.slice(0, 60),
        files: [{ name: "runs.csv", size: 412_300, type: "text/csv" }],
        runs: [{
          run_id: "20261012T091402000000-7f3a", question: QUESTION, answer: ANSWER,
          episode: episode("20261012T091402000000-7f3a", "a41f09c27be3"), events: script(),
        }],
      },
      {
        id: "5d0e7a44b1c2", created: at(-86500), title: FAIL_QUESTION.slice(0, 60),
        files: [{ name: "curve.csv", size: 18_420, type: "text/csv" }],
        runs: [FAILED_RUN],
      },
      { id: "0c3b9e615fd8", created: at(-172900), title: "New session", files: [], runs: [] },
    ];
  }
  return store;
}

function scenarioName(): string {
  if (typeof window === "undefined") return "";
  return new URLSearchParams(window.location.search).get("mock") ?? "";
}

const taskIds = (from: number) => Array.from({ length: 30 }, (_, i) => `dabench-${from + i * 7}`);
const NOT_SCORED = { score: null, errors: null, wins: null, losses: null, p_value: null };

// Three generations, shaped as rsi/evolve.py writes them: a row from before
// the screen and the confirmation draw existed, a generation whose three
// candidates stop at three different stages, and an adoption.
const HARNESSES: HarnessInfo = {
  current: "001.py",
  lineage_skipped: 0,
  limits: { model_calls: 12, tool_calls: 12, judge_calls: 12 },
  lineage: [
    {
      generation: 0, family: "dabench", parent: "000.py", parent_sha: "fa8cf48bee16fb4310baca31d0d7de20",
      val_seeds: taskIds(11), parent_score: 0.7667, null_score: 0.0667, n_val: 30, adopted: null, cost: 2.024436,
      ts: at(-260000),
      candidates: [
        { sha: "36760cb51f566946fdb8d4a426ea214a", valid: true, reason: "scored 67%, not above the parent's 77%", score: 0.6667, errors: 0, wins: 2, losses: 5, p_value: 0.453125 },
        { sha: "2e3cb0962995535cb4d983a300a0b3e9", valid: true, reason: "scored 63%, not above the parent's 77%", score: 0.6333, errors: 0, wins: 2, losses: 6, p_value: 0.289062 },
        { sha: "c0ffee1234567890c0ffee1234567890", valid: false, reason: "defines no top-level run(task, api)", ...NOT_SCORED },
      ],
    },
    {
      generation: 1, family: "dabench", parent: "000.py", parent_sha: "fa8cf48bee16fb4310baca31d0d7de20",
      val_seeds: taskIds(300), confirm_seeds: taskIds(600), parent_score: 0.7, null_score: 0.0667, n_val: 30,
      adopted: null, cost: 3.412208, ts: at(-200000),
      candidates: [
        {
          sha: "77b2c0e4a913d5f60718293a4b5c6d7e", valid: true, ...NOT_SCORED, confirm: null,
          rationale: "Raise the Jev check from every third step to every step, so a wrong hypothesis is caught before more tool calls are spent on it.",
          screen: { fixed: 0, broken: 1, n: 10 },
          reason: "dropped by the screen on 10 train tasks: fixed 0 of the parent's 5 failures, broke 1 of 5 passes",
        },
        {
          sha: "b83d1f20c6e54a3928170f6e5d4c3b2a", valid: true, score: 0.6667, errors: 0, wins: 2, losses: 3, p_value: 1, confirm: null,
          rationale: "Tell the model to print the dataframe's dtypes before any aggregation; three of the failed tasks summed a column that had been read as text.",
          screen: { fixed: 2, broken: 1, n: 10 },
          reason: "scored 67%, not above the parent's 70%",
        },
        {
          sha: "d03f6b1e55c7a8192b3c4d5e6f708192", valid: true, score: 0.7667, errors: 0, wins: 4, losses: 2, p_value: 0.6875,
          rationale: "When the question asks for an ANSWER line with a number of decimals, restate that format in the last-step message; the parent lost four tasks to rounding.",
          screen: { fixed: 3, broken: 0, n: 10 },
          confirm: { n: 30, parent_score: 0.7333, score: 0.6667, errors: 0, wins: 1, losses: 3, combined_wins: 5, combined_losses: 5, combined_p_value: 1 },
          reason: "won the first draw but not the confirmation draw: 67% against the parent's 73% with 0 errored episodes; over both draws 72% against 72%",
        },
      ],
    },
    {
      generation: 2, family: "dabench", parent: "000.py", parent_sha: "fa8cf48bee16fb4310baca31d0d7de20",
      val_seeds: taskIds(900), confirm_seeds: taskIds(1200), parent_score: 0.7333, null_score: 0.1, n_val: 30,
      adopted: "001.py", cost: 3.655017, ts: at(-140000),
      candidates: [
        {
          sha: "41c9d27e80aa5b6c7d8e9f0a1b2c3d4e", valid: true, score: 0.8667, errors: 0, wins: 5, losses: 1, p_value: 0.21875,
          rationale: "Send a reply that has no ANSWER line back once more with the exact format quoted from the question, instead of submitting it; the parent's failures were mostly right numbers in the wrong format.",
          screen: { fixed: 4, broken: 0, n: 12 },
          confirm: { n: 30, parent_score: 0.7, score: 0.8, errors: 0, wins: 4, losses: 1, combined_wins: 9, combined_losses: 2, combined_p_value: 0.06543 },
          reason: "adopted",
        },
        {
          sha: "6f10a3bb92d40e1f2a3b4c5d6e7f8091", valid: true, score: 0.8, errors: 0, wins: 3, losses: 1, p_value: 0.625, confirm: null,
          rationale: "Lower the Jev nudge threshold from 0.65 to 0.55 so the agent re-checks its hypothesis sooner.",
          screen: { fixed: 2, broken: 1, n: 12 },
          reason: "beat the parent on the first draw, but another candidate scored higher",
        },
        {
          sha: "02ad9f7713ce4b5a69788796a5b4c3d2", valid: false, ...NOT_SCORED, screen: null, confirm: null,
          rationale: "Retry a failed tool call once with the same arguments before reporting the error to the model.",
          reason: "preflight task ended with harness_error",
        },
      ],
    },
  ],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// A question containing "fail" replays the crashed run, which ends the way
// the real server does: an `error` message and then `done`.
function sseBody(session: SessionDetail, question: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const runId = `${new Date().toISOString().replace(/[-:.Z]/g, "")}000-${(++db().counter).toString(16).padStart(4, "0")}`;
  const fails = /fail/i.test(question);
  const events = (fails ? failedScript(question) : script(question)).map((e) => ({
    ...e, ts: new Date().toISOString(),
  }));
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (name: string, data: unknown) =>
        controller.enqueue(encoder.encode(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`));
      for (const event of events) {
        if (event.kind === "model") {
          // Every model call streams its text, not just the final answer.
          const text = event.content ?? "";
          for (let i = 0; i < text.length; i += 24) {
            emit("delta", { text: text.slice(i, i + 24) });
            await wait(25);
          }
        }
        await wait(event.kind === "model" ? 500 : 260);
        emit("event", event);
      }
      const ep = fails ? failedEpisode(runId, session.id) : episode(runId, session.id);
      session.runs.push({ run_id: runId, question, answer: ep.answer ?? null, episode: ep, events });
      if (session.title === "New session") session.title = question.slice(0, 60);
      if (fails) emit("error", { message: "the harness exited with code 1 before submitting" });
      emit("done", { episode: ep });
      controller.close();
    },
  });
}

export async function mockFetch(path: string, init?: RequestInit): Promise<Response> {
  await wait(180);
  if (scenarioName() === "error") {
    return json({ detail: "Mock failure: the API returned an error." }, 500);
  }
  const method = (init?.method ?? "GET").toUpperCase();
  const state = db();
  const parts = path.replace(/^\/api\//, "").split("/").map(decodeURIComponent);

  if (parts[0] === "harnesses") {
    return json(scenarioName() === "empty" ? { ...HARNESSES, current: "000.py", lineage: [] } : HARNESSES);
  }
  if (parts[0] !== "sessions") return json({ detail: "Not found" }, 404);

  if (parts.length === 1) {
    if (method === "POST") {
      const created: SessionDetail = {
        id: Date.now().toString(16).padStart(12, "0").slice(-12), created: new Date().toISOString(),
        title: "New session", files: [], runs: [],
      };
      state.sessions.unshift(created);
      return json({ id: created.id, created: created.created, title: created.title });
    }
    return json(state.sessions.map(({ id, created, title }) => ({ id, created, title })));
  }

  const session = state.sessions.find((s) => s.id === parts[1]);
  if (!session) return json({ detail: "Session not found" }, 404);
  if (parts.length === 2) return json(session);

  if (parts[2] === "files" && method === "POST") {
    const file = init?.body instanceof FormData ? init.body.get("file") : null;
    if (!(file instanceof File)) return json({ detail: "No file in the request" }, 400);
    if (file.size > 25 * 1024 * 1024) return json({ detail: "File is over the 25 MB limit" }, 413);
    const saved: SessionFile = {
      name: file.name.replace(/^.*[\\/]/, ""), size: file.size,
      type: file.type || "application/octet-stream",
    };
    const replaced = session.files.some((f) => f.name === saved.name);
    session.files = [...session.files.filter((f) => f.name !== saved.name), saved];
    return json({ ...saved, replaced });
  }

  if (parts[2] === "messages" && method === "POST") {
    const body = JSON.parse(typeof init?.body === "string" ? init.body : "{}") as { question?: string };
    return new Response(sseBody(session, body.question ?? ""), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }
  return json({ detail: "Not found" }, 404);
}
