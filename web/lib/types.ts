// Mirrors the backend HTTP contract (plan: "Events and episode", "HTTP API").
// Field names stay snake_case so a payload can be used as received.

export type EventKind =
  | "user"
  | "model"
  | "tool_call"
  | "tool_result"
  | "judge"
  | "submit"
  | "answer"
  | "error";

export type Claim = {
  text: string;
  evidence: number[];
};

export type TokenUsage = { in: number; out: number };

export type ToolCallRequest = {
  id?: string;
  function?: { name?: string; arguments?: string };
  [key: string]: unknown;
};

// One line of events.jsonl, exactly as the broker wrote it (rsi/broker.py).
// Only id, ts, step and kind are always present. `step` is the number of
// model calls so far, so several events share a step; order is by `id`.
// The index signature keeps fields this UI does not know about, so the detail
// view can show them instead of dropping them.
export type TrajectoryEvent = {
  id: number;
  ts: string;
  step: number;
  kind: EventKind;
  // user
  files?: string[];
  // user, submit, answer
  text?: string;
  // model
  content?: string | null;
  tool_calls?: ToolCallRequest[];
  tokens_in?: number;
  tokens_out?: number;
  tokens?: TokenUsage | number;
  cost?: number;
  // tool_call (args is the raw string when the model sent malformed JSON)
  tool?: string;
  args?: Record<string, unknown> | string | null;
  // tool_result
  call_id?: number;
  result?: string;
  chars_full?: number;
  chars_shown?: number;
  clipped?: boolean;
  file?: string;
  elapsed?: number;
  // judge
  source?: "harness" | "gate";
  questions?: string[];
  answers?: Record<string, unknown> | null;
  // submit
  claims?: Claim[];
  accepted?: boolean;
  reason?: string;
  p_supported?: number | null;
  // answer
  unsupported?: boolean;
  // error (judge events also carry `error` when Jev did not answer)
  message?: string;
  stderr?: string;
  error?: string;
  [key: string]: unknown;
};

export type StopReason =
  | "submitted"
  | "no_submit"
  | "harness_error"
  | "llm_error"
  | "run_limit"
  | "model_call_limit"
  | "tool_call_limit"
  | "judge_call_limit"
  | "broker_error";

export type Episode = {
  run_id: string;
  session_id?: string | null;
  task_id?: string;
  harness_sha256?: string;
  model?: string | null;
  calls?: number;
  tool_calls?: number;
  usage?: TokenUsage;
  cost?: number;
  // A reason the backend adds must be added to StopReason, which makes
  // STOP_REASONS in lib/events.ts fail the type-check until it has a text.
  stop_reason?: StopReason;
  unsupported?: boolean;
  // whether the pre-submit gate was on for this run; absent in older episodes
  gate?: boolean;
  answer?: string | null;
  reward?: number | null;
  elapsed?: number;
};

export type SessionSummary = {
  id: string;
  created: string;
  title: string;
  // meta.json could not be read; created and title were rebuilt by the server
  meta_unreadable?: boolean;
};

export type SessionFile = {
  name: string;
  size: number;
  type: string;
  // on the upload response: this upload replaced a stored file of the same name
  replaced?: boolean;
};

export type Run = {
  run_id: string;
  question: string;
  answer: string | null;
  // null while a run is still in flight
  episode: Episode | null;
  events: TrajectoryEvent[];
  // some lines of events.jsonl, or episode.json, could not be parsed and are missing here
  log_unreadable?: boolean;
};

export type SessionDetail = SessionSummary & {
  files: SessionFile[];
  runs: Run[];
};

// The screen: the candidate on the parent's train failures and as many of its passes.
export type LineageScreen = { fixed: number; broken: number; n: number };

// The confirmation draw: parent and first-draw winner on fresh validation tasks.
export type LineageConfirm = {
  n: number;
  parent_score: number;
  score: number;
  errors: number;
  wins: number;
  losses: number;
  combined_wins: number;
  combined_losses: number;
  combined_p_value: number;
};

// The decision: fresh runs of parent and draw-A winner on both validation draws,
// paired by task. Adopted when errors is 0 and net (wins - losses) >= margin.
export type LineageDecision = {
  n: number;
  wins: number;
  losses: number;
  net: number;
  margin: number;
  errors: number;
  p_value: number;
  parent_score: number;
  score: number;
};

// The free replay: the parent's right answers handed to the candidate unchanged.
export type LineageReplay = {
  n: number;
  broken: number;
  no_submit?: number;
  examples: { task: string; before: string; after: string }[];
};

// One candidate of a generation (rsi/evolve.py, _score and _decide). A stage
// it never reached is null: dropped by the screen means valid with score,
// errors, wins, losses and p_value all null. Rows written before the decision
// existed carry confirm instead of decision; older ones carry neither, nor
// rationale and screen. sha is null for a slot that produced no file.
export type LineageCandidate = {
  sha?: string | null;
  valid?: boolean;
  // the backend's own words for what happened; "adopted" for the adopted one
  reason?: string | null;
  score?: number | null;
  errors?: number | null;
  wins?: number | null;
  losses?: number | null;
  p_value?: number | null;
  rationale?: string | null;
  // "edit", "whole file", "hand-written" or "skip"
  form?: string | null;
  replay?: LineageReplay | null;
  screen?: LineageScreen | null;
  // episodes not run because the candidate could no longer pass that stage
  skipped?: { screen: number; val: number } | null;
  confirm?: LineageConfirm | null;
  decision?: LineageDecision | null;
};

// A row of harnesses/lineage.jsonl: one per generation.
export type LineageRow = {
  generation?: number;
  family?: string;
  parent?: string;
  parent_sha?: string;
  candidates?: LineageCandidate[];
  // generated families draw numeric seeds, dabench draws task ids like "dabench-321"
  val_seeds?: (number | string)[] | null;
  confirm_seeds?: (number | string)[] | null;
  parent_score?: number | null;
  null_score?: number | null;
  n_val?: number;
  // file name of the adopted harness, or null
  adopted?: string | null;
  // why the generation stopped early, when it did
  stopped?: string;
  cost?: number;
  ts?: string;
};

// The broker's per-run call limits (rsi/broker.py).
export type Limits = { model_calls: number; tool_calls: number; judge_calls: number };

export type HarnessInfo = {
  current: string;
  lineage: LineageRow[];
  // lines of lineage.jsonl that could not be parsed; absent from older servers
  lineage_skipped?: number;
  // absent from older servers
  limits?: Limits;
};

// SSE payloads for POST /api/sessions/{id}/messages.
export type DeltaPayload = { text: string };
export type DonePayload = { episode: Episode };
export type ErrorPayload = { message: string };

// UI-side state shared by the panes.
export type Loadable<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: T };

export type RunView = Run & {
  status: "done" | "streaming" | "error";
  error?: string;
};
