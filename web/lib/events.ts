// Pure helpers over trajectory events: which track an event sits on, how
// calls pair with results, and how values are formatted for display.

import type {
  Claim,
  EventKind,
  LineageCandidate,
  LineageRow,
  Run,
  StopReason,
  TrajectoryEvent,
} from "@/lib/types";

export type TrackId = "model" | "tools" | "jev" | "answer";

export const TRACKS: { id: TrackId; label: string; color: string }[] = [
  { id: "model", label: "Model", color: "var(--k-model)" },
  { id: "tools", label: "Tools", color: "var(--k-call)" },
  { id: "jev", label: "Jev", color: "var(--k-jev)" },
  { id: "answer", label: "Answer", color: "var(--k-answer)" },
];

export const KIND: Record<EventKind, { label: string; color: string; track: TrackId }> = {
  user: { label: "Question", color: "var(--k-model)", track: "model" },
  model: { label: "Model", color: "var(--k-model)", track: "model" },
  tool_call: { label: "Tool call", color: "var(--k-call)", track: "tools" },
  tool_result: { label: "Tool result", color: "var(--k-result)", track: "tools" },
  judge: { label: "Jev check", color: "var(--k-jev)", track: "jev" },
  submit: { label: "Gate verdict", color: "var(--k-jev)", track: "jev" },
  answer: { label: "Answer", color: "var(--k-answer)", track: "answer" },
  error: { label: "Error", color: "var(--k-error)", track: "model" },
};

export function kindOf(event: TrajectoryEvent) {
  return KIND[event.kind] ?? { label: String(event.kind), color: "var(--muted-foreground)", track: "model" as TrackId };
}

// An error has no track of its own: it is drawn where the run was when it
// failed, which is the track of the event before it.
export function trackOf(events: TrajectoryEvent[], index: number): TrackId {
  const event = events[index];
  if (event.kind !== "error") return kindOf(event).track;
  for (let i = index - 1; i >= 0; i--) {
    if (events[i].kind !== "error") return kindOf(events[i]).track;
  }
  return "model";
}

export function isRejected(event: TrajectoryEvent): boolean {
  return event.kind === "submit" && event.accepted === false;
}

// Maps each tool_call id to its tool_result id and back. The broker records
// call_id on every result; older or foreign logs fall back to order.
export function pairCalls(events: TrajectoryEvent[]): Map<number, number> {
  const pairs = new Map<number, number>();
  const open: TrajectoryEvent[] = [];
  for (const event of events) {
    if (event.kind === "tool_call") open.push(event);
    if (event.kind !== "tool_result") continue;
    let at = open.findIndex((call) => call.id === event.call_id);
    if (at === -1 && event.call_id === undefined) {
      at = open.findIndex((call) => call.tool === event.tool);
    }
    if (at === -1) continue;
    const [call] = open.splice(at, 1);
    pairs.set(call.id, event.id);
    pairs.set(event.id, call.id);
  }
  return pairs;
}

export function eventTitle(event: TrajectoryEvent): string {
  if (event.kind === "tool_call" || event.kind === "tool_result") {
    return event.tool ? `${kindOf(event).label}: ${event.tool}` : kindOf(event).label;
  }
  if (event.kind === "submit") {
    if (event.accepted === false) return "Gate verdict: rejected";
    // No estimate means the gate was off or Jev did not answer: nothing was checked.
    if (event.accepted === true) return typeof event.p_supported === "number" ? "Gate verdict: accepted" : "Submit: accepted unchecked";
  }
  if (event.kind === "judge" && event.source === "gate") return "Jev check: pre-submit gate";
  return kindOf(event).label;
}

// The prose an event carries: model events use `content`, the rest `text`.
export function textOf(event: TrajectoryEvent): string {
  return asText(event.kind === "model" ? (event.content ?? event.text) : (event.text ?? event.content));
}

export function lastError(events: TrajectoryEvent[]): TrajectoryEvent | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].kind === "error") return events[i];
  }
  return undefined;
}

// Record<StopReason, ...>: a reason added to the type without a text here fails the type-check.
export const STOP_REASONS: Record<StopReason, string> = {
  submitted: "Answer submitted",
  no_submit: "The harness exited without an accepted answer",
  harness_error: "The harness crashed",
  llm_error: "The model call failed",
  run_limit: "Run time limit reached",
  model_call_limit: "Model call limit reached",
  tool_call_limit: "Tool call limit reached",
  judge_call_limit: "Jev check limit reached",
  broker_error: "The broker itself failed; see the server log",
};

// The lookup still falls back to the raw string for a server newer than this build.
export function stopReasonText(reason: StopReason | undefined): string {
  return reason ? ((STOP_REASONS as Record<string, string>)[reason] ?? reason) : "-";
}

export function asText(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

// Tool arguments for display. The broker logs them as the model sent them,
// usually a JSON string; a string that is not valid JSON is shown as it is.
// Multi-line values (python code) are printed as text, not as one escaped line.
export function argsText(args: unknown): string {
  let value = args ?? {};
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return value as string;
    }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return asText(value);
  const entries = Object.entries(value);
  if (!entries.some(([, v]) => typeof v === "string" && v.includes("\n"))) return asText(value);
  return entries.map(([k, v]) => (typeof v === "string" ? `${k}:\n${v}` : `${k}: ${JSON.stringify(v)}`)).join("\n\n");
}

export function answerEvent(events: TrajectoryEvent[]): TrajectoryEvent | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].kind === "answer") return events[i];
  }
  return undefined;
}

export function claimsOf(run: Run): Claim[] {
  const claims = answerEvent(run.events)?.claims;
  return Array.isArray(claims) ? claims : [];
}

export function answerOf(run: Run): string {
  return run.answer || run.episode?.answer || answerEvent(run.events)?.text || "";
}

export function tokensOf(event: TrajectoryEvent): string {
  if (typeof event.tokens_in === "number" || typeof event.tokens_out === "number") {
    return `${fmtInt(event.tokens_in)} in, ${fmtInt(event.tokens_out)} out`;
  }
  const t = event.tokens;
  if (t === undefined || t === null) return "";
  if (typeof t === "number") return `${fmtInt(t)} tokens`;
  return `${fmtInt(t.in)} in, ${fmtInt(t.out)} out`;
}

export function fmtInt(n: number | undefined): string {
  return typeof n === "number" && Number.isFinite(n) ? n.toLocaleString("en-US") : "-";
}

export function fmtCost(n: number | undefined | null): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "-";
  return n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(4)}`;
}

export function fmtSeconds(n: number | undefined | null): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "-";
  if (n < 10) return `${n.toFixed(2)} s`;
  if (n < 60) return `${n.toFixed(1)} s`;
  return `${Math.floor(n / 60)} min ${Math.round(n % 60)} s`;
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10_240 ? 1 : 0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function fmtWhen(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString("en-GB", {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

// How far a candidate got. The order is the order of the stages.
export type CandidateStage = "invalid" | "screen" | "first" | "confirm" | "adopted";
const STAGE_ORDER: CandidateStage[] = ["invalid", "screen", "first", "confirm", "adopted"];

export type CandidateView = {
  candidate: LineageCandidate;
  // the stage that stopped it, or "adopted"
  stage: CandidateStage;
};

export type LineageView = {
  generation?: number;
  family?: string;
  parent?: string;
  parentScore?: number;
  nullScore?: number;
  candidates: CandidateView[];
  nVal?: number;
  nConfirm?: number;
  cost?: number;
  adopted: string | null;
  // one line for the generation: what was adopted, or the furthest any candidate got
  outcome: string;
};

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

export function readLineage(row: LineageRow): LineageView {
  const list = Array.isArray(row.candidates) ? row.candidates : [];
  const adopted = typeof row.adopted === "string" && row.adopted ? row.adopted : null;
  // Rows from before the confirmation draw do not mark the adopted candidate:
  // there it was the highest first-draw score.
  let winner = list.find((c) => c.reason === "adopted");
  if (!winner && adopted) {
    winner = list.filter((c) => num(c.score) !== undefined).sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0];
  }
  const candidates = list.map((candidate): CandidateView => {
    const stage: CandidateStage =
      candidate === winner ? "adopted"
      : candidate.valid === false ? "invalid"
      : num(candidate.score) === undefined ? (candidate.skipped?.val ? "first" : "screen")
      : candidate.confirm || candidate.decision ? "confirm"
      : "first";
    return { candidate, stage };
  });
  const furthest = Math.max(-1, ...candidates.map((c) => STAGE_ORDER.indexOf(c.stage)));
  const outcome = adopted
    ? `Adopted ${adopted}`
    : typeof row.stopped === "string" && row.stopped
      ? `Stopped early: ${row.stopped}`
      : [
          "No candidate was proposed",
          "No candidate was a valid harness",
          "Every valid candidate was dropped by the screen",
          "No candidate got past the first draw",
          "The first-draw winner failed the final check",
        ][furthest + 1] ?? "Not adopted";
  const count = (seeds: unknown) => (Array.isArray(seeds) ? seeds.length : undefined);
  return {
    generation: num(row.generation),
    family: typeof row.family === "string" ? row.family : undefined,
    parent: typeof row.parent === "string" ? row.parent : undefined,
    parentScore: num(row.parent_score),
    nullScore: num(row.null_score),
    candidates,
    nVal: num(row.n_val) ?? count(row.val_seeds),
    nConfirm: count(row.confirm_seeds),
    cost: num(row.cost),
    adopted,
    outcome,
  };
}

export function shortHash(hash: string | null | undefined): string {
  return hash ? hash.slice(0, 12) : "-";
}
