"use client";

import { ClipboardList, FileText, TriangleAlert } from "lucide-react";
import { EventChip } from "@/components/event-detail";
import { PaneEmpty, PaneError, PaneLoading } from "@/components/pane-state";
import { answerEvent, asText, claimsOf, fmtCost, fmtInt, fmtSeconds, lastError, shortHash, stopReasonText, textOf } from "@/lib/events";
import type { Limits, RunView } from "@/lib/types";

// `of` is the broker's limit for this count; without it (an older server) only the count is shown.
function Stat({ label, value, of }: { label: string; value: string; of?: { used: number; max: number } }) {
  return (
    <div className="flex flex-col gap-1 bg-surface px-3 py-2.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-heading text-[17px] leading-tight font-semibold tabular-nums">{value}</dd>
      {of ? (
        <div
          className="mt-0.5 h-1 overflow-hidden rounded-[1px] bg-line"
          role="meter"
          aria-label={`${label} used`}
          aria-valuemin={0}
          aria-valuemax={of.max}
          aria-valuenow={of.used}
        >
          <div className="h-full bg-foreground/70" style={{ width: `${Math.min(100, (of.used / of.max) * 100)}%` }} />
        </div>
      ) : null}
    </div>
  );
}

export function ResultsPanel({
  state,
  errorMessage,
  onRetry,
  run,
  sessionId,
  limits,
  onSelectEvent,
}: {
  state: "loading" | "error" | "ready";
  errorMessage?: string;
  onRetry?: () => void;
  run: RunView | undefined;
  sessionId: string | null;
  // from GET /api/harnesses; undefined until it loads or when the server does not send it
  limits?: Limits;
  onSelectEvent: (id: number) => void;
}) {
  if (state === "loading") return <PaneLoading label="Loading results" lines={6} />;
  if (state === "error") {
    return <PaneError title="Results did not load" message={errorMessage ?? ""} onRetry={onRetry} />;
  }
  if (!run) {
    return (
      <PaneEmpty icon={<ClipboardList aria-hidden />} title="No results yet">
        When a run finishes, its claims, the evidence behind each one, and what the run cost appear here.
      </PaneEmpty>
    );
  }

  const ep = run.episode;
  const claims = claimsOf(run);
  const events = run.events;
  const modelCalls = ep?.calls ?? events.filter((e) => e.kind === "model").length;
  const toolCalls = ep?.tool_calls ?? events.filter((e) => e.kind === "tool_call").length;
  const cost = ep?.cost ?? events.reduce((sum, e) => sum + (typeof e.cost === "number" ? e.cost : 0), 0);
  const unsupported = ep?.unsupported === true || answerEvent(events)?.unsupported === true;
  const clipped = events.filter((e) => e.clipped);
  const failure = lastError(events);
  const counted = (used: number, max: number | undefined) =>
    typeof max === "number" && max > 0 ? { value: `${used} of ${max}`, of: { used, max } } : { value: String(used) };
  const folder = sessionId && ep?.run_id ? `runs/sessions/${sessionId}/${ep.run_id}/` : null;

  return (
    <div className="flex flex-col gap-5 p-4">
      {run.status === "streaming" ? (
        <p className="text-[13px] text-muted-foreground">The run is still going. Totals below are so far.</p>
      ) : null}
      {run.status === "error" && run.error ? (
        <div className="-m-4 mb-0">
          <PaneError title="The run failed" message={run.error} />
        </div>
      ) : null}

      {run.log_unreadable ? (
        <div role="alert" className="flex items-start gap-2 rounded-md border border-k-jev/40 bg-k-jev/8 p-3 text-[13px]">
          <TriangleAlert className="mt-0.5 size-4 shrink-0 text-k-jev" aria-hidden />
          <p>
            Part of this run&apos;s log could not be read. The trajectory and the totals below leave out what was
            unreadable{ep ? "" : ", and the run summary is missing, so the run is shown as unfinished"}.
          </p>
        </div>
      ) : null}

      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-md border bg-line sm:grid-cols-3">
        <Stat label="Cost" value={fmtCost(cost)} />
        <Stat label="Model calls" {...counted(modelCalls, limits?.model_calls)} />
        <Stat label="Tool calls" {...counted(toolCalls, limits?.tool_calls)} />
        <Stat label="Elapsed" value={fmtSeconds(ep?.elapsed)} />
        <Stat label="Tokens in" value={fmtInt(ep?.usage?.in)} />
        <Stat label="Tokens out" value={fmtInt(ep?.usage?.out)} />
      </dl>

      {ep && ep.stop_reason && ep.stop_reason !== "submitted" ? (
        <div role="alert" className="flex items-start gap-2 rounded-md border border-k-error/40 bg-k-error/8 p-3 text-[13px]">
          <TriangleAlert className="mt-0.5 size-4 shrink-0 text-k-error" aria-hidden />
          <div className="min-w-0">
            <p className="font-semibold">
              No answer: {stopReasonText(ep.stop_reason)}{" "}
              <span className="font-normal text-muted-foreground">({ep.stop_reason})</span>
            </p>
            {failure ? (
              <p className="mt-1 flex flex-wrap items-center gap-2 break-words text-muted-foreground">
                {failure.message}
                <EventChip event={failure} onSelect={onSelectEvent} />
              </p>
            ) : null}
          </div>
        </div>
      ) : null}

      {unsupported ? (
        <div role="alert" className="flex items-start gap-2 rounded-md border border-k-error/40 bg-k-error/8 p-3 text-[13px]">
          <TriangleAlert className="mt-0.5 size-4 shrink-0 text-k-error" aria-hidden />
          <p>
            The gate judged this answer as not supported by the trajectory, twice. It was accepted on the second
            submit and is marked unsupported.
          </p>
        </div>
      ) : null}

      <section className="flex flex-col gap-2">
        <h3 className="font-heading text-[15px] font-semibold">Claims and evidence</h3>
        {claims.length === 0 ? (
          <p className="text-[13px] text-muted-foreground">
            {run.status === "streaming"
              ? "Claims appear once the answer is submitted."
              : "This run's answer carries no claims."}
          </p>
        ) : (
          <ol className="flex flex-col gap-2">
            {claims.map((claim, i) => {
              const evidence = (claim.evidence ?? []).flatMap((id) => events.find((e) => e.id === id) ?? []);
              return (
                <li key={i} className="flex flex-col gap-2 rounded-md border bg-surface p-3">
                  <p className="text-[13.5px] leading-snug">{claim.text}</p>
                  {evidence.length === 0 ? (
                    <p className="text-xs text-k-error">No evidence event is linked to this claim.</p>
                  ) : (
                    evidence.map((e) => (
                      <div key={e.id} className="flex flex-col items-start gap-1.5">
                        <EventChip event={e} onSelect={onSelectEvent} />
                        <pre className="line-clamp-3 w-full overflow-hidden font-mono text-[11.5px] leading-normal whitespace-pre-wrap text-muted-foreground">
                          {(asText(e.result) || textOf(e)).slice(0, 320)}
                        </pre>
                      </div>
                    ))
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="font-heading text-[15px] font-semibold">Files from this run</h3>
        {ep ? (
          <ul className="flex flex-col divide-y rounded-md border bg-surface text-[13px]">
            {["events.jsonl", "episode.json"].map((name) => (
              <li key={name} className="flex items-center gap-2 px-3 py-2">
                <FileText className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                <code className="font-mono text-xs">{name}</code>
                <span className="ml-auto text-xs text-muted-foreground">
                  {name === "events.jsonl" ? `${events.length} events` : "run summary"}
                </span>
              </li>
            ))}
            {clipped.map((e) => (
              <li key={e.id} className="flex flex-wrap items-center gap-2 px-3 py-2">
                <FileText className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                <code className="font-mono text-xs">{e.file ?? "full tool output"}</code>
                <EventChip event={e} onSelect={onSelectEvent} />
                <span className="ml-auto text-xs text-muted-foreground tabular-nums">
                  {fmtInt(e.chars_full)} characters
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[13px] text-muted-foreground">Files are listed when the run finishes.</p>
        )}
        {folder ? (
          <p className="text-xs text-muted-foreground">
            Saved under <code className="font-mono">{folder}</code>
          </p>
        ) : null}
      </section>

      {ep ? (
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
          <dt className="text-muted-foreground">Stop reason</dt>
          <dd>{stopReasonText(ep.stop_reason)}</dd>
          <dt className="text-muted-foreground">Pre-submit gate</dt>
          <dd>
            {ep.gate === true
              ? "On: Jev rates an answer before it is accepted"
              : ep.gate === false
                ? "Off: an answer is accepted without a Jev check"
                : "Not recorded"}
          </dd>
          <dt className="text-muted-foreground">Model</dt>
          <dd className="break-all">{ep.model ?? "-"}</dd>
          <dt className="text-muted-foreground">Harness</dt>
          <dd>
            <code className="font-mono">{shortHash(ep.harness_sha256)}</code>
          </dd>
          {typeof ep.reward === "number" ? (
            <>
              <dt className="text-muted-foreground">Reward</dt>
              <dd className="tabular-nums">{ep.reward}</dd>
            </>
          ) : null}
        </dl>
      ) : null}
    </div>
  );
}
