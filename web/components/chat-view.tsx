"use client";

import { CircleAlert, LoaderCircle, Orbit, Telescope, TriangleAlert } from "lucide-react";
import { useEffect, useRef } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Composer } from "@/components/composer";
import { EventGlyph } from "@/components/event-glyph";
import { PaneEmpty, PaneError, PaneLoading } from "@/components/pane-state";
import { Button } from "@/components/ui/button";
import { answerEvent, answerOf, claimsOf, eventTitle, fmtCost, fmtSeconds, isRejected, lastError, stopReasonText } from "@/lib/events";
import type { RunView, SessionFile, TrajectoryEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

// react-markdown passes the syntax node along; it must not reach the DOM.
function domProps<T extends { node?: unknown }>(props: T): Omit<T, "node"> {
  const copy = { ...props };
  delete copy.node;
  return copy;
}

function Answer({ text, dim }: { text: string; dim?: boolean }) {
  return (
    <div className={cn("prose-answer text-sm", dim && "text-muted-foreground")}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: (props) => (
            <div className="table-wrap">
              <table {...domProps(props)} />
            </div>
          ),
          a: (props) => <a {...domProps(props)} target="_blank" rel="noreferrer" />,
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}

function progress(last: TrajectoryEvent | undefined): string {
  if (!last) return "Starting the run";
  if (last.kind === "tool_call") return `Running ${last.tool ?? "a tool"}`;
  if (last.kind === "tool_result") return `Reading the ${last.tool ?? "tool"} result`;
  if (isRejected(last)) return "The gate rejected the draft; revising";
  if (last.kind === "submit") return "The answer was accepted";
  // A harness check is followed by the next model call, usually the longest wait.
  if (last.kind === "judge") return last.source === "gate" ? "The gate checked the answer" : "Jev checked progress; thinking";
  if (last.kind === "error") return "Something went wrong; finishing up";
  return "Thinking";
}

function RunBlock({
  run,
  active,
  selectedEventId,
  streamText,
  onSelectRun,
  onSelectEvidence,
}: {
  run: RunView;
  active: boolean;
  selectedEventId: number | null;
  streamText: string;
  onSelectRun: (runId: string) => void;
  onSelectEvidence: (runId: string, eventId: number) => void;
}) {
  const answer = answerOf(run);
  const claims = claimsOf(run);
  const last = run.events[run.events.length - 1];
  const ep = run.episode;
  const unsupported = ep?.unsupported === true || answerEvent(run.events)?.unsupported === true;

  return (
    <article className="flex flex-col gap-3" aria-label={`Question: ${run.question}`}>
      <div className="max-w-[88%] self-end rounded-md bg-raised px-3 py-2 text-sm leading-normal break-words whitespace-pre-wrap">
        {run.question}
      </div>

      {run.status === "streaming" ? (
        <div role="status" className="flex items-center gap-2 text-[13px] text-muted-foreground">
          <LoaderCircle className="size-3.5 shrink-0 animate-spin" aria-hidden />
          <span>
            {progress(last)}
            {last ? <span className="tabular-nums"> · step {last.step}</span> : null}
          </span>
        </div>
      ) : null}

      {answer ? (
        <Answer text={answer} />
      ) : run.status === "streaming" && streamText ? (
        // Provisional: the text of the model call in progress. It is replaced
        // by the answer event, or cleared when the next model call starts.
        <div className="rounded-md border border-dashed px-3 py-2" aria-label="Model output in progress">
          <Answer text={streamText} dim />
        </div>
      ) : null}

      {run.status === "error" ? (
        <div role="alert" className="flex items-start gap-2 rounded-md border border-k-error/40 bg-k-error/8 p-3 text-[13px]">
          <CircleAlert className="mt-0.5 size-4 shrink-0 text-k-error" aria-hidden />
          <div className="min-w-0">
            <p className="font-semibold">The run failed</p>
            <p className="break-words text-muted-foreground">{run.error}</p>
          </div>
        </div>
      ) : run.status === "done" && !answer ? (
        <div className="flex items-start gap-2 rounded-md border border-k-error/40 bg-k-error/8 p-3 text-[13px]">
          <CircleAlert className="mt-0.5 size-4 shrink-0 text-k-error" aria-hidden />
          {ep ? (
            <div className="min-w-0">
              <p className="font-semibold">No answer: {stopReasonText(ep.stop_reason)}</p>
              <p className="break-words text-muted-foreground">
                {lastError(run.events)?.message ?? "The trajectory shows where the run stopped."}
              </p>
            </div>
          ) : (
            <p>This run had not finished when the session was loaded. Reload the session to see its result.</p>
          )}
        </div>
      ) : null}

      {unsupported ? (
        <p className="flex items-start gap-2 text-[13px] text-k-error">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          The gate marked this answer as not supported by the trajectory.
        </p>
      ) : null}

      {claims.length > 0 ? (
        <section className="rounded-md border bg-surface" aria-label="Claims">
          <h3 className="border-b px-3 py-1.5 text-xs font-semibold text-muted-foreground">
            Claims and the events that support them
          </h3>
          <ul className="divide-y">
            {claims.map((claim, i) => (
              <li key={i} className="flex items-start gap-3 px-3 py-2">
                <p className="min-w-0 flex-1 text-[13px] leading-snug">{claim.text}</p>
                <div className="flex shrink-0 flex-wrap justify-end gap-1">
                  {(claim.evidence ?? []).map((id) => {
                    const event = run.events.find((e) => e.id === id);
                    const selected = active && selectedEventId === id;
                    return (
                      <button
                        key={id}
                        type="button"
                        aria-pressed={selected}
                        aria-label={`Show evidence: event ${id}${event ? `, ${eventTitle(event)}` : ""}`}
                        title={event ? eventTitle(event) : `Event ${id}`}
                        onClick={() => onSelectEvidence(run.run_id, id)}
                        className={cn(
                          "inline-flex h-6 items-center gap-1.5 rounded-[4px] border px-1.5 text-xs font-medium tabular-nums transition-colors",
                          selected ? "border-foreground bg-raised" : "border-line bg-ground hover:border-muted-foreground",
                        )}
                      >
                        {event ? <EventGlyph event={event} size={11} /> : null}
                        {id}
                      </button>
                    );
                  })}
                </div>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {run.status !== "streaming" ? (
        <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground tabular-nums">
          <span>
            {run.events.length} events
            {ep ? ` · ${fmtCost(ep.cost)} · ${fmtSeconds(ep.elapsed)}` : ""}
          </span>
          <Button
            variant="ghost"
            size="xs"
            className={cn("ml-auto", active && "text-foreground")}
            onClick={() => onSelectRun(run.run_id)}
          >
            <Orbit aria-hidden />
            {active ? "Trajectory shown" : "Show trajectory"}
          </Button>
        </footer>
      ) : null}
    </article>
  );
}

export function ChatView({
  state,
  errorMessage,
  onRetry,
  title,
  files,
  runs,
  activeRunId,
  selectedEventId,
  streamText,
  busy,
  sendError,
  onSelectRun,
  onSelectEvidence,
  onSend,
  onUpload,
}: {
  state: "idle" | "loading" | "error" | "ready";
  errorMessage?: string;
  onRetry?: () => void;
  title: string;
  files: SessionFile[];
  runs: RunView[];
  activeRunId: string | null;
  selectedEventId: number | null;
  streamText: string;
  busy: boolean;
  sendError: string | null;
  onSelectRun: (runId: string) => void;
  onSelectEvidence: (runId: string, eventId: number) => void;
  onSend: (question: string) => Promise<boolean>;
  onUpload: (file: File) => Promise<void>;
}) {
  const end = useRef<HTMLDivElement>(null);
  const lastRun = runs[runs.length - 1];
  const growth = `${runs.length}:${lastRun?.events.length ?? 0}:${lastRun?.status ?? ""}`;

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [growth]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-[45px] shrink-0 items-center gap-3 border-b px-4">
        <h2 className="min-w-0 truncate font-heading text-[15px] font-semibold">{title}</h2>
        {state === "ready" && runs.length > 0 ? (
          <span className="ml-auto shrink-0 text-xs text-muted-foreground tabular-nums">
            {runs.length} {runs.length === 1 ? "question" : "questions"}
          </span>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {state === "loading" ? (
          <PaneLoading label="Loading conversation" lines={7} />
        ) : state === "error" ? (
          <PaneError title="This session did not load" message={errorMessage ?? ""} onRetry={onRetry} />
        ) : runs.length === 0 ? (
          <PaneEmpty icon={<Telescope aria-hidden />} title="Ask a question about your data">
            Attach a CSV or a figure below, then ask. The agent works through its tools, and every number in the
            answer links to the tool result it came from.
          </PaneEmpty>
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-8 px-4 py-5">
            {runs.map((run) => (
              <RunBlock
                key={run.run_id}
                run={run}
                active={run.run_id === activeRunId}
                selectedEventId={selectedEventId}
                streamText={streamText}
                onSelectRun={onSelectRun}
                onSelectEvidence={onSelectEvidence}
              />
            ))}
            <div ref={end} />
          </div>
        )}
      </div>

      <div className="shrink-0 border-t bg-ground px-3 py-3">
        <div className="mx-auto max-w-3xl">
          {sendError ? (
            <p role="alert" className="mb-2 flex items-start gap-1.5 text-[13px] text-k-error">
              <CircleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              {sendError}
            </p>
          ) : null}
          <Composer files={files} busy={busy} onSend={onSend} onUpload={onUpload} />
          <p className="mt-1.5 text-xs text-muted-foreground max-sm:hidden">
            Enter sends. Shift and Enter adds a line.
          </p>
        </div>
      </div>
    </div>
  );
}
