"use client";

import { FlaskConical, LoaderCircle, Orbit, Scissors } from "lucide-react";
import { useEffect, useMemo, useRef, type CSSProperties, type KeyboardEvent } from "react";
import { EventDetail } from "@/components/event-detail";
import { EventGlyph } from "@/components/event-glyph";
import { PaneEmpty, PaneError, PaneLoading } from "@/components/pane-state";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TRACKS, eventTitle, kindOf, pairCalls, trackOf } from "@/lib/events";
import type { EventKind, RunView, TrajectoryEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

const LABEL_W = 62;
const COL_MIN = 30;
const STEP_ROW = 1;
const FIRST_TRACK_ROW = 2;
const ID_ROW = FIRST_TRACK_ROW + TRACKS.length;

const LEGEND: { kind: EventKind; label: string; accepted?: boolean }[] = [
  { kind: "user", label: "Question" },
  { kind: "model", label: "Model call" },
  { kind: "tool_call", label: "Tool call" },
  { kind: "tool_result", label: "Tool result" },
  { kind: "judge", label: "Jev check" },
  { kind: "submit", label: "Gate accepted", accepted: true },
  { kind: "submit", label: "Gate rejected", accepted: false },
  { kind: "answer", label: "Answer" },
  { kind: "error", label: "Error" },
];

type StepGroup = { step: number; start: number; end: number };

function stepGroups(events: TrajectoryEvent[]): StepGroup[] {
  const groups: StepGroup[] = [];
  events.forEach((event, i) => {
    const last = groups[groups.length - 1];
    if (last && last.step === event.step) last.end = i;
    else groups.push({ step: event.step, start: i, end: i });
  });
  return groups;
}

export function TrajectoryTimeline({
  state,
  errorMessage,
  onRetry,
  runs,
  run,
  selectedId,
  fresh,
  onSelect,
  onSelectRun,
}: {
  state: "loading" | "error" | "ready";
  errorMessage?: string;
  onRetry?: () => void;
  runs: RunView[];
  run: RunView | undefined;
  selectedId: number | null;
  fresh: ReadonlySet<number>;
  onSelect: (id: number) => void;
  onSelectRun: (runId: string) => void;
}) {
  const events = useMemo(() => run?.events ?? [], [run]);
  const pairs = useMemo(() => pairCalls(events), [events]);
  const groups = useMemo(() => stepGroups(events), [events]);
  const buttons = useRef(new Map<number, HTMLButtonElement>());

  const selectedIndex = events.findIndex((e) => e.id === selectedId);
  const selected = selectedIndex === -1 ? undefined : events[selectedIndex];

  useEffect(() => {
    if (selectedId === null) return;
    buttons.current.get(selectedId)?.scrollIntoView({ inline: "nearest", block: "nearest" });
  }, [selectedId, events.length]);

  if (state === "loading") return <PaneLoading label="Loading trajectory" lines={6} />;
  if (state === "error") {
    return <PaneError title="Trajectory did not load" message={errorMessage ?? ""} onRetry={onRetry} />;
  }
  if (!run) {
    return (
      <PaneEmpty icon={<Orbit aria-hidden />} title="No trajectory yet">
        Ask a question and each model call, tool call, Jev check and answer will appear here as a marker, in the order it happened.
      </PaneEmpty>
    );
  }

  const move = (from: number, to: number) => {
    const target = events[Math.max(0, Math.min(events.length - 1, to))];
    if (!target || target.id === events[from]?.id) return;
    onSelect(target.id);
    buttons.current.get(target.id)?.focus();
  };

  const onKey = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (e.key === "ArrowRight") move(index, index + 1);
    else if (e.key === "ArrowLeft") move(index, index - 1);
    else if (e.key === "Home") move(index, 0);
    else if (e.key === "End") move(index, events.length - 1);
    else return;
    e.preventDefault();
  };

  const streaming = run.status === "streaming";
  const stepCount = new Set(events.map((e) => e.step)).size;
  const grid: CSSProperties = {
    // Columns share the pane's width down to 30px each, then the strip scrolls.
    // The last column is filler so the track lines always reach the edge.
    gridTemplateColumns: `${LABEL_W}px repeat(${Math.max(events.length, 1)}, minmax(${COL_MIN}px, 58px)) minmax(8px, 1fr)`,
    gridTemplateRows: `22px repeat(${TRACKS.length}, 42px) 20px`,
    minWidth: LABEL_W + events.length * COL_MIN + 8,
  };

  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 pt-3 pb-2">
        {runs.length > 1 ? (
          <label className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
            Run
            <select
              value={run.run_id}
              onChange={(e) => onSelectRun(e.target.value)}
              className="h-7 max-w-[16rem] min-w-0 truncate rounded-md border bg-surface px-1.5 text-[13px] text-foreground"
            >
              {runs.map((r, i) => (
                <option key={r.run_id} value={r.run_id}>
                  {i + 1}. {r.question.slice(0, 48) || r.run_id}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <p className="text-xs text-muted-foreground tabular-nums">
          {events.length} {events.length === 1 ? "event" : "events"} over {stepCount}{" "}
          {stepCount === 1 ? "step" : "steps"}
        </p>
        {streaming ? (
          <p className="ml-auto flex items-center gap-1.5 text-xs text-k-model">
            <LoaderCircle className="size-3.5 animate-spin" aria-hidden />
            Running
          </p>
        ) : run.status === "error" ? (
          <p className="ml-auto text-xs text-k-error">Run failed</p>
        ) : null}
      </div>

      {events.length === 0 ? (
        <div className="tl-sky border-y px-4 py-8 text-[13px] text-muted-foreground">
          {streaming ? "Waiting for the first event." : "This run recorded no events."}
        </div>
      ) : (
        <div
          className="tl-sky sticky top-0 z-10 overflow-x-auto border-y"
          role="group"
          aria-label="Trajectory timeline. Use the left and right arrow keys to move between events."
        >
          <div className="grid w-full" style={grid}>
            {/* Step bands */}
            {groups.map((g, i) => (
              <div
                key={`band-${g.start}`}
                className={cn("border-l border-line/70", i % 2 === 1 && "bg-surface/55")}
                style={{ gridColumn: `${g.start + 2} / ${g.end + 3}`, gridRow: `${STEP_ROW} / -1` }}
              />
            ))}
            <div
              className="sticky left-0 z-20 flex items-center bg-ground pl-4 text-[11px] text-muted-foreground"
              style={{ gridColumn: 1, gridRow: STEP_ROW }}
            >
              Step
            </div>
            {groups.map((g) => (
              <div
                key={`step-${g.start}`}
                className="z-[1] flex items-center pl-1.5 text-[11px] font-medium text-muted-foreground tabular-nums"
                style={{ gridColumn: `${g.start + 2} / ${g.end + 3}`, gridRow: STEP_ROW }}
              >
                {g.step}
              </div>
            ))}

            {/* Tracks */}
            {TRACKS.map((track, t) => (
              <div
                key={`line-${track.id}`}
                className="h-px self-center bg-line"
                style={{ gridColumn: "2 / -1", gridRow: FIRST_TRACK_ROW + t }}
              />
            ))}
            {TRACKS.map((track, t) => (
              <div
                key={`label-${track.id}`}
                className="sticky left-0 z-20 flex items-center gap-1.5 bg-ground pl-4 text-xs font-medium text-foreground"
                style={{ gridColumn: 1, gridRow: FIRST_TRACK_ROW + t }}
              >
                {track.label}
              </div>
            ))}
            <div className="sticky left-0 z-20 bg-ground" style={{ gridColumn: 1, gridRow: ID_ROW }} />

            {/* Selected column beam */}
            {selected ? (
              <div
                className="pointer-events-none z-[1]"
                style={{
                  gridColumn: selectedIndex + 2,
                  gridRow: `${FIRST_TRACK_ROW} / -1`,
                  background: `linear-gradient(to bottom, color-mix(in srgb, ${kindOf(selected).color} 20%, transparent), color-mix(in srgb, ${kindOf(selected).color} 4%, transparent))`,
                }}
              />
            ) : null}

            {/* Call-to-result links on the tools track */}
            {events.map((event, i) => {
              if (event.kind !== "tool_call") return null;
              const resultId = pairs.get(event.id);
              const j = events.findIndex((e) => e.id === resultId);
              if (j <= i) return null;
              const half = 100 / (2 * (j - i + 1));
              return (
                <div
                  key={`link-${event.id}`}
                  className="relative z-[1] self-center"
                  style={{ gridColumn: `${i + 2} / ${j + 3}`, gridRow: FIRST_TRACK_ROW + 1, height: 2 }}
                >
                  <div
                    className="absolute inset-y-0"
                    style={{
                      left: `${half}%`,
                      right: `${half}%`,
                      background: "linear-gradient(to right, var(--k-call), var(--k-result))",
                      opacity: 0.75,
                    }}
                  />
                </div>
              );
            })}

            {/* Markers */}
            {events.map((event, i) => {
              const row = FIRST_TRACK_ROW + TRACKS.findIndex((t) => t.id === trackOf(events, i));
              const isSelected = event.id === selectedId;
              const tabbable = isSelected || (selectedIndex === -1 && i === events.length - 1);
              return (
                <div key={event.id} className="grid place-items-center" style={{ gridColumn: i + 2, gridRow: row }}>
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <button
                          type="button"
                          ref={(el) => {
                            if (el) buttons.current.set(event.id, el);
                            else buttons.current.delete(event.id);
                          }}
                          className="tl-marker"
                          style={{ "--c": kindOf(event).color } as CSSProperties}
                          aria-pressed={isSelected}
                          aria-label={`Event ${event.id}, step ${event.step}: ${eventTitle(event)}${event.clipped ? ", clipped" : ""}${event.simulated ? ", simulated" : ""}`}
                          data-fresh={fresh.has(event.id) ? "true" : undefined}
                          tabIndex={tabbable ? 0 : -1}
                          onClick={() => onSelect(event.id)}
                          onKeyDown={(e) => onKey(e, i)}
                        />
                      }
                    >
                      <EventGlyph event={event} size={event.kind === "answer" ? 22 : 17} />
                      {event.clipped || event.simulated ? (
                        <span className="absolute -top-1 -right-1 grid size-[15px] place-items-center rounded-[3px] border border-line bg-raised text-foreground">
                          {event.clipped ? <Scissors className="size-2.5" aria-hidden /> : <FlaskConical className="size-2.5" aria-hidden />}
                        </span>
                      ) : null}
                    </TooltipTrigger>
                    <TooltipContent className="border bg-raised text-foreground [&>div:last-child]:hidden">
                      Event {event.id}: {eventTitle(event)}
                      {event.clipped ? " (clipped)" : ""}
                      {event.simulated ? " (simulated)" : ""}
                    </TooltipContent>
                  </Tooltip>
                </div>
              );
            })}

            {/* Event ids */}
            {events.map((event, i) => (
              <div
                key={`id-${event.id}`}
                className={cn(
                  "z-[1] text-center text-[10.5px] leading-[18px] tabular-nums",
                  event.id === selectedId ? "font-semibold text-foreground" : "text-muted-foreground/80",
                )}
                style={{ gridColumn: i + 2, gridRow: ID_ROW }}
                aria-hidden
              >
                {event.id}
              </div>
            ))}
          </div>
        </div>
      )}

      <ul className="flex flex-wrap gap-x-3.5 gap-y-1 px-4 py-2.5 text-[11.5px] text-muted-foreground" aria-label="Legend">
        {LEGEND.map((item) => (
          <li key={item.label} className="flex items-center gap-1.5">
            <EventGlyph event={{ kind: item.kind, accepted: item.accepted }} size={11} />
            {item.label}
          </li>
        ))}
        <li className="flex items-center gap-1.5">
          <Scissors className="size-3 text-foreground" aria-hidden />
          Clipped output
        </li>
        <li className="flex items-center gap-1.5">
          <FlaskConical className="size-3 text-foreground" aria-hidden />
          Simulated tool
        </li>
      </ul>

      {run.status === "error" && run.error ? (
        <PaneError title="The run failed" message={run.error} />
      ) : null}

      <div className="border-t">
        {selected ? (
          <EventDetail
            event={selected}
            events={events}
            pairs={pairs}
            onSelect={onSelect}
          />
        ) : (
          <p className="px-4 py-6 text-[13px] text-muted-foreground">
            Select a marker to see that event&apos;s full input and output.
          </p>
        )}
      </div>
    </div>
  );
}
