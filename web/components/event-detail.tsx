"use client";

import { ChevronLeft, ChevronRight, FlaskConical, Scissors, ShieldX, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { EventGlyph } from "@/components/event-glyph";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { argsText, asText, eventTitle, fmtCost, fmtInt, fmtSeconds, isRejected, textOf, tokensOf } from "@/lib/events";
import type { Claim, TrajectoryEvent } from "@/lib/types";

function Block({ label, note, children }: { label: string; note?: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <h4 className="text-xs font-semibold text-foreground">{label}</h4>
        {note ? <span className="text-xs text-muted-foreground">{note}</span> : null}
      </div>
      {children}
    </section>
  );
}

// Raw tool input and output: the one place a monospace face is used.
function Raw({ text }: { text: string }) {
  return (
    <pre
      tabIndex={0}
      className="max-h-[26rem] overflow-auto rounded-md border bg-ground p-3 font-mono text-[12px] leading-[1.6] whitespace-pre text-foreground"
    >
      {text}
    </pre>
  );
}

function Prose({ text }: { text: string }) {
  return (
    <div className="rounded-md border bg-ground p-3 text-[13px] leading-relaxed break-words whitespace-pre-wrap">
      {text}
    </div>
  );
}

export function EventChip({
  event,
  onSelect,
}: {
  event: TrajectoryEvent;
  onSelect: (id: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(event.id)}
      className="inline-flex max-w-full items-center gap-1.5 rounded-md border bg-surface px-2 py-1 text-xs text-foreground transition-colors hover:bg-raised"
    >
      <EventGlyph event={event} size={11} />
      <span className="tabular-nums">Event {event.id}</span>
      <span className="truncate text-muted-foreground">{eventTitle(event)}</span>
    </button>
  );
}

function Chips({ events, onSelect }: { events: TrajectoryEvent[]; onSelect: (id: number) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {events.map((e) => (
        <EventChip key={e.id} event={e} onSelect={onSelect} />
      ))}
    </div>
  );
}

function ClaimList({
  claims,
  events,
  onSelect,
}: {
  claims: Claim[];
  events: TrajectoryEvent[];
  onSelect: (id: number) => void;
}) {
  return (
    <ul className="flex flex-col gap-2">
      {claims.map((claim, i) => (
        <li key={i} className="flex flex-col gap-1.5 rounded-md border bg-ground p-2.5 text-[13px]">
          <span>{claim.text}</span>
          <Chips
            events={(claim.evidence ?? []).flatMap((id) => events.find((e) => e.id === id) ?? [])}
            onSelect={onSelect}
          />
        </li>
      ))}
    </ul>
  );
}

export function EventDetail({
  event,
  events,
  pairs,
  onSelect,
}: {
  event: TrajectoryEvent;
  events: TrajectoryEvent[];
  pairs: Map<number, number>;
  onSelect: (id: number) => void;
}) {
  const index = events.findIndex((e) => e.id === event.id);
  const prev = events[index - 1];
  const next = events[index + 1];
  const partner = events.find((e) => e.id === pairs.get(event.id));
  const text = textOf(event);
  const claims = Array.isArray(event.claims) ? event.claims : [];

  const meta = [
    `Event ${event.id}`,
    `step ${event.step}`,
    typeof event.elapsed === "number" ? fmtSeconds(event.elapsed) : "",
    tokensOf(event),
    typeof event.cost === "number" ? fmtCost(event.cost) : "",
  ].filter(Boolean);

  let body: ReactNode;
  if (event.kind === "model") {
    let from = index - 1;
    while (from >= 0 && events[from].kind !== "model") from--;
    const seen = events.slice(from + 1, index);
    const requested: TrajectoryEvent[] = [];
    for (let i = index + 1; i < events.length && events[i].kind !== "model"; i++) {
      if (events[i].kind === "tool_call") requested.push(events[i]);
    }
    body = (
      <>
        <Block label="Input" note={seen.length ? "new since the previous model call" : "the conversation so far"}>
          {seen.length ? <Chips events={seen} onSelect={onSelect} /> : null}
        </Block>
        <Block label="Output">
          {text ? <Prose text={text} /> : <p className="text-[13px] text-muted-foreground">No text; the model only called tools.</p>}
        </Block>
        {requested.length ? (
          <Block label="Tool calls requested">
            <Chips events={requested} onSelect={onSelect} />
          </Block>
        ) : null}
      </>
    );
  } else if (event.kind === "tool_call" || event.kind === "tool_result") {
    const call = event.kind === "tool_call" ? event : partner;
    const result = event.kind === "tool_result" ? event : partner;
    body = (
      <>
        <Block label="Input" note={call && call !== event ? `arguments from event ${call.id}` : "arguments"}>
          {call ? <Raw text={argsText(call.args)} /> : <p className="text-[13px] text-muted-foreground">No matching tool call was recorded.</p>}
        </Block>
        <Block
          label="Output"
          note={
            result
              ? `${result !== event ? `event ${result.id}, ` : ""}${fmtInt(result.chars_shown ?? asText(result.result).length)} characters`
              : undefined
          }
        >
          {result ? <Raw text={asText(result.result)} /> : <p className="text-[13px] text-muted-foreground">No result yet.</p>}
        </Block>
      </>
    );
  } else if (event.kind === "judge") {
    const questions = Array.isArray(event.questions) ? event.questions : [];
    body = (
      <>
        <Block
          label="Input"
          note={event.source === "gate" ? "asked by the pre-submit gate" : event.source === "harness" ? "asked by the harness" : "questions sent to Jev"}
        >
          <Raw text={questions.length ? questions.join("\n") : asText(event.args ?? {})} />
        </Block>
        <Block label="Output">
          {event.answers === undefined || event.answers === null ? (
            <Prose text={`Jev did not answer${event.error ? `: ${event.error}` : ""}. The run treats this as no flag.`} />
          ) : (
            <Raw text={asText(event.answers)} />
          )}
        </Block>
      </>
    );
  } else if (event.kind === "submit") {
    body = (
      <>
        <Block label="Input" note="the answer the harness submitted">
          {text ? <Prose text={text} /> : null}
          {claims.length ? <ClaimList claims={claims} events={events} onSelect={onSelect} /> : null}
        </Block>
        <Block
          label="Output"
          note={
            typeof event.p_supported === "number"
              ? `Jev's support estimate ${event.p_supported.toFixed(2)}; below 0.50 is flagged`
              : prev?.kind === "judge" && prev.source === "gate"
                ? "Jev gave no usable estimate, so nothing was flagged"
                : "the pre-submit gate was off, so the answer was not checked"
          }
        >
          <Prose
            text={[
              event.accepted === true ? "Accepted." : event.accepted === false ? "Rejected and sent back to the harness." : "",
              asText(event.reason),
            ].filter(Boolean).join(" ") || "No verdict recorded."}
          />
        </Block>
      </>
    );
  } else if (event.kind === "answer") {
    body = (
      <>
        <Block label="Output" note="the final answer">
          <Prose text={text} />
        </Block>
        {claims.length ? (
          <Block label="Claims and evidence">
            <ClaimList claims={claims} events={events} onSelect={onSelect} />
          </Block>
        ) : null}
      </>
    );
  } else if (event.kind === "user") {
    const files = Array.isArray(event.files) ? event.files.map(String) : [];
    body = (
      <Block label="Input" note={files.length ? `with ${files.join(", ")}` : undefined}>
        <Prose text={text} />
      </Block>
    );
  } else {
    const message = asText(event.message ?? event.error ?? event.result ?? event.text);
    body = (
      <>
        <Block label="Output">
          <Prose text={message || "No message recorded."} />
        </Block>
        {event.stderr ? (
          <Block label="stderr">
            <Raw text={asText(event.stderr)} />
          </Block>
        ) : null}
      </>
    );
  }

  return (
    <article className="flex flex-col gap-4 p-4" aria-label={`Event ${event.id} detail`}>
      <header className="flex flex-col gap-2">
        <div className="flex items-start gap-2.5">
          <EventGlyph event={event} size={18} className="mt-0.5" />
          <div className="min-w-0 flex-1">
            <h3 className="font-heading text-base leading-tight font-semibold break-words">{eventTitle(event)}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">{meta.join(" · ")}</p>
          </div>
          <div className="flex shrink-0 gap-1">
            <Button variant="outline" size="icon-sm" aria-label="Previous event" disabled={!prev} onClick={() => prev && onSelect(prev.id)}>
              <ChevronLeft aria-hidden />
            </Button>
            <Button variant="outline" size="icon-sm" aria-label="Next event" disabled={!next} onClick={() => next && onSelect(next.id)}>
              <ChevronRight aria-hidden />
            </Button>
          </div>
        </div>
        {event.clipped || event.simulated || isRejected(event) || event.unsupported ? (
          <div className="flex flex-wrap gap-1.5">
            {event.clipped ? (
              <Badge variant="outline" className="h-auto min-h-5 bg-raised whitespace-normal">
                <Scissors aria-hidden />
                Clipped: {fmtInt(event.chars_shown)} of {fmtInt(event.chars_full)} characters shown. Full text saved as{" "}
                {event.file ?? "a file in the run folder"}.
              </Badge>
            ) : null}
            {event.simulated ? (
              <Badge variant="outline" className="h-auto min-h-5 bg-raised whitespace-normal">
                <FlaskConical aria-hidden />
                Simulated: fixture data, not a real lookup
              </Badge>
            ) : null}
            {isRejected(event) ? (
              <Badge variant="outline" className="h-auto min-h-5 border-k-error/50 bg-k-error/10 whitespace-normal">
                <ShieldX className="text-k-error" aria-hidden />
                Rejected by the pre-submit gate
              </Badge>
            ) : null}
            {event.unsupported ? (
              <Badge variant="outline" className="h-auto min-h-5 border-k-error/50 bg-k-error/10 whitespace-normal">
                <TriangleAlert className="text-k-error" aria-hidden />
                Marked unsupported by the gate
              </Badge>
            ) : null}
          </div>
        ) : null}
      </header>

      {body}

      <details className="group text-xs">
        <summary className="w-fit cursor-pointer rounded-sm text-muted-foreground hover:text-foreground">
          Raw event
        </summary>
        <div className="mt-1.5">
          <Raw text={JSON.stringify(event, null, 2)} />
        </div>
      </details>
    </article>
  );
}
