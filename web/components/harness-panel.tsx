"use client";

import { Check, ChevronRight, GitBranch, Minus, RotateCw, TriangleAlert, X } from "lucide-react";
import { PaneEmpty, PaneError, PaneLoading } from "@/components/pane-state";
import { Button } from "@/components/ui/button";
import { fmtCost, readLineage, shortHash, type CandidateStage, type CandidateView, type LineageView } from "@/lib/events";
import type { HarnessInfo, Loadable } from "@/lib/types";
import { cn } from "@/lib/utils";

// Scores are a share of validation tasks; a count is turned into one.
function pct(score: number | null | undefined, n?: number): string {
  if (typeof score !== "number") return "-";
  const share = score <= 1 ? score : n ? score / n : score / 100;
  return `${(share * 100).toFixed(1)}%`;
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const pValue = (p: number | null | undefined) => (typeof p === "number" ? `p = ${p.toFixed(3)}` : "p not recorded");

const STAGE_LABEL: Record<CandidateStage, string> = {
  invalid: "Not a valid harness",
  screen: "Dropped by the screen",
  first: "Stopped at the first draw",
  confirm: "Failed the final check",
  adopted: "Adopted",
};

type StepState = "passed" | "stopped" | "skipped";

// One of the three stages on a candidate card. `skipped` covers both a stage
// the candidate never reached and one this generation did not run; the text says which.
function Step({ name, state, children }: { name: string; state: StepState; children: React.ReactNode }) {
  const Icon = state === "passed" ? Check : state === "stopped" ? X : Minus;
  return (
    <li className="flex min-w-0 flex-col gap-0.5 bg-surface px-3 py-2.5">
      <p
        className={cn(
          "flex items-center gap-1.5 text-xs font-semibold",
          state === "passed" ? "text-k-answer" : state === "stopped" ? "text-k-error" : "text-muted-foreground",
        )}
      >
        <Icon className="size-3.5 shrink-0" aria-hidden />
        {name}
        <span className="sr-only">{state === "passed" ? "(passed)" : state === "stopped" ? "(stopped here)" : ""}</span>
      </p>
      <div className={cn("text-[13px] leading-snug tabular-nums", state === "skipped" && "text-muted-foreground")}>
        {children}
      </div>
    </li>
  );
}

function CandidateCard({ view, row }: { view: CandidateView; row: LineageView }) {
  const { candidate: c, stage } = view;
  const scored = typeof c.score === "number";
  const at = (s: CandidateStage, reached: boolean): StepState => (stage === s ? "stopped" : reached ? "passed" : "skipped");
  return (
    <li className="flex flex-col gap-2.5 rounded-md border bg-ground p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <p className="text-[13px] text-muted-foreground">
          Candidate <code className="font-mono text-xs text-foreground">{shortHash(c.sha).slice(0, 8)}</code>
        </p>
        <p className={cn("text-[13px] font-semibold", stage === "adopted" ? "text-k-answer" : "text-foreground")}>
          {STAGE_LABEL[stage]}
        </p>
      </div>

      {c.rationale ? (
        <blockquote className="border-l-2 border-line pl-3 text-[13px] leading-relaxed">
          <span className="block text-xs text-muted-foreground">The improver&apos;s rationale</span>
          {c.rationale}
        </blockquote>
      ) : (
        <p className="text-xs text-muted-foreground">No rationale was recorded for this candidate.</p>
      )}

      {stage === "invalid" ? null : (
        <ol className="grid gap-px overflow-hidden rounded-md border bg-line @lg:grid-cols-3">
          <Step name="Screen" state={c.screen ? at("screen", true) : "skipped"}>
            {c.screen ? (
              <>
                Fixed {c.screen.fixed}, broke {c.screen.broken}
                <span className="block text-xs text-muted-foreground">of {plural(c.screen.n, "train task")}</span>
              </>
            ) : scored ? (
              "Not run in this generation"
            ) : (
              "Not recorded"
            )}
          </Step>
          <Step name="First draw" state={scored ? at("first", true) : "skipped"}>
            {scored ? (
              <>
                {pct(c.score, row.nVal)} <span className="text-muted-foreground">against the parent&apos;s</span>{" "}
                {pct(row.parentScore, row.nVal)}
                <span className="block text-xs text-muted-foreground">
                  won {c.wins ?? "-"}, lost {c.losses ?? "-"}, {pValue(c.p_value)}
                  {c.errors ? `, ${plural(c.errors, "errored episode")}` : ""}
                </span>
              </>
            ) : c.skipped?.val ? (
              `Stopped ${plural(c.skipped.val, "task")} early: it could no longer finish ahead`
            ) : (
              "Not reached"
            )}
          </Step>
          <Step name={c.decision ? "Decision" : "Confirmation"} state={c.decision || c.confirm ? at("confirm", true) : "skipped"}>
            {c.decision ? (
              <>
                won {c.decision.wins}, lost {c.decision.losses}:{" "}
                net {c.decision.net} <span className="text-muted-foreground">of the</span> {c.decision.margin} needed
                <span className="block text-xs text-muted-foreground">
                  fresh runs of both on {plural(c.decision.n, "task")}: {pct(c.decision.score)} against the parent&apos;s{" "}
                  {pct(c.decision.parent_score)}
                </span>
                <span className="block text-xs text-muted-foreground">
                  {pValue(c.decision.p_value)}
                  {c.decision.errors ? `, ${plural(c.decision.errors, "errored episode")}` : ""}
                </span>
              </>
            ) : c.confirm ? (
              <>
                {pct(c.confirm.score, c.confirm.n)} <span className="text-muted-foreground">against the parent&apos;s</span>{" "}
                {pct(c.confirm.parent_score, c.confirm.n)}
                <span className="block text-xs text-muted-foreground">
                  won {c.confirm.wins}, lost {c.confirm.losses}
                  {c.confirm.errors ? `, ${plural(c.confirm.errors, "errored episode")}` : ""}
                </span>
                <span className="block text-xs text-muted-foreground">
                  both draws: won {c.confirm.combined_wins}, lost {c.confirm.combined_losses},{" "}
                  {pValue(c.confirm.combined_p_value)}
                </span>
              </>
            ) : stage === "adopted" ? (
              "Not run: this generation predates the confirmation draw"
            ) : (
              "Not reached"
            )}
          </Step>
        </ol>
      )}

      {c.replay?.broken ? (
        <p className="text-[13px] leading-snug">
          <span className="text-muted-foreground">Free replay: </span>
          changed {c.replay.broken} of {plural(c.replay.n, "right answer")} into wrong ones
          {c.replay.examples[0] ? (
            <span className="block font-mono text-xs text-muted-foreground">
              {c.replay.examples[0].task}: {c.replay.examples[0].before} became {c.replay.examples[0].after}
            </span>
          ) : null}
        </p>
      ) : null}

      {c.reason && c.reason !== "adopted" ? (
        <p className="text-[13px] leading-snug">
          <span className="text-muted-foreground">Recorded reason: </span>
          {c.reason}
        </p>
      ) : null}
    </li>
  );
}

function Generation({ row, index, open }: { row: LineageView; index: number; open: boolean }) {
  const draws = [
    row.family ? `Task family ${row.family}` : null,
    row.nVal !== undefined ? `first draw ${row.nVal} tasks` : null,
    row.nConfirm !== undefined ? `confirmation draw ${row.nConfirm} tasks` : null,
    row.nullScore !== undefined ? `guessing scores ${pct(row.nullScore, row.nVal)}` : null,
  ].filter(Boolean);
  return (
    <li>
      <details open={open} className="group">
        <summary className="grid cursor-pointer list-none grid-cols-[auto_1fr_auto] items-start gap-x-2.5 gap-y-0.5 px-3 py-2.5 hover:bg-surface/60 [&::-webkit-details-marker]:hidden">
          <ChevronRight
            className="mt-1 size-3.5 shrink-0 text-muted-foreground transition-transform group-open:rotate-90"
            aria-hidden
          />
          <div className="min-w-0">
            <p className="font-heading text-[15px] leading-snug font-semibold">Generation {row.generation ?? index + 1}</p>
            <p className={cn("flex items-start gap-1.5 text-[13px] leading-snug", row.adopted ? "text-k-answer" : "text-muted-foreground")}>
              {row.adopted ? <Check className="mt-0.5 size-3.5 shrink-0" aria-hidden /> : null}
              {row.outcome}
            </p>
          </div>
          <dl className="grid grid-cols-[auto_auto] gap-x-2 text-right text-xs text-muted-foreground tabular-nums">
            <dt>Parent{row.parent ? ` ${row.parent}` : ""}</dt>
            <dd className="font-semibold text-foreground">{pct(row.parentScore, row.nVal)}</dd>
            <dt>Cost</dt>
            <dd className="font-semibold text-foreground">{fmtCost(row.cost)}</dd>
          </dl>
        </summary>
        <div className="flex flex-col gap-2.5 border-t bg-surface/40 px-3 py-3">
          {draws.length ? <p className="text-xs text-muted-foreground">{draws.join(", ")}.</p> : null}
          {row.candidates.length === 0 ? (
            <p className="text-[13px] text-muted-foreground">No candidate was scored in this generation.</p>
          ) : (
            <ul className="flex flex-col gap-2.5">
              {row.candidates.map((view, i) => (
                <CandidateCard key={i} view={view} row={row} />
              ))}
            </ul>
          )}
        </div>
      </details>
    </li>
  );
}

export function HarnessPanel({
  harness,
  onRetry,
}: {
  harness: Loadable<HarnessInfo>;
  onRetry: () => void;
}) {
  if (harness.status === "loading") return <PaneLoading label="Loading harness lineage" lines={6} />;
  if (harness.status === "error") {
    return <PaneError title="Harness lineage did not load" message={harness.message} onRetry={onRetry} />;
  }

  const rows = harness.data.lineage.map(readLineage);
  const adopted = rows.filter((r) => r.adopted !== null).length;
  const skipped = harness.data.lineage_skipped ?? 0;

  return (
    <div className="@container flex flex-col gap-4 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs text-muted-foreground">Current harness</p>
          <p className="font-heading text-xl leading-tight font-semibold">{harness.data.current || "-"}</p>
          <p className="mt-1 max-w-[56ch] text-[13px] text-muted-foreground">
            The chat uses this file. An edit replaces it only after three stages: a screen on train tasks, a first
            draw of validation tasks where it must score above the parent, and a decision on fresh runs of both
            harnesses over two draws, where it must win a set margin more tasks than it loses with no errored episode.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RotateCw aria-hidden />
          Reload
        </Button>
      </div>

      {skipped > 0 ? (
        <p role="alert" className="flex items-start gap-2 rounded-md border border-k-error/40 bg-k-error/8 p-3 text-[13px]">
          <TriangleAlert className="mt-0.5 size-4 shrink-0 text-k-error" aria-hidden />
          {plural(skipped, "line")} of the lineage file could not be read and {skipped === 1 ? "is" : "are"} not shown below.
        </p>
      ) : null}

      {rows.length === 0 ? (
        <div className="rounded-md border">
          <PaneEmpty icon={<GitBranch aria-hidden />} title="No generations yet">
            Run <code className="rounded-[3px] bg-raised px-1 py-0.5 font-mono text-xs">rsi evolve</code> to
            propose and score rewrites. Each generation adds a row here, adopted or not.
          </PaneEmpty>
        </div>
      ) : (
        <>
          <p className="text-xs text-muted-foreground tabular-nums">
            {plural(rows.length, "generation")}, {adopted} adopted. Open a generation to see each candidate.
          </p>
          <ol className="divide-y overflow-hidden rounded-md border">
            {rows.map((row, i) => (
              <Generation key={i} row={row} index={i} open={i === rows.length - 1} />
            ))}
          </ol>
          <p className="max-w-[72ch] text-xs text-muted-foreground">
            Scores are the share of tasks answered correctly. The screen runs a candidate on the train tasks the parent
            failed and as many it passed; a candidate that fixes none, or breaks more than it fixes, is dropped before
            any validation task is spent. Only the best first-draw candidate goes to the decision (generations before it
            existed show a confirmation draw instead). Won and
            lost count the tasks where candidate and parent disagreed; the p-value is an exact McNemar test on those
            and is reported, not used as a condition.
          </p>
        </>
      )}
    </div>
  );
}
