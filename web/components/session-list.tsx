"use client";

import { LoaderCircle, MessagesSquare, Plus } from "lucide-react";
import { PaneEmpty, PaneError, PaneLoading } from "@/components/pane-state";
import { Button } from "@/components/ui/button";
import { fmtWhen } from "@/lib/events";
import type { Loadable, SessionSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

export function SessionList({
  sessions,
  activeId,
  streamingId,
  creating,
  onSelect,
  onCreate,
  onRetry,
}: {
  sessions: Loadable<SessionSummary[]>;
  activeId: string | null;
  streamingId: string | null;
  creating: boolean;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRetry: () => void;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-surface">
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2.5">
        <h2 className="font-heading text-[15px] font-semibold">Sessions</h2>
        <Button variant="outline" size="sm" onClick={onCreate} disabled={creating}>
          {creating ? <LoaderCircle className="animate-spin" aria-hidden /> : <Plus aria-hidden />}
          New
        </Button>
      </div>

      <nav aria-label="Sessions" className="min-h-0 flex-1 overflow-y-auto">
        {sessions.status === "loading" ? (
          <PaneLoading label="Loading sessions" lines={5} />
        ) : sessions.status === "error" ? (
          <PaneError title="Sessions did not load" message={sessions.message} onRetry={onRetry} />
        ) : sessions.data.length === 0 ? (
          <PaneEmpty icon={<MessagesSquare aria-hidden />} title="No sessions yet">
            Start one with New, or just ask a question in the chat.
          </PaneEmpty>
        ) : (
          <ul className="flex flex-col gap-px p-1.5">
            {sessions.data.map((session) => {
              const active = session.id === activeId;
              return (
                <li key={session.id}>
                  <button
                    type="button"
                    aria-current={active ? "true" : undefined}
                    onClick={() => onSelect(session.id)}
                    className={cn(
                      "flex w-full flex-col gap-0.5 rounded-md px-2.5 py-2 text-left transition-colors",
                      active ? "bg-raised text-foreground" : "text-muted-foreground hover:bg-raised/50 hover:text-foreground",
                    )}
                  >
                    <span className={cn("line-clamp-2 text-[13px] leading-snug", active ? "font-semibold" : "font-medium")}>
                      {session.title || "Untitled session"}
                    </span>
                    <span className="flex items-center gap-1.5 text-xs text-muted-foreground tabular-nums">
                      {fmtWhen(session.created)}
                      {session.meta_unreadable ? <span>details file unreadable</span> : null}
                      {session.id === streamingId ? (
                        <span className="inline-flex items-center gap-1 text-k-model">
                          <LoaderCircle className="size-3 animate-spin" aria-hidden />
                          running
                        </span>
                      ) : null}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </nav>
    </div>
  );
}
