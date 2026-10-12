"use client";

import { GitBranch, List, MessageSquare, Orbit, Telescope } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatView } from "@/components/chat-view";
import { HarnessPanel } from "@/components/harness-panel";
import { ResultsPanel } from "@/components/results-panel";
import { SessionList } from "@/components/session-list";
import { TrajectoryTimeline } from "@/components/trajectory-timeline";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  IS_MOCK,
  createSession,
  getHarnesses,
  getSession,
  listSessions,
  sendMessage,
  uploadFile,
} from "@/lib/api";
import { answerEvent } from "@/lib/events";
import type {
  HarnessInfo,
  Loadable,
  RunView,
  SessionDetail,
  SessionSummary,
  TrajectoryEvent,
} from "@/lib/types";
import { cn } from "@/lib/utils";

type Pane = "sessions" | "chat" | "inspect";
type RightTab = "trajectory" | "results" | "harness";

const PANES: { id: Pane; label: string; icon: typeof List }[] = [
  { id: "sessions", label: "Sessions", icon: List },
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "inspect", label: "Inspect", icon: Orbit },
];
const RIGHT_TABS: RightTab[] = ["trajectory", "results", "harness"];
const LIVE = "live";

const messageOf = (err: unknown) => (err instanceof Error ? err.message : "Unexpected error");

type Live = { sessionId: string; run: RunView; text: string };

export function AppShell() {
  const [sessions, setSessions] = useState<Loadable<SessionSummary[]>>({ status: "loading" });
  const [activeId, setActiveId] = useState<string | null>(null);
  // null means no session is selected.
  const [detail, setDetail] = useState<Loadable<SessionDetail> | null>({ status: "loading" });
  const [harness, setHarness] = useState<Loadable<HarnessInfo>>({ status: "loading" });
  const [live, setLive] = useState<Live | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<number | null>(null);
  const [fresh, setFresh] = useState<ReadonlySet<number>>(new Set());
  const [tab, setTab] = useState<RightTab>("trajectory");
  const [pane, setPane] = useState<Pane>("chat");
  const [creating, setCreating] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  const loadSeq = useRef(0);
  const abort = useRef<AbortController | null>(null);

  const loadDetail = useCallback(async (id: string) => {
    const seq = ++loadSeq.current;
    setDetail({ status: "loading" });
    try {
      const data = await getSession(id);
      if (seq === loadSeq.current) setDetail({ status: "ready", data });
    } catch (err) {
      if (seq === loadSeq.current) setDetail({ status: "error", message: messageOf(err) });
    }
  }, []);

  // The loaders below only set state after an await; callers that retry set
  // the loading state themselves first.
  const loadHarness = useCallback(async () => {
    try {
      const data = await getHarnesses();
      setHarness({ status: "ready", data });
    } catch (err) {
      setHarness({ status: "error", message: messageOf(err) });
    }
  }, []);

  const boot = useCallback(
    async (isCurrent: () => boolean = () => true) => {
      const params = new URLSearchParams(window.location.search);
      let list: SessionSummary[];
      try {
        list = await listSessions();
      } catch (err) {
        if (!isCurrent()) return;
        setSessions({ status: "error", message: messageOf(err) });
        setDetail(null);
        return;
      }
      if (!isCurrent()) return;
      // Deep links: ?session=, ?tab=, ?pane=, ?event=
      const wantTab = params.get("tab") as RightTab | null;
      const wantPane = params.get("pane") as Pane | null;
      const wantEvent = Number(params.get("event"));
      if (wantTab && RIGHT_TABS.includes(wantTab)) setTab(wantTab);
      if (wantPane && PANES.some((p) => p.id === wantPane)) setPane(wantPane);
      if (Number.isInteger(wantEvent) && wantEvent > 0) setSelectedEventId(wantEvent);
      setSessions({ status: "ready", data: list });
      const first = list.find((s) => s.id === params.get("session"))?.id ?? list[0]?.id ?? null;
      setActiveId(first);
      if (first) await loadDetail(first);
      else setDetail(null);
    },
    [loadDetail],
  );

  useEffect(() => {
    let current = true;
    // Both loaders set state only after their first await.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void boot(() => current);
    void loadHarness();
    return () => {
      current = false;
      abort.current?.abort();
    };
  }, [boot, loadHarness]);

  const selectSession = (id: string) => {
    setPane("chat");
    if (id === activeId && detail?.status === "ready") return;
    setActiveId(id);
    setActiveRunId(null);
    setSelectedEventId(null);
    setSendError(null);
    void loadDetail(id);
  };

  const createNew = async (): Promise<string | null> => {
    setCreating(true);
    setSendError(null);
    try {
      const created = await createSession();
      loadSeq.current++;
      setSessions((prev) => ({
        status: "ready",
        data: [created, ...(prev.status === "ready" ? prev.data : [])],
      }));
      setActiveId(created.id);
      setDetail({ status: "ready", data: { ...created, files: [], runs: [] } });
      setActiveRunId(null);
      setSelectedEventId(null);
      setPane("chat");
      return created.id;
    } catch (err) {
      setSendError(`Could not start a session. ${messageOf(err)}`);
      return null;
    } finally {
      setCreating(false);
    }
  };

  const upload = async (file: File) => {
    const sid = activeId ?? (await createNew());
    if (!sid) throw new Error("No session to attach to");
    const saved = await uploadFile(sid, file);
    setDetail((prev) =>
      prev?.status === "ready" && prev.data.id === sid
        ? {
            status: "ready",
            data: { ...prev.data, files: [...prev.data.files.filter((f) => f.name !== saved.name), saved] },
          }
        : prev,
    );
  };

  const send = async (question: string): Promise<boolean> => {
    const sid = activeId ?? (await createNew());
    if (!sid) return false;
    setSendError(null);

    const events: TrajectoryEvent[] = [];
    let text = "";
    const publish = (patch: Partial<RunView> = {}) =>
      setLive((prev) => ({
        sessionId: sid,
        text,
        run: {
          run_id: LIVE, question, answer: null, episode: null, status: "streaming",
          ...(prev?.sessionId === sid ? prev.run : {}),
          events: [...events],
          ...patch,
        },
      }));

    setLive({
      sessionId: sid, text: "",
      run: { run_id: LIVE, question, answer: null, episode: null, events: [], status: "streaming" },
    });
    setActiveRunId(LIVE);
    setSelectedEventId(null);
    setFresh(new Set());

    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;

    // `error` is not terminal: a run without an answer sends it and then
    // `done`. The message is only shown if the stream ends without `done`.
    let failure = "";
    const end = await sendMessage(
      sid,
      question,
      {
        onEvent: (event) => {
          events.push(event);
          // Deltas are the text of the model call in progress. A model event
          // ends that call (its text is on the event), so the bubble clears.
          if (event.kind === "model" || event.kind === "answer") text = "";
          setFresh((prev) => new Set(prev).add(event.id));
          publish();
        },
        onDelta: (chunk) => {
          text += chunk;
          publish();
        },
        onError: (message) => {
          failure = message;
        },
        onDone: (episode) => {
          const runId = episode?.run_id ?? `run-${Date.now()}`;
          const finished: RunView = {
            run_id: runId, question, episode: episode ?? null, events: [...events],
            answer: answerEvent(events)?.text ?? episode?.answer ?? null, status: "done",
          };
          setDetail((prev) =>
            prev?.status === "ready" && prev.data.id === sid
              ? { status: "ready", data: { ...prev.data, runs: [...prev.data.runs, finished] } }
              : prev,
          );
          setLive(null);
          setActiveRunId((prev) => (prev === LIVE ? runId : prev));
          window.setTimeout(() => setFresh(new Set()), 2500);
        },
      },
      controller.signal,
    );
    if (end === "ended") {
      text = "";
      publish({ status: "error", error: failure || "The run stopped unexpectedly." });
    }
    // The server titles a session from its first question, even if the run fails.
    if (end !== "aborted") {
      listSessions().then(
        (list) => setSessions({ status: "ready", data: list }),
        () => undefined,
      );
    }
    return true;
  };

  const runs = useMemo<RunView[]>(() => {
    const stored: RunView[] =
      detail?.status === "ready" ? detail.data.runs.map((r) => ({ ...r, status: "done" as const })) : [];
    return live && live.sessionId === activeId ? [...stored, live.run] : stored;
  }, [detail, live, activeId]);

  const activeRun = runs.find((r) => r.run_id === activeRunId) ?? runs[runs.length - 1];
  const lastEvent = activeRun?.events[activeRun.events.length - 1];
  const shownEventId =
    selectedEventId !== null && activeRun?.events.some((e) => e.id === selectedEventId)
      ? selectedEventId
      : (lastEvent?.id ?? null);

  const selectRun = (runId: string) => {
    setActiveRunId(runId);
    setSelectedEventId(null);
    setTab("trajectory");
    setPane("inspect");
  };

  const selectEvidence = (runId: string, eventId: number) => {
    setActiveRunId(runId);
    setSelectedEventId(eventId);
    setTab("trajectory");
    setPane("inspect");
  };

  const selectEvent = (eventId: number) => {
    setSelectedEventId(eventId);
    setTab("trajectory");
  };

  const busy = live?.run.status === "streaming";
  const paneState = detail === null ? "ready" : detail.status;
  const detailError = detail?.status === "error" ? detail.message : undefined;
  const retryDetail = activeId ? () => void loadDetail(activeId) : undefined;
  // The list is refreshed after every run, so it has the title the server gave.
  const listed = sessions.status === "ready" ? sessions.data.find((s) => s.id === activeId)?.title : undefined;
  const title =
    listed ||
    (detail?.status === "ready" ? detail.data.title || "Untitled session" : sessions.status === "ready" ? "New session" : "Session");

  return (
    <div className="flex h-dvh flex-col bg-ground">
      <header className="flex h-11 shrink-0 items-center gap-3 border-b bg-surface px-3">
        <Telescope className="size-[18px] text-foreground" aria-hidden />
        <h1 className="font-heading text-base leading-none font-bold tracking-tight">RSI</h1>
        <span className="text-[13px] text-muted-foreground max-sm:hidden">Science assistant</span>
        <div className="ml-auto flex items-center gap-2">
          {IS_MOCK ? (
            <span className="rounded-[4px] border border-k-jev/50 px-1.5 py-0.5 text-xs font-medium text-k-jev">
              Mock data
            </span>
          ) : null}
          <button
            type="button"
            onClick={() => {
              setTab("harness");
              setPane("inspect");
            }}
            className="flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs text-muted-foreground transition-colors hover:bg-raised hover:text-foreground"
          >
            <GitBranch className="size-3.5" aria-hidden />
            Harness
            <span className="font-medium text-foreground tabular-nums">
              {harness.status === "ready" ? harness.data.current || "none" : harness.status === "loading" ? "…" : "unavailable"}
            </span>
          </button>
        </div>
      </header>

      <nav aria-label="Panes" className="grid shrink-0 grid-cols-3 border-b bg-surface lg:hidden">
        {PANES.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            aria-current={pane === id ? "page" : undefined}
            onClick={() => setPane(id)}
            className={cn(
              "flex h-10 items-center justify-center gap-1.5 border-b-2 text-[13px] font-medium -outline-offset-2 transition-colors",
              pane === id ? "border-foreground text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            <Icon className="size-4" aria-hidden />
            {label}
          </button>
        ))}
      </nav>

      <main className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(0,1fr)] lg:grid-cols-[212px_minmax(0,1fr)_minmax(0,1.22fr)]">
        <aside className={cn("min-h-0 lg:block lg:border-r", pane !== "sessions" && "max-lg:hidden")}>
          <SessionList
            sessions={sessions}
            activeId={activeId}
            streamingId={busy ? (live?.sessionId ?? null) : null}
            creating={creating}
            onSelect={selectSession}
            onCreate={() => void createNew()}
            onRetry={() => {
              setSessions({ status: "loading" });
              void boot();
            }}
          />
        </aside>

        <section aria-label="Chat" className={cn("min-h-0 min-w-0 lg:block", pane !== "chat" && "max-lg:hidden")}>
          <ChatView
            state={detail === null ? "idle" : detail.status}
            errorMessage={detailError}
            onRetry={retryDetail}
            title={title}
            files={detail?.status === "ready" ? detail.data.files : []}
            runs={runs}
            activeRunId={activeRun?.run_id ?? null}
            selectedEventId={shownEventId}
            streamText={live?.text ?? ""}
            busy={busy}
            sendError={sendError}
            onSelectRun={selectRun}
            onSelectEvidence={selectEvidence}
            onSend={send}
            onUpload={upload}
          />
        </section>

        <section
          aria-label="Run inspector"
          className={cn("min-h-0 min-w-0 bg-surface/40 lg:block lg:border-l", pane !== "inspect" && "max-lg:hidden")}
        >
          <Tabs value={tab} onValueChange={(v) => setTab(v as RightTab)} className="h-full min-h-0 gap-0">
            <div className="flex h-[45px] shrink-0 items-end border-b px-3">
              <TabsList variant="line" className="h-9 gap-4 p-0">
                <TabsTrigger value="trajectory" className="px-0.5 font-heading text-[14px] font-semibold">
                  Compute Trajectory
                </TabsTrigger>
                <TabsTrigger value="results" className="px-0.5 font-heading text-[14px] font-semibold">
                  Results
                </TabsTrigger>
                <TabsTrigger value="harness" className="px-0.5 font-heading text-[14px] font-semibold">
                  Harness
                </TabsTrigger>
              </TabsList>
            </div>
            <TabsContent value="trajectory" className="min-h-0 overflow-y-auto">
              <TrajectoryTimeline
                state={paneState}
                errorMessage={detailError}
                onRetry={retryDetail}
                runs={runs}
                run={activeRun}
                selectedId={shownEventId}
                fresh={fresh}
                onSelect={selectEvent}
                onSelectRun={selectRun}
              />
            </TabsContent>
            <TabsContent value="results" className="min-h-0 overflow-y-auto">
              <ResultsPanel
                state={paneState}
                errorMessage={detailError}
                onRetry={retryDetail}
                run={activeRun}
                sessionId={activeId}
                limits={harness.status === "ready" ? harness.data.limits : undefined}
                onSelectEvent={selectEvent}
              />
            </TabsContent>
            <TabsContent value="harness" className="min-h-0 overflow-y-auto">
              <HarnessPanel
                harness={harness}
                onRetry={() => {
                  setHarness({ status: "loading" });
                  void loadHarness();
                }}
              />
            </TabsContent>
          </Tabs>
        </section>
      </main>
    </div>
  );
}
