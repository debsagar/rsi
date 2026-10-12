// Every backend call goes through request<T>() or streamSse().

import type {
  DonePayload,
  Episode,
  HarnessInfo,
  SessionDetail,
  SessionFile,
  SessionSummary,
  TrajectoryEvent,
} from "@/lib/types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function send(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${API_BASE}${path}`, init);
}

async function failure(res: Response): Promise<ApiError> {
  let message = `${res.status} ${res.statusText}`.trim();
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object") {
      const detail = (body as Record<string, unknown>).detail ??
        (body as Record<string, unknown>).message;
      if (typeof detail === "string") message = detail;
      else if (detail !== undefined) message = JSON.stringify(detail);
    }
  } catch {
    // Not JSON; the status line is the best message available.
  }
  return new ApiError(message, res.status);
}

function unreachable(err: unknown): Error {
  if (err instanceof ApiError) return err;
  if (err instanceof DOMException && err.name === "AbortError") return err;
  return new ApiError(
    `Cannot reach the API at ${API_BASE}. Is "rsi serve" running?`,
    0,
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await send(path, init);
  } catch (err) {
    throw unreachable(err);
  }
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

export type SseHandlers = {
  onEvent: (event: TrajectoryEvent) => void;
  onDelta: (text: string) => void;
  // Not terminal: a run that ends without an answer sends `error` and then
  // `done`. It is the last message only when the whole run crashed.
  onError: (message: string) => void;
  onDone: (episode: Episode) => void;
};

// "done": the server sent `done`. "ended": the stream closed without it
// (onError has been called with the reason). "aborted": the caller's signal.
export type StreamEnd = "done" | "ended" | "aborted";

// Parses one SSE block ("event: x\ndata: {...}") and routes it.
// Returns true only for `done`, the one terminal message.
function dispatch(block: string, handlers: SseHandlers): boolean {
  let name = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") name = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return false;
  let payload: unknown;
  try {
    payload = JSON.parse(data.join("\n"));
  } catch {
    handlers.onError(`Unreadable "${name}" message from the server.`);
    return false;
  }
  if (name === "event") {
    handlers.onEvent(payload as TrajectoryEvent);
  } else if (name === "delta") {
    handlers.onDelta(String((payload as { text?: unknown }).text ?? ""));
  } else if (name === "error") {
    const message = (payload as { message?: unknown }).message;
    handlers.onError(typeof message === "string" ? message : "The run failed.");
  } else if (name === "done") {
    handlers.onDone((payload as DonePayload).episode);
    return true;
  }
  return false;
}

// POSTs a JSON body and reads the response as server-sent events.
// EventSource cannot POST, so the stream is read with getReader().
async function streamSse(
  path: string,
  body: unknown,
  handlers: SseHandlers,
  signal?: AbortSignal,
): Promise<StreamEnd> {
  let res: Response;
  try {
    res = await send(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    const e = unreachable(err);
    if (e.name === "AbortError") return "aborted";
    handlers.onError(e.message);
    return "ended";
  }
  if (!res.ok || !res.body) {
    handlers.onError((await failure(res)).message);
    return "ended";
  }

  let reported = false;
  const wrapped: SseHandlers = {
    ...handlers,
    onError: (message) => {
      reported = true;
      handlers.onError(message);
    },
  };
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? "";
      for (const block of blocks) {
        if (dispatch(block, wrapped)) {
          await reader.cancel().catch(() => undefined);
          return "done";
        }
      }
      if (done) break;
    }
    if (buffer.trim() && dispatch(buffer, wrapped)) return "done";
    // Closed without `done`: the run crashed. Keep the server's own error
    // message if it sent one.
    if (!reported) handlers.onError("The stream ended before the run finished.");
    return "ended";
  } catch (err) {
    if (signal?.aborted) return "aborted";
    handlers.onError(
      err instanceof Error && err.message
        ? `The stream was interrupted: ${err.message}`
        : "The stream was interrupted.",
    );
    return "ended";
  }
}

export function createSession(): Promise<SessionSummary> {
  return request<SessionSummary>("/api/sessions", { method: "POST" });
}

export function listSessions(): Promise<SessionSummary[]> {
  return request<SessionSummary[]>("/api/sessions");
}

export function getSession(id: string): Promise<SessionDetail> {
  return request<SessionDetail>(`/api/sessions/${encodeURIComponent(id)}`);
}

export function uploadFile(id: string, file: File): Promise<SessionFile> {
  const form = new FormData();
  form.append("file", file);
  return request<SessionFile>(`/api/sessions/${encodeURIComponent(id)}/files`, {
    method: "POST",
    body: form,
  });
}

export function getHarnesses(): Promise<HarnessInfo> {
  return request<HarnessInfo>("/api/harnesses");
}

export function sendMessage(
  id: string,
  question: string,
  handlers: SseHandlers,
  signal?: AbortSignal,
): Promise<StreamEnd> {
  return streamSse(
    `/api/sessions/${encodeURIComponent(id)}/messages`,
    { question },
    handlers,
    signal,
  );
}
