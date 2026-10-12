"use client";

import { ArrowUp, CircleAlert, FileText, ImageIcon, LoaderCircle, Paperclip, X } from "lucide-react";
import { useRef, useState, type DragEvent, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { fmtBytes } from "@/lib/events";
import type { SessionFile } from "@/lib/types";
import { cn } from "@/lib/utils";

const MAX_BYTES = 25 * 1024 * 1024;

type Pending = { key: number; name: string; size: number; error?: string };

export function Composer({
  files,
  busy,
  onSend,
  onUpload,
}: {
  files: SessionFile[];
  busy: boolean;
  onSend: (question: string) => Promise<boolean>;
  onUpload: (file: File) => Promise<void>;
}) {
  const [text, setText] = useState("");
  const [pending, setPending] = useState<Pending[]>([]);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const nextKey = useRef(0);
  const dragDepth = useRef(0);

  const uploading = pending.some((p) => !p.error);
  const canSend = text.trim().length > 0 && !busy && !uploading;

  const add = (list: FileList | null) => {
    if (!list) return;
    for (const file of Array.from(list)) {
      const key = nextKey.current++;
      const fail = (error: string) =>
        setPending((prev) => prev.map((p) => (p.key === key ? { ...p, error } : p)));
      setPending((prev) => [...prev, { key, name: file.name, size: file.size }]);
      if (file.size > MAX_BYTES) {
        fail("Over the 25 MB limit");
        continue;
      }
      onUpload(file).then(
        () => setPending((prev) => prev.filter((p) => p.key !== key)),
        (err: unknown) => fail(err instanceof Error ? err.message : "Upload failed"),
      );
    }
  };

  const submit = async () => {
    if (!canSend) return;
    const question = text.trim();
    setText("");
    const ok = await onSend(question);
    if (!ok) setText(question);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submit();
    }
  };

  const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer.types).includes("Files");

  return (
    <form
      className={cn(
        "relative flex flex-col rounded-md border bg-surface transition-colors",
        dragging ? "border-foreground bg-raised" : "focus-within:border-foreground/60",
      )}
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
      onDragEnter={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dragDepth.current++;
        setDragging(true);
      }}
      onDragOver={(e) => {
        if (hasFiles(e)) e.preventDefault();
      }}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        add(e.dataTransfer.files);
      }}
    >
      <input
        ref={input}
        type="file"
        multiple
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        onChange={(e) => {
          add(e.target.files);
          e.target.value = "";
        }}
      />

      {/* Upload zone: click, or drop anywhere on the composer */}
      <button
        type="button"
        onClick={() => input.current?.click()}
        className={cn(
          "m-1.5 mb-0 flex items-center gap-2 rounded-[4px] border border-dashed px-2.5 py-1.5 text-left text-xs transition-colors",
          dragging
            ? "border-foreground text-foreground"
            : "border-line text-muted-foreground hover:border-muted-foreground hover:text-foreground",
        )}
      >
        <Paperclip className="size-3.5 shrink-0" aria-hidden />
        <span className="min-w-0 truncate">
          {dragging ? "Drop to attach" : "Drop files here, or click to attach data or figures"}
        </span>
        <span className="ml-auto shrink-0 max-sm:hidden">25 MB each</span>
      </button>

      {files.length > 0 || pending.length > 0 ? (
        <ul className="flex flex-wrap gap-1.5 px-1.5 pt-1.5" aria-label="Attached files">
          {files.map((file) => (
            <li
              key={file.name}
              className="flex max-w-full items-center gap-1.5 rounded-[4px] bg-raised px-2 py-1 text-xs"
            >
              {file.type.startsWith("image/") ? (
                <ImageIcon className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
              ) : (
                <FileText className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
              )}
              <span className="truncate font-medium">{file.name}</span>
              <span className="shrink-0 text-muted-foreground tabular-nums">{fmtBytes(file.size)}</span>
            </li>
          ))}
          {pending.map((p) => (
            <li
              key={p.key}
              className={cn(
                "flex max-w-full items-center gap-1.5 rounded-[4px] border px-2 py-[3px] text-xs",
                p.error ? "border-k-error/50 bg-k-error/10" : "border-line",
              )}
            >
              {p.error ? (
                <CircleAlert className="size-3.5 shrink-0 text-k-error" aria-hidden />
              ) : (
                <LoaderCircle className="size-3.5 shrink-0 animate-spin text-muted-foreground" aria-hidden />
              )}
              <span className="max-w-[45%] shrink-0 truncate font-medium">{p.name}</span>
              <span className="min-w-0 truncate text-muted-foreground" title={p.error}>
                {p.error ?? "Uploading"}
              </span>
              {p.error ? (
                <button
                  type="button"
                  aria-label={`Dismiss ${p.name}`}
                  className="-mr-1 grid size-5 shrink-0 place-items-center rounded-[3px] text-muted-foreground hover:bg-raised hover:text-foreground"
                  onClick={() => setPending((prev) => prev.filter((x) => x.key !== p.key))}
                >
                  <X className="size-3" aria-hidden />
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}

      <div className="flex items-end gap-2 p-1.5">
        <label htmlFor="question" className="sr-only">
          Question
        </label>
        <textarea
          id="question"
          value={text}
          rows={2}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about your data"
          className="max-h-40 min-h-[2.75rem] flex-1 resize-none bg-transparent px-1.5 py-1 text-sm leading-normal outline-none placeholder:text-muted-foreground [field-sizing:content]"
        />
        <Button type="submit" disabled={!canSend} aria-label="Send question">
          {busy ? <LoaderCircle className="animate-spin" aria-hidden /> : <ArrowUp aria-hidden />}
          <span className="max-sm:hidden">{busy ? "Running" : "Send"}</span>
        </Button>
      </div>
    </form>
  );
}
