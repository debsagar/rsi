import { CircleAlert, RotateCw } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";

// The three non-content states every pane can be in.

export function PaneLoading({ label, lines = 4 }: { label: string; lines?: number }) {
  return (
    <div role="status" aria-label={label} className="flex flex-col gap-2.5 p-4">
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className="skeleton h-3.5"
          style={{ width: `${[72, 94, 58, 83, 66][i % 5]}%` }}
        />
      ))}
      <span className="sr-only">{label}</span>
    </div>
  );
}

export function PaneEmpty({
  icon,
  title,
  children,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-start gap-1.5 px-4 py-8 text-muted-foreground">
      {icon ? <div className="mb-1 text-muted-foreground [&_svg]:size-5">{icon}</div> : null}
      <h3 className="font-heading text-[15px] font-semibold text-foreground">{title}</h3>
      {children ? <div className="max-w-[46ch] text-[13px] leading-relaxed">{children}</div> : null}
    </div>
  );
}

export function PaneError({
  title = "Something went wrong",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div role="alert" className="m-4 rounded-md border border-k-error/40 bg-k-error/8 p-3">
      <div className="flex items-start gap-2">
        <CircleAlert className="mt-0.5 size-4 shrink-0 text-k-error" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="text-[13px] font-semibold text-foreground">{title}</p>
          <p className="mt-0.5 text-[13px] break-words text-muted-foreground">{message}</p>
          {onRetry ? (
            <Button variant="outline" size="sm" className="mt-2.5" onClick={onRetry}>
              <RotateCw aria-hidden />
              Try again
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
