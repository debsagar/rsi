import { isRejected, kindOf } from "@/lib/events";
import type { TrajectoryEvent } from "@/lib/types";

// One shape per event kind, so kinds stay distinguishable without colour.
// Hollow shapes are inputs or requests; filled shapes are outputs.
export function EventGlyph({
  event,
  size = 16,
  className,
}: {
  event: Pick<TrajectoryEvent, "kind"> & { accepted?: unknown };
  size?: number;
  className?: string;
}) {
  const color = kindOf(event as TrajectoryEvent).color;
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 16 16",
    "aria-hidden": true,
    className,
    style: { color, flexShrink: 0 },
  } as const;
  const stroke = { fill: "none", stroke: "currentColor", strokeWidth: 1.75 } as const;

  switch (event.kind) {
    case "user":
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="4.6" {...stroke} />
        </svg>
      );
    case "model":
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="5.2" fill="currentColor" />
        </svg>
      );
    case "tool_call":
      return (
        <svg {...common}>
          <rect x="3.4" y="3.4" width="9.2" height="9.2" rx="1.6" {...stroke} />
        </svg>
      );
    case "tool_result":
      return (
        <svg {...common}>
          <rect x="3" y="3" width="10" height="10" rx="1.8" fill="currentColor" />
        </svg>
      );
    case "judge":
      return (
        <svg {...common}>
          <path d="M8 2.2 13.8 8 8 13.8 2.2 8Z" fill="currentColor" />
        </svg>
      );
    case "submit":
      return isRejected(event as TrajectoryEvent) ? (
        <svg {...common}>
          <path d="M8 2.6 13.4 8 8 13.4 2.6 8Z" {...stroke} />
          <path d="M5.6 5.6 10.4 10.4" {...stroke} strokeLinecap="round" />
        </svg>
      ) : (
        <svg {...common}>
          <path d="M8 2.6 13.4 8 8 13.4 2.6 8Z" {...stroke} />
          <circle cx="8" cy="8" r="1.7" fill="currentColor" />
        </svg>
      );
    case "answer":
      return (
        <svg {...common}>
          <path
            d="M8 0.8 9.7 6.3 15.2 8 9.7 9.7 8 15.2 6.3 9.7 0.8 8 6.3 6.3Z"
            fill="currentColor"
          />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <path d="M4 4 12 12M12 4 4 12" {...stroke} strokeWidth={2.2} strokeLinecap="round" />
        </svg>
      );
  }
}
