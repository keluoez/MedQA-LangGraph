import type { ReactNode } from "react";
import clsx from "clsx";
import type { NodeStatus } from "./usePipelineRun";

const ACCENTS = {
  sky: {
    border: "border-sky-500/40",
    glow: "shadow-[0_0_20px_rgba(56,189,248,0.08)]",
    bar: "bg-sky-400",
    dot: "bg-sky-400",
    title: "text-sky-300",
  },
  emerald: {
    border: "border-emerald-500/40",
    glow: "shadow-[0_0_20px_rgba(52,211,153,0.08)]",
    bar: "bg-emerald-400",
    dot: "bg-emerald-400",
    title: "text-emerald-300",
  },
  amber: {
    border: "border-amber-500/40",
    glow: "shadow-[0_0_20px_rgba(251,191,36,0.08)]",
    bar: "bg-amber-400",
    dot: "bg-amber-400",
    title: "text-amber-300",
  },
  yellow: {
    border: "border-yellow-500/40",
    glow: "shadow-[0_0_20px_rgba(250,204,21,0.06)]",
    bar: "bg-yellow-400",
    dot: "bg-yellow-400",
    title: "text-yellow-300",
  },
  rose: {
    border: "border-rose-500/40",
    glow: "shadow-[0_0_20px_rgba(251,113,133,0.08)]",
    bar: "bg-rose-400",
    dot: "bg-rose-400",
    title: "text-rose-300",
  },
} as const;

export type Accent = keyof typeof ACCENTS;

export default function AgentCard({
  title,
  subtitle,
  status,
  accent,
  badge,
  children,
  minHeight = true,
}: {
  title: string;
  subtitle: string;
  status: NodeStatus;
  accent: Accent;
  badge?: ReactNode;
  children?: ReactNode;
  minHeight?: boolean;
}) {
  const a = ACCENTS[accent];
  return (
    <div
      className={clsx(
        "relative rounded-xl bg-slate-900 border transition-all duration-300 overflow-hidden",
        status === "idle" && "border-slate-800 opacity-50",
        status === "running" && clsx(a.border, a.glow, "animate-breathing"),
        status === "done" && clsx("border-slate-700"),
      )}
    >
      {/* 顶部状态条 */}
      <div
        className={clsx(
          "h-0.5 transition-opacity duration-300",
          status === "running" ? a.bar : status === "done" ? "bg-slate-600" : "bg-slate-800",
          status === "running" && "animate-breathing",
        )}
      />
      <div className="p-3.5">
        <div className="flex items-center gap-2 mb-1">
          <span
            className={clsx(
              "w-1.5 h-1.5 rounded-full shrink-0",
              status === "idle" ? "bg-slate-600" : a.dot,
              status === "running" && "animate-breathing",
            )}
          />
          <h3 className={clsx("text-xs font-semibold tracking-wide", status === "idle" ? "text-slate-500" : a.title)}>
            {title}
          </h3>
          <div className="ml-auto">{badge}</div>
        </div>
        <p className="text-[10px] text-slate-500 mb-2">{subtitle}</p>
        <div
          className={clsx(
            "overflow-y-auto",
            minHeight && "max-h-64",
            status === "idle" && "opacity-40",
          )}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

/** 节点之间的流动箭头。active 时高亮 + 流动动画。 */
export function RouteArrow({
  active,
  label,
  tone = "emerald",
}: {
  active: boolean;
  label?: string;
  tone?: "emerald" | "amber" | "rose" | "slate";
}) {
  const tones = {
    emerald: "bg-emerald-400",
    amber: "bg-amber-400",
    rose: "bg-rose-400",
    slate: "bg-slate-700",
  };
  const line = active ? tones[tone] : "bg-slate-800";
  return (
    <div className="flex flex-col items-center justify-center gap-1 w-10 shrink-0 self-stretch pt-8">
      <div className={clsx("h-0.5 w-full rounded relative overflow-hidden", line)}>
        {active && (
          <div
            className="absolute inset-0 animate-flow"
            style={{
              backgroundImage:
                "repeating-linear-gradient(90deg, rgba(255,255,255,0.45) 0 8px, transparent 8px 16px)",
            }}
          />
        )}
      </div>
      <span
        className={clsx(
          "w-0 h-0 border-y-4 border-y-transparent border-l-[7px]",
          active
            ? tone === "amber"
              ? "border-l-amber-400"
              : tone === "rose"
                ? "border-l-rose-400"
                : "border-l-emerald-400"
            : "border-l-slate-800",
        )}
      />
      {label && (
        <span className="text-[9px] text-slate-600 mono">{label}</span>
      )}
    </div>
  );
}
