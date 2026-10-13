import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import clsx from "clsx";
import type { PipelineEvent } from "../../api/types";

function evSummary(ev: PipelineEvent): { tag: string; text: string; tone: string } {
  switch (ev.type) {
    case "meta":
      return {
        tag: "INIT",
        text: `mode=${ev.mode} pair=${ev.pair.drug}→${ev.pair.protein}`,
        tone: "text-slate-400",
      };
    case "node_output": {
      if (ev.node === "collect") {
        const n = ev.payload.evidence.pubmed_pmids.length;
        return { tag: "COLLECT", text: `${n} PMIDs whitelisted`, tone: "text-sky-400" };
      }
      if (ev.node === "generate")
        return {
          tag: "RESPOND",
          text: `draft ${ev.payload.answer.length} chars`,
          tone: "text-emerald-400",
        };
      if (ev.node === "verify")
        return {
          tag: "VERIFY",
          text: `score=${ev.payload.score} errors=${ev.payload.errors.length} critical=${ev.payload.critical_errors.length}`,
          tone: "text-amber-400",
        };
      return {
        tag: ev.node === "refine" ? "REFINE" : "REGEN",
        text: `revised answer (${ev.iteration})`,
        tone: ev.node === "refine" ? "text-yellow-400" : "text-rose-400",
      };
    }
    case "route":
      return {
        tag: "ROUTE",
        text: `→ ${ev.to} (${ev.score}) · ${ev.reason}`,
        tone:
          ev.to === "pass"
            ? "text-emerald-400"
            : ev.to === "refine"
              ? "text-yellow-400"
              : ev.to === "regenerate"
                ? "text-rose-400"
                : "text-slate-400",
      };
    case "final":
      return {
        tag: "FINAL",
        text: `quality=${ev.score} iterations=${ev.iterations} elapsed=${ev.elapsed}s`,
        tone: "text-emerald-400 font-bold",
      };
    case "error":
      return { tag: "ERROR", text: ev.message, tone: "text-rose-400" };
    default:
      return { tag: "?", text: "", tone: "text-slate-400" };
  }
}

export default function LogPanel({ events }: { events: PipelineEvent[] }) {
  const { t } = useTranslation();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div className="rounded-xl bg-slate-950/80 border border-slate-800">
      <div className="px-4 py-2 border-b border-slate-800 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
        <h3 className="text-xs font-semibold text-slate-400 tracking-wide">
          {t("pipeline.logs")}
        </h3>
        <span className="mono text-[10px] text-slate-600 ml-auto">{events.length} events</span>
      </div>
      <div className="h-44 overflow-y-auto p-3 space-y-1 mono text-xs">
        {events.length === 0 ? (
          <p className="text-slate-600">{t("pipeline.noLogs")}</p>
        ) : (
          events.map((ev, i) => {
            const s = evSummary(ev);
            return (
              <div key={i} className="flex gap-2 items-baseline animate-fade-up">
                <span className="text-slate-600 shrink-0">
                  [{ev.ts?.toFixed(1) ?? "?.?"}s]
                </span>
                <span className={clsx("shrink-0 w-16 font-semibold", s.tone)}>{s.tag}</span>
                <span className="text-slate-400 break-all">{s.text}</span>
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
