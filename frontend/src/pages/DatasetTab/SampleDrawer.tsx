import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../../api/client";
import type { SampleDetail } from "../../api/types";
import { DomainBadge, ScoreBadge, TierBadge } from "../../components/badges";
import SectionView from "../../components/SectionView";
import { useAppStore } from "../../store";

function SubscoreBar({ label, value, max }: { label: string; value: number; max: number }) {
  const { t } = useTranslation();
  const pct = Math.min(100, (value / max) * 100);
  const ratio = value / max;
  const color = ratio >= 0.85 ? "bg-emerald-400" : ratio >= 0.6 ? "bg-teal-400" : "bg-amber-400";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-slate-400 shrink-0">{t(`pipeline.${label}`)}</span>
      <div className="flex-1 h-1.5 rounded bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="mono text-slate-300 w-10 text-right">
        {value}/{max}
      </span>
    </div>
  );
}

export default function SampleDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const { t } = useTranslation();
  const [sample, setSample] = useState<SampleDetail | null>(null);
  const sendToPipeline = useAppStore((s) => s.sendToPipeline);

  useEffect(() => {
    api.getSample(id).then(setSample).catch(() => {});
  }, [id]);

  return (
    <div className="fixed inset-0 z-30 flex justify-end" onClick={onClose}>
      <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-2xl h-full bg-slate-900 border-l border-slate-800 overflow-y-auto animate-fade-up"
        onClick={(e) => e.stopPropagation()}
      >
        {!sample ? (
          <div className="h-full grid place-items-center text-slate-500 animate-breathing">
            {t("pipeline.running")}
          </div>
        ) : (
          <div className="p-6 space-y-5">
            {/* 头部 */}
            <div className="flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <DomainBadge domain={sample.domain} />
                  <TierBadge tier={sample.tier} />
                  {sample.correction_count > 0 && (
                    <span className="text-[10px] text-amber-400/90 mono">
                      ↻ {t("dataset.corrections", { count: sample.correction_count })}
                    </span>
                  )}
                </div>
                <h2 className="text-slate-100 text-base font-medium leading-snug">
                  {sample.question}
                </h2>
                <p className="mono text-xs text-slate-500 mt-1.5">
                  {sample.drug} → {sample.protein} · {sample.uniprot_id} ·{" "}
                  {t("dataset.elapsed", { sec: sample.elapsed_seconds })}
                </p>
              </div>
              <div className="text-right shrink-0">
                <ScoreBadge score={sample.quality_score} />
                <p className="text-[10px] text-slate-500 mt-0.5">{t("dataset.score")}</p>
              </div>
              <button
                onClick={onClose}
                className="ml-2 px-2 py-1 rounded text-slate-500 hover:text-slate-200 border border-slate-700 hover:border-slate-500 transition-colors text-xs"
              >
                ✕
              </button>
            </div>

            {/* 五段式回答 */}
            <div className="rounded-xl bg-slate-950/60 border border-slate-800 p-4">
              <SectionView sections={sample.sections} />
            </div>

            {/* 质量日志 */}
            <div className="rounded-xl bg-slate-950/60 border border-slate-800 p-4 space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                {t("dataset.qualityLog")}
              </h3>
              <div className="space-y-2">
                <SubscoreBar label="factuality" value={sample.subscores.factuality_effective ?? 0} max={4} />
                <SubscoreBar label="grounding" value={sample.subscores.grounding_llm_raw ?? 0} max={3} />
                <SubscoreBar label="format" value={sample.subscores.format_llm_raw ?? 0} max={2} />
                <SubscoreBar label="consistency" value={sample.subscores.consistency_llm_raw ?? 0} max={1} />
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <p className="text-slate-500">{t("dataset.errors")}</p>
                  <p className={sample.errors.length ? "text-amber-400 mono" : "text-slate-600"}>
                    {sample.errors.length ? sample.errors.join(" · ") : t("dataset.noErrors")}
                  </p>
                </div>
                <div>
                  <p className="text-slate-500">{t("dataset.criticalErrors")}</p>
                  <p className={sample.critical_errors.length ? "text-rose-400 mono" : "text-slate-600"}>
                    {sample.critical_errors.length
                      ? sample.critical_errors.join(" · ")
                      : t("dataset.noErrors")}
                  </p>
                </div>
              </div>
            </div>

            {/* Tab 联动 */}
            <button
              onClick={() =>
                sendToPipeline({
                  drug: sample.drug,
                  protein: sample.protein,
                  uniprot_id: sample.uniprot_id,
                  gene: sample.protein,
                })
              }
              className="w-full py-2.5 rounded-lg bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 text-sm font-medium hover:bg-emerald-500/25 transition-colors"
            >
              ▶ {t("dataset.sendToPipeline")}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
