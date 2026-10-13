import type { AnswerSection } from "../api/types";
import { useTranslation } from "react-i18next";

/** 把【Tag】五段式回答渲染为带高亮 PMID 的结构化区块。 */
function highlightPmids(text: string) {
  const parts = text.split(/(PMID:?\d+|UniProt:?\s?[OPQ][0-9A-Z]{5,9})/g);
  return parts.map((p, i) => {
    if (/^PMID:?\d+/.test(p) || /^UniProt:?\s?[OPQ][0-9A-Z]{5,9}$/.test(p)) {
      return (
        <span
          key={i}
          className="mono text-emerald-400 bg-emerald-400/10 rounded px-0.5"
          title={p}
        >
          {p}
        </span>
      );
    }
    return <span key={i}>{p}</span>;
  });
}

export default function SectionView({
  sections,
  compact = false,
}: {
  sections: AnswerSection[];
  compact?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <div className="space-y-3">
      {sections.map((s) => (
        <div key={s.title} className={compact ? "space-y-1" : "space-y-1.5"}>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-500/90 border border-emerald-500/30 bg-emerald-500/5 rounded px-1.5 py-0.5 mono">
              {t(`section.${s.title}`, s.title)}
            </span>
          </div>
          <p
            className={
              compact
                ? "text-xs text-slate-300 leading-relaxed whitespace-pre-wrap"
                : "text-sm text-slate-300 leading-relaxed whitespace-pre-wrap"
            }
          >
            {highlightPmids(s.text)}
          </p>
        </div>
      ))}
    </div>
  );
}
