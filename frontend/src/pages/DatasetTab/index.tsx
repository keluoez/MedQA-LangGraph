import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import clsx from "clsx";
import { api } from "../../api/client";
import type { SampleListResponse, SampleSummary } from "../../api/types";
import { DomainBadge, ScoreBadge, TierBadge } from "../../components/badges";
import SampleDrawer from "./SampleDrawer";

const DOMAINS = ["drug", "experimental", "pathological", "physiological"];
const TIERS = ["gold", "silver", "bronze"];

export default function DatasetTab() {
  const { t } = useTranslation();
  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  const [domain, setDomain] = useState("");
  const [tier, setTier] = useState("");
  const [sort, setSort] = useState("score");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<SampleListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const id = setTimeout(() => setDebouncedQ(q), 350);
    return () => clearTimeout(id);
  }, [q]);

  const fetchPage = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await api.listSamples({
        page,
        page_size: 20,
        domain: domain || undefined,
        tier: tier || undefined,
        q: debouncedQ || undefined,
        sort,
      });
      setData(resp);
    } finally {
      setLoading(false);
    }
  }, [page, domain, tier, debouncedQ, sort]);

  useEffect(() => {
    fetchPage();
  }, [fetchPage]);

  const pages = useMemo(
    () => (data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1),
    [data],
  );

  const select = (s: SampleSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelected(s.id);
  };

  return (
    <div className="space-y-4">
      {/* 筛选栏 */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setPage(1);
          }}
          placeholder={t("dataset.searchPlaceholder")}
          className="flex-1 min-w-[220px] max-w-md px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm placeholder:text-slate-600 focus:outline-none focus:border-emerald-500/60"
        />
        <select
          value={domain}
          onChange={(e) => {
            setDomain(e.target.value);
            setPage(1);
          }}
          className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm text-slate-300 focus:outline-none focus:border-emerald-500/60"
        >
          <option value="">{t("dataset.allDomains")}</option>
          {DOMAINS.map((d) => (
            <option key={d} value={d}>
              {t(`domain.${d}`)}
            </option>
          ))}
        </select>
        <select
          value={tier}
          onChange={(e) => {
            setTier(e.target.value);
            setPage(1);
          }}
          className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm text-slate-300 focus:outline-none focus:border-emerald-500/60"
        >
          <option value="">{t("dataset.allTiers")}</option>
          {TIERS.map((x) => (
            <option key={x} value={x}>
              {t(`tier.${x}`)}
            </option>
          ))}
        </select>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm text-slate-300 focus:outline-none focus:border-emerald-500/60"
        >
          <option value="score">{t("dataset.sortScore")}</option>
          <option value="original">{t("dataset.sortOriginal")}</option>
        </select>
        {data && (
          <span className="mono text-xs text-slate-500 ml-auto">
            {data.total.toLocaleString()} {t("stats.pairs")}
          </span>
        )}
      </div>

      {/* 列表 */}
      <div className="grid grid-cols-1 gap-2">
        {loading && !data ? (
          <div className="h-32 grid place-items-center text-slate-500 text-sm animate-breathing">
            {t("pipeline.running")}
          </div>
        ) : data && data.items.length === 0 ? (
          <div className="h-32 grid place-items-center text-slate-500 text-sm">
            {t("dataset.noResults")}
          </div>
        ) : (
          data?.items.map((s) => (
            <button
              key={s.id}
              onClick={(e) => select(s, e)}
              className={clsx(
                "text-left p-3 rounded-xl bg-slate-900 border border-slate-800 min-w-0",
                "hover:border-emerald-500/40 hover:bg-slate-900/70 transition-colors",
                selected === s.id && "border-emerald-500/50",
              )}
            >
              <div className="flex items-center gap-2 flex-wrap">
                <DomainBadge domain={s.domain} />
                <TierBadge tier={s.tier} />
                <span className="mono text-[11px] text-slate-500">
                  {s.drug} → {s.protein} · {s.uniprot_id}
                </span>
                <span className="ml-auto flex items-center gap-2">
                  {s.correction_count > 0 && (
                    <span className="text-[10px] text-amber-400/80 mono">
                      ↻ {t("dataset.corrections", { count: s.correction_count })}
                    </span>
                  )}
                  <ScoreBadge score={s.quality_score} />
                </span>
              </div>
              <p className="mt-1.5 text-sm text-slate-300 truncate">{s.question}</p>
            </button>
          ))
        )}
      </div>

      {/* 分页 */}
      {data && data.items.length > 0 && (
        <div className="flex items-center justify-center gap-3 text-sm text-slate-400">
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
            className="px-3 py-1.5 rounded-md border border-slate-700 disabled:opacity-30 hover:border-emerald-500/40 transition-colors"
          >
            {t("dataset.prev")}
          </button>
          <span className="mono text-xs">
            {t("dataset.page", { page, pages })}
          </span>
          <button
            disabled={page >= pages}
            onClick={() => setPage((p) => p + 1)}
            className="px-3 py-1.5 rounded-md border border-slate-700 disabled:opacity-30 hover:border-emerald-500/40 transition-colors"
          >
            {t("dataset.next")}
          </button>
        </div>
      )}

      {/* 详情抽屉 */}
      {selected && <SampleDrawer id={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
