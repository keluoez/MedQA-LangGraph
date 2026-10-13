import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import clsx from "clsx";
import { useAppStore } from "./store";
import DatasetTab from "./pages/DatasetTab";
import PipelineTab from "./pages/PipelineTab";
import MechanismTab from "./pages/MechanismTab";
import type { StatsResponse } from "./api/types";
import { api } from "./api/client";

function Header() {
  const { t, i18n } = useTranslation();
  const tab = useAppStore((s) => s.tab);
  const setTab = useAppStore((s) => s.setTab);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    api.stats().then(setStats).catch(() => {});
  }, []);

  const toggleLang = () => {
    const next = i18n.language === "zh" ? "en" : "zh";
    i18n.changeLanguage(next);
    localStorage.setItem("lang", next);
  };

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-20">
      <div className="mx-auto max-w-[1440px] px-6 py-3 flex items-center gap-6">
        <div className="flex items-center gap-3 min-w-0">
          {/* logo mark */}
          <div className="w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/40 grid place-items-center shrink-0">
            <span className="mono text-emerald-400 text-sm font-bold">M</span>
          </div>
          <div className="min-w-0">
            <h1 className="text-slate-100 font-semibold text-sm leading-tight truncate">
              {t("app.title")}
            </h1>
            <p className="text-[11px] text-slate-500 leading-tight truncate">
              {t("app.subtitle")}
            </p>
          </div>
        </div>

        {/* 统计徽章 */}
        {stats && (
          <div className="hidden md:flex items-center gap-2 text-xs mono">
            <span className="px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
              {stats.overall.total.toLocaleString()} {t("stats.pairs")}
            </span>
            <span className="px-2 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Gold {stats.overall.gold_pct}%
            </span>
            <span className="px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
              {t("stats.meanScore")} {stats.overall.mean_score}/10
            </span>
          </div>
        )}

        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={() => setTab("dataset")}
            className={clsx(
              "px-3 py-1.5 rounded-md text-sm transition-colors",
              tab === "dataset"
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/40"
                : "text-slate-400 hover:text-slate-200 border border-transparent",
            )}
          >
            {t("app.tabDataset")}
          </button>
          <button
            onClick={() => setTab("pipeline")}
            className={clsx(
              "px-3 py-1.5 rounded-md text-sm transition-colors",
              tab === "pipeline"
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/40"
                : "text-slate-400 hover:text-slate-200 border border-transparent",
            )}
          >
            {t("app.tabPipeline")}
          </button>
          <button
            onClick={() => setTab("mechanism")}
            className={clsx(
              "px-3 py-1.5 rounded-md text-sm transition-colors",
              tab === "mechanism"
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/40"
                : "text-slate-400 hover:text-slate-200 border border-transparent",
            )}
          >
            {t("app.tabMechanism")}
          </button>
          <button
            onClick={toggleLang}
            className="ml-2 px-2.5 py-1.5 rounded-md text-xs text-slate-400 hover:text-emerald-400 border border-slate-700 hover:border-emerald-500/40 transition-colors mono"
          >
            {t("app.langToggle")}
          </button>
        </div>
      </div>
    </header>
  );
}

export default function App() {
  const tab = useAppStore((s) => s.tab);
  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto max-w-[1440px] px-6 py-6">
        {tab === "dataset" ? <DatasetTab /> : tab === "pipeline" ? <PipelineTab /> : <MechanismTab />}
      </main>
    </div>
  );
}
