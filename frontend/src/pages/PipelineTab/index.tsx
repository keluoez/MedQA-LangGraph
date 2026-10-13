import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import clsx from "clsx";
import { api } from "../../api/client";
import type { Pair, Scenario } from "../../api/types";
import { useAppStore } from "../../store";
import SectionView from "../../components/SectionView";
import AgentCard, { RouteArrow } from "./AgentCard";
import LogPanel from "./LogPanel";
import { usePipelineRun, type RunState } from "./usePipelineRun";
import { parseAnswer } from "../../utils/parseAnswer";

// ── Collector 卡内容 ───────────────────────────────────────────────
function CollectorBody({ state }: { state: RunState }) {
  const { t } = useTranslation();
  const ev = state.collector.evidence;
  if (!ev) return <p className="text-xs text-slate-600 mono">…</p>;
  const uniprot = ev.uniprot as Record<string, unknown>;
  const proteinName = String(uniprot.protein_name ?? "");
  const func = String(uniprot.function ?? "");
  return (
    <div className="space-y-2.5">
      {/* PMID 白名单 */}
      <div>
        <p className="text-[10px] text-slate-500 mb-1">{t("pipeline.pmidWhitelist")}</p>
        <div className="flex flex-wrap gap-1">
          {ev.pubmed_pmids.map((p) => (
            <span key={p} className="mono text-[10px] text-emerald-400 bg-emerald-400/10 border border-emerald-400/25 rounded px-1 py-px">
              PMID:{p}
            </span>
          ))}
          {ev.pubmed_pmids.length === 0 && (
            <span className="text-[10px] text-slate-600">0</span>
          )}
        </div>
      </div>
      {/* UniProt */}
      {proteinName && (
        <div>
          <p className="text-[10px] text-slate-500 mb-0.5">{t("pipeline.uniprotInfo")}</p>
          <p className="text-[11px] text-slate-300 leading-snug">
            <span className="text-sky-300">{proteinName}</span>
            {func && <span className="text-slate-500"> — {func.slice(0, 160)}{func.length > 160 ? "…" : ""}</span>}
          </p>
        </div>
      )}
      {/* ChEMBL 机制 */}
      {ev.chembl_mechanisms.length > 0 && (
        <div>
          <p className="text-[10px] text-slate-500 mb-0.5">{t("pipeline.chemblMechanisms")}</p>
          {ev.chembl_mechanisms.slice(0, 2).map((mch, i) => (
            <p key={i} className="text-[11px] text-slate-400 leading-snug mono">
              [{mch.action_type || "?"}] {mch.mechanism_of_action?.slice(0, 100) || mch.target_name}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Responder / Refiner / Regenerator 共用回答展示 ──────────────────
function AnswerBody({ answer }: { answer: string }) {
  const sections = useMemo(() => parseAnswer(answer), [answer]);
  if (sections.length === 0) {
    return <p className="text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">{answer}</p>;
  }
  return <SectionView sections={sections} compact />;
}

// ── Verifier 卡内容 ────────────────────────────────────────────────
function VerifierBody({ state }: { state: RunState }) {
  const { t } = useTranslation();
  const latest = state.verifier.audits.at(-1);
  if (!latest) return <p className="text-xs text-slate-600 mono">…</p>;
  const s = latest.subscores;
  const bars = [
    { label: "factuality", value: s.subscores.factuality_effective ?? 0, max: 4 },
    { label: "grounding", value: s.subscores.grounding_llm_raw ?? 0, max: 3 },
    { label: "format", value: s.subscores.format_llm_raw ?? 0, max: 2 },
    { label: "consistency", value: s.subscores.consistency_llm_raw ?? 0, max: 1 },
  ];
  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-2">
        <span
          className={clsx(
            "mono text-2xl font-bold",
            latest.score >= 8 ? "text-emerald-400" : latest.score >= 6 ? "text-amber-400" : "text-rose-400",
          )}
        >
          {latest.score.toFixed(2)}
        </span>
        <span className="text-[10px] text-slate-500">/ 10 · {t("pipeline.iteration", { n: latest.iteration })}</span>
      </div>
      <div className="space-y-1.5">
        {bars.map((b) => {
          const pct = Math.min(100, (b.value / b.max) * 100);
          const color = b.value / b.max >= 0.85 ? "bg-emerald-400" : b.value / b.max >= 0.6 ? "bg-teal-400" : "bg-amber-400";
          return (
            <div key={b.label} className="flex items-center gap-2 text-[11px]">
              <span className="w-16 text-slate-400 shrink-0">{t(`pipeline.${b.label}`)}</span>
              <div className="flex-1 h-1 rounded bg-slate-800 overflow-hidden">
                <div className={`h-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
              </div>
              <span className="mono text-slate-400 w-9 text-right">{b.value}/{b.max}</span>
            </div>
          );
        })}
      </div>
      {(s.critical_errors.length > 0 || s.errors.length > 0) && (
        <div className="space-y-0.5">
          {s.critical_errors.map((e, i) => (
            <p key={`c${i}`} className="text-[10px] text-rose-400 mono truncate" title={e}>✕ {e}</p>
          ))}
          {s.errors.map((e, i) => (
            <p key={`e${i}`} className="text-[10px] text-amber-400/80 mono truncate" title={e}>⚠ {e}</p>
          ))}
        </div>
      )}
    </div>
  );
}

// ── 主页面 ─────────────────────────────────────────────────────────
export default function PipelineTab() {
  const { t } = useTranslation();
  const { state, events, start, stop, reset } = usePipelineRun();
  const consumePendingPair = useAppStore((s) => s.consumePendingPair);
  const startedForRef = useRef<string | null>(null);

  const [mode, setMode] = useState<"mock" | "live">("mock");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [speed, setSpeed] = useState(2);
  const [pair, setPair] = useState<Pair>({ drug: "", protein: "", uniprot_id: "", gene: "" });
  const [maxCorrections, setMaxCorrections] = useState(2);

  useEffect(() => {
    api.scenarios().then((r) => {
      setScenarios(r.scenarios);
      if (r.scenarios.length > 0) setScenarioId((prev) => prev || r.scenarios[0].scenario_id);
    });
  }, []);

  // Tab1 → Tab2 联动
  useEffect(() => {
    const pending = consumePendingPair();
    if (pending) {
      setMode("live");
      setPair({ ...pending.pair });
      if (pending.autoRun) {
        const key = JSON.stringify(pending.pair);
        startedForRef.current = key;
        start({ mode: "live", pair: pending.pair, max_corrections: 2 });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const doRun = () => {
    if (mode === "mock") {
      start({ mode: "mock", scenario_id: scenarioId, speed });
    } else {
      if (!pair.drug || !pair.protein) return;
      start({ mode: "live", pair, max_corrections: maxCorrections });
    }
  };

  const running = state.phase === "running";
  const routeTo = state.lastRoute?.to;
  const iterationBadge = (n: number) => (
    <span className="text-[9px] mono px-1 py-px rounded bg-slate-800 text-slate-400 border border-slate-700">
      {t("pipeline.iteration", { n })}
    </span>
  );

  const latestResponder = state.responder.rounds.at(-1);
  const latestRefiner = state.refiner.rounds.at(-1);
  const latestRegen = state.regenerator.rounds.at(-1);
  const finalSections =
    state.final?.sample.sections ??
    (state.final ? parseAnswer(state.final.sample.answer) : undefined);

  return (
    <div className="space-y-4">
      {/* 控制条 */}
      <div className="rounded-xl bg-slate-900 border border-slate-800 p-3 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs mono">
          {(["mock", "live"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={clsx(
                "px-3 py-1.5 transition-colors",
                mode === m ? "bg-emerald-500/20 text-emerald-400" : "text-slate-500 hover:text-slate-300",
              )}
            >
              {t(`pipeline.${m}`)}
            </button>
          ))}
        </div>

        {mode === "mock" ? (
          <>
            <select
              value={scenarioId}
              onChange={(e) => setScenarioId(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-slate-300 focus:outline-none focus:border-emerald-500/60 min-w-[240px]"
            >
              {scenarios.map((s) => (
                <option key={s.scenario_id} value={s.scenario_id}>
                  {s.scenario_id} · {s.description}
                </option>
              ))}
            </select>
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="px-2 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-slate-300 focus:outline-none"
            >
              {[1, 2, 4].map((x) => (
                <option key={x} value={x}>
                  {x}x
                </option>
              ))}
            </select>
          </>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <PairInput pair={pair} onChange={setPair} />
            <label className="flex items-center gap-1 text-[10px] text-slate-500">
              {t("pipeline.maxCorrections")}
              <input
                type="number"
                min={1}
                max={4}
                value={maxCorrections}
                onChange={(e) => setMaxCorrections(Number(e.target.value))}
                className="w-12 px-1.5 py-1 rounded bg-slate-950 border border-slate-700 text-xs text-slate-300 focus:outline-none mono"
              />
            </label>
            <span className="text-[10px] text-slate-600">{t("pipeline.liveNeedsKey")}</span>
          </div>
        )}

        <div className="ml-auto flex items-center gap-2">
          {state.phase !== "idle" && (
            <button
              onClick={reset}
              className="px-3 py-1.5 rounded-lg text-xs text-slate-400 border border-slate-700 hover:border-slate-500 transition-colors"
            >
              {t("pipeline.reset")}
            </button>
          )}
          {running ? (
            <button
              onClick={stop}
              className="px-4 py-1.5 rounded-lg text-xs font-medium text-rose-300 bg-rose-500/15 border border-rose-500/40 hover:bg-rose-500/25 transition-colors"
            >
              ■ {t("pipeline.stop")}
            </button>
          ) : (
            <button
              onClick={doRun}
              disabled={mode === "live" && (!pair.drug || !pair.protein)}
              className="px-4 py-1.5 rounded-lg text-xs font-medium text-emerald-300 bg-emerald-500/15 border border-emerald-500/40 hover:bg-emerald-500/25 transition-colors disabled:opacity-40"
            >
              ▶ {t("pipeline.run")}
            </button>
          )}
        </div>
      </div>

      {/* 问题条 */}
      {(state.question || state.meta) && (
        <div className="rounded-xl bg-slate-900/60 border border-slate-800 px-4 py-3 flex items-center gap-3 animate-fade-up">
          <span className="text-[10px] mono text-emerald-500/80 uppercase tracking-wider shrink-0">Q</span>
          <p className="text-sm text-slate-200 flex-1">{state.question || state.meta?.pair.drug + " → " + state.meta?.pair.protein}</p>
          {state.meta && (
            <span className="mono text-[10px] text-slate-500 shrink-0">
              {state.meta.mode} · {state.meta.model}
            </span>
          )}
        </div>
      )}

      {/* 三主卡 + 流动箭头 */}
      <div className="flex items-stretch gap-0">
        <div className="flex-1 min-w-0">
          <AgentCard
            title={t("pipeline.collector")}
            subtitle={t("pipeline.collectorSub")}
            status={state.collector.status}
            accent="sky"
          >
            <CollectorBody state={state} />
          </AgentCard>
        </div>
        <RouteArrow
          active={state.collector.status === "done" && state.responder.status !== "idle"}
          label="evidence"
        />
        <div className="flex-1 min-w-0">
          <AgentCard
            title={t("pipeline.responder")}
            subtitle={t("pipeline.responderSub")}
            status={state.responder.status}
            accent="emerald"
            badge={latestResponder && latestResponder.iteration > 1 ? iterationBadge(latestResponder.iteration) : undefined}
          >
            {latestResponder ? <AnswerBody answer={latestResponder.answer} /> : <p className="text-xs text-slate-600 mono">…</p>}
          </AgentCard>
        </div>
        <RouteArrow
          active={state.responder.status === "done" && state.verifier.status !== "idle"}
          label="draft"
        />
        <div className="flex-1 min-w-0">
          <AgentCard
            title={t("pipeline.verifier")}
            subtitle={t("pipeline.verifierSub")}
            status={state.verifier.status}
            accent="amber"
          >
            <VerifierBody state={state} />
          </AgentCard>
        </div>
      </div>

      {/* 修正循环分支 */}
      <div className="flex items-center gap-3">
        <div className="w-10 flex justify-center shrink-0">
          <div
            className={clsx(
              "w-px h-full min-h-16 border-l-2 border-dashed",
              routeTo === "refine" || routeTo === "regenerate" ? "border-amber-400/60" : "border-slate-800",
            )}
          />
        </div>
        <div className="grid grid-cols-2 gap-3 flex-1">
          <AgentCard
            title={t("pipeline.refiner")}
            subtitle={t("pipeline.refinerSub")}
            status={state.refiner.status}
            accent="yellow"
            minHeight={false}
            badge={latestRefiner ? iterationBadge(latestRefiner.iteration) : undefined}
          >
            {latestRefiner ? <AnswerBody answer={latestRefiner.answer} /> : null}
          </AgentCard>
          <AgentCard
            title={t("pipeline.regenerator")}
            subtitle={t("pipeline.regeneratorSub")}
            status={state.regenerator.status}
            accent="rose"
            minHeight={false}
            badge={latestRegen ? iterationBadge(latestRegen.iteration) : undefined}
          >
            {latestRegen ? <AnswerBody answer={latestRegen.answer} /> : null}
          </AgentCard>
        </div>
        <div className="shrink-0 w-24 text-center">
          {state.refiner.status === "done" || state.regenerator.status === "done" ? (
            <span className="mono text-[10px] text-amber-400/90 animate-breathing block">
              ↺ back to Verifier
            </span>
          ) : (
            <span className="mono text-[10px] text-slate-700">correction loop</span>
          )}
        </div>
      </div>

      {/* 路由决策可视化 */}
      {state.lastRoute && (
        <div className="rounded-xl border px-4 py-2.5 flex items-center gap-3 animate-fade-up"
          style={{
            borderColor:
              routeTo === "pass" ? "rgba(52,211,153,0.4)" :
              routeTo === "stop" ? "rgba(100,116,139,0.4)" :
              routeTo === "refine" ? "rgba(250,204,21,0.4)" : "rgba(251,113,133,0.4)",
            background:
              routeTo === "pass" ? "rgba(52,211,153,0.06)" :
              routeTo === "stop" ? "rgba(100,116,139,0.06)" :
              routeTo === "refine" ? "rgba(250,204,21,0.06)" : "rgba(251,113,133,0.06)",
          }}
        >
          <span className="text-[10px] mono uppercase tracking-wider text-slate-400 shrink-0">
            {t("pipeline.routeDecision")}
          </span>
          <span
            className={clsx(
              "mono text-sm font-bold",
              routeTo === "pass" ? "text-emerald-400" :
              routeTo === "stop" ? "text-slate-300" :
              routeTo === "refine" ? "text-yellow-300" : "text-rose-400",
            )}
          >
            {routeTo?.toUpperCase()}
          </span>
          <span className="text-xs text-slate-500 truncate">{state.lastRoute.reason}</span>
          {state.final && (
            <span className="ml-auto mono text-xs text-emerald-300 shrink-0">
              {t("pipeline.finalized")} · {state.final.score.toFixed(2)} · {state.final.iterations} iter · {state.final.elapsed}s
            </span>
          )}
        </div>
      )}

      {/* 错误条 */}
      {state.error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 animate-fade-up">
          <p className="mono text-xs text-rose-300 break-all">{state.error.message}</p>
        </div>
      )}

      {/* 日志面板 */}
      <LogPanel events={events} />

      {/* 最终样本 */}
      {state.final && finalSections && finalSections.length > 0 && (
        <details className="rounded-xl bg-slate-900/60 border border-emerald-500/25">
          <summary className="px-4 py-2.5 cursor-pointer text-xs text-emerald-400 font-medium select-none">
            {t("pipeline.finalized")} · {state.final.score.toFixed(2)}/10 · {t("pipeline.iteration", { n: state.final.iterations })}
          </summary>
          <div className="px-4 pb-4">
            <SectionView sections={finalSections} />
          </div>
        </details>
      )}
    </div>
  );
}

// ── live 模式交互对输入 ────────────────────────────────────────────
function PairInput({ pair, onChange }: { pair: Pair; onChange: (p: Pair) => void }) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Pair[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    clearTimeout(timer.current);
    if (!query.trim()) {
      setResults([]);
      return;
    }
    timer.current = setTimeout(async () => {
      try {
        const r = await api.pairs({ q: query, limit: 8 });
        setResults(r.items);
      } catch {
        setResults([]);
      }
    }, 300);
  }, [query]);

  const field = (key: keyof Pair, label: string, w = "w-28") => (
    <input
      value={String(pair[key] ?? "")}
      onChange={(e) => onChange({ ...pair, [key]: e.target.value })}
      placeholder={label}
      className={clsx(w, "px-2 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-emerald-500/60 mono")}
    />
  );

  return (
    <div className="flex flex-wrap items-center gap-2 relative">
      <div className="relative">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t("pipeline.searchPair")}
          className="w-52 px-2.5 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-emerald-500/60"
        />
        {results.length > 0 && (
          <div className="absolute top-full mt-1 left-0 right-0 z-10 rounded-lg bg-slate-950 border border-slate-700 shadow-xl overflow-hidden">
            {results.map((p, i) => (
              <button
                key={i}
                onClick={() => {
                  onChange({ ...p, gene: p.gene ?? p.protein });
                  setQuery("");
                  setResults([]);
                }}
                className="block w-full text-left px-2.5 py-1.5 text-xs text-slate-300 hover:bg-emerald-500/10 mono truncate"
              >
                {p.drug} → {p.protein}
              </button>
            ))}
          </div>
        )}
      </div>
      {field("drug", t("pipeline.drug"))}
      {field("protein", t("pipeline.protein"))}
      {field("uniprot_id", t("pipeline.uniprotId"), "w-24")}
      {field("gene", t("pipeline.gene"), "w-24")}
    </div>
  );
}
