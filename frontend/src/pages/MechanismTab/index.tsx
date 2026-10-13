import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import clsx from "clsx";
import { api } from "../../api/client";
import type { DomainMechanism, IntrospectResponse } from "../../api/types";

const DOMAIN_IDS = ["drug", "experimental", "pathological", "physiological"] as const;

const ROUTE_STYLE: Record<string, string> = {
  pass: "bg-emerald-400/10 text-emerald-400 border-emerald-400/30",
  refine: "bg-amber-400/10 text-amber-400 border-amber-400/30",
  regenerate: "bg-rose-400/10 text-rose-400 border-rose-400/30",
  stop: "bg-slate-700/30 text-slate-400 border-slate-600/50",
};

function Card({
  title,
  sub,
  className,
  children,
}: {
  title: string;
  sub?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={clsx("rounded-xl bg-slate-900 border border-slate-800 p-4", className)}>
      <h2 className="text-sm font-semibold text-slate-100">{title}</h2>
      {sub && <p className="text-[11px] text-slate-500 mt-0.5 mb-3">{sub}</p>}
      {!sub && <div className="mb-3" />}
      {children}
    </section>
  );
}

// ── Agent 流程条 ───────────────────────────────────────────────────────
function AgentFlow({ agents }: { agents: IntrospectResponse["agents"] }) {
  const { t } = useTranslation();
  const byId = Object.fromEntries(agents.map((a) => [a.id, a]));
  const main = ["collect", "generate", "verify"].map((id) => byId[id]);
  const branches = ["refine", "regenerate"].map((id) => byId[id]);
  return (
    <div className="rounded-xl bg-slate-900 border border-slate-800 p-3">
      <p className="text-[10px] text-slate-500 mb-2">{t("mechanism.flow")}</p>
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        {main.map((a, i) => (
          <span key={a.id} className="flex items-center gap-1.5">
            {i > 0 && <span className="text-slate-600">→</span>}
            <span
              title={a.desc}
              className="px-2 py-1 rounded bg-slate-800 border border-slate-700 text-slate-200"
            >
              {a.role}
            </span>
          </span>
        ))}
        <span className="text-slate-600">⇉</span>
        {branches.map((a) => (
          <span
            key={a.id}
            title={a.desc}
            className="px-2 py-1 rounded bg-amber-400/10 border border-amber-400/30 text-amber-400"
          >
            {a.role} ↺
          </span>
        ))}
        <span className="text-slate-600">|</span>
        <span
          title={byId["finalize"].desc}
          className="px-2 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400"
        >
          {byId["finalize"].role}
        </span>
      </div>
    </div>
  );
}

// ── Collector 检索机制 ─────────────────────────────────────────────────
function RetrievalCard({ d }: { d: DomainMechanism }) {
  const { t } = useTranslation();
  return (
    <Card title={t("mechanism.retrievalTitle")} sub={t("mechanism.retrievalSub")}>
      <div className="grid md:grid-cols-2 gap-x-6 gap-y-2">
        {d.retrieval.steps.map((s, i) => (
          <div key={i} className="flex gap-2.5">
            <span className="shrink-0 w-5 h-5 rounded-full bg-emerald-500/15 border border-emerald-500/40 grid place-items-center text-[10px] mono text-emerald-400">
              {i + 1}
            </span>
            <div className="min-w-0">
              <p className="text-[12px] text-slate-200 leading-snug">{s.title}</p>
              <p className="text-[11px] text-slate-500 leading-snug break-words">{s.detail}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-3 border-t border-slate-800">
        <p className="text-[10px] text-slate-500 mb-1.5">{t("mechanism.tools")}</p>
        <div className="space-y-1.5">
          {d.tools.map((tool) => (
            <div key={tool.name} className="text-[11px] leading-snug">
              <span className="text-sky-300">{tool.name}</span>
              <span className="text-slate-500"> · </span>
              <span className="mono text-slate-400 break-all">{tool.endpoint}</span>
              <p className="text-slate-500">{tool.note}</p>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

// ── PMID 排序算法 ──────────────────────────────────────────────────────
function RankCard({ d }: { d: DomainMechanism }) {
  const { t } = useTranslation();
  const maxAbs = Math.max(...d.retrieval.rank_weights.map((w) => Math.abs(w.weight)));
  return (
    <Card title={t("mechanism.rankTitle")} sub={t("mechanism.rankSub")}>
      <div className="space-y-1.5">
        {d.retrieval.rank_weights.map((w, i) => (
          <div key={i} className="flex items-center gap-2 text-[11px]">
            <div className="flex-1 min-w-0 text-slate-300 truncate" title={w.pattern}>
              {w.pattern}
            </div>
            <div className="w-20 h-1.5 rounded bg-slate-800 overflow-hidden flex justify-end">
              <div
                className={clsx("h-full", w.weight >= 0 ? "bg-emerald-400" : "bg-rose-400")}
                style={{ width: `${(Math.abs(w.weight) / maxAbs) * 100}%` }}
              />
            </div>
            <span
              className={clsx(
                "mono w-10 text-right",
                w.weight >= 0 ? "text-emerald-400" : "text-rose-400",
              )}
            >
              {w.weight >= 0 ? `+${w.weight.toFixed(1)}` : w.weight.toFixed(1)}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] text-slate-500 border-t border-slate-800 pt-2.5">
        {t("mechanism.whitelist", { n: d.retrieval.whitelist_size, m: d.retrieval.detail_articles })}
      </p>
    </Card>
  );
}

// ── Responder 上下文拼接 ───────────────────────────────────────────────
function ContextCard({ d }: { d: DomainMechanism }) {
  const { t } = useTranslation();
  return (
    <Card title={t("mechanism.contextTitle")} sub={t("mechanism.contextSub")}>
      <div className="space-y-1 mb-3">
        {d.context_blocks.map((b, i) => (
          <div key={i} className="text-[11px] leading-snug flex gap-1.5">
            <span className="text-emerald-400/70 shrink-0">▸</span>
            <div className="min-w-0">
              <span className="text-slate-200">{b.block}</span>
              <span className="text-slate-500"> — {b.source}</span>
            </div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-500 mb-1">{t("mechanism.previewNote")}</p>
      <pre className="mono text-[11px] text-slate-400 whitespace-pre-wrap leading-relaxed max-h-72 overflow-auto rounded-lg bg-slate-950 border border-slate-800 p-3">
        {d.context_preview}
      </pre>
      <div className="mt-2.5">
        <p className="text-[10px] text-slate-500 mb-1">{t("mechanism.verifierCtx")}</p>
        <p className="text-[11px] text-slate-500 leading-snug">
          {d.verifier_blocks.map((b) => `${b.block} (${b.source})`).join(" + ")}
        </p>
      </div>
    </Card>
  );
}

// ── Verifier 评分与路由 ────────────────────────────────────────────────
function ScoringCard({ d }: { d: DomainMechanism }) {
  const { t } = useTranslation();
  return (
    <Card title={t("mechanism.scoringTitle")} sub={t("mechanism.scoringSub")}>
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-slate-500 text-left">
            <th className="font-normal pb-1 pr-2">{t("mechanism.dimension")}</th>
            <th className="font-normal pb-1 pr-2 w-14">{t("mechanism.rangeCol")}</th>
            <th className="font-normal pb-1">{t("mechanism.fusionCol")}</th>
          </tr>
        </thead>
        <tbody>
          {d.scoring.dimensions.map((dim) => (
            <tr key={dim.key} className="border-t border-slate-800/70 align-top">
              <td className="py-1.5 pr-2 text-slate-200 whitespace-nowrap">
                {dim.label}
                <span className="mono text-slate-600 ml-1">{dim.key}</span>
              </td>
              <td className="py-1.5 pr-2 mono text-emerald-400">{dim.range}</td>
              <td className="py-1.5 text-slate-500 leading-snug">{dim.source}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-3">
        <p className="text-[10px] text-slate-500 mb-1">{t("mechanism.detChecks")}</p>
        <ul className="space-y-0.5">
          {d.scoring.det_checks.map((c, i) => (
            <li key={i} className="text-[11px] text-slate-400 leading-snug flex gap-1.5">
              <span className="text-amber-400/70 shrink-0">✓</span>
              {c}
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-3">
        <p className="text-[10px] text-slate-500 mb-1">{t("mechanism.formula")}</p>
        <p className="mono text-[11px] text-teal-300 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 leading-relaxed break-words">
          {d.scoring.formula}
        </p>
      </div>

      <div className="mt-3">
        <p className="text-[10px] text-slate-500 mb-1.5">{t("mechanism.routing")}</p>
        <div className="space-y-1">
          {d.scoring.routing.map((r, i) => (
            <div key={i} className="flex items-center gap-2 text-[11px]">
              <span
                className={clsx(
                  "mono px-1.5 py-px rounded border shrink-0 w-20 text-center",
                  ROUTE_STYLE[r.route],
                )}
              >
                {r.route}
              </span>
              <span className="text-slate-400 leading-snug">{r.condition}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

// ── Prompt 全文查看器 ───────────────────────────────────────────────────
function PromptCard({ d }: { d: DomainMechanism }) {
  const { t } = useTranslation();
  const [role, setRole] = useState<"responder" | "verifier" | "refiner">("responder");
  const roles = ["responder", "verifier", "refiner"] as const;
  return (
    <Card title={t("mechanism.promptsTitle")} sub={t("mechanism.promptsSub")}>
      <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs mono mb-3 w-fit">
        {roles.map((r) => (
          <button
            key={r}
            onClick={() => setRole(r)}
            className={clsx(
              "px-3 py-1.5 transition-colors capitalize",
              role === r
                ? "bg-emerald-500/20 text-emerald-400"
                : "text-slate-500 hover:text-slate-300",
            )}
          >
            {r}
          </button>
        ))}
      </div>
      <pre className="mono text-[11px] text-slate-400 whitespace-pre-wrap leading-relaxed max-h-96 overflow-auto rounded-lg bg-slate-950 border border-slate-800 p-3">
        {d.prompts[role]}
      </pre>
    </Card>
  );
}

// ── 主页面 ─────────────────────────────────────────────────────────────
export default function MechanismTab() {
  const { t } = useTranslation();
  const [data, setData] = useState<IntrospectResponse | null>(null);
  const [failed, setFailed] = useState(false);
  const [domainId, setDomainId] = useState<(typeof DOMAIN_IDS)[number]>("drug");

  useEffect(() => {
    api.introspect().then(setData).catch(() => setFailed(true));
  }, []);

  if (failed) {
    return <p className="text-sm text-rose-400 p-6">{t("mechanism.fetchError")}</p>;
  }
  const d = data?.domains[domainId];
  if (!data || !d) {
    return <p className="text-sm text-slate-500 p-6">{t("mechanism.loading")}</p>;
  }

  return (
    <div className="space-y-4">
      {/* 领域切换 + 元信息 */}
      <div className="rounded-xl bg-slate-900 border border-slate-800 p-3 space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10px] text-slate-500">{t("mechanism.domain")}</span>
          <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs">
            {DOMAIN_IDS.map((id) => (
              <button
                key={id}
                onClick={() => setDomainId(id)}
                className={clsx(
                  "px-3 py-1.5 transition-colors",
                  domainId === id
                    ? "bg-emerald-500/20 text-emerald-400"
                    : "text-slate-500 hover:text-slate-300",
                )}
              >
                {t(`domain.${id}`)}
              </button>
            ))}
          </div>
          <span className="ml-auto mono text-[10px] px-2 py-1 rounded bg-slate-800 border border-slate-700 text-sky-300">
            {t("mechanism.model")}: {d.model}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
          <span className="text-slate-500">{t("mechanism.questionTemplate")}:</span>
          <span className="mono text-slate-300">{d.question_template}</span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] text-slate-500">{t("mechanism.outputSections")}:</span>
          {d.sections.map((tag) => {
            const inner = tag.replace(/[【】]/g, "");
            return (
              <span
                key={tag}
                className="mono text-[10px] px-1.5 py-px rounded bg-slate-800 border border-slate-700 text-slate-300"
              >
                {t(`section.${inner}`)}
              </span>
            );
          })}
        </div>
      </div>

      <AgentFlow agents={data.agents} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <RetrievalCard d={d} />
        <div className="grid grid-cols-1 gap-4">
          <RankCard d={d} />
          <ContextCard d={d} />
        </div>
        <ScoringCard d={d} />
        <PromptCard d={d} />
      </div>
    </div>
  );
}
