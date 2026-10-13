import clsx from "clsx";
import type { Tier } from "../api/types";
import { useTranslation } from "react-i18next";

export function TierBadge({ tier, className }: { tier: Tier; className?: string }) {
  const { t } = useTranslation();
  const style = {
    gold: "bg-amber-400/10 text-amber-300 border-amber-400/40",
    silver: "bg-slate-400/10 text-slate-300 border-slate-400/40",
    bronze: "bg-orange-800/20 text-orange-300 border-orange-500/40",
  }[tier];
  return (
    <span
      className={clsx(
        "px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide border",
        style,
        className,
      )}
    >
      {t(`tier.${tier}`)}
    </span>
  );
}

export function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 9
      ? "text-emerald-400"
      : score >= 8
        ? "text-teal-400"
        : score >= 6
          ? "text-amber-400"
          : "text-rose-400";
  return (
    <span className={clsx("mono font-bold text-lg leading-none", color)}>
      {score.toFixed(2)}
    </span>
  );
}

export function DomainBadge({ domain }: { domain: string }) {
  const { t } = useTranslation();
  return (
    <span className="px-1.5 py-0.5 rounded text-[10px] text-sky-300/90 bg-sky-500/10 border border-sky-500/25 whitespace-nowrap">
      {t(`domain.${domain}`, domain)}
    </span>
  );
}

export function DomainBar({ domain }: { domain: string }) {
  const { t } = useTranslation();
  return (
    <span className="px-1.5 py-0.5 rounded text-[10px] text-slate-400 bg-slate-800/80 border border-slate-700 whitespace-nowrap">
      {t(`domain.${domain}`, domain)}
    </span>
  );
}
