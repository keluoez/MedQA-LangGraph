import { create } from "zustand";
import type { Pair } from "./api/types";

export type TabId = "dataset" | "pipeline" | "mechanism";

interface AppState {
  tab: TabId;
  /** Tab1 → Tab2 联动：待注入流水线的交互对 */
  pendingPair: Pair | null;
  pendingAutoRun: boolean;
  setTab: (tab: TabId) => void;
  sendToPipeline: (pair: Pair, autoRun?: boolean) => void;
  consumePendingPair: () => { pair: Pair; autoRun: boolean } | null;
}

export const useAppStore = create<AppState>((set, get) => ({
  tab: "dataset",
  pendingPair: null,
  pendingAutoRun: false,
  setTab: (tab) => set({ tab }),
  sendToPipeline: (pair, autoRun = true) =>
    set({ pendingPair: pair, pendingAutoRun: autoRun, tab: "pipeline" }),
  consumePendingPair: () => {
    const { pendingPair, pendingAutoRun } = get();
    if (pendingPair) set({ pendingPair: null, pendingAutoRun: false });
    return pendingPair ? { pair: pendingPair, autoRun: pendingAutoRun } : null;
  },
}));
