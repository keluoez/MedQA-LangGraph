import { useCallback, useRef, useState } from "react";
import { streamSSE } from "../../api/sse";
import type {
  CollectOutputEvent,
  ErrorEvent,
  EvidencePackage,
  FinalEvent,
  GenerateOutputEvent,
  MetaEvent,
  PipelineEvent,
  RouteEvent,
  RunRequest,
  VerifyOutputEvent,
} from "../../api/types";

export type NodeStatus = "idle" | "running" | "done";

export interface Audit {
  iteration: number;
  score: number;
  subscores: VerifyOutputEvent["payload"];
}

export interface RevisionRound {
  iteration: number;
  answer: string;
}

export interface RunState {
  phase: "idle" | "running" | "done" | "error";
  meta: MetaEvent | null;
  question: string;
  collector: { status: NodeStatus; evidence: EvidencePackage | null };
  responder: { status: NodeStatus; rounds: RevisionRound[] };
  verifier: { status: NodeStatus; audits: Audit[] };
  refiner: { status: NodeStatus; rounds: RevisionRound[] };
  regenerator: { status: NodeStatus; rounds: RevisionRound[] };
  lastRoute: RouteEvent | null;
  final: FinalEvent | null;
  error: ErrorEvent | null;
}

const EMPTY: RunState = {
  phase: "idle",
  meta: null,
  question: "",
  collector: { status: "idle", evidence: null },
  responder: { status: "idle", rounds: [] },
  verifier: { status: "idle", audits: [] },
  refiner: { status: "idle", rounds: [] },
  regenerator: { status: "idle", rounds: [] },
  lastRoute: null,
  final: null,
  error: null,
};

/** 把 SSE 事件流归约为流水线 UI 状态（增量 reducer）。 */
function applyEvent(state: RunState, ev: PipelineEvent): RunState {
  switch (ev.type) {
    case "meta":
      return {
        ...state,
        phase: "running",
        meta: ev,
        question: ev.question ?? state.question,
        collector: { status: "running", evidence: null },
      };
    case "node_output": {
      if (ev.node === "collect") {
        const c = ev as CollectOutputEvent;
        return {
          ...state,
          question: c.payload.question || state.question,
          collector: { status: "done", evidence: c.payload.evidence },
          responder: { ...state.responder, status: "running" },
        };
      }
      if (ev.node === "generate") {
        const g = ev as GenerateOutputEvent;
        return {
          ...state,
          responder: {
            status: "done",
            rounds: [...state.responder.rounds, { iteration: g.iteration, answer: g.payload.answer }],
          },
          verifier: { ...state.verifier, status: "running" },
        };
      }
      if (ev.node === "verify") {
        const v = ev as VerifyOutputEvent;
        return {
          ...state,
          verifier: {
            status: "done",
            audits: [
              ...state.verifier.audits,
              {
                iteration: v.iteration,
                score: v.payload.score,
                subscores: v.payload,
              },
            ],
          },
        };
      }
      // refine / regenerate
      const r = ev as GenerateOutputEvent;
      const target =
        ev.node === "refine" ? state.refiner : state.regenerator;
      return {
        ...state,
        [ev.node]: {
          status: "done" as NodeStatus,
          rounds: [...target.rounds, { iteration: r.iteration, answer: r.payload.answer }],
        },
        // 修正后回流 Verifier 再审计
        verifier: { ...state.verifier, status: "running" },
      };
    }
    case "route": {
      const route = ev as RouteEvent;
      const next: RunState = { ...state, lastRoute: route };
      if (route.to === "refine") next.refiner = { ...next.refiner, status: "running" };
      if (route.to === "regenerate")
        next.regenerator = { ...next.regenerator, status: "running" };
      return next;
    }
    case "final": {
      const f = ev as FinalEvent;
      return { ...state, phase: "done", final: f };
    }
    case "error": {
      const e = ev as ErrorEvent;
      return { ...state, phase: "error", error: e };
    }
    default:
      return state;
  }
}

export function usePipelineRun() {
  const [state, setState] = useState<RunState>(EMPTY);
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(async (req: RunRequest) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setState({ ...EMPTY, phase: "running" });
    setEvents([]);
    try {
      await streamSSE(
        "/api/qa/run",
        req,
        (ev) => {
          setEvents((prev) => [...prev, ev]);
          setState((prev) => applyEvent(prev, ev));
        },
        ac.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setState((prev) =>
          prev.phase === "idle"
            ? prev
            : {
                ...prev,
                phase: "error",
                error: { type: "error", message: String(err), ts: 0 },
              },
        );
      }
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState((prev) => ({ ...prev, phase: "done" }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(EMPTY);
    setEvents([]);
  }, []);

  return { state, events, start, stop, reset };
}
