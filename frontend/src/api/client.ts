import type {
  PairsResponse,
  Pair,
  RunRequest,
  SampleDetail,
  SampleListResponse,
  Scenario,
  StatsResponse,
  IntrospectResponse,
} from "./types";

async function get<T>(url: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== "") qs.set(k, String(v));
  }
  const q = qs.toString();
  const resp = await fetch(q ? `${url}?${q}` : url);
  if (!resp.ok) throw new Error(`GET ${url} -> ${resp.status}`);
  return resp.json() as Promise<T>;
}

export const api = {
  listSamples(params: {
    page?: number;
    page_size?: number;
    domain?: string;
    tier?: string;
    q?: string;
    sort?: string;
  }): Promise<SampleListResponse> {
    return get("/api/dataset/samples", params);
  },
  getSample(id: string): Promise<SampleDetail> {
    return get(`/api/dataset/samples/${id}`);
  },
  stats(): Promise<StatsResponse> {
    return get("/api/dataset/stats");
  },
  scenarios(): Promise<{ scenarios: Scenario[] }> {
    return get("/api/qa/scenarios");
  },
  pairs(params: { limit?: number; offset?: number; q?: string }): Promise<PairsResponse> {
    return get("/api/qa/pairs", params);
  },
  introspect(): Promise<IntrospectResponse> {
    return get("/api/pipeline/introspect");
  },
  runUrl: "/api/qa/run",
};

export type { Pair, RunRequest };
