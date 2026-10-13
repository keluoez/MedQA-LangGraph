/** 与后端 api/ 层契约对应的 TypeScript 类型定义。 */

// ── 数据集（/api/dataset） ──────────────────────────────────────────────

export type Tier = "gold" | "silver" | "bronze";
export type DomainId = "drug" | "experimental" | "pathological" | "physiological";

export interface Subscores {
  factuality_base?: number;
  factuality_llm_raw?: number;
  factuality_effective?: number;
  grounding_llm_raw?: number;
  format_llm_raw?: number;
  consistency_llm_raw?: number;
  [k: string]: number | undefined;
}

export interface SampleSummary {
  id: string;
  domain: string;
  domain_label: string;
  question: string;
  drug: string;
  protein: string;
  uniprot_id: string;
  quality_score: number;
  tier: Tier;
  correction_count: number;
  last_route: string;
  subscores: Subscores;
  errors: string[];
  critical_errors: string[];
  elapsed_seconds: number;
}

export interface AnswerSection {
  title: string;
  text: string;
}

export interface SampleDetail extends SampleSummary {
  answer: string;
  sections: AnswerSection[];
}

export interface SampleListResponse {
  total: number;
  page: number;
  page_size: number;
  items: SampleSummary[];
}

export interface DomainStats {
  total: number;
  mean_score: number;
  gold: number;
  silver: number;
  bronze: number;
  gold_pct: number;
  silver_pct: number;
  bronze_pct: number;
  label?: string;
}

export interface StatsResponse {
  overall: DomainStats;
  by_domain: Record<string, DomainStats & { label: string }>;
}

// ── 流水线（/api/qa） ─────────────────────────────────────────────────

export interface Pair {
  drug: string;
  protein: string;
  uniprot_id?: string;
  gene?: string;
  chembl_id?: string;
  drugbank_id?: string;
}

export interface Scenario {
  scenario_id: string;
  description: string;
  pair: Pair;
  iterations: number;
  duration_ms: number;
  source: string;
}

export interface PairsResponse {
  total: number;
  items: Pair[];
}

export type RunMode = "mock" | "live";

export interface RunRequest {
  mode: RunMode;
  scenario_id?: string;
  pair?: Pair;
  speed?: number;
  max_corrections?: number;
}

// ── SSE 事件协议（与 api/adapters/stream_adapter.py 一致） ─────────────

export type NodeType = "collect" | "generate" | "verify" | "refine" | "regenerate";
export type RouteTo = "pass" | "refine" | "regenerate" | "stop";

export interface EvidenceArticle {
  pmid: string;
  title: string;
  journal: string;
  year: string;
  abstract: string;
}

export interface EvidencePackage {
  uniprot: Record<string, unknown>;
  chembl: { pref_name: string };
  chembl_mechanisms: {
    action_type: string;
    target_name: string;
    mechanism_of_action: string;
    reference: string;
  }[];
  pubmed_pmids: string[];
  pubmed_titles: Record<string, string>;
  pubmed_articles: EvidenceArticle[];
  drugbank_id: string;
}

export interface MetaEvent {
  type: "meta";
  run_id: string;
  mode: RunMode | string;
  pair: Pair;
  question?: string;
  model?: string;
  ts: number;
  scenario_id?: string;
}

export interface CollectOutputEvent {
  type: "node_output";
  node: "collect";
  iteration: number;
  ts: number;
  payload: { question: string; evidence: EvidencePackage };
}

export interface GenerateOutputEvent {
  type: "node_output";
  node: "generate" | "refine" | "regenerate";
  iteration: number;
  ts: number;
  payload: { answer: string; correction_count?: number };
}

export interface VerifyOutputEvent {
  type: "node_output";
  node: "verify";
  iteration: number;
  ts: number;
  payload: {
    score: number;
    subscores: Subscores;
    errors: string[];
    critical_errors: string[];
    revision_advice: string[];
  };
}

export type NodeOutputEvent =
  | CollectOutputEvent
  | GenerateOutputEvent
  | VerifyOutputEvent;

export interface RouteEvent {
  type: "route";
  from: string;
  to: RouteTo;
  score: number;
  reason: string;
  iteration: number;
  ts: number;
}

export interface FinalEvent {
  type: "final";
  score: number;
  iterations: number;
  elapsed: number;
  sample: {
    question: string;
    answer: string;
    drug?: string;
    protein?: string;
    uniprot_id?: string;
    quality_score: number;
    sections?: AnswerSection[];
    tier?: Tier;
  };
  ts: number;
}

export interface ErrorEvent {
  type: "error";
  message: string;
  ts: number;
}

export type PipelineEvent =
  | MetaEvent
  | NodeOutputEvent
  | RouteEvent
  | FinalEvent
  | ErrorEvent;

// ── 机制自省（/api/pipeline/introspect） ────────────────────────────────

export interface MechanismAgent {
  id: string;
  role: string;
  desc: string;
}

export interface MechanismTool {
  name: string;
  endpoint: string;
  note: string;
}

export interface RetrievalStep {
  title: string;
  detail: string;
}

export interface RankWeight {
  pattern: string;
  weight: number;
}

export interface RetrievalSpec {
  steps: RetrievalStep[];
  rank_weights: RankWeight[];
  whitelist_size: number;
  detail_articles: number;
}

export interface ContextBlock {
  block: string;
  source: string;
}

export interface ScoreDimension {
  key: string;
  label: string;
  range: string;
  source: string;
}

export interface RoutingRule {
  route: string;
  condition: string;
}

export interface ScoringSpec {
  dimensions: ScoreDimension[];
  det_checks: string[];
  formula: string;
  routing: RoutingRule[];
  max_score: number;
}

export interface DomainMechanism {
  label: string;
  model: string;
  question_template: string;
  sections: string[];
  tools: MechanismTool[];
  retrieval: RetrievalSpec;
  context_blocks: ContextBlock[];
  verifier_blocks: ContextBlock[];
  scoring: ScoringSpec;
  prompts: { responder: string; verifier: string; refiner: string };
  context_preview: string;
}

export interface IntrospectResponse {
  agents: MechanismAgent[];
  domains: Record<string, DomainMechanism>;
}
