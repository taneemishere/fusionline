/** Anthropic Tool Search Tool as a strategy under test.
 *
 *  Sends the task query to the Messages API with all registry tools marked
 *  `defer_loading: true` plus a server-side tool_search tool; collects the
 *  tool names the API returns in `tool_search_tool_result` blocks, in
 *  discovery order, as the predicted ranking.
 *
 *  IMPORTANT: Anthropic's tool search is lexical (BM25 or regex variants),
 *  not embedding-based — the apples-to-apples comparison is our `bm25`
 *  strategy, and the interesting question is their multi-turn agentic
 *  search loop vs. our single-shot retrieval.
 *
 *  Budget: ANTHROPIC_BUDGET_USD env var (default $2) hard-caps estimated
 *  spend; the run aborts with BudgetExceededError when exceeded.
 *  Haiku 4.5 is the default model — the retrieval is server-side, so the
 *  model only formulates search queries; cheap model isolates retriever
 *  quality from model capability.
 */

import type { RegistryView, Strategy, Task } from "./types.js";

const API_URL = "https://api.anthropic.com/v1/messages";
export const SEARCH_TOOL = { type: "tool_search_tool_bm25_20251119", name: "tool_search_tool_bm25" };
const NAME_RE = /^[a-zA-Z0-9_-]{1,64}$/;

// Claude Haiku 4.5 rates ($/MTok).
const PRICE_IN = 1.0;
const PRICE_OUT = 5.0;

export class BudgetExceededError extends Error {
  override name = "BudgetExceededError";
}

function safeName(raw: string, fallback: string, taken: Set<string>): string {
  let n = raw.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  if (!NAME_RE.test(n)) n = fallback.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  while (taken.has(n)) n = (n.slice(0, 60) + "_" + taken.size).slice(0, 64);
  taken.add(n);
  return n;
}

export function toApiTool(t: { toolId: string; name: string; description: string; documentation: string },
                   taken: Set<string>, nameToId: Map<string, string>) {
  const name = safeName(t.name, t.toolId, taken);
  nameToId.set(name, t.toolId);
  let inputSchema: unknown = { type: "object", properties: {} };
  try {
    const doc = JSON.parse(t.documentation);
    for (const k of ["input_schema", "parameters", "doc_arguments", "arguments"]) {
      if (doc[k] && typeof doc[k] === "object" && doc[k].type === "object") {
        inputSchema = doc[k];
        break;
      }
    }
  } catch { /* keep default */ }
  return { name, description: (t.description || t.name || t.toolId).slice(0, 1024),
           input_schema: inputSchema, defer_loading: true };
}

function discoveredIds(resp: any, nameToId: Map<string, string>, out: string[]): void {
  for (const block of resp?.content ?? []) {
    const refs = block?.tool_references ?? block?.content?.tool_references ?? [];
    for (const r of refs) {
      const id = nameToId.get(r?.tool_name ?? r?.name);
      if (id && !out.includes(id)) out.push(id);
    }
    const matched = block?.matched_tools ?? [];
    for (const r of matched) {
      const id = nameToId.get(r?.tool_name ?? r?.name);
      if (id && !out.includes(id)) out.push(id);
    }
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class AnthropicToolSearch implements Strategy {
  name = "anthropic";
  readonly model: string;
  private apiKey: string;
  private maxTurns = 4;
  private budgetUsd: number;
  usage = { input_tokens: 0, output_tokens: 0, calls: 0 };

  constructor(opts: { model?: string; budgetUsd?: number } = {}) {
    this.apiKey = process.env.ANTHROPIC_API_KEY ?? "";
    if (!this.apiKey) throw new Error("anthropic strategy requires ANTHROPIC_API_KEY env var");
    this.model = opts.model ?? "claude-haiku-4-5";
    this.budgetUsd = opts.budgetUsd
      ?? Number(process.env.ANTHROPIC_BUDGET_USD ?? "2.0");
  }

  get spentUsd(): number {
    return (this.usage.input_tokens * PRICE_IN + this.usage.output_tokens * PRICE_OUT) / 1e6;
  }

  private async call(body: unknown): Promise<any> {
    for (let attempt = 0; attempt < 5; attempt++) {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": this.apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify(body),
      });
      if (res.ok) return await res.json();
      const text = (await res.text()).slice(0, 300);
      if (res.status === 429 || res.status >= 500) {
        await sleep(2000 * 2 ** attempt);
        continue;
      }
      throw new Error(`Anthropic API ${res.status}: ${text}`);
    }
    throw new Error("Anthropic API: retries exhausted (429/5xx)");
  }

  async select(task: Task, registry: RegistryView, k: number): Promise<string[]> {
    if (this.spentUsd >= this.budgetUsd) {
      throw new BudgetExceededError(
        `budget cap $${this.budgetUsd} reached (spent $${this.spentUsd.toFixed(3)})`);
    }
    const taken = new Set<string>();
    const nameToId = new Map<string, string>();
    const apiTools = [SEARCH_TOOL,
      ...[...registry.tools.values()].map((t) => toApiTool(t, taken, nameToId))];

    const messages: unknown[] = [{ role: "user", content: task.query }];
    const found: string[] = [];

    for (let turn = 0; turn < this.maxTurns; turn++) {
      const data = await this.call({
        model: this.model,
        max_tokens: 1024,
        messages,
        tools: apiTools,
      });
      this.usage.calls++;
      this.usage.input_tokens += data.usage?.input_tokens ?? 0;
      this.usage.output_tokens += data.usage?.output_tokens ?? 0;
      discoveredIds(data, nameToId, found);
      messages.push({ role: "assistant", content: data.content });
      if (this.spentUsd >= this.budgetUsd) {
        throw new BudgetExceededError(
          `budget cap $${this.budgetUsd} reached mid-task (spent $${this.spentUsd.toFixed(3)})`);
      }
      if (data.stop_reason === "pause_turn") continue;
      if (data.stop_reason === "tool_use") {
        // Claude called a discovered client tool — reply with stub results so
        // the multi-turn search loop can continue (we only measure discovery).
        const stubs = (data.content ?? [])
          .filter((b: any) => b?.type === "tool_use")
          .map((b: any) => ({ type: "tool_result", tool_use_id: b.id, content: "ok" }));
        if (!stubs.length) break;
        messages.push({ role: "user", content: stubs });
        continue;
      }
      break;
    }
    return found.slice(0, k);
  }
}
