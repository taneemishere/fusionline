import { createWriteStream, mkdirSync, type WriteStream } from "node:fs";
import { dirname } from "node:path";
import { getEncoding } from "js-tiktoken";
import { scoreAll } from "./metrics.js";
import { RegistryManager } from "./registry.js";
import type { RegistryView, Strategy, Task, ToolRecord } from "./types.js";

const enc = getEncoding("cl100k_base");
const estTokens = (s: string) => enc.encode(s).length;

export interface EvalRow {
  strategy: string;
  model: string;
  registry_mode: string;
  registry_size: number;
  k: number;
  task_id: string;
  source: string;
  category: string;
  n_gt: number;
  exposed_tools: number;
  est_tokens: number;
  latency_ms: number;
  recall: number;
  hit: number;
  precision: number;
  f1: number;
  ndcg: number;
  mrr: number;
  error: string;
}

export async function runEval(
  tasks: Task[],
  allTools: Map<string, ToolRecord>,
  strategies: Strategy[],
  registryMode: string,
  ks: number[],
  outCsv?: string,
): Promise<EvalRow[]> {
  const mgr = new RegistryManager(allTools, registryMode);
  const tokCache = new Map<string, number>();
  for (const [id, t] of allTools) tokCache.set(id, estTokens(t.documentation));
  const maxK = Math.max(...ks);

  let out: WriteStream | undefined;
  let cols: (keyof EvalRow)[] | undefined;
  if (outCsv) {
    mkdirSync(dirname(outCsv), { recursive: true });
    out = createWriteStream(outCsv);
    cols = ["strategy", "model", "registry_mode", "registry_size", "k", "task_id",
            "source", "category", "n_gt", "exposed_tools", "est_tokens",
            "latency_ms", "recall", "hit", "precision", "f1", "ndcg", "mrr", "error"];
    out.write(cols.join(",") + "\n");
  }

  const rows: EvalRow[] = [];
  let consecErrors = 0;
  for (const strat of strategies) {
    for (const task of tasks) {
      const reg: RegistryView = mgr.forTask(task);
      const isAll = strat.name === "all";
      // One select() at max k per (strategy, task); smaller k reuse the prefix.
      const t0 = performance.now();
      let preds: string[] = [];
      let err = "";
      try {
        preds = await strat.select(task, reg, isAll ? reg.tools.size : maxK);
        consecErrors = 0;
      } catch (e) {
        err = String((e as Error).message ?? e).replace(/[\n,]/g, " ").slice(0, 200);
        consecErrors++;
        if ((e as Error).name === "BudgetExceededError" || consecErrors >= 5) {
          console.error(`aborting '${strat.name}': ${err}`);
          if (out) await new Promise((r) => out!.end(r));
          return rows;
        }
        console.error(`select() error on ${task.taskId}: ${err}`);
      }
      const latencyMs = performance.now() - t0;
      for (const k of isAll ? [reg.tools.size] : ks) {
        const m = scoreAll(task.gtIds, preds, k);
        let est = 0;
        for (const p of preds.slice(0, k)) est += tokCache.get(p) ?? 0;
        const row: EvalRow = {
          strategy: strat.name,
          model: strat.model ?? "",
          registry_mode: registryMode,
          registry_size: reg.tools.size,
          k,
          task_id: task.taskId,
          source: task.source,
          category: task.category,
          n_gt: task.gtIds.length,
          exposed_tools: Math.min(preds.length, k),
          est_tokens: est,
          latency_ms: Math.round(latencyMs * 1000) / 1000,
          ...m,
          error: err,
        };
        rows.push(row);
        if (out && cols) {
          out.write(cols.map((c) => JSON.stringify(row[c] ?? "")).join(",") + "\n");
        }
      }
    }
  }

  if (out) await new Promise((r) => out!.end(r));
  return rows;
}

export function summarize(rows: EvalRow[]): void {
  const agg = new Map<string, EvalRow[]>();
  for (const r of rows) {
    const key = `${r.strategy}|${r.registry_mode}|${r.k}`;
    let g = agg.get(key);
    if (!g) agg.set(key, (g = []));
    g.push(r);
  }
  const mean = (rs: EvalRow[], f: keyof EvalRow) =>
    rs.reduce((a, r) => a + (r[f] as number), 0) / rs.length;

  const header =
    `${"strategy".padEnd(10)} ${"mode".padEnd(10)} ${"k".padStart(5)} ${"n".padStart(5)} ` +
    `${"recall".padStart(7)} ${"hit".padStart(6)} ${"f1".padStart(6)} ${"ndcg".padStart(6)} ` +
    `${"mrr".padStart(6)} ${"tokens".padStart(9)} ${"ms".padStart(7)}`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const key of [...agg.keys()].sort()) {
    const rs = agg.get(key)!;
    const [s, mode, k] = key.split("|");
    console.log(
      `${s.padEnd(10)} ${mode.padEnd(10)} ${k.padStart(5)} ${String(rs.length).padStart(5)} ` +
        `${mean(rs, "recall").toFixed(3).padStart(7)} ${mean(rs, "hit").toFixed(3).padStart(6)} ` +
        `${mean(rs, "f1").toFixed(3).padStart(6)} ${mean(rs, "ndcg").toFixed(3).padStart(6)} ` +
        `${mean(rs, "mrr").toFixed(3).padStart(6)} ${mean(rs, "est_tokens").toFixed(0).padStart(9)} ` +
        `${mean(rs, "latency_ms").toFixed(1).padStart(7)}`,
    );
  }
}
