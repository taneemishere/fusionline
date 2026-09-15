#!/usr/bin/env tsx
/** Run the tool-selection benchmark (TypeScript harness).
 *
 *  Example (from ts-harness/):
 *    npx tsx src/run_eval.ts --strategies all,random,bm25 --registry full --ks 5,10 --limit 30
 */

import { loadTasksJsonl, loadToolsJsonl } from "./data.js";
import { runEval, summarize } from "./eval.js";
import { makeStrategy } from "./strategies.js";
import { join } from "node:path";
import type { Task } from "./types.js";

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const flag = (name: string) => process.argv.includes(`--${name}`);

/** Deterministic per-source split: even-index tasks -> train, odd -> test. */
function splitTasks(tasks: Task[]): { train: Task[]; test: Task[] } {
  const idx = new Map<string, number>();
  const train: Task[] = [];
  const test: Task[] = [];
  for (const t of tasks) {
    const i = idx.get(t.source) ?? 0;
    idx.set(t.source, i + 1);
    (i % 2 === 0 ? train : test).push(t);
  }
  return { train, test };
}

async function main() {
  const dataDir = arg("data-dir", "../data/toolret")!;
  const strategiesArg = arg("strategies", "all,random,bm25")!;
  const registryMode = arg("registry", "full")!;
  const ks = arg("ks", "5,10")!.split(",").map(Number);
  const configs = arg("configs")?.split(",");
  const limit = arg("limit") ? Number(arg("limit")) : undefined;
  const instruction = flag("instruction");
  const category = arg("category");
  const totalLimit = arg("total-limit") ? Number(arg("total-limit")) : undefined;
  const out = arg("out", "../results/eval-ts.csv")!;
  const embedModel = arg("embed-model", "all-MiniLM-L6-v2")!;
  const split = arg("split", "all")!;

  console.log(`Loading tools from ${dataDir}/tools.jsonl ...`);
  const tools = loadToolsJsonl(join(dataDir, "tools.jsonl"));
  console.log(`  ${tools.size} tools`);

  let allTasks = loadTasksJsonl(join(dataDir, "tasks.jsonl"), {
    configs,
    limitPerConfig: limit,
  });
  if (category) allTasks = allTasks.filter((t) => t.category === category);
  if (totalLimit !== undefined) allTasks = allTasks.slice(0, totalLimit);
  const { train, test } = splitTasks(allTasks);
  const tasks = split === "train" ? train : split === "test" ? test : allTasks;
  console.log(`  ${tasks.length} tasks from ${new Set(tasks.map((t) => t.source)).size} configs (split=${split})`);

  const embDir = join(dataDir, "embeddings", embedModel);
  const strategies = [];
  for (const n of strategiesArg.split(",")) {
    strategies.push(await makeStrategy(n.trim(), { instruction, embDir, trainTasks: train }));
  }

  const rows = await runEval(tasks, tools, strategies, registryMode, ks, out);
  console.log(`\nWrote ${rows.length} rows -> ${out}\n`);
  for (const s of strategies) {
    const u = (s as any).usage;
    const spent = (s as any).spentUsd;
    if (u) console.log(`[${s.name}] api calls=${u.calls} in=${u.input_tokens} ` +
      `out=${u.output_tokens}` + (spent !== undefined ? ` spent=$${spent.toFixed(4)}` : ""));
  }
  summarize(rows);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
