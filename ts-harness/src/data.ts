import { readFileSync } from "node:fs";
import type { Task, ToolRecord } from "./types.js";

export function loadToolsJsonl(path: string): Map<string, ToolRecord> {
  const tools = new Map<string, ToolRecord>();
  for (const line of readFileSync(path, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    tools.set(r.tool_id, {
      toolId: r.tool_id,
      name: r.name,
      description: r.description,
      searchText: r.search_text,
      documentation: r.documentation,
      category: r.category,
    });
  }
  return tools;
}

export function loadTasksJsonl(
  path: string,
  opts: { configs?: string[]; limitPerConfig?: number } = {},
): Task[] {
  const tasks: Task[] = [];
  const seen = new Map<string, number>();
  for (const line of readFileSync(path, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    if (opts.configs && !opts.configs.includes(r.source)) continue;
    const n = seen.get(r.source) ?? 0;
    if (opts.limitPerConfig !== undefined && n >= opts.limitPerConfig) continue;
    seen.set(r.source, n + 1);
    tasks.push({
      taskId: r.task_id,
      source: r.source,
      query: r.query,
      instruction: r.instruction ?? "",
      gtIds: r.gt_ids,
      category: r.category,
    });
  }
  return tasks;
}
