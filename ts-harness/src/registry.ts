import type { RegistryView, Task, ToolRecord } from "./types.js";

/** Builds and caches registry views per task.
 *  Modes:
 *   - full:     every task sees the entire corpus
 *   - category: tools in the task's format category (web/customized/code)
 *   - source:   tools in the task's own source namespace (~tens of tools)
 */
export class RegistryManager {
  private cache = new Map<string, RegistryView>();

  constructor(
    private allTools: Map<string, ToolRecord>,
    private mode: string = "full",
  ) {}

  forTask(task: Task): RegistryView {
    let key: string;
    if (this.mode === "full") {
      key = "full";
    } else if (this.mode === "category") {
      key = `category:${task.category}`;
    } else if (this.mode === "source") {
      const ns = task.taskId.replace(/_query_\d+$/, "");
      key = `source:${ns}`;
    } else {
      throw new Error(`unknown registry mode: ${this.mode}`);
    }

    let view = this.cache.get(key);
    if (!view) {
      if (this.mode === "full") {
        view = { key, tools: this.allTools };
      } else if (this.mode === "category") {
        view = { key, tools: filter(this.allTools, (_id, t) => t.category === task.category) };
      } else {
        const prefix = key.slice("source:".length) + "_tool_";
        view = { key, tools: filter(this.allTools, (id) => id.startsWith(prefix)) };
      }
      this.cache.set(key, view);
    }
    return view;
  }
}

function filter(
  all: Map<string, ToolRecord>,
  pred: (id: string, t: ToolRecord) => boolean,
): Map<string, ToolRecord> {
  const out = new Map<string, ToolRecord>();
  for (const [id, t] of all) if (pred(id, t)) out.set(id, t);
  return out;
}
