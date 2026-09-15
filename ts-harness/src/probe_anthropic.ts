#!/usr/bin/env tsx
/** One-shot raw-response probe for the Anthropic tool-search calibration.
 *  Makes ONE call with the customized pool + one real task query, and dumps
 *  the fields that verify the mechanism actually engaged:
 *    response.model, stop_reason, usage (incl. server_tool_use),
 *    content block types, and discovered tool names.
 *  Cost: ~$0.003.
 */

import { join } from "node:path";
import { loadTasksJsonl, loadToolsJsonl } from "./data.js";
import { SEARCH_TOOL, toApiTool } from "./anthropic.js";

const dataDir = process.argv[2] ?? "../data/toolret";

const tools = [...loadToolsJsonl(join(dataDir, "tools.jsonl")).values()]
  .filter((t) => t.category === "customized");
const task = loadTasksJsonl(join(dataDir, "tasks.jsonl"))
  .find((t) => t.category === "customized")!;

const taken = new Set<string>();
const nameToId = new Map<string, string>();
const apiTools: unknown[] = [
  SEARCH_TOOL,
  ...tools.map((t) => toApiTool(t, taken, nameToId)),
];

const res = await fetch("https://api.anthropic.com/v1/messages", {
  method: "POST",
  headers: {
    "content-type": "application/json",
    "x-api-key": process.env.ANTHROPIC_API_KEY ?? "",
    "anthropic-version": "2023-06-01",
  },
  body: JSON.stringify({
    model: "claude-haiku-4-5",
    max_tokens: 1024,
    messages: [{ role: "user", content: task.query }],
    tools: apiTools,
  }),
});
const data: any = await res.json();

console.log("=== VERIFICATION ===");
console.log("http_status:", res.status);
console.log("response.model:", data.model);
console.log("stop_reason:", data.stop_reason);
console.log("usage:", JSON.stringify(data.usage, null, 2));
console.log("content block types:", (data.content ?? []).map((b: any) => b.type).join(", "));
for (const b of data.content ?? []) {
  if (b.type === "server_tool_use")
    console.log("server_tool_use:", b.name, "input:", JSON.stringify(b.input));
  if (b.type === "tool_search_tool_result") {
    const refs = b.content?.tool_references ?? [];
    console.log("tool_search_tool_result: refs ->",
      refs.map((r: any) => `${r.tool_name}=${nameToId.get(r.tool_name)}`));
  }
  if (b.type === "text")
    console.log("text:", String(b.text).slice(0, 200));
}
