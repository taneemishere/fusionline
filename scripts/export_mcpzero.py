#!/usr/bin/env python3
"""Export MCP-Zero's MCP-tools corpus for the TypeScript harness.

Source: data/mcpzero/mcp_tools_with_embedding.json
  (308 servers, ~2.8K tools, bundled text-embedding-3-large vectors)

Outputs:
    data/mcpzero/tools.jsonl  - {tool_id, name, description, search_text,
                                 documentation, category}
    data/mcpzero/tasks.jsonl  - one task per server: query = server summary,
                                gt_ids = that server's tool ids
    data/mcpzero/embeddings/text-embedding-3-large/  (bundled vectors)
    data/mcpzero/embeddings/<model_tag>/             (with --minilm etc.)

Task design: server summary -> that server's tools (the capability-routing
problem MCP-Zero's own hierarchical routing addresses). gt sets are
multi-tool by construction (median 5 tools/server).

Run from the repo root:
    python3 scripts/export_mcpzero.py
    python3 scripts/export_mcpzero.py --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path("data/mcpzero")
SRC = DATA / "mcp_tools_with_embedding.json"


def build() -> tuple[list[dict], list[dict], list[dict]]:
    d = json.load(open(SRC))
    tools_rows: list[dict] = []
    tasks_rows: list[dict] = []
    servers: list[dict] = d
    for i, s in enumerate(servers):
        srv = f"srv{i}"
        sname = s.get("name") or f"server{i}"
        ids: list[str] = []
        for t in s.get("tools", []):
            tid = f"{srv}/{t['name']}"
            params = json.dumps(t.get("parameter") or {})
            tools_rows.append({
                "tool_id": tid,
                "name": f"{sname}/{t['name']}",
                "description": t.get("description") or "",
                "search_text": f"{sname} {t['name']}\n{t.get('description') or ''}\n{params}".strip(),
                "documentation": json.dumps({
                    "server": sname, "name": t["name"],
                    "description": t.get("description") or "",
                    "parameter": t.get("parameter") or {},
                }),
                "category": "mcp",
                "source": srv,
            })
            ids.append(tid)
        if not ids:
            continue
        tasks_rows.append({
            "task_id": f"mcpzero_{srv}",
            "source": srv,
            "query": (s.get("summary") or s.get("description") or "").strip(),
            "instruction": "",
            "gt_ids": ids,
            "category": "mcp",
            "_server_name": sname,
        })
    return tools_rows, tasks_rows, servers


def norm_rows(vecs: list[list[float]]) -> np.ndarray:
    m = np.asarray(vecs, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


def write_openai_embeddings(tools_rows, tasks_rows, servers) -> None:
    out = DATA / "embeddings" / "text-embedding-3-large"
    out.mkdir(parents=True, exist_ok=True)
    tvec = norm_rows([t["description_embedding"]
                      for s in servers for t in s.get("tools", [])])
    qvec = norm_rows([s["summary_embedding"] for s in servers if s.get("tools")])
    assert tvec.shape[0] == len(tools_rows) and qvec.shape[0] == len(tasks_rows)
    tvec.tofile(out / "tools.f32")
    qvec.tofile(out / "tasks.f32")
    inst = out / "tasks_inst.f32"
    inst.unlink(missing_ok=True)
    inst.symlink_to("tasks.f32")  # corpus has no instructions; inst == tasks
    (out / "tools.ids.json").write_text(json.dumps([r["tool_id"] for r in tools_rows]))
    (out / "tasks.ids.json").write_text(json.dumps([r["task_id"] for r in tasks_rows]))
    (out / "meta.json").write_text(json.dumps({
        "model": "text-embedding-3-large (bundled with MCP-tools)",
        "dim": int(tvec.shape[1]), "n_tools": len(tools_rows),
        "n_tasks": len(tasks_rows), "normalized": True}, indent=2))
    print(f"openai embeddings -> {out} (dim={tvec.shape[1]})")


def encode_st(model_name: str, tools_rows, tasks_rows) -> None:
    from sentence_transformers import SentenceTransformer
    tag = model_name.rsplit("/", 1)[-1]
    out = DATA / "embeddings" / tag
    out.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    enc = lambda xs: model.encode(xs, batch_size=256, convert_to_numpy=True,
                                  normalize_embeddings=True).astype(np.float32)
    tv = enc([r["search_text"] for r in tools_rows])
    qv = enc([r["query"] for r in tasks_rows])
    tv.tofile(out / "tools.f32")
    qv.tofile(out / "tasks.f32")
    inst = out / "tasks_inst.f32"
    inst.unlink(missing_ok=True)
    inst.symlink_to("tasks.f32")  # corpus has no instructions; inst == tasks
    (out / "tools.ids.json").write_text(json.dumps([r["tool_id"] for r in tools_rows]))
    (out / "tasks.ids.json").write_text(json.dumps([r["task_id"] for r in tasks_rows]))
    (out / "meta.json").write_text(json.dumps({
        "model": model_name, "dim": int(tv.shape[1]),
        "n_tools": len(tools_rows), "n_tasks": len(tasks_rows),
        "normalized": True}, indent=2))
    print(f"{tag} embeddings -> {out} (dim={tv.shape[1]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="also precompute sentence-transformers embeddings")
    args = ap.parse_args()

    tools_rows, tasks_rows, servers = build()
    with (DATA / "tools.jsonl").open("w") as f:
        for r in tools_rows:
            f.write(json.dumps(r) + "\n")
    with (DATA / "tasks.jsonl").open("w") as f:
        for r in tasks_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(tools_rows)} tools, {len(tasks_rows)} tasks")

    write_openai_embeddings(tools_rows, tasks_rows, servers)
    if args.model:
        encode_st(args.model, tools_rows, tasks_rows)


if __name__ == "__main__":
    main()
