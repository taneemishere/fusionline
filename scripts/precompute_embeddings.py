#!/usr/bin/env python3
"""Precompute sentence-transformer embeddings for the TS `embed` strategy.

Writes float32 matrices + aligned id lists:
    data/toolret/embeddings/<model_tag>/tools.f32
    data/toolret/embeddings/<model_tag>/tools.ids.json
    data/toolret/embeddings/<model_tag>/tasks.f32        (query only, w/o inst)
    data/toolret/embeddings/<model_tag>/tasks_inst.f32   (query + instruction)
    data/toolret/embeddings/<model_tag>/tasks.ids.json

Run from the repo root (GPU recommended for the dense models; uncomment
sentence-transformers in requirements.txt first):
    PYTHONPATH=. python3 scripts/precompute_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolsel.data import load_tasks, load_tools


def encode(model, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts, batch_size=256, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    tag = args.model.rsplit("/", 1)[-1]
    out_dir = Path("data/toolret/embeddings") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tools = load_tools()
    tasks = load_tasks()
    tool_ids = list(tools)
    task_ids = [t.task_id for t in tasks]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    print(f"Encoding {len(tool_ids)} tools with {args.model} ...")
    tools_emb = encode(model, [tools[t].search_text for t in tool_ids])
    tools_emb.tofile(out_dir / "tools.f32")

    print(f"Encoding {len(task_ids)} tasks (w/o inst) ...")
    tasks_emb = encode(model, [t.query for t in tasks])
    tasks_emb.tofile(out_dir / "tasks.f32")

    print(f"Encoding {len(task_ids)} tasks (w/ inst) ...")
    tasks_inst_emb = encode(model, [f"{t.query}\n{t.instruction}".strip() for t in tasks])
    tasks_inst_emb.tofile(out_dir / "tasks_inst.f32")

    (out_dir / "tools.ids.json").write_text(json.dumps(tool_ids))
    (out_dir / "tasks.ids.json").write_text(json.dumps(task_ids))
    (out_dir / "meta.json").write_text(json.dumps({
        "model": args.model,
        "dim": int(tools_emb.shape[1]),
        "n_tools": len(tool_ids),
        "n_tasks": len(task_ids),
        "normalized": True,
    }, indent=2))
    print(f"done -> {out_dir} (dim={tools_emb.shape[1]})")


if __name__ == "__main__":
    main()
