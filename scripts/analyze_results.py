#!/usr/bin/env python3
"""Failure breakdown over the eval CSVs.

Run from the repo root:
    python3 scripts/analyze_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_all() -> pd.DataFrame:
    frames = []
    for f in sorted(RESULTS.glob("eval-*.csv")):
        df = pd.read_csv(f)
        parts = f.stem.split("-")
        df["matrix_mode"] = parts[1]
        df["matrix_inst"] = parts[2]
        df["matrix_model"] = parts[3] if len(parts) > 3 else "all-MiniLM-L6-v2"
        if "model" not in df.columns:
            df["model"] = df["matrix_model"].where(df.strategy.isin(["embed", "hybrid"]), "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_all()
    k10 = df[(df.k == 10)].copy()

    print("=" * 72)
    print("1. PER-SOURCE hit@10 (hybrid, w/ inst, ToolRet-BGE) — hardest & easiest")
    print("=" * 72)
    sel = k10[(k10.strategy == "hybrid") & (k10.matrix_inst == "wi")
              & (k10.matrix_model == "bge") & (k10.matrix_mode == "full")]
    g = sel.groupby("source").agg(hit=("hit", "mean"), n=("task_id", "count"),
                                  n_gt=("n_gt", "mean")).round(3).sort_values("hit")
    print(g.head(10).to_string())
    print("...")
    print(g.tail(5).to_string())

    print("\n" + "=" * 72)
    print("2. SINGLE vs MULTI-tool tasks (full mode, k=10, w/o inst, MiniLM)")
    print("=" * 72)
    sel = k10[(k10.matrix_mode == "full") & (k10.matrix_inst == "wo")
              & (k10.matrix_model == "all-MiniLM-L6-v2")]
    sel["gt_type"] = (sel.n_gt > 1).map({True: "multi", False: "single"})
    g = sel.groupby(["gt_type", "strategy"])[["recall", "hit", "ndcg"]].mean().round(3)
    print(g.to_string())
    print(f"\nsingle n={int((sel.gt_type=='single').sum()/sel.strategy.nunique())} "
          f"multi n={int((sel.gt_type=='multi').sum()/sel.strategy.nunique())}")

    print("\n" + "=" * 72)
    print("3. BY CATEGORY (full mode, k=10, w/ inst, ToolRet-BGE)")
    print("=" * 72)
    sel = k10[(k10.matrix_mode == "full") & (k10.matrix_inst == "wi")
              & (k10.matrix_model == "bge")]
    g = sel.groupby(["category", "strategy"])[["recall", "hit", "ndcg"]].mean().round(3)
    print(g.to_string())

    print("\n" + "=" * 72)
    print("4. STRATEGY AGREEMENT (full mode, k=10, w/o inst, MiniLM)")
    print("=" * 72)
    sel = k10[(k10.matrix_mode == "full") & (k10.matrix_inst == "wo")
              & (k10.matrix_model == "all-MiniLM-L6-v2")]
    piv = sel.pivot_table(index="task_id", columns="strategy", values="hit")
    for a, b in [("bm25", "embed"), ("bm25", "hybrid"), ("embed", "hybrid")]:
        both = ((piv[a] == 1) & (piv[b] == 1)).mean()
        a_only = ((piv[a] == 1) & (piv[b] == 0)).mean()
        b_only = ((piv[a] == 0) & (piv[b] == 1)).mean()
        neither = ((piv[a] == 0) & (piv[b] == 0)).mean()
        print(f"{a:>7} vs {b:<7}: both={both:.3f}  {a}-only={a_only:.3f}  "
              f"{b}-only={b_only:.3f}  neither={neither:.3f}")
    print("\ntasks where NOTHING hits (all strategies miss):")
    sel_any = piv[[c for c in piv.columns if c != "all"]]
    print(f"  {((sel_any == 0).all(axis=1)).mean():.3f}")

    print("\n" + "=" * 72)
    print("5. TOKENS-vs-RECALL tradeoff (full mode, w/ inst, ToolRet-BGE)")
    print("=" * 72)
    sel = df[(df.matrix_mode == "full") & (df.matrix_inst == "wi")
             & (df.matrix_model == "bge") & (df.k <= 15)]
    g = sel.groupby(["strategy", "k"])[["recall", "est_tokens"]].mean().round(3)
    print(g.to_string())


if __name__ == "__main__":
    main()
