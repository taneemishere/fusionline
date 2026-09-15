# FusionLine

A reproducible benchmark harness for MCP tool selection. It runs eight
selection strategies (dump-all, random, BM25, dense embedding, equal-weight
and weighted reciprocal-rank fusion, a set-aware co-occurrence baseline, and
Anthropic's managed Tool Search Tool) against shared registries (ToolRet's
44K-tool catalog, plus MCP-Zero as a transfer check) and reports retrieval
quality, prompt-token cost, and latency side by side.

Full writeup, figures, and findings:
**[taneemishere.github.io/fusionline](https://taneemishere.github.io/fusionline/)**

## Layout

| Path | What it is |
|---|---|
| `ts-harness/` | TypeScript eval harness (Node 18+, tsx): strategies, registry modes, metrics, CSV output |
| `toolsel/` + `scripts/` | Python data prep: dataset download, JSONL export, embedding precompute |
| `data/` | `toolret/` and `mcpzero/` exports (tools.jsonl, tasks.jsonl, embeddings/) |
| `results/` | Per-run CSVs |

## Datasets

**ToolRet** ([mangopy/ToolRet](https://huggingface.co/mangopy)): 44,453 tools
and 7,961 labeled queries across 34 source configurations, in three format
categories (`code`, `customized`, `web`). `scripts/download_toolret.py`
pulls it from HuggingFace:

```bash
pip install -r requirements.txt
python scripts/download_toolret.py   # -> data/toolret/{tools,queries}/<config>/
```

**MCP-Zero** ([MCP-Zero](https://arxiv.org/abs/2506.01056)): 2,797 tools from
308 real MCP servers, bundled with OpenAI text-embedding-3-large vectors.
Download `mcp_tools_with_embedding.json` from the link in the MCP-Zero
repository and place it at `data/mcpzero/mcp_tools_with_embedding.json`.

## Run it

The Python side turns the datasets into the uniform JSONL the harness reads
(`tools.jsonl`: id, name, description, search_text, documentation, category;
`tasks.jsonl`: id, query, instruction, gt_ids), then precomputes the
embeddings the dense and fusion strategies score against.

```bash
# 1. Export to JSONL (run from the repo root)
PYTHONPATH=. python scripts/export_toolret_jsonl.py   # -> data/toolret/{tools,tasks}.jsonl
python scripts/export_mcpzero.py                      # -> data/mcpzero/{tools,tasks}.jsonl
                                                      #    + bundled OpenAI embeddings

# 2. Precompute embeddings (needs: pip install sentence-transformers)
PYTHONPATH=. python scripts/precompute_embeddings.py \
    --model sentence-transformers/all-MiniLM-L6-v2    # -> data/toolret/embeddings/<tag>/
python scripts/export_mcpzero.py --model all-MiniLM-L6-v2  # same for MCP-Zero

# 3. Eval (TypeScript)
cd ts-harness && npm install
npx tsx src/run_eval.ts \
  --strategies all,random,bm25,embed,hybrid \
  --registry full --ks 5,10 --out ../results/run.csv
```

Strategies: `all`, `random`, `bm25`, `embed`, `hybrid` (equal-weight RRF),
`hybridw` (weighted RRF), `setaware` (co-occurrence baseline), `anthropic`.

Useful flags: `--registry full|category|source` (pool size the strategy sees),
`--instruction` (append task instructions), `--category customized`,
`--total-limit N`, `--configs a,b` (subset by source), `--embed-model <tag>`,
`--split train|test` (held-out split for `setaware`), `--data-dir <dir>`
(point at `../data/mcpzero` for the transfer corpus).

`scripts/analyze_results.py` groups the CSVs in `results/` into per-source,
per-category, and strategy-agreement breakdowns.

## Anthropic strategy (optional, paid)

Why it's here: the vendor baseline. Anthropic's Tool Search Tool is the
managed answer to this exact problem. Tool schemas are sent with
`defer_loading: true` so they never enter the prompt, and a server-side BM25
retriever surfaces candidates only when the model asks. FusionLine treats it
as just another strategy under test: every registry tool is mapped to an API
tool definition, the task query goes to the Messages API, and the tools
returned in `tool_search_tool_result` blocks, in discovery order over up to
four turns, become the predicted ranking. Same tasks, same metrics
(hit/recall/nDCG), plus per-call token accounting and latency, so the managed
retriever sits in the same table as the 60-line local BM25.

Two platform constraints it exposes: a hard 10,000 deferred-tools-per-request
ceiling (the full 44K catalog cannot be served; use `--registry category` or
`--total-limit`), and real per-call cost. The harness hard-caps estimated
spend via `ANTHROPIC_BUDGET_USD` and aborts the run once the cap is reached.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BUDGET_USD=1.50 npx tsx src/run_eval.ts \
  --strategies anthropic,bm25,embed --embed-model ToolRet-trained-bge-large-en-v1.5 \
  --registry category --category customized --total-limit 100 \
  --out ../results/anthropic-calib.csv
```

`src/probe_anthropic.ts` makes one raw call (~$0.003) and dumps the fields
that verify the mechanism: `response.model`, `server_tool_use` blocks,
discovered tool references, and token usage.
