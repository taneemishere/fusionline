import { readFileSync } from "node:fs";
import { join } from "node:path";
import { BM25 } from "./bm25.js";
import type { RegistryView, Strategy, Task } from "./types.js";

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
}

export class AllTools implements Strategy {
  name = "all";
  select(_task: Task, registry: RegistryView, _k: number): string[] {
    return [...registry.tools.keys()];
  }
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class RandomTools implements Strategy {
  name = "random";
  private rand = mulberry32(0);
  select(_task: Task, registry: RegistryView, k: number): string[] {
    const ids = [...registry.tools.keys()];
    for (let i = ids.length - 1; i > 0; i--) {
      const j = Math.floor(this.rand() * (i + 1));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }
    return ids.slice(0, Math.min(k, ids.length));
  }
}

export class BM25Tools implements Strategy {
  name = "bm25";
  private cache = new Map<string, { bm25: BM25; ids: string[] }>();
  constructor(private useInstruction = false) {}

  private index(registry: RegistryView) {
    let hit = this.cache.get(registry.key);
    if (!hit) {
      const ids = [...registry.tools.keys()];
      const corpus = ids.map((id) => tokenize(registry.tools.get(id)!.searchText));
      hit = { bm25: new BM25(corpus), ids };
      this.cache.set(registry.key, hit);
    }
    return hit;
  }

  /** Full ranked id list over the registry. */
  rank(task: Task, registry: RegistryView): string[] {
    const { bm25, ids } = this.index(registry);
    const q = task.query + (this.useInstruction ? "\n" + task.instruction : "");
    const scores = bm25.scores(tokenize(q));
    return topK(scores, ids, ids.length);
  }

  select(task: Task, registry: RegistryView, k: number): string[] {
    return this.rank(task, registry).slice(0, k);
  }
}

export class EmbeddingTools implements Strategy {
  name = "embed";
  readonly model: string;
  private dim = 0;
  private toolEmb!: Float32Array;
  private toolRowOf = new Map<string, number>();
  private taskEmb!: Float32Array;
  private taskRowOf = new Map<string, number>();
  private viewCache = new Map<string, { emb: Float32Array; ids: string[] }>();
  private loaded = false;

  constructor(private embDir: string, private useInstruction = false) {
    this.model = embDir.split("/").filter(Boolean).pop() ?? "embed";
  }

  private load() {
    if (this.loaded) return;
    const meta = JSON.parse(readFileSync(join(this.embDir, "meta.json"), "utf8"));
    this.dim = meta.dim;
    const toolIds: string[] = JSON.parse(readFileSync(join(this.embDir, "tools.ids.json"), "utf8"));
    this.toolEmb = readF32(join(this.embDir, "tools.f32"));
    toolIds.forEach((id, i) => this.toolRowOf.set(id, i));
    const taskIds: string[] = JSON.parse(readFileSync(join(this.embDir, "tasks.ids.json"), "utf8"));
    const taskFile = this.useInstruction ? "tasks_inst.f32" : "tasks.f32";
    this.taskEmb = readF32(join(this.embDir, taskFile));
    taskIds.forEach((id, i) => this.taskRowOf.set(id, i));
    this.loaded = true;
  }

  private index(registry: RegistryView) {
    let hit = this.viewCache.get(registry.key);
    if (!hit) {
      const ids = [...registry.tools.keys()];
      const emb = new Float32Array(ids.length * this.dim);
      ids.forEach((id, i) => {
        const src = this.toolRowOf.get(id)! * this.dim;
        emb.set(this.toolEmb.subarray(src, src + this.dim), i * this.dim);
      });
      hit = { emb, ids };
      this.viewCache.set(registry.key, hit);
    }
    return hit;
  }

  /** Full ranked id list over the registry. */
  rank(task: Task, registry: RegistryView): string[] {
    this.load();
    const { emb, ids } = this.index(registry);
    const qrow = this.taskRowOf.get(task.taskId);
    if (qrow === undefined) return [];
    const q = this.taskEmb.subarray(qrow * this.dim, (qrow + 1) * this.dim);
    const scores = new Float64Array(ids.length);
    for (let i = 0; i < ids.length; i++) {
      const off = i * this.dim;
      let s = 0;
      for (let d = 0; d < this.dim; d++) s += emb[off + d] * q[d];
      scores[i] = s;
    }
    return topK(scores, ids, ids.length);
  }

  select(task: Task, registry: RegistryView, k: number): string[] {
    return this.rank(task, registry).slice(0, k);
  }
}

/** Reciprocal Rank Fusion of BM25 + dense embedding rankings (RRF, c=60).
 *  Rank fusion avoids normalizing the two score scales.
 *  `w` weights each parent ranking; `hybridw` uses [0.3, 0.7] (embed-favored). */
export class HybridTools implements Strategy {
  readonly name: string;
  readonly model: string;

  constructor(
    private bm25: BM25Tools,
    private embed: EmbeddingTools,
    private c = 60,
    private w: [number, number] = [1, 1],
    name = "hybrid",
  ) {
    this.name = name;
    this.model = embed.model;
  }

  rank(task: Task, registry: RegistryView): string[] {
    const rankings = [this.bm25.rank(task, registry), this.embed.rank(task, registry)]
      .filter((r) => r.length > 0);
    const fused = new Map<string, number>();
    rankings.forEach((r, ri) => {
      const w = this.w[ri] ?? 1;
      r.forEach((id, i) => {
        fused.set(id, (fused.get(id) ?? 0) + w / (this.c + i + 1));
      });
    });
    return [...fused.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id);
  }

  select(task: Task, registry: RegistryView, k: number): string[] {
    return this.rank(task, registry).slice(0, k);
  }
}

/** Set-aware (HYSET-style) baseline: boost tools that co-occur with the
 *  base ranker's top-W in *train-split* label sets. Held-out by construction —
 *  co-occurrence is built only from train-half tasks passed at construction. */
export class SetAwareTools implements Strategy {
  name = "setaware";
  readonly model: string;
  private cooc = new Map<string, Map<string, number>>();

  constructor(
    private base: { rank(t: Task, r: RegistryView): string[]; model?: string },
    trainTasks: Task[],
    private topW = 20,
    private lambda = 0.5,
    private c = 60,
  ) {
    this.model = base.model ?? "";
    for (const t of trainTasks) {
      for (const a of t.gtIds) {
        for (const b of t.gtIds) {
          if (a === b) continue;
          let m = this.cooc.get(a);
          if (!m) this.cooc.set(a, (m = new Map()));
          m.set(b, (m.get(b) ?? 0) + 1);
        }
      }
    }
  }

  select(task: Task, registry: RegistryView, k: number): string[] {
    const r = this.base.rank(task, registry);
    const boost = new Map<string, number>();
    for (const s of r.slice(0, this.topW)) {
      const m = this.cooc.get(s);
      if (!m) continue;
      for (const [t, cnt] of m) boost.set(t, (boost.get(t) ?? 0) + cnt);
    }
    // co-occurrence as a third ranked list: boost>0 tools by boost desc,
    // then the rest in base order; fused via RRF with weight `lambda`.
    const coocRank = [...boost.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([id]) => id);
    const seen = new Set(coocRank);
    for (const id of r) if (!seen.has(id)) coocRank.push(id);

    const fused = new Map<string, number>();
    r.forEach((id, i) => fused.set(id, (fused.get(id) ?? 0) + 1 / (this.c + i + 1)));
    coocRank.forEach((id, i) =>
      fused.set(id, (fused.get(id) ?? 0) + this.lambda / (this.c + i + 1)));
    return [...fused.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .map(([id]) => id);
  }
}

function readF32(path: string): Float32Array {
  const buf = readFileSync(path);
  return new Float32Array(
    buf.buffer, buf.byteOffset, buf.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );
}

function topK(scores: Float64Array, ids: string[], k: number): string[] {
  const order = Array.from({ length: scores.length }, (_, i) => i);
  order.sort((a, b) => scores[b] - scores[a]);
  return order.slice(0, Math.min(k, ids.length)).map((i) => ids[i]);
}

export async function makeStrategy(
  name: string,
  opts: { instruction?: boolean; embDir?: string; trainTasks?: Task[] },
): Promise<Strategy> {
  switch (name) {
    case "all":
      return new AllTools();
    case "random":
      return new RandomTools();
    case "bm25":
      return new BM25Tools(opts.instruction);
    case "embed":
      if (!opts.embDir) throw new Error("embed strategy requires --embed-dir");
      return new EmbeddingTools(opts.embDir, opts.instruction);
    case "hybrid":
      if (!opts.embDir) throw new Error("hybrid strategy requires --embed-dir");
      return new HybridTools(
        new BM25Tools(opts.instruction),
        new EmbeddingTools(opts.embDir, opts.instruction),
      );
    case "hybridw":
      if (!opts.embDir) throw new Error("hybridw strategy requires --embed-dir");
      return new HybridTools(
        new BM25Tools(opts.instruction),
        new EmbeddingTools(opts.embDir, opts.instruction),
        60, [0.3, 0.7], "hybridw",
      );
    case "setaware":
      if (!opts.embDir) throw new Error("setaware strategy requires --embed-dir");
      if (!opts.trainTasks) throw new Error("setaware requires train tasks");
      return new SetAwareTools(
        new HybridTools(
          new BM25Tools(opts.instruction),
          new EmbeddingTools(opts.embDir, opts.instruction),
        ),
        opts.trainTasks,
      );
    case "anthropic": {
      const { AnthropicToolSearch } = await import("./anthropic.js");
      return new AnthropicToolSearch();
    }
    default:
      throw new Error(`unknown strategy '${name}'. Available: all, random, bm25, embed, hybrid, hybridw, setaware, anthropic`);
  }
}
