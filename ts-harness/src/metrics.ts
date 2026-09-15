export function recallAtK(gt: string[], pred: string[], k: number): number {
  if (gt.length === 0) return 0;
  const g = new Set(gt);
  let hit = 0;
  for (const p of pred.slice(0, k)) if (g.has(p)) hit++;
  return hit / g.size;
}

export function precisionAtK(gt: string[], pred: string[], k: number): number {
  const top = pred.slice(0, k);
  if (top.length === 0) return 0;
  const g = new Set(gt);
  let hit = 0;
  for (const p of top) if (g.has(p)) hit++;
  return hit / top.length;
}

export function setF1AtK(gt: string[], pred: string[], k: number): number {
  const p = precisionAtK(gt, pred, k);
  const r = recallAtK(gt, pred, k);
  return p + r === 0 ? 0 : (2 * p * r) / (p + r);
}

export function hitAtK(gt: string[], pred: string[], k: number): number {
  const g = new Set(gt);
  return pred.slice(0, k).some((p) => g.has(p)) ? 1 : 0;
}

export function ndcgAtK(gt: string[], pred: string[], k: number): number {
  const g = new Set(gt);
  let dcg = 0;
  pred.slice(0, k).forEach((p, i) => {
    if (g.has(p)) dcg += 1 / Math.log2(i + 2);
  });
  let ideal = 0;
  for (let i = 0; i < Math.min(g.size, k); i++) ideal += 1 / Math.log2(i + 2);
  return ideal === 0 ? 0 : dcg / ideal;
}

export function mrrAtK(gt: string[], pred: string[], k: number): number {
  const g = new Set(gt);
  const top = pred.slice(0, k);
  for (let i = 0; i < top.length; i++) if (g.has(top[i])) return 1 / (i + 1);
  return 0;
}

export function scoreAll(gt: string[], pred: string[], k: number) {
  return {
    recall: recallAtK(gt, pred, k),
    hit: hitAtK(gt, pred, k),
    precision: precisionAtK(gt, pred, k),
    f1: setF1AtK(gt, pred, k),
    ndcg: ndcgAtK(gt, pred, k),
    mrr: mrrAtK(gt, pred, k),
  };
}
