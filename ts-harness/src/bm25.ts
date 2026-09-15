/** BM25Okapi matching rank_bm25 defaults: k1=1.5, b=0.75, epsilon=0.25
 *  with negative-idf flooring at epsilon * average_idf. */
export class BM25 {
  private postings = new Map<string, Array<[number, number]>>();
  private idf = new Map<string, number>();
  private docLen: number[];
  private avgdl = 0;

  constructor(docs: string[][], private k1 = 1.5, private b = 0.75, private epsilon = 0.25) {
    const n = docs.length;
    this.docLen = new Array(n);
    const df = new Map<string, number>();

    docs.forEach((doc, di) => {
      this.docLen[di] = doc.length;
      this.avgdl += doc.length / n;
      const tf = new Map<string, number>();
      for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const [term, f] of tf) {
        df.set(term, (df.get(term) ?? 0) + 1);
        let list = this.postings.get(term);
        if (!list) this.postings.set(term, (list = []));
        list.push([di, f]);
      }
    });

    let idfSum = 0;
    const negative: string[] = [];
    for (const [term, f] of df) {
      const idf = Math.log(n - f + 0.5) - Math.log(f + 0.5);
      this.idf.set(term, idf);
      idfSum += idf;
      if (idf < 0) negative.push(term);
    }
    const floor = this.epsilon * (idfSum / Math.max(df.size, 1));
    for (const term of negative) this.idf.set(term, floor);
  }

  scores(query: string[]): Float64Array {
    const out = new Float64Array(this.docLen.length);
    for (const term of new Set(query)) {
      const idf = this.idf.get(term);
      const plist = this.postings.get(term);
      if (idf === undefined || !plist) continue;
      for (const [di, f] of plist) {
        const dl = this.docLen[di];
        out[di] += idf * ((f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl)));
      }
    }
    return out;
  }
}
