import type { Sentence } from './text.ts';

export type Similarity = (i: number, j: number) => number;

/** PageRank sobre o grafo de similaridade entre frases (TextRank). */
export function textRank(n: number, sim: Similarity, iterations = 40, damping = 0.85): Float64Array {
  const weights: Float64Array[] = [];
  const outSum = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n);
    for (let j = 0; j < i; j++) {
      const w = Math.max(0, sim(i, j));
      row[j] = w;
      weights[j][i] = w;
      outSum[i] += w;
      outSum[j] += w;
    }
    weights.push(row);
  }
  let scores = new Float64Array(n).fill(1 / n);
  for (let it = 0; it < iterations; it++) {
    const next = new Float64Array(n).fill((1 - damping) / n);
    for (let j = 0; j < n; j++) {
      if (!outSum[j]) continue;
      const share = (damping * scores[j]) / outSum[j];
      const row = weights[j];
      for (let i = 0; i < n; i++) if (row[i]) next[i] += row[i] * share;
    }
    scores = next;
  }
  return scores;
}

/**
 * Escolhe as k frases de maior pontuação evitando redundância (MMR simplificado)
 * e devolve os índices na ordem em que aparecem no texto.
 */
export function pickSentences(
  scores: ArrayLike<number>,
  sim: Similarity,
  k: number,
  eligible: (i: number) => boolean,
  redundancy = 0.6,
): number[] {
  const order = Array.from({ length: scores.length }, (_, i) => i)
    .filter(eligible)
    .sort((a, b) => scores[b] - scores[a]);
  const chosen: number[] = [];
  for (const i of order) {
    if (chosen.length >= k) break;
    if (chosen.some((c) => sim(i, c) > redundancy)) continue;
    chosen.push(i);
  }
  return chosen.sort((a, b) => a - b);
}

type Sparse = Map<string, number>;

function tfidfVectors(sents: Sentence[]): Sparse[] {
  const df = new Map<string, number>();
  for (const s of sents) for (const t of new Set(s.terms)) df.set(t, (df.get(t) ?? 0) + 1);
  const n = sents.length;
  return sents.map((s) => {
    const v: Sparse = new Map();
    for (const t of s.terms) v.set(t, (v.get(t) ?? 0) + 1);
    let norm = 0;
    for (const [t, tf] of v) {
      const w = (1 + Math.log(tf)) * Math.log((1 + n) / (1 + df.get(t)!) + 1);
      v.set(t, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [t, w] of v) v.set(t, w / norm);
    return v;
  });
}

function cosine(a: Sparse, b: Sparse): number {
  if (a.size > b.size) [a, b] = [b, a];
  let dot = 0;
  for (const [t, w] of a) {
    const o = b.get(t);
    if (o) dot += w * o;
  }
  return dot;
}

/** Frases "de verdade": tamanho razoável e pontuação final (descarta títulos, legendas e referências). */
export function summaryFilter(sents: { text: string }[]): (i: number) => boolean {
  const sized = (t: string) => {
    const n = t.split(/\s+/).length;
    return n >= 6 && n <= 80;
  };
  const strict = (t: string) =>
    sized(t) && /[.!?…]["”')\]]?$/.test(t) && !/^[»•▪◦■]/.test(t) && !/https?:|www\.|\[Internet\]|acesso em/i.test(t);
  const pool = sents.filter((s) => strict(s.text)).length >= 3 ? strict : sized;
  return (i) => pool(sents[i].text);
}

/** Acima disso o grafo completo (n²) fica caro; usamos centralidade em relação ao centroide. */
const MAX_GRAPH = 1500;

export function summarize(sents: Sentence[], k: number): number[] {
  if (!sents.length) return [];
  const vecs = tfidfVectors(sents);
  const sim: Similarity = (i, j) => cosine(vecs[i], vecs[j]);
  let scores: ArrayLike<number>;
  if (sents.length <= MAX_GRAPH) {
    scores = textRank(sents.length, sim);
  } else {
    const centroid: Sparse = new Map();
    for (const v of vecs) for (const [t, w] of v) centroid.set(t, (centroid.get(t) ?? 0) + w);
    scores = vecs.map((v) => cosine(v, centroid));
  }
  return pickSentences(scores, sim, k, summaryFilter(sents));
}
