/// <reference lib="webworker" />
import { analyze, type Analysis } from '../core/analyze.ts';
import { summarize } from '../core/summarize.ts';
import { search } from '../core/search.ts';
import type { AnalysisRequest, AnalysisResponse, Keyphrase } from './protocol.ts';

let last: Analysis | null = null;

const post = (msg: AnalysisResponse) => self.postMessage(msg);

self.onmessage = (e: MessageEvent<AnalysisRequest>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'analyze': {
      const t0 = performance.now();
      last = analyze(msg.text);
      const { sentences, ...rest } = last;
      post({
        type: 'analysis',
        id: msg.id,
        ms: performance.now() - t0,
        result: { ...rest, sentences: sentences.map(({ text, para }) => ({ text, para })) },
      });
      break;
    }
    case 'summarize':
      post({ type: 'summary', id: msg.id, indices: last ? summarize(last.sentences, msg.k) : [] });
      break;
    case 'search':
      post({ type: 'search', id: msg.id, ...search(last?.sentences ?? [], msg.query, msg.wholeWord) });
      break;
    case 'keyphrases':
      post({ type: 'keyphrases', id: msg.id, phrases: last ? extractKeyphrases(last, msg.topN) : [] });
      break;
  }
};

// ---------------------------------------------------------------------------
// Frases-chave: TF-IDF entre frases + MMR para diversidade (zero latência)
// ---------------------------------------------------------------------------

function extractKeyphrases(analysis: Analysis, topN: number): Keyphrase[] {
  const sentences = analysis.sentences;
  if (!sentences.length) return [];

  const tokenSets = sentences.map((s) =>
    (s.text.toLowerCase().match(/\b[\p{L}]{3,}\b/giu) ?? [])
  );
  const df = new Map<string, number>();
  for (const tokens of tokenSets)
    for (const t of new Set(tokens)) df.set(t, (df.get(t) ?? 0) + 1);
  const N = sentences.length || 1;

  const candidateScores = new Map<string, number>();
  for (const tokens of tokenSets) {
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    const len = tokens.length || 1;
    for (const [t, count] of tf) {
      const idf = Math.log((N + 1) / ((df.get(t) ?? 0) + 1)) + 1;
      candidateScores.set(t, Math.max(candidateScores.get(t) ?? 0, (count / len) * idf));
    }
    for (let i = 0; i < tokens.length - 1; i++) {
      const bg = `${tokens[i]} ${tokens[i + 1]}`;
      const idf1 = Math.log((N + 1) / ((df.get(tokens[i]) ?? 0) + 1)) + 1;
      const idf2 = Math.log((N + 1) / ((df.get(tokens[i + 1]) ?? 0) + 1)) + 1;
      candidateScores.set(bg, Math.max(candidateScores.get(bg) ?? 0, ((1 / len) * (idf1 + idf2)) / 2));
    }
  }

  const sorted = [...candidateScores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN * 5)
    .map(([phrase, score]) => ({ phrase, score }));

  const selected: Keyphrase[] = [];
  const remaining = [...sorted];
  const lambda = 0.6;
  while (selected.length < topN && remaining.length) {
    let best = -Infinity, bestIdx = 0;
    for (let i = 0; i < remaining.length; i++) {
      const maxSim = selected.reduce((m, s) => Math.max(m, jaccardWords(remaining[i].phrase, s.phrase)), 0);
      const mmr = lambda * remaining[i].score - (1 - lambda) * maxSim;
      if (mmr > best) { best = mmr; bestIdx = i; }
    }
    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }
  return selected;
}

function jaccardWords(a: string, b: string): number {
  const sa = new Set(a.split(' '));
  const sb = new Set(b.split(' '));
  const inter = [...sa].filter((x) => sb.has(x)).length;
  return new Set([...sa, ...sb]).size ? inter / new Set([...sa, ...sb]).size : 0;
}
