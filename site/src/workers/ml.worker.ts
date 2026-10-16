/// <reference lib="webworker" />
import { pipeline, env, type FeatureExtractionPipeline, type TextClassificationPipeline } from '@huggingface/transformers';
import { textRank, pickSentences, summaryFilter } from '../core/summarize.ts';
import { GENRES } from '../genres.ts';
import type { MLRequest, MLResponse, SentimentResult } from './protocol.ts';

// os modelos vêm do Hugging Face Hub e ficam no Cache Storage do navegador
env.allowLocalModels = false;
env.useBrowserCache = true;

const EMBED_MODEL = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
const SENTIMENT_MODEL = 'Xenova/distilbert-base-multilingual-cased-sentiments-student';
/** limite de frases indexadas: acima disso amostramos de forma uniforme pelo texto */
const MAX_INDEX = 1200;
const BATCH = 16;
const MAX_SENTIMENT = 200;
const DIM = 384;

const post = (msg: MLResponse) => self.postMessage(msg);

function progressReporter(id: number, label: string) {
  const files = new Map<string, { loaded: number; total: number }>();
  return (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (p.status !== 'progress' || !p.file || !p.total) return;
    files.set(p.file, { loaded: p.loaded ?? 0, total: p.total });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    post({ type: 'progress', id, stage: 'download', label, progress: total ? loaded / total : 0 });
  };
}

let embedder: Promise<FeatureExtractionPipeline> | null = null;
function getEmbedder(id: number) {
  embedder ??= pipeline('feature-extraction', EMBED_MODEL, {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: progressReporter(id, 'Baixando modelo de embeddings (~118 MB, só na primeira vez)'),
  }).catch((err) => {
    embedder = null;
    throw err;
  }) as Promise<FeatureExtractionPipeline>;
  return embedder;
}

let sentimentModel: Promise<TextClassificationPipeline> | null = null;
function getSentiment(id: number) {
  sentimentModel ??= pipeline('text-classification', SENTIMENT_MODEL, {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: progressReporter(id, 'Baixando modelo de sentimento (~136 MB, só na primeira vez)'),
  }).catch((err) => {
    sentimentModel = null;
    throw err;
  }) as Promise<TextClassificationPipeline>;
  return sentimentModel;
}

async function embed(model: FeatureExtractionPipeline, texts: string[]): Promise<Float32Array> {
  const out = await model(texts, { pooling: 'mean', normalize: true });
  return out.data as Float32Array;
}

function dot(a: Float32Array, ai: number, b: Float32Array, bi: number) {
  let s = 0;
  const ao = ai * DIM;
  const bo = bi * DIM;
  for (let k = 0; k < DIM; k++) s += a[ao + k] * b[bo + k];
  return s;
}

function sampleIndices(n: number, max: number): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = n / max;
  return Array.from({ length: max }, (_, i) => Math.floor(i * step));
}

interface Index {
  key: string;
  /** índice da frase original para cada linha da matriz */
  rows: number[];
  texts: string[];
  vectors: Float32Array;
  /** true quando o texto tinha mais frases que MAX_INDEX */
  sampled: boolean;
}

let index: Index | null = null;
let indexing: { key: string; promise: Promise<Index> } | null = null;

/**
 * Embeddings já calculados, por frase. Ao editar o texto, só as frases novas ou
 * alteradas passam pelo modelo — é o que permite reindexar quase instantaneamente.
 */
const cache = new Map<string, Float32Array>();
const CACHE_MAX = 20_000;

/** Frases que valem indexar: títulos e fragmentos ("4.2 Uso Ético") só poluem busca e resumo. */
function isIndexable(s: string) {
  const n = (s.match(/\p{L}{2,}/gu) ?? []).length;
  return n >= 5 && (n >= 10 || /[.!?…:;]["”')\]]?$/.test(s));
}

async function buildIndex(id: number, key: string, sentences: string[]): Promise<Index> {
  const model = await getEmbedder(id);
  const eligible = sentences.flatMap((s, i) => (isIndexable(s) ? [i] : []));
  const pool = eligible.length >= 3 ? eligible : sentences.map((_, i) => i);
  const rows = sampleIndices(pool.length, MAX_INDEX).map((i) => pool[i]);
  const texts = rows.map((r) => sentences[r]);

  // lotes com frases de tamanho parecido: o modelo preenche cada lote até a
  // frase mais longa, então misturar curtas e longas desperdiça processamento
  const missing = [...new Set(texts.filter((t) => !cache.has(t)))].sort((a, b) => a.length - b.length);
  for (let i = 0; i < missing.length; i += BATCH) {
    // outro texto foi enviado enquanto indexávamos: abandona este
    if (indexing?.key !== key) throw new Error('cancelado');
    const batch = missing.slice(i, i + BATCH);
    const out = await embed(model, batch);
    batch.forEach((t, j) => cache.set(t, out.slice(j * DIM, (j + 1) * DIM)));
    post({ type: 'progress', id, stage: 'compute', label: 'Calculando embeddings das frases', progress: Math.min(1, (i + BATCH) / missing.length) });
  }

  const vectors = new Float32Array(rows.length * DIM);
  texts.forEach((t, r) => vectors.set(cache.get(t)!, r * DIM));
  if (cache.size > CACHE_MAX) {
    // descarta as entradas mais antigas (Map preserva a ordem de inserção)
    const keep = new Set(texts);
    for (const k of cache.keys()) {
      if (cache.size <= CACHE_MAX / 2) break;
      if (!keep.has(k)) cache.delete(k);
    }
  }
  return { key, rows, texts, vectors, sampled: pool.length > rows.length };
}

function requireIndex(key: string): Index {
  if (!index || index.key !== key) throw new Error('O texto ainda não foi indexado.');
  return index;
}

let genreVectors: Float32Array | null = null;

async function classify(id: number, key: string) {
  const idx = requireIndex(key);
  const model = await getEmbedder(id);
  const prototypes = GENRES.flatMap((g) => g.prototypes);
  genreVectors ??= await embed(model, prototypes);

  const docVec = new Float32Array(DIM);
  const n = idx.rows.length;
  for (let r = 0; r < n; r++) for (let k = 0; k < DIM; k++) docVec[k] += idx.vectors[r * DIM + k];
  const norm = Math.hypot(...docVec) || 1;
  for (let k = 0; k < DIM; k++) docVec[k] /= norm;

  let p = 0;
  const sims = GENRES.map((g) => {
    let best = -1;
    for (let j = 0; j < g.prototypes.length; j++, p++) best = Math.max(best, dot(docVec, 0, genreVectors!, p));
    return best;
  });
  // softmax com temperatura baixa: as similaridades brutas ficam todas próximas
  const T = 0.03;
  const max = Math.max(...sims);
  const exps = sims.map((s) => Math.exp((s - max) / T));
  const sum = exps.reduce((a, b) => a + b, 0);
  return GENRES.map((g, i) => ({ label: g.label, score: exps[i] / sum })).sort((a, b) => b.score - a.score);
}

async function sentiment(id: number, sentences: string[]): Promise<SentimentResult> {
  const model = await getSentiment(id);
  const eligible = sentences.map((s, i) => [s, i] as const).filter(([s]) => s.split(/\s+/).length >= 4);
  const picked = sampleIndices(eligible.length, MAX_SENTIMENT).map((i) => eligible[i]);
  const dist = { positive: 0, neutral: 0, negative: 0 };
  const polarity: { sentence: number; score: number }[] = [];
  for (let i = 0; i < picked.length; i += BATCH) {
    const batch = picked.slice(i, i + BATCH);
    const out = (await model(batch.map(([s]) => s), { top_k: null })) as unknown as { label: string; score: number }[][];
    out.forEach((labels, j) => {
      const by = Object.fromEntries(labels.map((l) => [l.label, l.score])) as Record<string, number>;
      dist.positive += by.positive ?? 0;
      dist.neutral += by.neutral ?? 0;
      dist.negative += by.negative ?? 0;
      polarity.push({ sentence: batch[j][1], score: (by.positive ?? 0) - (by.negative ?? 0) });
    });
    post({ type: 'progress', id, stage: 'compute', label: 'Analisando sentimento das frases', progress: Math.min(1, (i + BATCH) / picked.length) });
  }
  const n = picked.length || 1;
  polarity.sort((a, b) => b.score - a.score);
  return {
    distribution: { positive: dist.positive / n, neutral: dist.neutral / n, negative: dist.negative / n },
    analyzed: picked.length,
    mostPositive: polarity.filter((p) => p.score > 0.3).slice(0, 3),
    mostNegative: polarity.filter((p) => p.score < -0.3).slice(-3).reverse(),
  };
}

self.onmessage = async (e: MessageEvent<MLRequest>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'index': {
        if (index?.key !== msg.key) {
          if (indexing?.key !== msg.key) indexing = { key: msg.key, promise: buildIndex(msg.id, msg.key, msg.sentences) };
          const mine = indexing!;
          try {
            index = await mine.promise;
          } finally {
            if (indexing === mine) indexing = null;
          }
        }
        post({ type: 'indexed', id: msg.id, key: msg.key, count: index.rows.length, total: msg.sentences.length, sampled: index.sampled });
        break;
      }
      case 'ask': {
        const idx = requireIndex(msg.key);
        const q = await embed(await getEmbedder(msg.id), [msg.question]);
        const scored = idx.rows.map((sentence, r) => ({ sentence, score: dot(idx.vectors, r, q, 0) }));
        scored.sort((a, b) => b.score - a.score);
        post({ type: 'answer', id: msg.id, matches: scored.slice(0, msg.k) });
        break;
      }
      case 'classify':
        post({ type: 'genres', id: msg.id, genres: await classify(msg.id, msg.key) });
        break;
      case 'semantic-summary': {
        const idx = requireIndex(msg.key);
        const sim = (i: number, j: number) => dot(idx.vectors, i, idx.vectors, j);
        const scores = textRank(idx.rows.length, (i, j) => Math.max(0, sim(i, j) - 0.2));
        const chosen = pickSentences(scores, sim, msg.k, summaryFilter(idx.texts.map((text) => ({ text }))), 0.8);
        post({ type: 'summary', id: msg.id, indices: chosen.map((r) => idx.rows[r]) });
        break;
      }
      case 'sentiment':
        post({ type: 'sentiment', id: msg.id, result: await sentiment(msg.id, msg.sentences) });
        break;
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    post({
      type: 'error',
      id: msg.id,
      message: /while trying to load file|Failed to fetch|NetworkError/i.test(message)
        ? 'Não foi possível baixar o modelo do Hugging Face. Verifique a conexão e tente novamente.'
        : message,
    });
  }
};
