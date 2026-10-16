import { sentences, words, isContentWord, type Sentence } from './text.ts';
import { readability, type Readability } from './readability.ts';

export interface Term {
  term: string;
  count: number;
  score: number;
}

export interface Stats {
  chars: number;
  charsNoSpaces: number;
  words: number;
  uniqueWords: number;
  contentWords: number;
  sentences: number;
  paragraphs: number;
  /** comprimento das palavras (em letras) */
  wordLength: { mean: number; median: number; std: number };
  wordsPerSentence: number;
  syllablesPerWord: number;
  /** razão tipo/token (vocabulário distinto ÷ total) */
  lexicalDiversity: number;
  /** palavras que aparecem uma única vez */
  hapax: number;
  /** Flesch adaptado ao português (Martins et al., 1996) — atalho para readability.flesch */
  flesch: number;
  readability: Readability;
  readingMinutes: number;
  speakingMinutes: number;
  /** histograma de comprimento: índice = nº de letras (último bucket agrupa 15+) */
  lengthHistogram: number[];
}

export interface Analysis {
  stats: Stats;
  frequencies: Term[];
  keywords: Term[];
  ngrams: Term[];
  sentences: Sentence[];
}

const HIST_MAX = 15;

function lengthStats(lengths: number[]) {
  if (!lengths.length) return { mean: 0, median: 0, std: 0 };
  const sorted = [...lengths].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const median = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  // desvio padrão amostral, igual ao pandas.Series.std() da versão original
  const variance = n > 1 ? sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
  return { mean, median, std: Math.sqrt(variance) };
}

/**
 * TF-IDF de verdade exige mais de um documento. Aqui cada trecho do texto
 * (parágrafo, ou grupo de frases quando há poucos parágrafos) é um documento;
 * termos concentrados em partes específicas ganham peso, termos espalhados
 * por todo o texto perdem.
 */
function segmentsFor(sents: Sentence[]): string[][] {
  const byPara = new Map<number, string[]>();
  for (const s of sents) {
    const seg = byPara.get(s.para);
    if (seg) seg.push(...s.terms);
    else byPara.set(s.para, [...s.terms]);
  }
  if (byPara.size >= 5) return [...byPara.values()];
  const size = Math.max(1, Math.ceil(sents.length / 12));
  const segs: string[][] = [];
  for (let i = 0; i < sents.length; i += size) {
    segs.push(sents.slice(i, i + size).flatMap((s) => s.terms));
  }
  return segs;
}

export function keywordScores(sents: Sentence[], counts: Map<string, number>, limit: number): Term[] {
  const segs = segmentsFor(sents);
  const df = new Map<string, number>();
  for (const seg of segs) for (const t of new Set(seg)) df.set(t, (df.get(t) ?? 0) + 1);
  const N = segs.length;
  const total = [...counts.values()].reduce((a, b) => a + b, 0) || 1;
  const out: Term[] = [];
  for (const [term, count] of counts) {
    const idf = Math.log((1 + N) / (1 + (df.get(term) ?? 0))) + 1;
    // sublinear tf evita que termos muito repetidos dominem a lista
    out.push({ term, count, score: ((1 + Math.log(count)) / Math.log(1 + total)) * idf });
  }
  out.sort((a, b) => b.score - a.score || b.count - a.count);
  const top = out.slice(0, limit);
  const max = top[0]?.score || 1;
  for (const t of top) t.score /= max;
  return top;
}

/** Bigramas e trigramas que não começam nem terminam em stopword. */
function ngramCounts(sents: Sentence[], limit: number): Term[] {
  const counts = new Map<string, number>();
  for (const { words: ws } of sents) {
    for (let i = 0; i < ws.length; i++) {
      if (!isContentWord(ws[i])) continue;
      for (const n of [2, 3]) {
        const last = ws[i + n - 1];
        if (!last || !isContentWord(last)) continue;
        // em trigramas o meio pode ser stopword ("processamento de linguagem")
        const key = ws.slice(i, i + n).join(' ');
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }
  }
  const out: Term[] = [];
  for (const [term, count] of counts) {
    if (count < 2) continue;
    out.push({ term, count, score: count * (term.split(' ').length === 3 ? 1.5 : 1) });
  }
  out.sort((a, b) => b.score - a.score);
  // remove bigramas totalmente contidos num trigrama com a mesma contagem
  const kept: Term[] = [];
  for (const t of out) {
    if (kept.some((k) => k.count >= t.count && k.term.includes(t.term))) continue;
    kept.push(t);
    if (kept.length >= limit) break;
  }
  return kept;
}

export function analyze(text: string): Analysis {
  const sents = sentences(text);
  const allWords = words(text);
  const lower = allWords.map((w) => w.toLowerCase());

  const allCounts = new Map<string, number>();
  for (const w of lower) allCounts.set(w, (allCounts.get(w) ?? 0) + 1);

  const contentCounts = new Map<string, number>();
  for (const s of sents) for (const t of s.terms) contentCounts.set(t, (contentCounts.get(t) ?? 0) + 1);

  const lengths = allWords.map((w) => w.length);
  const histogram = new Array(HIST_MAX + 1).fill(0);
  for (const l of lengths) histogram[Math.min(l, HIST_MAX)]++;

  const nWords = allWords.length;
  const read = readability(sents);

  let hapax = 0;
  for (const c of allCounts.values()) if (c === 1) hapax++;

  const frequencies: Term[] = [...contentCounts]
    .map(([term, count]) => ({ term, count, score: count }))
    .sort((a, b) => b.count - a.count || a.term.localeCompare(b.term, 'pt'));

  return {
    stats: {
      chars: text.length,
      charsNoSpaces: text.replace(/\s/g, '').length,
      words: nWords,
      uniqueWords: allCounts.size,
      contentWords: [...contentCounts.values()].reduce((a, b) => a + b, 0),
      sentences: sents.length,
      paragraphs: new Set(sents.map((s) => s.para)).size,
      wordLength: lengthStats(lengths),
      wordsPerSentence: read.wordsPerSentence,
      syllablesPerWord: read.syllablesPerWord,
      // TTR cai naturalmente em textos longos; usamos a média em janelas de 1000
      // palavras (MSTTR) para que textos de tamanhos diferentes sejam comparáveis
      lexicalDiversity: msttr(lower, 1000),
      hapax,
      flesch: read.flesch,
      readability: read,
      readingMinutes: nWords / 230,
      speakingMinutes: nWords / 140,
      lengthHistogram: histogram,
    },
    frequencies: frequencies.slice(0, 200),
    keywords: keywordScores(sents, contentCounts, 50),
    ngrams: ngramCounts(sents, 30),
    sentences: sents,
  };
}

function msttr(ws: string[], window: number): number {
  if (!ws.length) return 0;
  if (ws.length < window * 1.5) return new Set(ws).size / ws.length;
  let sum = 0;
  let n = 0;
  for (let i = 0; i + window <= ws.length; i += window) {
    sum += new Set(ws.slice(i, i + window)).size / window;
    n++;
  }
  return sum / n;
}
