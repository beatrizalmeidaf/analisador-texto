import { STOPWORDS } from './stopwords.ts';

// palavras: letras (com acentos) e hífens/apóstrofos internos, ex.: "pré-treinado", "d'água"
const WORD_RE = /[\p{L}\p{M}]+(?:[-'’][\p{L}\p{M}]+)*/gu;

export interface Sentence {
  text: string;
  /** índice do parágrafo de origem */
  para: number;
  /** palavras em minúsculas (inclui stopwords) */
  words: string[];
  /** palavras de conteúdo: sem stopwords e com 2+ letras */
  terms: string[];
}

export function words(text: string): string[] {
  return text.match(WORD_RE) ?? [];
}

export function isContentWord(w: string): boolean {
  return w.length > 1 && !STOPWORDS.has(w);
}

/** Remove acentos e caixa — usado para buscas tolerantes. */
export function fold(s: string): string {
  return s.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();
}

export function paragraphs(text: string): string[] {
  return text
    .replace(/\r\n?/g, '\n')
    .split(/\n\s*\n/)
    .map((p) => p.replace(/\s*\n\s*/g, ' ').trim())
    .filter(Boolean);
}

const segmenter =
  typeof Intl !== 'undefined' && 'Segmenter' in Intl
    ? new Intl.Segmenter('pt', { granularity: 'sentence' })
    : null;

function splitSentences(paragraph: string): string[] {
  if (segmenter) {
    return Array.from(segmenter.segment(paragraph), (s) => s.segment.trim()).filter(Boolean);
  }
  return paragraph.split(/(?<=[.!?…])\s+(?=[\p{Lu}\d"“(])/u).filter(Boolean);
}

export function sentences(text: string): Sentence[] {
  const out: Sentence[] = [];
  paragraphs(text).forEach((p, para) => {
    for (const s of splitSentences(p)) {
      const ws = words(s).map((w) => w.toLowerCase());
      if (!ws.length) continue;
      out.push({ text: s, para, words: ws, terms: ws.filter(isContentWord) });
    }
  });
  return out;
}
