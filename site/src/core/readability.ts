import { words } from './text.ts';

// ---------------------------------------------------------------- sílabas

const VOWELS = 'aeiouáéíóúâêôãõàüy';
const isVowel = (c: string | undefined) => !!c && VOWELS.includes(c);
/** semivogais candidatas: i/u sem acento agudo formam ditongo com a vogal anterior */
const GLIDES = 'iuy';
const NASAL_DIPHTHONG = /^(ão|ãe|õe|õi|ãi)$/;

/**
 * Conta sílabas de uma palavra em português pelas regras de ditongo e hiato:
 * - "qu"/"gu" + vogal: o u não forma sílaba (que-ro, á-gua, guer-ra);
 * - vogal + i/u átono é ditongo decrescente (mui-to, cai-xa, céu), exceto
 *   quando o i/u fecha sílaba com l, m, n, r, z ou vem antes de "nh"
 *   (ra-iz, ru-im, ra-i-nha, ca-ir);
 * - ão, ãe, õe são ditongos nasais (pão, mãe, ele-i-ções);
 * - nos demais encontros de vogais há hiato (di-a, le-ão, po-e-ta, vo-o).
 */
const memo = new Map<string, number>();

/** syllables() com cache: o vocabulário de um texto é pequeno perto do número de palavras. */
export function syllablesCached(word: string): number {
  let n = memo.get(word);
  if (n === undefined) {
    n = syllables(word);
    if (memo.size > 200_000) memo.clear();
    memo.set(word, n);
  }
  return n;
}

export function syllables(word: string): number {
  const w = word.toLowerCase().replace(/(?<=[qg])u(?=[aeoáéóâêôãõií])/g, '');
  let count = 0;
  for (let i = 0; i < w.length; i++) {
    const c = w[i];
    if (!isVowel(c)) continue;
    const prev = w[i - 1];
    if (!isVowel(prev)) {
      count++;
      continue;
    }
    if (NASAL_DIPHTHONG.test(prev + c)) continue;
    if (GLIDES.includes(c) && c !== prev) {
      // i/u depois de vogal: ditongo, a menos que fechem sílaba própria (hiato)
      const next = w[i + 1];
      const after = w[i + 2];
      const beforeNh = next === 'n' && after === 'h';
      const closesSyllable = !!next && 'lmnrz'.includes(next) && !isVowel(after);
      if (!beforeNh && !closesSyllable) continue;
    }
    count++;
  }
  return Math.max(1, count);
}

// ---------------------------------------------------------------- legibilidade

export interface HardSentence {
  /** índice na lista de frases da análise */
  sentence: number;
  score: number;
  words: number;
  syllablesPerWord: number;
}

export interface Readability {
  /** Flesch adaptado ao português (Martins et al., 1996), limitado a 0–100 */
  flesch: number;
  /** valor sem limite, útil para comparar textos nos extremos */
  fleschRaw: number;
  band: Band;
  wordsPerSentence: number;
  syllablesPerWord: number;
  /** frases com mais de 30 palavras */
  longSentenceRatio: number;
  /** palavras com 4 ou mais sílabas (polissílabas) */
  polysyllableRatio: number;
  /** frases de prosa consideradas (títulos e itens soltos ficam de fora) */
  sentencesUsed: number;
  wordsUsed: number;
  /** pontos perdidos por frases longas e por palavras longas, em relação a um texto fácil */
  penalties: { sentences: number; words: number };
  reliable: boolean;
  hardest: HardSentence[];
}

export interface Band {
  label: 'Muito fácil' | 'Fácil' | 'Difícil' | 'Muito difícil';
  audience: string;
}

export function band(flesch: number): Band {
  if (flesch >= 75) return { label: 'Muito fácil', audience: 'leitores dos anos iniciais do ensino fundamental' };
  if (flesch >= 50) return { label: 'Fácil', audience: 'leitores dos anos finais do ensino fundamental' };
  if (flesch >= 25) return { label: 'Difícil', audience: 'leitores do ensino médio ou da graduação' };
  return { label: 'Muito difícil', audience: 'leitores especializados (textos acadêmicos, técnicos ou jurídicos)' };
}

const flesch = (wps: number, spw: number) => 248.835 - 1.015 * wps - 84.6 * spw;

/** Limites de um texto "fácil" (Flesch ≈ 50): até 20 palavras por frase e 2,1 sílabas por palavra. */
export const TARGET = { wordsPerSentence: 20, syllablesPerWord: 2.1 };
const LONG_SENTENCE = 30;

/**
 * Frase de prosa: termina com pontuação ou é longa o bastante para não ser
 * um título. Títulos, legendas ("Figura 1 - ...") e itens de sumário encurtam
 * artificialmente a média de palavras por frase.
 */
function isProse(text: string, nWords: number) {
  if (nWords < 3) return false;
  if (/^(figura|tabela|quadro|gráfico|fonte|imagem)\s*\d/i.test(text)) return false;
  return /[.!?…;:]["”')\]]?$/.test(text) || nWords >= 12;
}

export function readability(sentences: { text: string; words?: string[] }[]): Readability {
  const per = sentences.map((s, i) => {
    // as frases da análise já trazem as palavras em minúsculas
    const ws = s.words ?? words(s.text).map((w) => w.toLowerCase());
    let syl = 0;
    let poly = 0;
    for (const w of ws) {
      const n = syllablesCached(w);
      syl += n;
      if (n >= 4) poly++;
    }
    return { i, n: ws.length, syl, poly, prose: isProse(s.text.trim(), ws.length) };
  });
  const prose = per.filter((p) => p.prose);
  // sem frases de prosa suficientes (lista, poema, texto muito curto): usa tudo
  const used = prose.length >= 3 ? prose : per.filter((p) => p.n > 0);

  const nSent = used.length;
  const nWords = used.reduce((a, p) => a + p.n, 0);
  const nSyl = used.reduce((a, p) => a + p.syl, 0);
  const nPoly = used.reduce((a, p) => a + p.poly, 0);
  const wps = nSent ? nWords / nSent : 0;
  const spw = nWords ? nSyl / nWords : 0;
  const raw = nWords ? flesch(wps, spw) : 0;

  const hardest = per
    .filter((p) => p.prose && p.n >= 8)
    .map((p) => ({ sentence: p.i, score: flesch(p.n, p.syl / p.n), words: p.n, syllablesPerWord: p.syl / p.n }))
    .sort((a, b) => a.score - b.score)
    .slice(0, 5);

  const value = Math.max(0, Math.min(100, raw));
  return {
    flesch: value,
    fleschRaw: raw,
    band: band(value),
    wordsPerSentence: wps,
    syllablesPerWord: spw,
    longSentenceRatio: nSent ? used.filter((p) => p.n > LONG_SENTENCE).length / nSent : 0,
    polysyllableRatio: nWords ? nPoly / nWords : 0,
    sentencesUsed: nSent,
    wordsUsed: nWords,
    penalties: {
      sentences: 1.015 * Math.max(0, wps - TARGET.wordsPerSentence),
      words: 84.6 * Math.max(0, spw - TARGET.syllablesPerWord),
    },
    // a fórmula é uma média: com poucas frases ou palavras ela oscila demais
    reliable: nSent >= 5 && nWords >= 100,
    hardest: hardest.filter((h) => h.score < 50),
  };
}

const n1 = (x: number) => x.toLocaleString('pt-BR', { maximumFractionDigits: 1 });
const n2 = (x: number) => x.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/** Explica em linguagem simples o que torna o texto fácil ou difícil. */
export function explainReadability(r: Readability): string[] {
  const out: string[] = [];
  if (!r.wordsUsed) return out;
  const { sentences: ps, words: pw } = r.penalties;
  const sentHard = r.wordsPerSentence > TARGET.wordsPerSentence;
  const wordHard = r.syllablesPerWord > TARGET.syllablesPerWord;

  if (!sentHard && !wordHard) {
    out.push(`Frases curtas (${n1(r.wordsPerSentence)} palavras em média) e palavras simples (${n2(r.syllablesPerWord)} sílabas em média) deixam o texto fluido.`);
  } else {
    const main = ps >= pw ? 'frases' : 'palavras';
    if (main === 'frases') {
      out.push(`O que mais pesa são as frases longas: ${n1(r.wordsPerSentence)} palavras em média (o recomendado para leitura fácil é até ${TARGET.wordsPerSentence}).`);
      if (wordHard) out.push(`O vocabulário também contribui: ${n2(r.syllablesPerWord)} sílabas por palavra (textos fáceis ficam perto de ${n1(TARGET.syllablesPerWord)}).`);
    } else {
      out.push(`O que mais pesa é o vocabulário: ${n2(r.syllablesPerWord)} sílabas por palavra em média (textos fáceis ficam perto de ${n1(TARGET.syllablesPerWord)}), com ${n1(r.polysyllableRatio * 100)}% de palavras com 4 ou mais sílabas.`);
      if (sentHard) out.push(`As frases também são longas: ${n1(r.wordsPerSentence)} palavras em média (o recomendado é até ${TARGET.wordsPerSentence}).`);
    }
  }
  if (r.longSentenceRatio >= 0.1) {
    out.push(`${n1(r.longSentenceRatio * 100)}% das frases têm mais de ${LONG_SENTENCE} palavras; dividi-las é a forma mais rápida de facilitar a leitura.`);
  }
  return out;
}
