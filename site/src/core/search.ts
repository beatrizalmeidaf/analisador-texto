import { fold } from './text.ts';

export interface Hit {
  sentence: number;
  /** trechos [início, fim) da frase que casam com a busca */
  ranges: [number, number][];
}

function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Busca exata insensível a acentos e caixa. Funciona sobre a versão "dobrada"
 * da frase — como a remoção de acentos (NFD + strip de marcas) preserva o número
 * de caracteres-base, mapeamos os índices de volta para o texto original.
 */
export function search(sentences: { text: string }[], query: string, wholeWord: boolean, limit = 200) {
  const q = fold(query.trim());
  if (!q) return { total: 0, hits: [] as Hit[] };
  const pattern = wholeWord ? `(?<![\\p{L}\\p{M}])${escapeRe(q)}(?![\\p{L}\\p{M}])` : escapeRe(q);
  const re = new RegExp(pattern, 'gu');
  const hits: Hit[] = [];
  let total = 0;
  sentences.forEach(({ text }, idx) => {
    const { folded, map } = foldWithMap(text);
    const ranges: [number, number][] = [];
    for (const m of folded.matchAll(re)) {
      ranges.push([map[m.index], map[m.index + m[0].length]]);
    }
    if (ranges.length) {
      total += ranges.length;
      if (hits.length < limit) hits.push({ sentence: idx, ranges });
    }
  });
  return { total, hits };
}

function foldWithMap(text: string) {
  let folded = '';
  const map: number[] = [];
  let i = 0;
  for (const ch of text) {
    const f = fold(ch);
    for (let k = 0; k < f.length; k++) {
      map.push(i);
      folded += f[k];
    }
    i += ch.length;
  }
  map.push(i);
  return { folded, map };
}
