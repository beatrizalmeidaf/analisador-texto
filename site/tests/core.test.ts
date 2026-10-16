import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { analyze } from '../src/core/analyze.ts';
import { summarize } from '../src/core/summarize.ts';
import { search } from '../src/core/search.ts';
import { sentences } from '../src/core/text.ts';
import { syllables } from '../src/core/readability.ts';

test('sílabas aproximadas', () => {
  assert.equal(syllables('casa'), 2);
  assert.equal(syllables('saúde'), 3);
  assert.equal(syllables('pão'), 1);
  assert.equal(syllables('linguagem'), 3);
});

test('estatísticas básicas', () => {
  const a = analyze('O gato subiu no telhado. O gato desceu!\n\nOutro parágrafo aqui.');
  assert.equal(a.stats.words, 11);
  assert.equal(a.stats.sentences, 3);
  assert.equal(a.stats.paragraphs, 2);
  assert.equal(a.frequencies[0].term, 'gato');
  assert.equal(a.frequencies[0].count, 2);
});

test('texto vazio não quebra', () => {
  const a = analyze('');
  assert.equal(a.stats.words, 0);
  assert.deepEqual(summarize(a.sentences, 3), []);
});

test('busca ignora acentos e mapeia índices', () => {
  const s = sentences('A Informação é útil. Informacao sem acento.');
  const r = search(s, 'informação', true);
  assert.equal(r.total, 2);
  const [a, b] = r.hits[0].ranges[0];
  assert.equal(s[0].text.slice(a, b), 'Informação');
});

test('arquivo de exemplo', () => {
  const text = readFileSync(new URL('../../data/teste.txt', import.meta.url), 'utf8');
  const t0 = performance.now();
  const a = analyze(text);
  const idx = summarize(a.sentences, 5);
  const ms = performance.now() - t0;
  console.log({ ms: Math.round(ms), ...a.stats, lengthHistogram: undefined });
  console.log('keywords', a.keywords.slice(0, 12).map((k) => k.term).join(', '));
  console.log('ngrams', a.ngrams.slice(0, 10).map((k) => `${k.term}(${k.count})`).join(', '));
  console.log('resumo:\n- ' + idx.map((i) => a.sentences[i].text).join('\n- '));
  assert.equal(idx.length, 5);
});
