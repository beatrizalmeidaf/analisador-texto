import { test } from 'node:test';
import assert from 'node:assert/strict';
import { syllables, readability, explainReadability } from '../src/core/readability.ts';
import { sentences } from '../src/core/text.ts';

test('divisão silábica: ditongos, hiatos, qu/gu e nasais', () => {
  const known: Record<string, number> = {
    casa: 2, saúde: 3, pão: 1, linguagem: 3, país: 2, raiz: 2, rainha: 3, muito: 2, poeta: 3, voo: 2,
    dia: 2, quatro: 2, aqui: 2, guerra: 2, leão: 2, coração: 3, ruim: 2, caixa: 2, europa: 3, saída: 3,
    piano: 3, teatro: 3, frequente: 3, quando: 2, água: 2, iguais: 2, ideia: 3, eleições: 3, construção: 3,
    processamento: 5, linguística: 4, queijo: 2, juiz: 2, baú: 2, moeda: 3, oito: 2, sereia: 3, feira: 2,
    herói: 2, também: 2, ninguém: 2, mãe: 1, irmãos: 2, chapéu: 2, ainda: 3, cuidado: 3, paulo: 2, raul: 2,
    saudade: 3, cair: 2, mais: 1, saia: 2,
  };
  for (const [w, n] of Object.entries(known)) assert.equal(syllables(w), n, w);
});

const INFANTIL = 'O menino tem um cachorro. O cachorro se chama Bob. Bob gosta de correr no parque. Todo dia eles vão ao parque. O menino joga a bola e Bob pega. Depois eles voltam para casa. A mãe faz um lanche. Bob dorme no tapete. O menino lê um livro.';
const NOTICIA = 'A prefeitura anunciou nesta segunda-feira a ampliação do horário de funcionamento dos postos de saúde da cidade. Segundo a secretaria, a medida deve atender cerca de dez mil pessoas por semana. As unidades vão abrir às sete horas e fechar às vinte e duas horas. A mudança começa no próximo mês e vale para todos os bairros. Os moradores podem consultar a lista de endereços no site oficial da prefeitura. Quem tiver dúvidas pode ligar para a central de atendimento.';
const JURIDICO = 'Considerando a necessidade de regulamentação das disposições constitucionais concernentes à administração pública, fica estabelecido que os procedimentos administrativos instaurados no âmbito da administração direta, autárquica e fundacional observarão, obrigatoriamente, os princípios da legalidade, impessoalidade, moralidade, publicidade e eficiência, sem prejuízo das demais disposições legais e regulamentares aplicáveis à espécie, ressalvadas as hipóteses expressamente previstas na legislação específica.';

test('ordena textos por dificuldade e acerta as faixas', () => {
  const inf = readability(sentences(INFANTIL));
  const not = readability(sentences(NOTICIA));
  const jur = readability(sentences(JURIDICO));
  assert.ok(inf.fleschRaw > not.fleschRaw && not.fleschRaw > jur.fleschRaw);
  assert.equal(inf.band.label, 'Muito fácil');
  assert.ok(['Fácil', 'Difícil'].includes(not.band.label), not.band.label);
  assert.equal(jur.band.label, 'Muito difícil');
  assert.equal(jur.reliable, false); // uma frase só
  const why = explainReadability(jur).join(' ');
  assert.match(why, /vocabulário/);
  assert.match(why, /frases também são longas/);
});

test('títulos e legendas não entram na média de palavras por frase', () => {
  const corpo = NOTICIA;
  const comTitulos = '1 Introdução\n\n1.1 Objetivos\n\nFigura 1 - Arquitetura do sistema\n\n' + corpo;
  const a = readability(sentences(corpo));
  const b = readability(sentences(comTitulos));
  assert.equal(b.sentencesUsed, a.sentencesUsed);
  assert.equal(b.flesch, a.flesch);
});

test('frases mais difíceis apontam a frase longa', () => {
  const r = readability(sentences(NOTICIA + ' ' + JURIDICO));
  assert.equal(r.hardest.length >= 1, true);
  assert.match(sentences(NOTICIA + ' ' + JURIDICO)[r.hardest[0].sentence].text, /^Considerando/);
});
