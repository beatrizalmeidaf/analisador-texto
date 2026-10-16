/**
 * Relatório HTML autocontido da análise, pensado para leitura e impressão
 * (A4, retrato). Gerado no navegador, sem dependências externas.
 */

import type { AnalysisResult, Keyphrase, NerEntity, SentimentResult, Genre } from './workers/protocol.ts';
import { explainReadability } from './core/readability.ts';
import { GENRES } from './genres.ts';

export interface ReportData {
  text: string;
  result: AnalysisResult;
  summaryLines?: string[];
  keyphrases?: Keyphrase[];
  sentiment?: SentimentResult;
  genres?: Genre[];
  entities?: NerEntity[];
  cloudCanvas?: HTMLCanvasElement;
  /** cor de fundo com que a nuvem foi desenhada (o canvas é transparente) */
  cloudBackground?: string;
}

const ENTITY_LABELS: Record<string, string> = {
  PER: 'Pessoas',
  ORG: 'Organizações',
  LOC: 'Locais',
  MISC: 'Outras entidades',
};

// ---------------------------------------------------------------- formatação

function esc(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
const num = (n: number, digits = 0) =>
  n.toLocaleString('pt-BR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const pct = (n: number, digits = 1) => `${num(n * 100, digits)}%`;

function duration(min: number) {
  if (min < 1) return `${Math.max(1, Math.round(min * 60))} s`;
  if (min < 60) return `${Math.round(min)} min`;
  return `${Math.floor(min / 60)} h ${Math.round(min % 60)} min`;
}

/** Trecho inicial legível: junta letras espaçadas de PDFs ("A N E X O") e normaliza espaços. */
function excerpt(text: string, max = 700) {
  const clean = text
    .replace(/\b(?:\p{L} ){3,}\p{L}\b/gu, (m) => m.replace(/ /g, ''))
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  if (clean.length <= max) return clean;
  const cut = clean.slice(0, max);
  return `${cut.slice(0, cut.lastIndexOf(' ') > max * 0.8 ? cut.lastIndexOf(' ') : max)}…`;
}

/** Remove marcadores de lista ("»", "•", "-") do início de trechos citados. */
const unbullet = (t: string) => t.replace(/^[»•▪◦■\-–—]\s*/, '');

function meter(value: number) {
  return `<span class="meter" aria-hidden="true"><span style="width:${Math.max(0, Math.min(100, value * 100)).toFixed(1)}%"></span></span>`;
}

function table(head: string[], rows: string[][], numeric: number[] = []) {
  const cls = (i: number) => (numeric.includes(i) ? ' class="num"' : '');
  return `<table>
    <thead><tr>${head.map((h, i) => `<th${cls(i)}>${h}</th>`).join('')}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td${cls(i)}>${c}</td>`).join('')}</tr>`).join('')}</tbody>
  </table>`;
}

function cloudImage(canvas: HTMLCanvasElement | undefined, background: string | undefined) {
  if (!canvas || canvas.width < 10 || canvas.height < 10) return '';
  try {
    // o canvas da página é transparente: compõe sobre o fundo em que foi desenhado
    const out = document.createElement('canvas');
    out.width = canvas.width;
    out.height = canvas.height;
    const ctx = out.getContext('2d')!;
    ctx.fillStyle = background || '#ffffff';
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(canvas, 0, 0);
    return `<img src="${out.toDataURL('image/png')}" alt="Nuvem com as palavras mais frequentes do texto" />`;
  } catch {
    return '';
  }
}

// ---------------------------------------------------------------- relatório

export function generateHtmlReport(data: ReportData): string {
  const { result, text, summaryLines, keyphrases, sentiment, genres, entities } = data;
  const s = result.stats;
  const rd = s.readability;
  const sents = result.sentences;
  const generated = new Date();
  const dateStr = generated.toLocaleDateString('pt-BR', { day: '2-digit', month: 'long', year: 'numeric' });
  const timeStr = generated.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });

  // ---- síntese executiva: uma frase por achado principal
  const findings: string[] = [
    `Texto com ${num(s.words)} palavras, ${num(s.sentences)} frases e ${num(s.paragraphs)} parágrafos; tempo estimado de leitura de ${duration(s.readingMinutes)}.`,
    `Legibilidade <strong>${rd.band.label.toLowerCase()}</strong> (índice ${num(rd.flesch, 1)} de 100), indicada para ${rd.band.audience}${rd.reliable ? '' : ' — estimativa de baixa confiabilidade, pois o texto é curto'}.`,
  ];
  if (genres?.length) findings.push(`Gênero textual predominante: <strong>${esc(genres[0].label)}</strong> (${pct(genres[0].score, 0)} de afinidade).`);
  if (sentiment) {
    const d = sentiment.distribution;
    const main = [
      ['positivo', d.positive],
      ['neutro', d.neutral],
      ['negativo', d.negative],
    ].sort((a, b) => (b[1] as number) - (a[1] as number))[0];
    findings.push(`Tom predominantemente <strong>${main[0]}</strong> (${pct(main[1] as number, 0)} das frases analisadas, em média).`);
  }
  // expressões recorrentes descrevem o tema melhor que palavras soltas
  const themes = (result.ngrams.length >= 3 ? result.ngrams : result.frequencies).slice(0, 5).map((k) => esc(k.term));
  if (themes.length) findings.push(`Temas recorrentes: ${themes.join(', ')}.`);

  // ---- seções (só entram as análises executadas)
  const sections: { title: string; body: string }[] = [];

  sections.push({
    title: 'Visão geral',
    body: `
      <div class="kpis">
        ${[
          ['Palavras', num(s.words)],
          ['Palavras distintas', num(s.uniqueWords)],
          ['Frases', num(s.sentences)],
          ['Parágrafos', num(s.paragraphs)],
          ['Tempo de leitura', duration(s.readingMinutes)],
          ['Diversidade lexical', pct(s.lexicalDiversity)],
        ]
          .map(([k, v]) => `<div class="kpi"><div class="kpi-v">${v}</div><div class="kpi-k">${k}</div></div>`)
          .join('')}
      </div>
      ${table(
        ['Indicador', 'Valor'],
        [
          ['Caracteres (com espaços)', num(s.chars)],
          ['Palavras de conteúdo (sem stopwords)', num(s.contentWords)],
          ['Palavras que aparecem uma única vez', num(s.hapax)],
          ['Tamanho médio da palavra', `${num(s.wordLength.mean, 2)} letras`],
          ['Tempo de leitura em voz alta', duration(s.speakingMinutes)],
        ],
        [1],
      )}`,
  });

  const why = explainReadability(rd);
  sections.push({
    title: 'Legibilidade',
    body: `
      <p class="lead">Classificação: <strong>${rd.band.label}</strong> — índice ${num(rd.flesch, 1)} de 100 (quanto maior, mais fácil), adequado a ${rd.band.audience}.</p>
      <div class="scale" aria-hidden="true">
        ${['Muito difícil', 'Difícil', 'Fácil', 'Muito fácil']
          .map((b) => `<span class="${b === rd.band.label ? 'on' : ''}">${b}</span>`)
          .join('')}
        <i style="left:${Math.min(99, Math.max(1, rd.flesch)).toFixed(1)}%"></i>
      </div>
      ${!rd.reliable ? `<p class="caution">Estimativa de baixa confiabilidade: o índice é uma média e requer ao menos 5 frases e 100 palavras (o texto tem ${num(rd.sentencesUsed)} frases de prosa e ${num(rd.wordsUsed)} palavras).</p>` : ''}
      ${table(
        ['Fator', 'Valor no texto', 'Referência para leitura fácil'],
        [
          ['Palavras por frase', num(rd.wordsPerSentence, 1), 'até 20'],
          ['Sílabas por palavra', num(rd.syllablesPerWord, 2), 'cerca de 2,1'],
          ['Frases com mais de 30 palavras', pct(rd.longSentenceRatio), '—'],
          ['Palavras com 4 ou mais sílabas', pct(rd.polysyllableRatio), '—'],
        ],
        [1],
      )}
      ${why.length ? `<h3>Interpretação</h3>${why.map((w) => `<p>${esc(w)}</p>`).join('')}` : ''}
      ${
        rd.hardest.length
          ? `<h3>Frases de leitura mais difícil</h3>
             <ol class="quotes">${rd.hardest
               .map(
                 (h) =>
                   `<li><blockquote>${esc(unbullet(sents[h.sentence]?.text ?? ''))}</blockquote><div class="cap">${num(h.words)} palavras · ${num(h.syllablesPerWord, 2)} sílabas por palavra</div></li>`,
               )
               .join('')}</ol>`
          : ''
      }`,
  });

  const freq = result.frequencies.slice(0, 15);
  const kws = result.keywords.slice(0, 15);
  const grams = result.ngrams.slice(0, 10);
  sections.push({
    title: 'Vocabulário',
    body: `
      <div class="cols">
        <div>
          <h3>Palavras mais frequentes</h3>
          ${table(['#', 'Palavra', 'Ocorrências'], freq.map((t, i) => [String(i + 1), esc(t.term), num(t.count)]), [0, 2])}
        </div>
        <div>
          <h3>Termos característicos (TF-IDF)</h3>
          ${table(['#', 'Termo', 'Relevância'], kws.map((t, i) => [String(i + 1), esc(t.term), num(t.score, 2)]), [0, 2])}
        </div>
      </div>
      ${grams.length ? `<h3>Expressões recorrentes</h3>${table(['Expressão', 'Ocorrências'], grams.map((g) => [esc(g.term), num(g.count)]), [1])}` : ''}
      ${keyphrases?.length ? `<h3>Frases-chave</h3><p>${keyphrases.map((k) => esc(k.phrase)).join(' · ')}</p>` : ''}`,
  });

  if (summaryLines?.length) {
    sections.push({
      title: 'Resumo extrativo',
      body: `<p class="note">Frases do próprio texto selecionadas por centralidade (TextRank), na ordem em que aparecem.</p>
        <ol class="summary">${summaryLines.map((l) => `<li>${esc(l)}</li>`).join('')}</ol>`,
    });
  }

  if (genres?.length || sentiment) {
    let body = '';
    if (genres?.length) {
      const def = GENRES.find((g) => g.label === genres[0].label);
      body += `<h3>Gênero textual</h3>
        <p class="lead"><strong>${esc(genres[0].label)}</strong>${def ? ` — ${esc(def.description)}` : ''}</p>
        ${table(['Gênero', 'Afinidade', ''], genres.slice(0, 5).map((g) => [esc(g.label), pct(g.score), meter(g.score)]), [1])}`;
    }
    if (sentiment) {
      const d = sentiment.distribution;
      const quote = (i: number) => `<li><blockquote>${esc(unbullet(sents[i]?.text ?? ''))}</blockquote></li>`;
      body += `<h3>Sentimento</h3>
        ${table(
          ['Polaridade', 'Proporção média', ''],
          [
            ['Positivo', pct(d.positive), meter(d.positive)],
            ['Neutro', pct(d.neutral), meter(d.neutral)],
            ['Negativo', pct(d.negative), meter(d.negative)],
          ],
          [1],
        )}
        <p class="note">Média sobre ${num(sentiment.analyzed)} frases amostradas ao longo do texto.</p>
        ${sentiment.mostPositive.length ? `<h4>Trechos mais positivos</h4><ul class="quotes">${sentiment.mostPositive.map((p) => quote(p.sentence)).join('')}</ul>` : ''}
        ${sentiment.mostNegative.length ? `<h4>Trechos mais negativos</h4><ul class="quotes">${sentiment.mostNegative.map((p) => quote(p.sentence)).join('')}</ul>` : ''}`;
    }
    sections.push({ title: 'Classificação e sentimento', body });
  }

  if (entities?.length) {
    const byType = new Map<string, Map<string, number>>();
    for (const e of entities) {
      const m = byType.get(e.entity) ?? new Map<string, number>();
      m.set(e.word, (m.get(e.word) ?? 0) + 1);
      byType.set(e.entity, m);
    }
    sections.push({
      title: 'Entidades mencionadas',
      body: `${table(
        ['Tipo', 'Entidades'],
        [...byType].map(([type, m]) => [
          ENTITY_LABELS[type] ?? esc(type),
          [...m].sort((a, b) => b[1] - a[1]).map(([w, c]) => `${esc(w)}${c > 1 ? ` <span class="muted">(${c})</span>` : ''}`).join(', '),
        ]),
      )}
      <p class="note">Entidades reconhecidas automaticamente nos primeiros 2.000 caracteres do texto.</p>`,
    });
  }

  const cloud = cloudImage(data.cloudCanvas, data.cloudBackground);
  if (cloud) sections.push({ title: 'Nuvem de palavras', body: `<figure>${cloud}<figcaption>Tamanho proporcional à frequência de cada palavra de conteúdo.</figcaption></figure>` });

  sections.push({
    title: 'Trecho inicial do texto analisado',
    body: `<div class="excerpt">${esc(excerpt(text))}</div>`,
  });

  const notes = [
    'Legibilidade: índice de Flesch adaptado ao português (Martins et al., 1996): 248,835 − 1,015 × (palavras por frase) − 84,6 × (sílabas por palavra). Títulos, legendas e itens soltos são desconsiderados.',
    'Termos característicos: TF-IDF calculado entre os parágrafos do próprio texto, destacando termos concentrados em partes específicas.',
    'Resumo: seleção extrativa por TextRank com remoção de frases redundantes; nenhuma frase é reescrita.',
  ];
  if (genres?.length) notes.push('Gênero textual: classificação zero-shot por similaridade semântica (modelo paraphrase-multilingual-MiniLM-L12-v2); trata-se de uma estimativa.');
  if (sentiment) notes.push('Sentimento: modelo distilbert-base-multilingual-cased-sentiments-student aplicado frase a frase.');
  notes.push('Todo o processamento foi realizado localmente no navegador; o texto não foi enviado a servidores.');

  const toc = sections.map((sec, i) => `<li><a href="#s${i + 1}">${i + 1}. ${sec.title}</a></li>`).join('');

  return `<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Relatório de análise textual — ${dateStr}</title>
<style>
  :root { --ink:#1a1a1a; --ink2:#4a4a48; --muted:#77756f; --rule:#dcdad3; --soft:#f3f2ee; --accent:#1f4e79; }
  * { box-sizing: border-box; }
  html { background: #e9e8e4; }
  body { margin: 0; color: var(--ink); font: 10.5pt/1.55 Georgia, 'Times New Roman', serif; }
  .page { max-width: 210mm; margin: 24px auto; background: #fff; padding: 22mm 20mm; box-shadow: 0 1px 6px rgba(0,0,0,.12); }
  h1, h2, h3, h4, .kpi-v, th, .meta, .toc, .cap, figcaption, .note, .scale, footer {
    font-family: 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif;
  }
  header { border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 18px; }
  .kicker { font: 600 8pt/1 'Segoe UI', system-ui, sans-serif; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }
  h1 { font-size: 20pt; margin: 6px 0 4px; font-weight: 650; letter-spacing: -.01em; }
  .meta { color: var(--muted); font-size: 8.5pt; }
  h2 { font-size: 12.5pt; font-weight: 650; margin: 26px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--rule); color: var(--accent); }
  h3 { font-size: 10.5pt; font-weight: 650; margin: 16px 0 6px; }
  h4 { font-size: 9.5pt; font-weight: 600; margin: 12px 0 4px; color: var(--ink2); }
  p { margin: 0 0 8px; }
  .lead { font-size: 11pt; }
  .muted { color: var(--muted); }
  .note { font-size: 8.5pt; color: var(--muted); }
  .synthesis { background: var(--soft); border-left: 3px solid var(--accent); padding: 12px 16px; margin: 0 0 6px; }
  .synthesis h2 { margin-top: 0; border: 0; padding: 0; }
  .synthesis ul { margin: 0; padding-left: 18px; }
  .synthesis li { margin: 0 0 4px; }
  .toc { columns: 2; font-size: 9pt; margin: 10px 0 0; padding: 0; list-style: none; }
  .toc a { color: var(--ink2); text-decoration: none; }
  .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--rule); border: 1px solid var(--rule); margin: 4px 0 14px; }
  .kpi { background: #fff; padding: 10px 12px; }
  .kpi-v { font-size: 15pt; font-weight: 600; }
  .kpi-k { font: 8pt 'Segoe UI', system-ui, sans-serif; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  table { width: 100%; border-collapse: collapse; margin: 4px 0 10px; font-size: 9.5pt; }
  th { text-align: left; font-size: 8pt; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); border-bottom: 1px solid var(--ink); padding: 5px 8px 4px 0; }
  td { padding: 5px 8px 5px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
  .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .meter { display: inline-block; width: 120px; height: 7px; background: var(--soft); vertical-align: middle; }
  .meter span { display: block; height: 100%; background: var(--accent); }
  .scale { position: relative; display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; margin: 6px 0 22px; font-size: 7.5pt; text-align: center; }
  .scale span { background: var(--soft); color: var(--muted); padding: 4px 2px; }
  .scale span.on { background: var(--accent); color: #fff; font-weight: 600; }
  .scale i { position: absolute; top: calc(100% + 3px); transform: translateX(-50%); border: 5px solid transparent; border-bottom-color: var(--ink); border-top: 0; }
  .caution { border: 1px solid var(--rule); border-left: 3px solid #b5552b; padding: 8px 12px; font-size: 9pt; }
  blockquote { margin: 0; padding: 0 0 0 12px; border-left: 2px solid var(--rule); font-style: italic; color: var(--ink2); }
  .quotes { padding-left: 18px; margin: 4px 0 10px; }
  .quotes li { margin: 0 0 10px; }
  .cap { font-size: 8pt; color: var(--muted); margin: 3px 0 0 14px; }
  .summary { padding-left: 20px; }
  .summary li { margin: 0 0 8px; }
  figure { margin: 6px 0; }
  figure img { width: 100%; max-height: 110mm; object-fit: contain; border: 1px solid var(--rule); }
  figcaption { font-size: 8pt; color: var(--muted); margin-top: 4px; }
  .excerpt { white-space: pre-wrap; font-size: 9pt; color: var(--ink2); border: 1px solid var(--rule); padding: 10px 12px; background: #fcfcfa; }
  footer { margin-top: 28px; padding-top: 10px; border-top: 1px solid var(--rule); font-size: 8pt; color: var(--muted); }
  footer ol { padding-left: 16px; margin: 6px 0; }
  footer li { margin-bottom: 3px; }
  @media (max-width: 700px) {
    .page { padding: 20px 16px; margin: 0; }
    .cols { grid-template-columns: 1fr; }
    .kpis { grid-template-columns: repeat(2, 1fr); }
    .toc { columns: 1; }
  }
  @page { size: A4; margin: 16mm 15mm; }
  @media print {
    html { background: #fff; }
    .page { box-shadow: none; margin: 0; padding: 0; max-width: none; }
    h2, h3, h4 { break-after: avoid; }
    table, figure, blockquote, .kpis, .scale, .synthesis { break-inside: avoid; }
    a { color: inherit; text-decoration: none; }
  }
</style>
</head>
<body>
<main class="page">
  <header>
    <div class="kicker">Analisador de Texto</div>
    <h1>Relatório de análise textual</h1>
    <div class="meta">Emitido em ${dateStr}, às ${timeStr} · ${num(s.words)} palavras analisadas</div>
  </header>

  <section class="synthesis">
    <h2>Síntese</h2>
    <ul>${findings.map((f) => `<li>${f}</li>`).join('')}</ul>
    <ol class="toc">${toc}</ol>
  </section>

  ${sections.map((sec, i) => `<section id="s${i + 1}"><h2>${i + 1}. ${sec.title}</h2>${sec.body}</section>`).join('\n')}

  <footer>
    <strong>Notas metodológicas</strong>
    <ol>${notes.map((n) => `<li>${n}</li>`).join('')}</ol>
    <div>Gerado por Analisador de Texto · https://beatrizalmeidaf.github.io/analisador-texto/</div>
  </footer>
</main>
</body>
</html>`;
}

export function downloadHtml(html: string, filename = `relatorio-analise-${new Date().toISOString().slice(0, 10)}.html`) {
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
