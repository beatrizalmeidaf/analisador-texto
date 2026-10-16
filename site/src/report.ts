/**
 * Gerador de relatório HTML para o Analisador de Texto (versão web).
 * Roda no main thread — sem dependências externas.
 */

import type { AnalysisResult, Keyphrase, NerEntity, SentimentResult } from './workers/protocol.ts';
import type { Genre } from './workers/protocol.ts';

export interface ReportData {
  text: string;
  result: AnalysisResult;
  summaryLines?: string[];
  keyphrases?: Keyphrase[];
  sentiment?: SentimentResult;
  genres?: Genre[];
  entities?: NerEntity[];
  cloudCanvas?: HTMLCanvasElement;
}

const ENTITY_COLORS: Record<string, string> = {
  PER:  '#6366f1',
  ORG:  '#0891b2',
  LOC:  '#16a34a',
  MISC: '#d97706',
};
const ENTITY_LABELS: Record<string, string> = {
  PER:  'Pessoa',
  ORG:  'Organização',
  LOC:  'Local',
  MISC: 'Misc',
};

function esc(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function badge(entity: string) {
  const color = ENTITY_COLORS[entity] ?? '#6b7280';
  const label = ENTITY_LABELS[entity] ?? entity;
  return `<span style="display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.75rem;
    font-weight:600;background:${color}22;border:1px solid ${color};color:${color};
    margin:2px;">${label}</span>`;
}

function bar(value: number, color: string, label: string) {
  const w = Math.round(value * 100);
  return `<div style="margin:4px 0;">
    <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#64748b;margin-bottom:2px;">
      <span>${label}</span><span>${w}%</span>
    </div>
    <div style="background:#f1f5f9;border-radius:4px;height:8px;overflow:hidden;">
      <div style="width:${w}%;height:100%;background:${color};border-radius:4px;"></div>
    </div>
  </div>`;
}

export function generateHtmlReport(data: ReportData): string {
  const { result, text, summaryLines, keyphrases, sentiment, genres, entities, cloudCanvas } = data;
  const s = result.stats;
  const now = new Date().toLocaleString('pt-BR');
  const preview = esc(text.slice(0, 600)) + (text.length > 600 ? '…' : '');

  // Cloud image
  let cloudImg = '';
  try {
    if (cloudCanvas) cloudImg = `<img src="${cloudCanvas.toDataURL()}" style="max-width:100%;border-radius:8px;" alt="Nuvem de palavras"/>`;
  } catch { /* cross-origin or no canvas */ }

  // Top words table
  const topWords = result.frequencies.slice(0, 20);
  const topWordsRows = topWords.map((t) =>
    `<tr><td>${esc(t.term)}</td><td style="text-align:right;font-variant-numeric:tabular-nums;">${t.count}</td></tr>`
  ).join('');

  // Keywords table
  const topKw = result.keywords.slice(0, 15);
  const kwRows = topKw.map((t) =>
    `<tr><td>${esc(t.term)}</td><td style="text-align:right;font-variant-numeric:tabular-nums;">${t.score.toFixed(3)}</td></tr>`
  ).join('');

  // Summary
  const summaryHtml = summaryLines?.length
    ? `<ol style="padding-left:1.4em;margin:0;">${summaryLines.map((l) => `<li style="margin:0 0 8px;line-height:1.6;">${esc(l)}</li>`).join('')}</ol>`
    : '<p style="color:#94a3b8;"><em>Resumo não gerado nesta sessão.</em></p>';

  // Keyphrases
  const kpHtml = keyphrases?.length
    ? keyphrases.map((k) =>
        `<span style="display:inline-block;margin:4px;padding:5px 12px;border-radius:20px;
          background:#6366f122;border:1px solid #6366f1;font-size:0.88rem;">
          ${esc(k.phrase)}
          <em style="color:#6366f1;margin-left:4px;">(${k.score.toFixed(3)})</em>
        </span>`
      ).join('')
    : '<p style="color:#94a3b8;"><em>Frases-chave não extraídas nesta sessão.</em></p>';

  // Sentiment
  let sentHtml = '<p style="color:#94a3b8;"><em>Análise de sentimento não realizada nesta sessão.</em></p>';
  if (sentiment) {
    const d = sentiment.distribution;
    sentHtml = `
      ${bar(d.positive, '#22c55e', '😊 Positivo')}
      ${bar(d.neutral,  '#94a3b8', '😐 Neutro')}
      ${bar(d.negative, '#ef4444', '😞 Negativo')}
      <p style="font-size:0.8rem;color:#64748b;margin-top:6px;">
        Média sobre ${sentiment.analyzed} frases analisadas.
      </p>`;
  }

  // Genres
  let genreHtml = '<p style="color:#94a3b8;"><em>Gênero não classificado nesta sessão.</em></p>';
  if (genres?.length) {
    const top = genres[0];
    genreHtml = `<p style="font-size:1.1rem;font-weight:700;color:#4f46e5;margin-bottom:8px;">${esc(top.label)}</p>
      ${genres.slice(0, 5).map((g) => bar(g.score, '#4f46e5', esc(g.label))).join('')}`;
  }

  // NER
  let nerHtml = '<p style="color:#94a3b8;"><em>Entidades não extraídas nesta sessão.</em></p>';
  if (entities?.length) {
    const byType: Record<string, string[]> = {};
    for (const e of entities) {
      (byType[e.entity] ??= []).push(e.word);
    }
    nerHtml = Object.entries(byType).map(([type, words]) =>
      `<div style="margin-bottom:12px;">
        <div style="font-size:0.8rem;font-weight:600;color:#64748b;margin-bottom:6px;">
          ${badge(type)} ${ENTITY_LABELS[type] ?? type}
        </div>
        <div>${[...new Set(words)].map((w) => `<span style="display:inline-block;margin:2px 4px;padding:3px 10px;background:#f1f5f9;border-radius:6px;font-size:0.88rem;">${esc(w)}</span>`).join('')}</div>
      </div>`
    ).join('');
  }

  return `<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Relatório — Analisador de Texto</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',system-ui,sans-serif;background:#f8fafc;color:#1e293b;padding:32px 20px;}
    h1{font-size:1.8rem;color:#4f46e5;margin-bottom:4px;}
    h2{font-size:1.15rem;color:#4f46e5;margin:28px 0 10px;border-bottom:2px solid #e2e8f0;padding-bottom:6px;}
    .meta{color:#64748b;font-size:0.85rem;margin-bottom:28px;}
    .card{background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 4px #0001;}
    .metrics{display:flex;flex-wrap:wrap;gap:12px;}
    .metric{background:#f1f5f9;border-radius:8px;padding:14px 18px;text-align:center;flex:1;min-width:110px;}
    .metric .val{font-size:1.6rem;font-weight:700;color:#4f46e5;}
    .metric .lbl{font-size:0.75rem;color:#64748b;margin-top:2px;}
    table{width:100%;border-collapse:collapse;font-size:0.88rem;}
    th{background:#f1f5f9;padding:7px 10px;text-align:left;font-weight:600;}
    td{padding:6px 10px;border-bottom:1px solid #f1f5f9;}
    .preview{background:#f8fafc;border-left:4px solid #6366f1;padding:10px 14px;
             border-radius:0 8px 8px 0;font-size:0.88rem;color:#475569;white-space:pre-wrap;word-break:break-word;}
    footer{text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:32px;}
  </style>
</head>
<body>
  <h1>📊 Relatório de Análise de Texto</h1>
  <p class="meta">Gerado em ${now} · Analisador de Texto (versão web)</p>

  <div class="card">
    <h2>📄 Prévia do Texto</h2>
    <p class="preview">${preview}</p>
  </div>

  <div class="card">
    <h2>📏 Métricas</h2>
    <div class="metrics">
      <div class="metric"><div class="val">${s.words.toLocaleString('pt-BR')}</div><div class="lbl">Palavras</div></div>
      <div class="metric"><div class="val">${s.uniqueWords.toLocaleString('pt-BR')}</div><div class="lbl">Distintas</div></div>
      <div class="metric"><div class="val">${s.sentences.toLocaleString('pt-BR')}</div><div class="lbl">Frases</div></div>
      <div class="metric"><div class="val">${s.paragraphs.toLocaleString('pt-BR')}</div><div class="lbl">Parágrafos</div></div>
      <div class="metric"><div class="val">${s.flesch.toFixed(1)}</div><div class="lbl">Legibilidade (Flesch) · ${s.readability.band.label}</div></div>
      <div class="metric"><div class="val">${(s.lexicalDiversity * 100).toFixed(1)}%</div><div class="lbl">Diversidade lexical</div></div>
    </div>
  </div>

  <div class="card">
    <h2>🏷️ Gênero Textual</h2>
    ${genreHtml}
  </div>

  <div class="card">
    <h2>💬 Sentimento</h2>
    ${sentHtml}
  </div>

  <div class="card">
    <h2>🔑 Frases-Chave</h2>
    ${kpHtml}
  </div>

  <div class="card">
    <h2>🧩 Entidades Nomeadas (NER)</h2>
    ${nerHtml}
  </div>

  <div class="card">
    <h2>📝 Resumo Automático</h2>
    ${summaryHtml}
  </div>

  <div class="card">
    <h2>📊 Top 20 Palavras Mais Frequentes</h2>
    <table>
      <thead><tr><th>Palavra</th><th style="text-align:right;">Ocorrências</th></tr></thead>
      <tbody>${topWordsRows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>🔎 Top 15 Termos TF-IDF</h2>
    <table>
      <thead><tr><th>Termo</th><th style="text-align:right;">TF-IDF</th></tr></thead>
      <tbody>${kwRows}</tbody>
    </table>
  </div>

  ${cloudImg ? `<div class="card"><h2>☁️ Nuvem de Palavras</h2>${cloudImg}</div>` : ''}

  <footer>Gerado pelo Analisador de Texto · beatrizalmeidaf.github.io/analisador-texto</footer>
</body>
</html>`;
}

export function downloadHtml(html: string, filename = 'relatorio-analise.html') {
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
