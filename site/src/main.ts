import './style.css';
import WordCloud from 'wordcloud';
import AnalysisWorker from './workers/analysis.worker.ts?worker';
import type { AnalysisRequest, AnalysisResponse, AnalysisResult, MLRequest, MLResponse } from './workers/protocol.ts';
import type { Term } from './core/analyze.ts';
import { explainReadability, type Readability } from './core/readability.ts';
import { GENRES } from './genres.ts';
import { $, h, fmt, pct, duration, hbars, columns, stack, progress, highlighted, cssVar } from './ui.ts';
import { generateHtmlReport, downloadHtml } from './report.ts';
import type { Keyphrase, NerEntity, SentimentResult, Genre } from './workers/protocol.ts';

// ---------------------------------------------------------------- workers

/** Omit que distribui sobre uniões (o Omit padrão achata a união). */
type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

type Pending<R> = { resolve: (r: R) => void; reject: (e: Error) => void; onProgress?: (label: string, p: number) => void };

/** RPC simples sobre postMessage: cada pedido recebe um id e uma Promise. */
function rpc<Req extends { id: number }, Res extends { id: number; type: string }>(worker: Worker) {
  let seq = 0;
  const pending = new Map<number, Pending<Res>>();
  worker.onmessage = (e: MessageEvent<Res>) => {
    const m = e.data as Res & { label?: string; progress?: number; message?: string };
    const p = pending.get(m.id);
    if (!p) return;
    if (m.type === 'progress') return p.onProgress?.(m.label!, m.progress!);
    pending.delete(m.id);
    if (m.type === 'error') p.reject(new Error(m.message));
    else p.resolve(m);
  };
  return (msg: DistOmit<Req, 'id'>, onProgress?: Pending<Res>['onProgress']) =>
    new Promise<Res>((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ ...msg, id });
    });
}

const analysis = rpc<AnalysisRequest, AnalysisResponse>(new AnalysisWorker());

// o worker de IA (e o bundle do Transformers.js) só é carregado quando necessário
let mlCall: ReturnType<typeof rpc<MLRequest, MLResponse>> | null = null;
const ml: ReturnType<typeof rpc<MLRequest, MLResponse>> = (msg, onProgress) => {
  mlCall ??= rpc<MLRequest, MLResponse>(new Worker(new URL('./workers/ml.worker.ts', import.meta.url), { type: 'module' }));
  return mlCall(msg, onProgress);
};

// ---------------------------------------------------------------- estado

type Tab = 'stats' | 'words' | 'cloud' | 'summary' | 'search' | 'ai' | 'keyphrases' | 'ner' | 'export';

const state = {
  text: '',
  key: '',
  result: null as AnalysisResult | null,
  tab: 'stats' as Tab,
  dirty: new Set<Tab>(),
  wordsMode: 'freq' as 'freq' | 'tfidf' | 'ngrams',
  summaryMode: 'stat' as 'stat' | 'ai',
  searchMode: 'exact' as 'exact' | 'ai',
  indexedKey: '',
  // cached results for report
  lastSummaryLines: [] as string[],
  lastKeyphrases: null as Keyphrase[] | null,
  lastSentiment: null as SentimentResult | null,
  lastGenres: null as Genre[] | null,
  lastEntities: null as NerEntity[] | null,
};

const store = {
  get(k: string) {
    try {
      return localStorage.getItem(`analisador:${k}`);
    } catch {
      return null;
    }
  },
  set(k: string, v: string) {
    try {
      localStorage.setItem(`analisador:${k}`, v);
    } catch {
      /* armazenamento indisponível ou cheio: segue sem persistir */
    }
  },
};

function hashText(s: string) {
  let h1 = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h1 ^= s.charCodeAt(i);
    h1 = Math.imul(h1, 0x01000193);
  }
  return `${s.length}-${(h1 >>> 0).toString(36)}`;
}

// ---------------------------------------------------------------- entrada

const input = $<HTMLTextAreaElement>('#input');
const status = $('#status');

let timer = 0;
let latestAnalyze = 0;

function schedule() {
  clearTimeout(timer);
  const n = input.value.length;
  // textos maiores levam mais tempo para analisar: espera o usuário parar de digitar
  const delay = n < 20_000 ? 120 : n < 300_000 ? 350 : 800;
  timer = window.setTimeout(runAnalysis, delay);
}

async function runAnalysis() {
  const text = input.value;
  state.text = text;
  const ticket = ++latestAnalyze;
  if (!text.trim()) {
    state.result = null;
    state.key = '';
    status.textContent = 'Aguardando texto…';
    renderAll();
    store.set('text', '');
    return;
  }
  if (text.length > 300_000) status.textContent = 'Analisando…';
  const res = await analysis({ type: 'analyze', text });
  if (ticket !== latestAnalyze || res.type !== 'analysis') return;
  state.result = res.result;
  state.key = hashText(text);
  status.textContent = `${fmt(res.result.stats.words)} palavras · analisado em ${fmt(res.ms)} ms`;
  renderAll();
  scheduleBackgroundIndex();
  if (text.length < 2_000_000) store.set('text', text);
}

input.addEventListener('input', schedule);

function setText(text: string) {
  input.value = text;
  input.scrollTop = 0;
  clearTimeout(timer);
  runAnalysis();
}

async function decodeText(buf: ArrayBuffer) {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(buf);
  } catch {
    return new TextDecoder('windows-1252').decode(buf);
  }
}

async function loadFile(file: File) {
  try {
    const buf = await file.arrayBuffer();
    if (file.type === 'application/pdf' || /\.pdf$/i.test(file.name)) {
      status.textContent = `Lendo ${file.name}…`;
      const { extractPdf } = await import('./pdf.ts');
      const text = await extractPdf(buf, (p, total) => (status.textContent = `Extraindo texto do PDF: página ${p} de ${total}`));
      if (!text.trim()) {
        status.textContent = 'Nenhum texto encontrado no PDF (pode ser um documento escaneado, só com imagens).';
        return;
      }
      setText(text);
    } else {
      setText(await decodeText(buf));
    }
  } catch (err) {
    status.textContent = `Não foi possível ler o arquivo: ${err instanceof Error ? err.message : err}`;
  }
}

$<HTMLInputElement>('#file').addEventListener('change', (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) loadFile(f);
  (e.target as HTMLInputElement).value = '';
});

$('#sample').addEventListener('click', async () => {
  status.textContent = 'Carregando exemplo…';
  const res = await fetch(`${import.meta.env.BASE_URL}exemplo.txt`);
  setText(await res.text());
});

$('#clear').addEventListener('click', () => {
  setText('');
  input.focus();
});

const drop = $('#drop');
const editor = $('.editor');
let dragDepth = 0;
editor.addEventListener('dragenter', (e) => {
  if (!e.dataTransfer?.types.includes('Files')) return;
  e.preventDefault();
  dragDepth++;
  drop.hidden = false;
});
editor.addEventListener('dragover', (e) => e.preventDefault());
editor.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    drop.hidden = true;
  }
});
editor.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  drop.hidden = true;
  const f = e.dataTransfer?.files[0];
  if (f) loadFile(f);
});

// ---------------------------------------------------------------- abas e controles

const tabs = document.querySelectorAll<HTMLButtonElement>('[role=tab]');
function selectTab(tab: Tab) {
  state.tab = tab;
  tabs.forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  document.querySelectorAll<HTMLElement>('[data-panel]').forEach((p) => (p.hidden = p.dataset.panel !== tab));
  if (state.dirty.has(tab)) renderPanel(tab);
}
tabs.forEach((b) => b.addEventListener('click', () => selectTab(b.dataset.tab as Tab)));

function segmented<T extends string>(sel: string, onChange: (v: T) => void) {
  const root = $(sel);
  root.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('button');
    if (!btn) return;
    root.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
    onChange(btn.dataset.v as T);
  });
}

segmented<typeof state.wordsMode>('#words-mode', (v) => {
  state.wordsMode = v;
  renderWords();
});
segmented<typeof state.summaryMode>('#summary-mode', (v) => {
  state.summaryMode = v;
  renderSummary();
});
segmented<typeof state.searchMode>('#search-mode', (v) => {
  state.searchMode = v;
  $('#whole-wrap').hidden = v === 'ai';
  $<HTMLInputElement>('#query').placeholder = v === 'ai' ? 'Faça uma pergunta ou descreva o que procura' : 'Busque uma palavra ou expressão';
  $('#search-out').replaceChildren();
  renderSearch();
});

function renderAll() {
  renderTiles();
  state.dirty = new Set(['stats', 'words', 'cloud', 'summary', 'search', 'ai', 'keyphrases', 'ner', 'export']);
  renderPanel(state.tab);
}

function renderPanel(tab: Tab) {
  state.dirty.delete(tab);
  ({ stats: renderStats, words: renderWords, cloud: renderCloud, summary: renderSummary, search: renderSearch, ai: renderAI, keyphrases: renderKeyphrases, ner: renderNer, export: renderExport })[tab]();
}

const empty = () => h('p', { class: 'empty' }, 'Digite ou carregue um texto para ver a análise.');

// ---------------------------------------------------------------- blocos de resultado

function renderTiles() {
  const r = state.result;
  const tiles = $('#tiles');
  if (!r) return tiles.replaceChildren();
  const s = r.stats;
  const tile = (k: string, v: string, sub: string) => h('div', { class: 'tile' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 's' }, sub));
  tiles.replaceChildren(
    tile('Palavras', fmt(s.words), `${fmt(s.uniqueWords)} distintas`),
    tile('Frases', fmt(s.sentences), `${fmt(s.paragraphs)} parágrafos`),
    tile('Leitura', duration(s.readingMinutes), `${duration(s.speakingMinutes)} em voz alta`),
    tile('Legibilidade', s.readability.band.label, `índice ${fmt(s.flesch)} de 100${s.readability.reliable ? '' : ' · estimativa'}`),
    tile('Diversidade lexical', pct(s.lexicalDiversity), `${fmt(s.hapax)} palavras únicas`),
  );
}

function renderStats() {
  const body = $('#stats-body');
  const r = state.result;
  if (!r) return body.replaceChildren(empty());
  const s = r.stats;
  const item = (k: string, v: string) => h('div', {}, h('dt', {}, k), h('dd', {}, v));
  const hist = s.lengthHistogram.slice(1);
  const totalWords = s.words || 1;
  body.replaceChildren(
    h(
      'dl',
      { class: 'stat-grid' },
      item('Caracteres', fmt(s.chars)),
      item('Caracteres sem espaços', fmt(s.charsNoSpaces)),
      item('Palavras de conteúdo', fmt(s.contentWords)),
      item('Tamanho médio da palavra', `${fmt(s.wordLength.mean, 2)} letras`),
      item('Mediana do tamanho', `${fmt(s.wordLength.median, 1)} letras`),
      item('Desvio padrão do tamanho', fmt(s.wordLength.std, 2)),
    ),
    readabilityBlock(s.readability, r.sentences),
    h('div', { class: 'section-title' }, 'Distribuição do tamanho das palavras'),
    columns(
      hist,
      hist.map((_, i) => (i === hist.length - 1 ? `${i + 1}+` : String(i + 1))),
      (i) => `${i === hist.length - 1 ? `${i + 1} ou mais` : i + 1} letras: ${fmt(hist[i])} palavras (${pct(hist[i] / totalWords)})`,
    ),
    h('p', { class: 'muted small' }, 'Número de letras por palavra (eixo horizontal) e quantidade de palavras com aquele tamanho.'),
  );
}

function readabilityBlock(rd: Readability, sentences: { text: string }[]) {
  const bands = ['Muito difícil', 'Difícil', 'Fácil', 'Muito fácil'];
  const current = Math.min(3, Math.floor(rd.flesch / 25));
  return h(
    'section',
    { class: 'readability' },
    h('div', { class: 'section-title' }, `Legibilidade: ${rd.band.label.toLowerCase()}`),
    h('p', { class: 'read-lead' }, `Índice ${fmt(rd.flesch)} de 100 — quanto maior, mais fácil. Indicado para ${rd.band.audience}.`),
    h(
      'div',
      { class: 'bands', role: 'meter', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': Math.round(rd.flesch), 'aria-valuetext': `${fmt(rd.flesch)}, ${rd.band.label}` },
      ...bands.map((b, i) => h('span', { class: i === current ? 'on' : '' }, b)),
      h('i', { class: 'marker', style: `left:${Math.min(99, Math.max(1, rd.flesch))}%` }),
    ),
    !rd.reliable
      ? h('p', { class: 'warn small' }, `Estimativa pouco confiável: o índice é uma média e precisa de pelo menos 5 frases e 100 palavras (este texto tem ${fmt(rd.sentencesUsed)} frase${rd.sentencesUsed === 1 ? '' : 's'} e ${fmt(rd.wordsUsed)} palavras).`)
      : null,
    h(
      'dl',
      { class: 'stat-grid' },
      ...[
        ['Palavras por frase', fmt(rd.wordsPerSentence, 1), 'até 20 em textos fáceis'],
        ['Sílabas por palavra', fmt(rd.syllablesPerWord, 2), 'perto de 2,1 em textos fáceis'],
        ['Frases com mais de 30 palavras', pct(rd.longSentenceRatio), ''],
        ['Palavras com 4+ sílabas', pct(rd.polysyllableRatio), ''],
      ].map(([k, v, hint]) => h('div', {}, h('dt', {}, k), h('dd', {}, v), hint ? h('dd', { class: 'hint' }, hint) : null)),
    ),
    h('div', { class: 'note' }, ...explainReadability(rd).map((t) => h('p', {}, t))),
    rd.hardest.length ? h('div', { class: 'section-title' }, 'Frases mais difíceis de ler') : null,
    ...rd.hardest.map((hs) =>
      h(
        'div',
        { class: 'quote neg' },
        sentences[hs.sentence]?.text ?? '',
        h('div', { class: 'muted small' }, `${fmt(hs.words)} palavras · ${fmt(hs.syllablesPerWord, 2)} sílabas por palavra · índice ${fmt(Math.max(0, hs.score))}`),
      ),
    ),
    h(
      'p',
      { class: 'muted small' },
      `Índice de Flesch adaptado ao português (Martins et al., 1996): 248,8 − 1,015 × palavras por frase − 84,6 × sílabas por palavra. Calculado sobre ${fmt(rd.sentencesUsed)} frases de prosa; títulos, legendas e itens soltos ficam de fora.`,
    ),
  );
}

const WORDS_HELP = {
  freq: 'Palavras de conteúdo mais repetidas (sem artigos, preposições e outras stopwords).',
  tfidf: 'TF-IDF calculado entre os parágrafos (ou trechos) do texto: destaca termos que caracterizam partes específicas, e não os que aparecem em todo lugar. Pontuação relativa ao primeiro termo.',
  ngrams: 'Sequências de 2 ou 3 palavras que se repetem pelo menos duas vezes.',
};

function currentTerms(): Term[] {
  const r = state.result!;
  return state.wordsMode === 'freq' ? r.frequencies : state.wordsMode === 'tfidf' ? r.keywords : r.ngrams;
}

function renderWords() {
  const chart = $('#words-chart');
  $('#words-help').textContent = WORDS_HELP[state.wordsMode];
  if (!state.result) return chart.replaceChildren(empty());
  const terms = currentTerms().slice(0, 25);
  if (!terms.length) return chart.replaceChildren(h('p', { class: 'empty' }, 'Nada a mostrar para este texto.'));
  const content = state.result.stats.contentWords || 1;
  chart.replaceChildren(
    hbars(
      terms.map((t) => ({
        label: t.term,
        value: t.score,
        display: state.wordsMode === 'tfidf' ? fmt(t.score, 2) : fmt(t.count),
        tooltip:
          state.wordsMode === 'tfidf'
            ? `${t.term}: relevância ${fmt(t.score, 2)} · ${fmt(t.count)} ocorrências`
            : `${t.term}: ${fmt(t.count)} ocorrências${state.wordsMode === 'freq' ? ` (${pct(t.count / content)} das palavras de conteúdo)` : ''}`,
      })),
      searchFor,
    ),
  );
}

$('#export-csv').addEventListener('click', () => {
  if (!state.result) return;
  const rows = [['termo', 'ocorrencias', 'pontuacao'], ...currentTerms().map((t) => [t.term, String(t.count), t.score.toFixed(4)])];
  const csv = rows.map((r) => r.map((c) => `"${c.replace(/"/g, '""')}"`).join(',')).join('\n');
  const a = h('a', { href: URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv' })), download: `palavras-${state.wordsMode}.csv` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

function renderCloud() {
  const canvas = $<HTMLCanvasElement>('#cloud');
  const ctx = canvas.getContext('2d')!;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const words = state.result?.frequencies.slice(0, 120) ?? [];
  if (!words.length || !width) return;
  const max = words[0].count;
  const maxSize = Math.min(canvas.width / 13, canvas.height / 5);
  const minSize = 11 * dpr;
  const ramp = cssVar('--cloud-ramp').split(',').map((c) => c.trim());
  WordCloud(canvas, {
    list: words.map((w): WordCloud.ListEntry => [w.term, minSize + (maxSize - minSize) * Math.sqrt(w.count / max)]),
    weightFactor: 1,
    fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
    fontWeight: '600',
    // tom único do mais claro (raro) ao mais escuro (frequente) — escala sequencial
    color: (_word: string, weight: string | number) => {
      const t = (Number(weight) - minSize) / (maxSize - minSize || 1);
      const dark = document.documentElement.dataset.theme === 'dark' || (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
      const i = Math.round((dark ? 1 - t : t) * (ramp.length - 1));
      return ramp[Math.max(0, Math.min(ramp.length - 1, i))];
    },
    backgroundColor: 'transparent',
    rotateRatio: 0.15,
    rotationSteps: 2,
    shrinkToFit: true,
    drawOutOfBound: false,
    gridSize: Math.round(6 * dpr),
    click: (item) => searchFor(String(item[0])),
    hover: (item) => {
      canvas.style.cursor = item ? 'pointer' : 'default';
    },
  });
}

let resizeTimer = 0;
new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => state.tab === 'cloud' && state.result && renderCloud(), 200);
}).observe($('.cloud-wrap'));

// ---------------------------------------------------------------- resumo

const summaryK = $<HTMLInputElement>('#summary-k');
summaryK.addEventListener('input', () => {
  $('#summary-k-out').textContent = summaryK.value;
  renderSummary();
});

let summaryTicket = 0;
let lastSummary: string[] = [];

/**
 * Indexação (embeddings das frases) compartilhada: a indexação em segundo plano
 * e os pedidos da interface aguardam o mesmo trabalho, e todos os indicadores
 * de progresso abertos recebem as atualizações.
 */
let indexJob: { key: string; promise: Promise<MLResponse>; listeners: Set<HTMLElement>; last: [string, number] | null } | null = null;

/** Depois do primeiro uso o modelo fica no cache do navegador: aí vale a pena antecipar trabalho. */
const EMBED_READY = 'embed-ready';

async function ensureIndex(progressEl?: HTMLElement) {
  const key = state.key;
  if (state.indexedKey === key) return key;
  if (!indexJob || indexJob.key !== key) {
    const job: NonNullable<typeof indexJob> = { key, listeners: new Set(), last: null, promise: null! };
    job.promise = ml({ type: 'index', key, sentences: state.result!.sentences.map((s) => s.text) }, (label, p) => {
      job.last = [label, p];
      job.listeners.forEach((el) => progress(el, label, p));
    });
    indexJob = job;
  }
  const job = indexJob;
  if (progressEl) {
    job.listeners.add(progressEl);
    if (job.last) progress(progressEl, ...job.last);
    else progress(progressEl, 'Preparando modelo', null);
  }
  try {
    let res: MLResponse;
    try {
      res = await job.promise;
    } catch (err) {
      // o texto mudou durante a indexação: o trabalho antigo foi descartado, espera o do texto novo
      if ((err as Error).message === 'cancelado' && state.result && state.key !== key) return ensureIndex(progressEl);
      throw err;
    }
    if (res.type === 'indexed') {
      if (res.key === state.key) state.indexedKey = res.key;
      store.set(EMBED_READY, '1');
      const note = res.sampled ? `Texto longo: ${fmt(res.count)} de ${fmt(res.total)} frases foram indexadas, amostradas ao longo do texto.` : '';
      job.listeners.forEach((el) => el.replaceChildren(...(note ? [h('p', { class: 'muted small' }, note)] : [])));
    }
  } finally {
    if (indexJob === job) indexJob = null;
    if (progressEl) job.listeners.delete(progressEl);
  }
  return key;
}

let warmed = false;
/** Carrega o modelo de embeddings na memória antes do primeiro pedido (só se já estiver em cache). */
function warmEmbedder() {
  if (warmed || !store.get(EMBED_READY)) return;
  warmed = true;
  ml({ type: 'warm' }).catch(() => (warmed = false));
}

let backgroundTimer = 0;
/** Com o modelo em cache, indexa o texto quando a digitação para — a busca semântica fica pronta antes do clique. */
function scheduleBackgroundIndex() {
  clearTimeout(backgroundTimer);
  if (!store.get(EMBED_READY) || !state.result || state.result.stats.sentences < 2) return;
  backgroundTimer = window.setTimeout(() => {
    if (state.result && state.indexedKey !== state.key) ensureIndex().catch(() => {});
  }, 1500);
}

async function renderSummary() {
  const out = $('#summary-out');
  const prog = $('#summary-progress');
  const copy = $('#copy-summary');
  $('#summary-help').textContent =
    state.summaryMode === 'stat'
      ? 'Resumo extrativo com TextRank: as frases mais centrais do texto, escolhidas por similaridade de vocabulário e sem repetição.'
      : 'TextRank sobre embeddings semânticos: considera frases parecidas mesmo quando usam palavras diferentes. Usa o modelo MiniLM (~118 MB no primeiro uso).';
  if (!state.result) {
    prog.replaceChildren();
    copy.hidden = true;
    return out.replaceChildren(empty());
  }
  const ticket = ++summaryTicket;
  const k = Number(summaryK.value);
  const sents = state.result.sentences;
  try {
    let indices: number[];
    if (state.summaryMode === 'stat') {
      prog.replaceChildren();
      const res = await analysis({ type: 'summarize', k });
      indices = res.type === 'summary' ? res.indices : [];
    } else {
      const key = await ensureIndex(prog);
      const res = await ml({ type: 'semantic-summary', key, k });
      indices = res.type === 'summary' ? res.indices : [];
    }
    if (ticket !== summaryTicket) return;
    lastSummary = indices.map((i) => sents[i]?.text).filter(Boolean);
    out.replaceChildren(...(lastSummary.length ? lastSummary.map((t) => h('li', {}, t)) : [h('p', { class: 'empty' }, 'O texto é curto demais para resumir.')]));
    copy.hidden = !lastSummary.length;
  } catch (err) {
    if (ticket === summaryTicket) prog.replaceChildren(h('p', { class: 'error' }, `Erro: ${(err as Error).message}`));
  }
}

$('#copy-summary').addEventListener('click', async (e) => {
  const btn = e.currentTarget as HTMLButtonElement;
  try {
    await navigator.clipboard.writeText(lastSummary.join('\n'));
    btn.textContent = 'Copiado!';
  } catch {
    btn.textContent = 'Não foi possível copiar';
  }
  setTimeout(() => (btn.textContent = 'Copiar resumo'), 1500);
});

// cache lastSummary into state for the report
const _origLastSummary = lastSummary;
Object.defineProperty(lastSummary, 'push', {
  value: (...args: string[]) => { state.lastSummaryLines = [..._origLastSummary, ...args]; return Array.prototype.push.call(lastSummary, ...args); }
});

// ---------------------------------------------------------------- busca

const query = $<HTMLInputElement>('#query');
let searchTimer = 0;
let searchTicket = 0;

function searchFor(term: string) {
  query.value = term;
  if (state.searchMode !== 'exact') $<HTMLButtonElement>('#search-mode [data-v=exact]').click();
  selectTab('search');
  renderSearch();
}

query.addEventListener('input', () => {
  if (state.searchMode !== 'exact') return;
  clearTimeout(searchTimer);
  searchTimer = window.setTimeout(renderSearch, 150);
});
$('#whole').addEventListener('change', () => renderSearch());
$('#search-form').addEventListener('submit', (e) => {
  e.preventDefault();
  renderSearch(true);
});

async function renderSearch(submitted = false) {
  const out = $('#search-out');
  const prog = $('#search-progress');
  const help = $('#search-help');
  const q = query.value.trim();
  help.textContent =
    state.searchMode === 'exact'
      ? 'Ignora acentos e maiúsculas: “informacao” encontra “Informação”.'
      : 'Encontra as frases com significado mais próximo da pergunta, mesmo sem as mesmas palavras. Usa o modelo MiniLM (~118 MB no primeiro uso).';
  if (!state.result) return out.replaceChildren(empty());
  if (!q) return out.replaceChildren();
  const ticket = ++searchTicket;
  const sents = state.result.sentences;
  try {
    if (state.searchMode === 'exact') {
      const res = await analysis({ type: 'search', query: q, wholeWord: $<HTMLInputElement>('#whole').checked });
      if (ticket !== searchTicket || res.type !== 'search') return;
      out.replaceChildren(
        h('p', { class: 'small' }, res.total ? `${fmt(res.total)} ocorrência${res.total > 1 ? 's' : ''} em ${fmt(res.hits.length)}${res.hits.length >= 200 ? '+' : ''} frase${res.hits.length > 1 ? 's' : ''}` : 'Nenhuma ocorrência.'),
        ...res.hits.map((hit) =>
          h('div', { class: 'hit' }, h('div', { class: 'meta' }, `Parágrafo ${sents[hit.sentence].para + 1}`), highlighted(sents[hit.sentence].text, hit.ranges)),
        ),
      );
    } else {
      if (!submitted) return;
      const key = await ensureIndex(prog);
      const res = await ml({ type: 'ask', key, question: q, k: 8 });
      if (ticket !== searchTicket || res.type !== 'answer') return;
      out.replaceChildren(
        ...res.matches.map((m) => {
          const prev = sents[m.sentence - 1];
          const next = sents[m.sentence + 1];
          const sameP = (s?: { para: number }) => s && s.para === sents[m.sentence].para;
          return h(
            'div',
            { class: 'hit' },
            h('div', { class: 'meta' }, `Similaridade ${pct(Math.max(0, m.score))} · parágrafo ${sents[m.sentence].para + 1}`),
            sameP(prev) ? h('span', { class: 'ctx' }, `${prev!.text} `) : null,
            h('strong', {}, sents[m.sentence].text),
            sameP(next) ? h('span', { class: 'ctx' }, ` ${next!.text}`) : null,
          );
        }),
      );
    }
  } catch (err) {
    if (ticket === searchTicket) prog.replaceChildren(h('p', { class: 'error' }, `Erro: ${(err as Error).message}`));
  }
}

// ---------------------------------------------------------------- IA: gênero e sentimento

function staleNote(container: HTMLElement) {
  if (container.dataset.key && container.dataset.key !== state.key && container.childElementCount && !container.querySelector('.stale')) {
    container.prepend(h('p', { class: 'muted small stale' }, 'O texto mudou desde esta análise. Rode novamente para atualizar.'));
  }
}

function renderAI() {
  staleNote($('#genre-out'));
  staleNote($('#sentiment-out'));
  $<HTMLButtonElement>('#run-genre').disabled = !state.result;
  $<HTMLButtonElement>('#run-sentiment').disabled = !state.result;
}

async function withButton(btn: HTMLButtonElement, fn: () => Promise<void>) {
  btn.disabled = true;
  try {
    await fn();
  } finally {
    btn.disabled = !state.result;
  }
}

$('#run-genre').addEventListener('click', (e) =>
  withButton(e.currentTarget as HTMLButtonElement, async () => {
    const out = $('#genre-out');
    const prog = $('#genre-progress');
    if (!state.result) return;
    try {
      const key = await ensureIndex(prog);
      const res = await ml({ type: 'classify', key });
      if (res.type !== 'genres') return;
      state.lastGenres = res.genres; // cache for report
      const top = res.genres[0];
      const def = GENRES.find((g) => g.label === top.label);
      out.dataset.key = key;
      out.replaceChildren(
        h('div', { class: 'genre-top' }, top.label),
        h('p', { class: 'muted small' }, def?.description ?? ''),
        hbars(
          res.genres.map((g) => ({ label: g.label, value: g.score, display: pct(g.score), tooltip: `${g.label}: ${pct(g.score)} de afinidade` })),
        ),
        h('p', { class: 'muted small' }, 'Afinidade relativa entre os gêneros (soma 100%). É uma estimativa por similaridade, não um classificador treinado para esta tarefa.'),
      );
    } catch (err) {
      prog.replaceChildren(h('p', { class: 'error' }, `Erro: ${(err as Error).message}`));
    }
  }),
);

$('#run-sentiment').addEventListener('click', (e) =>
  withButton(e.currentTarget as HTMLButtonElement, async () => {
    const out = $('#sentiment-out');
    const prog = $('#sentiment-progress');
    const r = state.result;
    if (!r) return;
    const key = state.key;
    try {
      progress(prog, 'Preparando modelo', null);
      const res = await ml({ type: 'sentiment', key, sentences: r.sentences.map((s) => s.text) }, (label, p) => progress(prog, label, p));
      prog.replaceChildren();
      if (res.type !== 'sentiment') return;
      state.lastSentiment = res.result; // cache for report
      const { distribution: d, mostPositive, mostNegative, analyzed } = res.result;
      const quote = (cls: string, i: number) => h('div', { class: `quote ${cls}` }, r.sentences[i]?.text ?? '');
      out.dataset.key = key;
      const parts: (Node | null)[] = [
        stack([
          { label: 'Positivo', value: d.positive, color: 'var(--series-1)' },
          { label: 'Neutro', value: d.neutral, color: 'var(--neu)' },
          { label: 'Negativo', value: d.negative, color: 'var(--neg)' },
        ]),
        h('p', { class: 'muted small' }, `Média sobre ${fmt(analyzed)} frases.`),
        mostPositive.length ? h('div', { class: 'section-title' }, 'Frases mais positivas') : null,
        ...mostPositive.map((p) => quote('pos', p.sentence)),
        mostNegative.length ? h('div', { class: 'section-title' }, 'Frases mais negativas') : null,
        ...mostNegative.map((p) => quote('neg', p.sentence)),
      ];
      out.replaceChildren(...parts.filter((n): n is Node => n !== null));
    } catch (err) {
      prog.replaceChildren(h('p', { class: 'error' }, `Erro: ${(err as Error).message}`));
    }
  }),
);


// ---------------------------------------------------------------- frases-chave

const kpN = $<HTMLInputElement>('#kp-n');
kpN.addEventListener('input', () => {
  $<HTMLOutputElement>('#kp-n-out').value = kpN.value;
  renderKeyphrases();
});

async function renderKeyphrases() {
  const out = $('#kp-out');
  if (!state.result) return out.replaceChildren(empty());
  const n = Number(kpN.value);
  const res = await analysis({ type: 'keyphrases', topN: n });
  if (res.type !== 'keyphrases') return;
  state.lastKeyphrases = res.phrases; // cache for report
  if (!res.phrases.length) return out.replaceChildren(h('p', { class: 'empty' }, 'Nada a mostrar.'));
  const max = res.phrases[0].score || 1;
  out.replaceChildren(
    hbars(
      res.phrases.map((kp) => ({
        label: kp.phrase,
        value: kp.score / max,
        display: kp.score.toFixed(3),
        tooltip: `"${kp.phrase}" — relevância ${kp.score.toFixed(3)}`,
      })),
    ),
  );
}

// ---------------------------------------------------------------- NER

const ENTITY_COLORS: Record<string, string> = {
  PER: '#6366f1', ORG: '#0891b2', LOC: '#16a34a', MISC: '#d97706',
};
const ENTITY_LABELS: Record<string, string> = {
  PER: 'Pessoa', ORG: 'Organização', LOC: 'Local', MISC: 'Misc',
};

function renderNer() {
  $<HTMLButtonElement>('#run-ner').disabled = !state.result;
}

$('#run-ner').addEventListener('click', (e) =>
  withButton(e.currentTarget as HTMLButtonElement, async () => {
    const out = $('#ner-out');
    const prog = $('#ner-progress');
    if (!state.result) return;
    try {
      progress(prog, 'Preparando modelo NER', null);
      const res = await ml({ type: 'ner', text: state.text }, (label, p) => progress(prog, label, p));
      prog.replaceChildren();
      if (res.type !== 'ner') return;
      state.lastEntities = res.entities;
      if (!res.entities.length) {
        out.replaceChildren(h('p', { class: 'empty' }, 'Nenhuma entidade detectada no trecho analisado.'));
        return;
      }
      // Group by entity type
      const byType = new Map<string, string[]>();
      for (const e of res.entities) {
        const list = byType.get(e.entity) ?? [];
        list.push(e.word);
        byType.set(e.entity, list);
      }
      const sections: HTMLElement[] = [];
      for (const [type, words] of byType) {
        const color = ENTITY_COLORS[type] ?? '#6b7280';
        const label = ENTITY_LABELS[type] ?? type;
        sections.push(
          h('div', { class: 'ai-block' },
            h('div', { class: 'section-title', style: `color:${color}` }, label),
            h('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;' },
              ...[...new Set(words)].map((w) =>
                h('span', {
                  style: `padding:3px 10px;border-radius:20px;font-size:0.88rem;
                    background:${color}18;border:1px solid ${color};color:${color};`
                }, w)
              )
            )
          )
        );
      }
      out.replaceChildren(
        h('p', { class: 'muted small' }, `${res.entities.length} entidade${res.entities.length > 1 ? 's' : ''} detectada${res.entities.length > 1 ? 's' : ''} nos primeiros 2000 caracteres.`),
        h('div', { class: 'ai-grid' }, ...sections)
      );
    } catch (err) {
      prog.replaceChildren(h('p', { class: 'error' }, `Erro: ${(err as Error).message}`));
    }
  }),
);

// ---------------------------------------------------------------- exportar relatório

function renderExport() {
  $<HTMLButtonElement>('#run-export').disabled = !state.result;
}

$('#run-export').addEventListener('click', () => {
  const r = state.result;
  if (!r) return;
  const statusEl = $('#export-status');
  statusEl.textContent = 'Gerando relatório…';
  try {
    const html = generateHtmlReport({
      text: state.text,
      result: r,
      summaryLines: lastSummary.length ? [...lastSummary] : undefined,
      keyphrases: state.lastKeyphrases ?? undefined,
      sentiment: state.lastSentiment ?? undefined,
      genres: state.lastGenres ?? undefined,
      entities: state.lastEntities ?? undefined,
      cloudCanvas: $<HTMLCanvasElement>('#cloud'),
    });
    downloadHtml(html);
    statusEl.textContent = 'Relatório baixado com sucesso!';
    setTimeout(() => (statusEl.textContent = ''), 3000);
  } catch (err) {
    statusEl.textContent = `Erro ao gerar relatório: ${(err as Error).message}`;
  }
});

// ---------------------------------------------------------------- tema

const root = document.documentElement;
const savedTheme = store.get('theme');
if (savedTheme === 'light' || savedTheme === 'dark') root.dataset.theme = savedTheme;
$('#theme').addEventListener('click', () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  store.set('theme', root.dataset.theme);
  // a nuvem é desenhada em canvas: precisa ser redesenhada com a nova paleta
  state.dirty.add('cloud');
  if (state.tab === 'cloud') renderPanel('cloud');
});

// ---------------------------------------------------------------- início

const saved = store.get('text');
if (saved) setText(saved);
else renderAll();

// abrir qualquer recurso de IA já começa a carregar o modelo (quando ele está em cache)
document.querySelectorAll('[data-tab=ai], [data-tab=search], [data-tab=summary], [data-v=ai]').forEach((el) =>
  el.addEventListener('pointerenter', warmEmbedder, { once: true }),
);
