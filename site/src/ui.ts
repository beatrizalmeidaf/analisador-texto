// Pequenos componentes de interface e gráficos em HTML puro (sem bibliotecas).

export const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector<T>(sel)!;

type Attrs = Record<string, string | number | boolean | EventListener | undefined>;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: (Node | string | null | undefined | false)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    if (typeof v === 'function') el.addEventListener(k.replace(/^on/, ''), v);
    else if (k === 'style') el.setAttribute('style', String(v));
    else if (k === 'class') el.className = String(v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined && c !== false) el.append(c);
  return el;
}

const nf = new Intl.NumberFormat('pt-BR');
export const fmt = (n: number, digits = 0) =>
  digits ? n.toLocaleString('pt-BR', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : nf.format(Math.round(n));
export const pct = (n: number) => `${(n * 100).toLocaleString('pt-BR', { maximumFractionDigits: 1 })}%`;

export function duration(min: number) {
  if (min < 1) return `${Math.max(1, Math.round(min * 60))} s`;
  if (min < 60) return `${Math.round(min)} min`;
  return `${Math.floor(min / 60)} h ${Math.round(min % 60)} min`;
}

// tooltip compartilhado
const tip = () => $('#tooltip');
export function withTooltip(el: HTMLElement, text: () => string) {
  el.addEventListener('pointermove', (e) => {
    const t = tip();
    t.textContent = text();
    t.hidden = false;
    const pad = 14;
    const { width, height } = t.getBoundingClientRect();
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    if (x + width > innerWidth - 8) x = e.clientX - width - pad;
    if (y + height > innerHeight - 8) y = e.clientY - height - pad;
    t.style.left = `${x}px`;
    t.style.top = `${y}px`;
  });
  el.addEventListener('pointerleave', () => (tip().hidden = true));
}

export interface BarItem {
  label: string;
  value: number;
  display: string;
  tooltip: string;
}

/** Barras horizontais ordenadas; rótulos clicáveis. */
export function hbars(items: BarItem[], onPick?: (label: string) => void): HTMLElement {
  const max = Math.max(...items.map((i) => i.value), 1e-9);
  const root = h('div', { class: 'hbars' });
  for (const it of items) {
    const track = h(
      'div',
      { class: 'track' },
      h('span', { class: 'bar', style: `width:${(it.value / max) * 85}%` }),
      h('span', { class: 'val' }, it.display),
    );
    withTooltip(track, () => it.tooltip);
    root.append(
      h('button', { class: 'lbl', type: 'button', title: `Buscar “${it.label}” no texto`, onclick: () => onPick?.(it.label) }, it.label),
      track,
    );
  }
  return root;
}

/** Histograma em colunas. */
export function columns(values: number[], labels: string[], tooltip: (i: number) => string): HTMLElement {
  const max = Math.max(...values, 1);
  const cols = h('div', { class: 'cols', role: 'img' });
  values.forEach((v, i) => {
    const col = h('div', { class: 'col' }, h('span', { style: `height:${(v / max) * 100}%` }));
    withTooltip(col, () => tooltip(i));
    cols.append(col);
  });
  const x = h('div', { class: 'cols-x', 'aria-hidden': 'true' }, ...labels.map((l) => h('span', {}, l)));
  return h('div', {}, cols, x);
}

export interface StackPart {
  label: string;
  value: number;
  color: string;
}

/** Barra empilhada 100% com legenda (identidade nunca só pela cor). */
export function stack(parts: StackPart[]): HTMLElement {
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  const bar = h('div', { class: 'stack', role: 'img', 'aria-label': parts.map((p) => `${p.label} ${pct(p.value / total)}`).join(', ') });
  for (const p of parts) {
    if (!p.value) continue;
    const seg = h('span', { style: `flex:${p.value / total};background:${p.color}` });
    withTooltip(seg, () => `${p.label}: ${pct(p.value / total)}`);
    bar.append(seg);
  }
  const legend = h(
    'div',
    { class: 'legend' },
    ...parts.map((p) => h('span', {}, h('i', { style: `background:${p.color}` }), `${p.label} ${pct(p.value / total)}`)),
  );
  return h('div', {}, bar, legend);
}

export function progress(container: HTMLElement, label: string, value: number | null) {
  container.replaceChildren(
    h(
      'div',
      { class: 'progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': value === null ? undefined : Math.round(value * 100) },
      `${label}${value === null ? '…' : ` — ${Math.round(value * 100)}%`}`,
      h('div', { class: 'pbar' }, h('span', { style: `width:${value === null ? 15 : value * 100}%` })),
    ),
  );
}

/** Texto com trechos destacados em <mark>. */
export function highlighted(text: string, ranges: [number, number][]): DocumentFragment {
  const frag = document.createDocumentFragment();
  let pos = 0;
  for (const [a, b] of ranges) {
    if (a > pos) frag.append(text.slice(pos, a));
    frag.append(h('mark', {}, text.slice(a, b)));
    pos = b;
  }
  frag.append(text.slice(pos));
  return frag;
}

export function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
