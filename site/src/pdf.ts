// build "legacy": o build padrão do pdf.js 6 exige APIs muito recentes
// (ex.: Uint8Array.prototype.toHex) e quebra em navegadores de poucos meses atrás
import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs';
import type { TextItem } from 'pdfjs-dist/types/src/display/api';
import workerUrl from 'pdfjs-dist/legacy/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

type Item = Pick<TextItem, 'str' | 'hasEOL' | 'transform' | 'height'>;

const isPageNumber = (line: string) => /^\s*(p(á|a)g(ina)?\.?\s*)?\d{1,4}(\s*(de|\/)\s*\d{1,4})?\s*$/i.test(line);

/**
 * Reconstrói linhas e parágrafos de uma página a partir das posições dos
 * trechos de texto: um salto vertical maior que ~1,6× a altura da linha vira
 * quebra de parágrafo. Números de página isolados são descartados.
 */
function pageText(items: Item[]): string {
  const lines: { text: string; y: number; h: number }[] = [];
  let current = '';
  let y = NaN;
  let h = 0;
  for (const it of items) {
    if (Number.isNaN(y)) y = it.transform[5];
    current += it.str;
    h = Math.max(h, it.height || 0);
    if (it.hasEOL) {
      lines.push({ text: current, y, h });
      current = '';
      y = NaN;
      h = 0;
    }
  }
  if (current) lines.push({ text: current, y, h });

  let out = '';
  let prev: (typeof lines)[number] | null = null;
  for (const line of lines) {
    const text = line.text.replace(/\s+/g, ' ').trim();
    if (!text || isPageNumber(text)) continue;
    if (prev) {
      const gap = Math.abs(prev.y - line.y);
      const lh = prev.h || line.h || 10;
      if (gap > lh * 1.6) out += '\n\n';
      else if (/[-‐]$/.test(out) && /^\p{Ll}/u.test(text)) out = out.replace(/[-‐]$/, ''); // palavra hifenizada
      else out += '\n';
    }
    out += text;
    prev = line;
  }
  return out;
}

export async function extractPdf(data: ArrayBuffer, onProgress?: (page: number, total: number) => void): Promise<string> {
  const task = pdfjs.getDocument({ data });
  const doc = await task.promise;
  const pages: string[] = [];
  try {
    for (let p = 1; p <= doc.numPages; p++) {
      const page = await doc.getPage(p);
      const content = await page.getTextContent();
      pages.push(pageText(content.items.filter((i): i is TextItem => 'str' in i)));
      page.cleanup();
      onProgress?.(p, doc.numPages);
    }
  } finally {
    await task.destroy();
  }
  return pages.filter(Boolean).join('\n\n');
}
