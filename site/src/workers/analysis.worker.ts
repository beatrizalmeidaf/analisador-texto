/// <reference lib="webworker" />
import { analyze, type Analysis } from '../core/analyze.ts';
import { summarize } from '../core/summarize.ts';
import { search } from '../core/search.ts';
import type { AnalysisRequest, AnalysisResponse } from './protocol.ts';

let last: Analysis | null = null;

const post = (msg: AnalysisResponse) => self.postMessage(msg);

self.onmessage = (e: MessageEvent<AnalysisRequest>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'analyze': {
      const t0 = performance.now();
      last = analyze(msg.text);
      const { sentences, ...rest } = last;
      post({
        type: 'analysis',
        id: msg.id,
        ms: performance.now() - t0,
        result: { ...rest, sentences: sentences.map(({ text, para }) => ({ text, para })) },
      });
      break;
    }
    case 'summarize':
      post({ type: 'summary', id: msg.id, indices: last ? summarize(last.sentences, msg.k) : [] });
      break;
    case 'search':
      post({ type: 'search', id: msg.id, ...search(last?.sentences ?? [], msg.query, msg.wholeWord) });
      break;
  }
};
