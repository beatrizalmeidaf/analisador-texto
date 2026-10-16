import type { Analysis } from '../core/analyze.ts';
import type { Hit } from '../core/search.ts';

export interface LiteSentence {
  text: string;
  para: number;
}

export type AnalysisResult = Omit<Analysis, 'sentences'> & { sentences: LiteSentence[] };



export interface Genre {
  label: string;
  score: number;
}

export interface SentimentResult {
  /** médias de positivo/neutro/negativo sobre as frases analisadas */
  distribution: { positive: number; neutral: number; negative: number };
  analyzed: number;
  mostPositive: { sentence: number; score: number }[];
  mostNegative: { sentence: number; score: number }[];
}

export interface Keyphrase {
  phrase: string;
  score: number;
}

export interface NerEntity {
  word: string;
  entity: string; // PER | ORG | LOC | MISC
  score: number;
  start: number;
  end: number;
}

export type AnalysisRequest =
  | { type: 'analyze'; id: number; text: string }
  | { type: 'summarize'; id: number; k: number }
  | { type: 'search'; id: number; query: string; wholeWord: boolean }
  | { type: 'keyphrases'; id: number; topN: number };

export type AnalysisResponse =
  | { type: 'analysis'; id: number; ms: number; result: AnalysisResult }
  | { type: 'summary'; id: number; indices: number[] }
  | { type: 'search'; id: number; total: number; hits: Hit[] }
  | { type: 'keyphrases'; id: number; phrases: Keyphrase[] };

export type MLRequest =
  | { type: 'index'; id: number; key: string; sentences: string[] }
  | { type: 'ask'; id: number; key: string; question: string; k: number }
  | { type: 'classify'; id: number; key: string }
  | { type: 'semantic-summary'; id: number; key: string; k: number }
  | { type: 'sentiment'; id: number; key: string; sentences: string[] }
  | { type: 'ner'; id: number; text: string }
  | { type: 'warm'; id: number };

export type MLResponse =
  | { type: 'progress'; id: number; stage: 'download' | 'compute'; label: string; progress: number }
  | { type: 'indexed'; id: number; key: string; count: number; total: number; sampled: boolean }
  | { type: 'answer'; id: number; matches: { sentence: number; score: number }[] }
  | { type: 'genres'; id: number; genres: Genre[] }
  | { type: 'summary'; id: number; indices: number[] }
  | { type: 'sentiment'; id: number; result: SentimentResult }
  | { type: 'ner'; id: number; entities: NerEntity[] }
  | { type: 'warmed'; id: number }
  | { type: 'error'; id: number; message: string };
