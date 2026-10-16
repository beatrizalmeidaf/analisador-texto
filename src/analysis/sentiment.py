"""
Análise de sentimento por parágrafo/trecho.

Usa o modelo compacto DistilBERT multilingual fine-tuned em sentimentos:
  lxyuan/distilbert-base-multilingual-cased-sentiments-student

Suporta: positivo / negativo / neutro
Latência: ~50-150ms por trecho em CPU (modelo ~250 MB, cache com @lru_cache)
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import TypedDict


class SentimentResult(TypedDict):
    texto: str
    label: str
    score: float
    emoji: str


_LABEL_MAP = {
    "positive": ("Positivo", "😊"),
    "neutral":  ("Neutro",   "😐"),
    "negative": ("Negativo", "😞"),
}


@lru_cache(maxsize=1)
def _load_pipeline():
    """Carrega o pipeline de sentimento com cache (instancia apenas uma vez)."""
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        top_k=None,          # retorna todos os labels
        truncation=True,
        max_length=512,
    )


def _split_chunks(text: str, max_chars: int = 500) -> list[str]:
    """
    Divide o texto em parágrafos ou trechos menores.
    Prioriza quebras de parágrafo; divide frases se o trecho ainda for grande.
    """
    # divide por parágrafo primeiro
    chunks = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            # quebra por frase
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        result.append(current)
                    current = sent
            if current:
                result.append(current)
    return result or [text[:max_chars]]


def analisar_sentimento(text: str) -> list[SentimentResult]:
    """
    Analisa o sentimento de cada trecho do texto.

    Retorna uma lista de dicts com:
      - texto  : trecho original
      - label  : "Positivo" | "Negativo" | "Neutro"
      - score  : confiança (0–1)
      - emoji  : emoji representativo
    """
    pipe = _load_pipeline()
    chunks = _split_chunks(text)
    results: list[SentimentResult] = []

    for chunk in chunks:
        if not chunk:
            continue
        output = pipe(chunk[:512])  # garante truncamento seguro
        # output é uma lista de dicts [{label, score}, ...]
        top = max(output[0], key=lambda x: x["score"])
        raw_label = top["label"].lower()
        label_pt, emoji = _LABEL_MAP.get(raw_label, ("Neutro", "😐"))
        results.append(SentimentResult(
            texto=chunk,
            label=label_pt,
            score=top["score"],
            emoji=emoji,
        ))

    return results


def sentimento_geral(results: list[SentimentResult]) -> SentimentResult:
    """
    Agrega os resultados por chunk e retorna o sentimento geral do texto
    ponderado pelo tamanho de cada trecho.
    """
    if not results:
        return SentimentResult(texto="", label="Neutro", score=0.0, emoji="😐")

    from collections import defaultdict
    scores: dict[str, float] = defaultdict(float)
    total_weight = 0.0

    for r in results:
        weight = len(r["texto"])
        scores[r["label"]] += r["score"] * weight
        total_weight += weight

    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label] / total_weight
    _, emoji = next(
        (v for v in _LABEL_MAP.values() if v[0] == best_label),
        ("Neutro", "😐"),
    )
    return SentimentResult(
        texto="(Texto completo)",
        label=best_label,
        score=best_score,
        emoji=emoji,
    )
