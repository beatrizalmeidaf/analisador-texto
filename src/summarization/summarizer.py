"""
Sumarizador de texto com dois modos:
  - Estatístico: TextRank baseado em TF-IDF (instantâneo, zero GPU)
  - Semântico  : TextRank sobre embeddings MiniLM (≈ 118 MB no primeiro uso,
                 mas roda em CPU em poucos segundos para textos normais)

Ambos os modos são extrativos, sem geração, garantindo baixa latência.
"""

import re
import math
import heapq
import numpy as np
from functools import lru_cache
from collections import Counter
from typing import Literal

# ---------------------------------------------------------------------------
# Utilitários compartilhados
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Divide o texto em frases usando heurística simples."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _clean_tokens(sentence: str) -> list[str]:
    """Remove pontuação e converte para minúsculas."""
    return re.findall(r'\b[a-záàâãéêíóôõúüç]+\b', sentence.lower())


# ---------------------------------------------------------------------------
# Modo 1 – Estatístico (TextRank + TF-IDF)  ← instantâneo
# ---------------------------------------------------------------------------

def _tfidf_similarity(s1_tokens: list[str], s2_tokens: list[str],
                       idf: dict[str, float]) -> float:
    """Cosseno entre vetores TF-IDF de duas frases."""
    vocab = set(s1_tokens) | set(s2_tokens)
    if not vocab:
        return 0.0

    def vec(tokens):
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {w: (tf[w] / total) * idf.get(w, 1.0) for w in vocab}

    v1, v2 = vec(s1_tokens), vec(s2_tokens)
    dot = sum(v1[w] * v2[w] for w in vocab)
    n1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    n2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def _textrank_statistical(sentences: list[str], n: int) -> list[str]:
    """TextRank baseado em similaridade TF-IDF — sem dependências pesadas."""
    tokenized = [_clean_tokens(s) for s in sentences]

    # IDF sobre o "corpus" das frases
    N = len(sentences)
    df: dict[str, int] = {}
    for tokens in tokenized:
        for w in set(tokens):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (freq + 1)) + 1 for w, freq in df.items()}

    # Matriz de adjacência
    scores = [1.0] * N
    d = 0.85
    for _ in range(30):
        new_scores = [0.0] * N
        for i in range(N):
            sim_sum = 0.0
            for j in range(N):
                if i != j:
                    sim = _tfidf_similarity(tokenized[i], tokenized[j], idf)
                    new_scores[i] += d * sim * scores[j]
                    sim_sum += sim
            new_scores[i] = (1 - d) + new_scores[i]
        scores = new_scores

    top_idx = heapq.nlargest(n, range(N), key=lambda i: scores[i])
    top_idx_sorted = sorted(top_idx)
    return [sentences[i] for i in top_idx_sorted]


# ---------------------------------------------------------------------------
# Modo 2 – Semântico (TextRank + embeddings MiniLM)  ← ~118 MB 1ª vez
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_semantic_model():
    """Carrega o modelo MiniLM com cache (já usado pelo KeyBERT)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embeddings / norms
    return normed @ normed.T


def _textrank_semantic(sentences: list[str], n: int) -> list[str]:
    """TextRank com embeddings semânticos."""
    model = _load_semantic_model()
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
    sim_matrix = _cosine_matrix(np.array(embeddings))
    np.fill_diagonal(sim_matrix, 0)

    N = len(sentences)
    scores = np.ones(N)
    d = 0.85
    for _ in range(30):
        new_scores = sim_matrix @ scores
        scores = (1 - d) + d * new_scores

    top_idx = heapq.nlargest(n, range(N), key=lambda i: float(scores[i]))
    top_idx_sorted = sorted(top_idx)
    return [sentences[i] for i in top_idx_sorted]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def resumir_texto(
    texto: str,
    n_sentences: int = 5,
    mode: Literal["estatistico", "semantico"] = "estatistico",
) -> list[str]:
    """
    Retorna uma lista com as `n_sentences` frases mais representativas.

    Parâmetros
    ----------
    texto       : texto de entrada
    n_sentences : número de frases no resumo
    mode        : "estatistico" (zero latência) ou "semantico" (MiniLM)
    """
    sentences = _split_sentences(texto)
    if not sentences:
        return []

    n = min(n_sentences, len(sentences))

    if mode == "semantico":
        return _textrank_semantic(sentences, n)
    else:
        return _textrank_statistical(sentences, n)
