"""
Extração de frases-chave com KeyBERT + MiniLM.

Modelo: all-MiniLM-L6-v2  (~90 MB, compartilhado com o sumarizador semântico)
Retorna frases (n-gramas) em vez de palavras isoladas, capturando contexto real.

Latência típica: < 1 s em CPU para textos de até 5000 caracteres.
"""

from __future__ import annotations
from functools import lru_cache
from typing import TypedDict


class Keyphrase(TypedDict):
    frase: str
    score: float


@lru_cache(maxsize=1)
def _load_keybert():
    """Instancia o KeyBERT com MiniLM (cache singleton)."""
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return KeyBERT(model=model)


def extrair_frases_chave(
    text: str,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 3),
    diversity: float = 0.5,
) -> list[Keyphrase]:
    """
    Extrai as `top_n` frases-chave mais relevantes do texto.

    Parâmetros
    ----------
    text         : texto de entrada
    top_n        : número de frases a retornar
    ngram_range  : (min, max) tamanho dos n-gramas  (padrão: 1 a 3 palavras)
    diversity    : diversidade MMR entre 0 (redundante) e 1 (máximo diverso)

    Retorna lista de dicts {"frase": str, "score": float} em ordem decrescente.
    """
    if not text or not text.strip():
        return []

    kw_model = _load_keybert()

    # Limita o texto para manter latência baixa (KeyBERT com MMR é O(n²))
    snippet = text[:8000]

    keyphrases = kw_model.extract_keywords(
        snippet,
        keyphrase_ngram_range=ngram_range,
        stop_words=None,       # modelo lida bem sem lista de stopwords manual
        use_mmr=True,
        diversity=diversity,
        top_n=top_n,
    )

    return [Keyphrase(frase=kp, score=round(sc, 4)) for kp, sc in keyphrases]
