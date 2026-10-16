import re
from collections import Counter
from functools import lru_cache

import nltk


@lru_cache(maxsize=1)
def get_stopwords():
    """ Carrega as stopwords em português (baixa o corpus do NLTK apenas se necessário) """
    try:
        return frozenset(nltk.corpus.stopwords.words('portuguese'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        return frozenset(nltk.corpus.stopwords.words('portuguese'))


def tokenizer(text):
    """ Tokenização para dividir um texto em unidades menores e remoção
     de palavras de parada """

    # o texto já chega sem pontuação (clean_text), então separar por espaços basta
    # e evita depender do modelo punkt do NLTK
    stopwords = get_stopwords()
    return [x for x in re.split(r'\s+', text) if x and x not in stopwords]

def word_counter(tokens):
   """ Contagem de frequência de palavras importantes
   """
   return Counter(tokens)


if __name__ == '__main__':
    from .cleaner import clean_text

    with open('data/teste.txt', encoding='utf-8') as f:
        print(tokenizer(clean_text(f.read())))
