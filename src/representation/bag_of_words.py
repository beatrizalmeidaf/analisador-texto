from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def bow(tokens):
    """
    Gera a representação Bag of Words (BoW) a partir de uma lista de documentos.

    Parâmetros:
        tokens (list of str): Lista de sentenças/documentos em formato de strings.

    Retorna:
        tuple: Uma tupla contendo:
            - array BoW (numpy.ndarray): Matriz onde cada linha representa um documento 
              e cada coluna representa a frequência de uma palavra.
            - vectorizer (CountVectorizer): O vetor de características treinado.
    """
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(tokens)
    return x.toarray(), vectorizer

def cooccurrence(tokens, window_size=2):
    """
    Calcula a matriz de coocorrência de palavras com base em uma janela de contexto deslizante.

    Parâmetros:
        tokens (list of str): Lista de sentenças/documentos em formato de strings.
        window_size (int): Tamanho da janela de contexto para considerar palavras vizinhas.

    Retorna:
        numpy.ndarray: Matriz quadrada de coocorrência onde cada célula (i, j) representa
        quantas vezes a palavra i apareceu no contexto da palavra j.
    """
    _, vectorizer = bow(tokens)

    vocab = vectorizer.get_feature_names_out()
    vocab_size = len(vocab)

    ccr = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    tokenized_text = " ".join(tokens).split()

    for i, token in enumerate(tokenized_text):
        if token not in vectorizer.vocabulary_:
            continue

        token_index = vectorizer.vocabulary_[token]

        start = max(0, i - window_size)
        end = min(len(tokenized_text), i + window_size + 1)

        for j in range(start, end):
            if i == j:
                continue

            context_token = tokenized_text[j]
            if context_token in vectorizer.vocabulary_:
                context_index = vectorizer.vocabulary_[context_token]
                ccr[token_index, context_index] += 1

    return ccr
