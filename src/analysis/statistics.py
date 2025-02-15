# Cálculo de métricas do texto

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_metrics(tokens):
    """ Cálculo de métricas estatísticas
    """
    if not tokens:
        return {"media": None, "mediana": None, "desvio_padrao": None}
    
    tamanhos = [len(x) for x in tokens]
    serie = pd.Series(tamanhos)
    
    return {
        "media": serie.mean(),
        "mediana": serie.median(),
        "desvio_padrao": serie.std()
    }

def calculate_tfidf(tokens):
    """ Calcula o TF-IDF para os tokens e retorna os valores em um DataFrame ordenado por relevância. 
    """

    if not tokens:
        print("Lista de tokens vazia. Nenhum TF-IDF gerado.")
        return pd.DataFrame()
    
    documento = " ".join(tokens)

    vectorizer = TfidfVectorizer()
    tfidf_matriz = vectorizer.fit_transform([documento])

    # extraindo termos
    termos = vectorizer.get_feature_names_out()
    valores = tfidf_matriz.toarray()[0]

    df_tfidf = pd.DataFrame({"termo": termos, "tfidf": valores})
    df_tfidf = df_tfidf.sort_values(by="tfidf", ascending=False)  # ordenar por relevância

    return df_tfidf