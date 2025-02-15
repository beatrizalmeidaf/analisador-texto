# Geração de gráficos e nuvens de palavras

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

def plot_metrics(media, mediana, std):
    """Gera gráfico de métricas ordenadas do maior para o menor.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metricas = ["Média", "Mediana", "Desvio Padrão"]
    valores = [media, mediana, std]
    
    # ordenação do maior para o menor
    metricas, valores = zip(*sorted(zip(metricas, valores), key=lambda x: x[1], reverse=True))
    
    # gradiente de cores azuis 
    cmap = sns.color_palette("Blues", as_cmap=True)
    norm_values = np.interp(valores, (min(valores), max(valores)), (0.4, 1))  
    colors = [cmap(v) for v in norm_values]

    sns.barplot(x=metricas, y=valores, palette=colors, ax=ax)
    
    # adicionando os valores acima das barras
    for i, v in enumerate(valores):
        ax.text(i, v + (max(valores) * 0.02), f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.title("Métricas do Comprimento dos Tokens")
    plt.ylim(0, max(valores) * 1.1)  # ajuste para dar espaço ao texto
    plt.tight_layout()
    
    return fig


def generate_wordcloud(tokens):
    """ Gera e retorna uma nuvem de palavras 
    """
    if not tokens:
        return None
    
    all_text = " ".join(tokens)
    wordcloud = WordCloud(
        background_color="white",
        width=800,
        height=400,
        colormap="viridis"
    ).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_axis_off()
    plt.title("Nuvem de Palavras")
    
    return fig

def plot_tfidf(df_tfidf):
    """Gera e retorna gráfico TF-IDF com melhor contraste de cores."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # normaliza os valores 
    min_val, max_val = np.log1p(df_tfidf["tfidf"].min()), np.log1p(df_tfidf["tfidf"].max())
    norm_values = np.interp(np.log1p(df_tfidf["tfidf"]), (min_val, max_val), (0.2, 1))
    colors = plt.cm.Blues(norm_values)  

    sns.barplot(x="tfidf", y="termo", data=df_tfidf.head(15), palette=colors, ax=ax)
    
    plt.title("Top 15 Termos por TF-IDF")
    plt.xlabel("TF-IDF")
    plt.ylabel("")
    plt.tight_layout()
    
    return fig

def plot_word_frequency(word_counts):
    """Gera gráfico das 15 palavras mais frequentes com cores mais equilibradas."""
    
    top_words = word_counts.most_common(15)
    df = pd.DataFrame(top_words, columns=['Palavra', 'Frequência'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # normaliza e ajusta a paleta
    min_val, max_val = np.log1p(df['Frequência'].min()), np.log1p(df['Frequência'].max())
    norm_values = np.interp(np.log1p(df['Frequência']), (min_val, max_val), (0.2, 1))
    colors = plt.cm.Blues(norm_values)  

    bars = ax.barh(df['Palavra'], df['Frequência'], color=colors)
    ax.set_title('Top 15 Palavras Mais Frequentes')
    ax.set_xlabel('Frequência')
    ax.invert_yaxis()
    
    # adiciona os valores ao lado das barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig