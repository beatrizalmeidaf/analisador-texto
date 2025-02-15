# Geração de gráficos e nuvens de palavras

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

def plot_metrics(media, mediana, std):
    """Gera gráfico de métricas
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    metricas = ["Média", "Mediana", "Desvio Padrão"]
    valores = [media, mediana, std]
    
    sns.barplot(x=metricas, y=valores, palette="Blues_r", ax=ax)
    
    for i, v in enumerate(valores):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.title("Métricas do Comprimento dos Tokens")
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
    """ Gera e retorna gráfico TF-IDF 
    """
    
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x="tfidf", y="termo", data=df_tfidf.head(15), palette="Blues_r")
    plt.title("Top 15 Termos por TF-IDF")
    plt.xlabel("TF-IDF")
    plt.ylabel("")
    plt.tight_layout()
    
    return fig
   
def plot_word_frequency(word_counts):
    """ Gera gráfico das 15 palavras mais frequentes
    """
    top_words = word_counts.most_common(15)
    df = pd.DataFrame(top_words, columns=['Palavra', 'Frequência'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # normaliza as frequências para mapear na paleta de cores
    norm = plt.Normalize(df['Frequência'].min(), df['Frequência'].max())
    colors = plt.cm.Blues(norm(df['Frequência'])) 

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