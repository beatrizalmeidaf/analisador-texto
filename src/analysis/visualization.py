import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

# parametros globais
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Blues")

def set_custom_style():
    """ Parametros globais de estilo 
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['figure.facecolor'] = '#F0F2F6'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.alpha'] = 0.2

def plot_metrics(media, mediana, std):
    """ Gera gráfico de métricas ordenadas do maior para o menor.
    """

    set_custom_style()
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    
    ax1 = fig.add_subplot(gs[0])
    metricas = ["Média", "Mediana", "Desvio Padrão"]
    valores = [media, mediana, std]
    
    # ordenação do maior para o menor
    metricas, valores = zip(*sorted(zip(metricas, valores), key=lambda x: x[1], reverse=True))
    
    colors = sns.color_palette("Blues", n_colors=3)[::-1]
    bars = ax1.bar(metricas, valores, color=colors)
   
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    ax1.set_title("Métricas de Análise Textual", pad=20, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # chart
    ax2 = fig.add_subplot(gs[1], projection='polar')
    angles = np.linspace(0, 2*np.pi, len(metricas), endpoint=False)
    valores_norm = [(v - min(valores))/(max(valores) - min(valores)) for v in valores]
    valores_norm = np.concatenate((valores_norm, [valores_norm[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    ax2.plot(angles, valores_norm, 'o-', linewidth=2, color=colors[0])
    ax2.fill(angles, valores_norm, alpha=0.25, color=colors[0])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metricas)
    
    plt.tight_layout()
    return fig

def generate_wordcloud(tokens):
    """ Gera e retorna uma nuvem de palavras
    """

    if not tokens:
        return None
    
    set_custom_style()
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0])
    
    # gera texto a partir dos tokens
    text = " ".join(tokens)
    
    # gerar cores
    def color_func(*args, **kwargs):
        return "#%02x%02x%02x" % tuple(np.random.randint(0, 255, size=3))
    
    # gera nuvem de palavras
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        color_func=color_func,
        max_words=100,
        min_font_size=10,
        max_font_size=60,
        prefer_horizontal=0.7,
        relative_scaling=0.5,
        random_state=42
    ).generate(text)
   
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()
    ax.set_title("Análise de Frequência de Palavras", pad=20, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_tfidf(df_tfidf):
    """ Gera e retorna gráfico TF-IDF com melhor contraste de cores.
    """

    set_custom_style()
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0])
    
   
    n_bars = min(15, len(df_tfidf))
    colors = sns.color_palette("Blues", n_colors=n_bars)
    
    bars = ax.barh(range(n_bars), 
                   df_tfidf['tfidf'].head(n_bars)[::-1],
                   color=colors,
                   height=0.6)

    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(df_tfidf['termo'].head(n_bars)[::-1])
    
    max_width = max(df_tfidf['tfidf'].head(n_bars))  
    spacing = 0.01 * max_width  
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + spacing, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}',
                ha='left', va='center',
                fontsize=10)
    
    ax.set_title("Análise TF-IDF: Termos Mais Relevantes", pad=20, fontweight='bold')
    ax.set_xlabel("Valor TF-IDF")
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_word_frequency(word_counts):
    """ Gera gráfico das 15 palavras mais frequentes com cores mais equilibradas.
    """

    set_custom_style()
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0])
    
    # dados
    top_words = word_counts.most_common(15)
    words, freqs = zip(*top_words)
    y_pos = np.arange(len(words))
    
    colors = sns.color_palette("Blues", n_colors=len(words))[::-1] 
    
    bars = ax.barh(y_pos, freqs, align='center', color=colors, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    
    max_width = max(freqs)
    spacing = 0.01 * max_width 
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + spacing, bar.get_y() + bar.get_height()/2,
                f'{int(width)}',
                ha='left', va='center',
                fontsize=10)
    
    ax.set_title("Frequência das Palavras Mais Comuns", pad=20, fontweight='bold')
    ax.set_xlabel("Frequência")
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig
