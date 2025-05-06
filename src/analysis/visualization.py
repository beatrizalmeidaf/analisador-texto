import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects

# configurações globais de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Blues")

def set_custom_style():
    """ Definição de customizações"""

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['figure.facecolor'] = '#F0F2F6'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.alpha'] = 0.2

def plot_metrics(media, mediana, std):
    """
    Gera um gráfico de métricas estatísticas com visualização
    aprimorada, incluindo gráfico de barras e radar.
    """
    # configuração específica das metricas estatisticas 
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    

    fig = plt.figure(figsize=(14, 8), facecolor='white')
    
    # título principal para o relatório
    fig.suptitle('ANÁLISE ESTATÍSTICA DO CORPUS TEXTUAL', 
                fontsize=18, fontweight='bold', y=0.98, color='#333333')
    
    gs = GridSpec(2, 3, figure=fig, height_ratios=[4, 1], width_ratios=[2, 1, 1])
    
    metricas = ["Média", "Mediana", "Desvio Padrão"]
    valores = [media, mediana, std]
    descricoes = [
        "Valor médio de palavras por documento",
        "Valor central (50%) da distribuição", 
        "Dispersão dos dados em relação à média"
    ]
    
    # ordenar do maior para o menor
    indices_ordenados = np.argsort(valores)[::-1]
    metricas = [metricas[i] for i in indices_ordenados]
    valores = [valores[i] for i in indices_ordenados]
    descricoes = [descricoes[i] for i in indices_ordenados]
    
    custom_colors = ["#3f37c9", "#4361ee", "#3a86ff"]
    
    max_valor = max(valores)
    min_valor = min(valores)
    range_valor = max_valor - min_valor
    
    # gráfico de barras principal
    ax1 = fig.add_subplot(gs[0, 0])
    
    bars = ax1.bar(
        metricas, 
        valores, 
        color=custom_colors,
        width=0.65,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.85
    )

    for i, bar in enumerate(bars):
        ax1.add_patch(plt.Rectangle(
            (bar.get_x(), 0), 
            bar.get_width(), 
            bar.get_height(), 
            fill=True, 
            alpha=0.1,
            color='white',
            linewidth=0
        ))
        
        height = bar.get_height()
        txt = ax1.text(
            bar.get_x() + bar.get_width()/2, 
            height * 1.02,
            f'{height:.2f}',
            ha='center', 
            va='bottom',
            fontsize=12, 
            fontweight='bold',
            color=custom_colors[i]
        )
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=2, foreground='white')
        ])
        
        ax1.text(
            i, 
            -max_valor * 0.15, 
            descricoes[i],
            ha='center',
            va='center',
            fontsize=9,
            color='#555555',
            style='italic',
            wrap=True
        )
    
    ax1.set_ylim(0, max_valor * 1.25)
    ax1.set_title("Principais Métricas Estatísticas", 
                  pad=20, fontweight='bold', fontsize=16, color='#333333')
    ax1.set_ylabel("Valor", fontsize=12, fontweight='bold', labelpad=10)
    
    ax1.axhline(y=np.mean(valores), color='#ff6361', 
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Média das métricas: {np.mean(valores):.2f}')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
 
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
    
    ax2 = fig.add_subplot(gs[0, 1:], polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(metricas), endpoint=False).tolist()
    
    # normalizar valores para o radar (escala 0-1)
    valores_norm = [(v - min_valor)/(max_valor - min_valor + 1e-10) for v in valores]
    
    # fechar o polígono
    valores_norm.append(valores_norm[0])
    angles.append(angles[0])
    metricas_radar = metricas + [metricas[0]]
    valores_radar = valores + [valores[0]]

    for i in range(len(metricas)):
    
        these_angles = [angles[0], angles[i], angles[(i+1)%len(metricas)]]
        these_values = [0.0, valores_norm[i], valores_norm[(i+1)%len(metricas)]]
   
        these_angles.append(these_angles[0])
        these_values.append(these_values[0])
        
        # plotar o segmento
        ax2.fill(these_angles, these_values, alpha=0.65, color=custom_colors[i])
        
        ax2.plot(these_angles, these_values, color=custom_colors[i], 
                 linewidth=2, alpha=0.8)

    ax2.scatter(angles[:-1], valores_norm[:-1], s=120, 
                color='white', edgecolor=custom_colors, linewidth=2, zorder=10)
  
    for i, (angle, value, valor_real) in enumerate(zip(angles[:-1], valores_norm[:-1], valores[:-1])):
        # calcular posição ajustada
        x = angle
        y = value + 0.05 if value < 0.9 else value - 0.15
        
        txt = ax2.text(x, y, f'{valor_real:.2f}',
                   ha='center', va='center', fontsize=11, 
                   fontweight='bold', color=custom_colors[i])
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white')
        ])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metricas, fontsize=12, fontweight='bold')
    ax2.yaxis.grid(False)
    ax2.xaxis.grid(False)
    ax2.set_yticks([])
    
    ax2.set_title("Comparação Relativa", 
                 pad=25, fontweight='bold', fontsize=16, color='#333333')
    
    for r in [0.25, 0.5, 0.75, 1.0]:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', 
                          alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.add_artist(circle)

        valor_referencia = r * (max_valor - min_valor) + min_valor
        ax2.text(np.pi/4, r, f'{int(r*100)}%', 
                color='gray', alpha=0.7, fontsize=8, ha='left', va='center')
    
    # tabela de resumo
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    dados_tabela = [
        ['Métrica', 'Valor', 'Percentual do Máximo', 'Interpretação'],
    ]
    
    for i, (metrica, valor) in enumerate(zip(metricas, valores)):
        percentual = (valor / max_valor) * 100
        # determinar interpretação básica
        if metrica == "Média":
            interpretacao = "Valor central esperado"
        elif metrica == "Mediana":
            if mediana < media:
                interpretacao = "Distribuição com viés positivo"
            else:
                interpretacao = "Distribuição com viés negativo"
        else:  # desvio padrão
            if std < media * 0.2:
                interpretacao = "Baixa variabilidade"
            elif std < media * 0.5:
                interpretacao = "Variabilidade moderada"
            else:
                interpretacao = "Alta variabilidade"
                
        dados_tabela.append([
            metrica, 
            f'{valor:.2f}', 
            f'{percentual:.1f}%',
            interpretacao
        ])
    
    tabela = ax3.table(
        cellText=dados_tabela[1:],
        colLabels=dados_tabela[0],
        loc='center',
        cellLoc='center',
        bbox=[0.05, 0.0, 0.9, 0.9]
    )
    
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)

    for i, key in enumerate(tabela._cells):
        cell = tabela._cells[key]
        if key[0] == 0: 
            cell.set_text_props(
                weight='bold', 
                color='white'
            )
            cell.set_facecolor('#333333')
        else:  
            if i % 4 == 1: 
                cell.set_text_props(weight='bold', color=custom_colors[key[0]-1])
    
    fig.text(0.5, 0.02, 
             'Análise estatística baseada em processamento do corpus textual completo',
             fontsize=9, ha='center', color='#555555', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
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
    
    plt.tight_layout()
    return fig


def plot_tfidf(df_tfidf):
    """
    Gera um gráfico TF-IDF com design profissional e interativo.
    """
    set_custom_style()
    
    n_bars = min(15, len(df_tfidf))
    terms = df_tfidf['termo'].head(n_bars)[::-1]
    values = df_tfidf['tfidf'].head(n_bars)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_blue', ['#1a75ff', '#00264d'], N=n_bars)
    colors = [custom_cmap(i / n_bars) for i in range(n_bars)]
    
    # adicionar barras com bordas suaves
    bars = ax.barh(range(n_bars), values, color=colors, height=0.65, 
                  alpha=0.85, edgecolor='white', linewidth=0.7)
    
    # adicionar uma linha de referência vertical
    median_value = np.median(values)
    ax.axvline(x=median_value, color='#444444', linestyle='--', alpha=0.5, 
              label=f'Mediana: {median_value:.4f}')
    
    # configurar eixos
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(terms, fontsize=11)

    max_width = max(values)
    spacing = 0.005 * max_width
    
    for bar in bars:
        width = bar.get_width()
        percentage = (width / max_width) * 100

        if width > max_width * 0.25:
            text_color = 'white'
            x_position = width - spacing * 5
            ha_align = 'right'
        else:
            text_color = '#333333'
            x_position = width + spacing * 5
            ha_align = 'left'
            
        ax.text(x_position, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f} ({percentage:.1f}%)', 
                ha=ha_align, va='center', fontsize=9,
                color=text_color, fontweight='bold')

    ax.set_title("Análise TF-IDF: Termos Mais Relevantes no Documento", 
                pad=20, fontweight='bold', fontsize=16, color='#333333')
    ax.set_xlabel("Valor TF-IDF", fontsize=12, fontweight='bold', labelpad=10)
    
    fig.text(0.01, 0.01, 
             "TF-IDF (Term Frequency-Inverse Document Frequency) mede a importância de um termo em um documento\n"
             "em relação a uma coleção de documentos.", 
             fontsize=9, color='#555555', ha='left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    ax.set_xlim(0, max_width * 1.15)
    
    ax.legend(loc='lower right', frameon=True, framealpha=0.8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def plot_word_frequency(word_counts):
    """ Gera gráfico das 15 palavras mais frequentes.
    """

    set_custom_style()
    
    top_words = word_counts.most_common(15)
    words, freqs = zip(*top_words)
    y_pos = np.arange(len(words))

    fig, ax = plt.subplots(figsize=(10, 6))

    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_blue', ['#00264d','#1a75ff'], N=len(words))
    colors = [custom_cmap(i/len(words)) for i in range(len(words))]
    
    # adicionar barras com estilo profissional
    bars = ax.barh(y_pos, freqs, align='center', color=colors, 
                  height=0.65, alpha=0.9, edgecolor='white', linewidth=0.7)
    
    # adicionar linha de média
    mean_freq = np.mean(freqs)
    ax.axvline(x=mean_freq, color='#ff6666', linestyle='--', alpha=0.7,
              linewidth=1.5, label=f'Média: {int(mean_freq)}')
    
    # configurar eixos
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis() 
    
    # formatar eixo x para usar separadores de milhares
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.'))
    )
    
    # adicionar rótulos nas barras
    for i, (bar, freq) in enumerate(zip(bars, freqs)):
        width = bar.get_width()
        
        percentage = (freq / sum(freqs)) * 100

        if width > max(freqs) * 0.4:
            text_color = 'white'
            x_position = width * 0.93
            ha_align = 'right'
        else:
            text_color = '#333333'
            x_position = width + max(freqs) * 0.02
            ha_align = 'left'
            
        ax.text(x_position, bar.get_y() + bar.get_height()/2, 
                f'{int(freq):,} ({percentage:.1f}%)'.replace(',', '.'), 
                ha=ha_align, va='center', fontsize=9,
                color=text_color, fontweight='bold')
 
    ax.set_title("Análise de Frequência das Palavras Mais Comuns", 
                pad=20, fontweight='bold', fontsize=16, color='#333333')
    ax.set_xlabel("Frequência (contagem)", fontsize=12, fontweight='bold', labelpad=10)
    

    for i, word in enumerate(words):
        y = y_pos[i]
        ax.axhspan(y - 0.3, y + 0.3, color=colors[i], alpha=0.05)
    

    ax.legend(loc='lower right', frameon=True, framealpha=0.8)
    
    fig.text(0.01, 0.01, 
             "Palavras mais frequentes encontradas após remoção de stopwords e normalização do texto.", 
             fontsize=9, color='#555555', ha='left')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig