import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

LABELS_MAP_PT = {
    'Other': 'Outro',
    'Information/Explanation': 'Informativo/Explicativo',
    'News': 'Notícia',
    'Instruction': 'Instrução',
    'Opinion/Argumentation': 'Opinião/Argumentação',
    'Forum': 'Fórum',
    'Prose/Lyrical': 'Prosa/Lírico',
    'Legal': 'Jurídico',
    'Promotion': 'Promocional'
}

GENRE_COLORS = {
    'Outro': '#4C78A8',
    'Informativo/Explicativo': '#72B7B2',
    'Notícia': '#54A24B',
    'Instrução': '#EECA3B',
    'Opinião/Argumentação': '#F58518',
    'Fórum': '#E45756',
    'Prosa/Lírico': '#B279A2',
    'Jurídico': '#9D755D',
    'Promocional': '#BAB0AC'
}

@st.cache_resource
def load_classifiers():
    """Carrega o modelo de classificação de gênero textual."""
    classifier_genero = pipeline("text-classification", 
                                model="classla/xlm-roberta-base-multilingual-text-genre-classifier", 
                                top_k=None)  
    return classifier_genero

def classificador(text_input):
    """Classifica o texto por gênero textual e retorna os resultados ordenados por confiança."""
    try:
        classifier_genero = load_classifiers()
        genero_result = classifier_genero(text_input)
        
        if isinstance(genero_result, list) and isinstance(genero_result[0], list):
            genero = genero_result[0]
        elif isinstance(genero_result, list):
            genero = genero_result
        else:
            genero = [genero_result]
        
        genero = sorted(genero, key=lambda x: x['score'], reverse=True)
        
        return genero
    except Exception as e:
        st.error(f"Erro ao classificar o texto: {str(e)}")
        return None

def create_plotly_visualization(genero_results):
    """Cria uma visualização interativa e moderna usando Plotly."""
    if not genero_results:
        return None
    
    # extrair os top 5 resultados
    top_results = genero_results[:5]
    
    # preparar os dados para o gráfico
    labels = [LABELS_MAP_PT.get(item['label'], item['label']) for item in top_results]
    scores = [item['score'] for item in top_results]
    colors = [GENRE_COLORS.get(label, '#636EFA') for label in labels]
    
    df = pd.DataFrame({
        'Gênero': labels,
        'Confiança': scores
    })
    
    fig = px.bar(
        df,
        x='Confiança', 
        y='Gênero',
        orientation='h',
        color='Gênero',
        color_discrete_map={label: color for label, color in zip(labels, colors)},
        text=df['Confiança'].apply(lambda x: f'{x:.1%}'),
        title='Classificação por Gênero Textual',
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=18, family="Arial, sans-serif"),
        xaxis=dict(
            title='Nível de Confiança',
            tickformat='.0%',
            range=[0, 1],
            gridcolor='#EEEEEE'
        ),
        yaxis=dict(
            title='',
            autorange="reversed"  
        ),
        margin=dict(l=20, r=20, t=70, b=50)
    )
    
    fig.update_traces(
        textposition='auto',
        texttemplate='%{text}',
        hovertemplate='<b>%{y}</b><br>Confiança: %{x:.1%}<extra></extra>'
    )
    
    return fig

