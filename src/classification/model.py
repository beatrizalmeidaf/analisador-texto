import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import pipeline

@st.cache_resource
def load_classifiers():

    classifier_genero = pipeline("text-classification", 
                                model="classla/xlm-roberta-base-multilingual-text-genre-classifier", 
                                top_k=1)
    return classifier_genero

def classificador(text_input):
    """Classifica o texto por categoria e gênero textual"""
    try:
        classifier_genero = load_classifiers()
        genero_result = classifier_genero(text_input)
        
        if isinstance(genero_result, list) and isinstance(genero_result[0], list):
            genero = genero_result[0]
        elif isinstance(genero_result, list):
            genero = genero_result
        else:
            genero = [genero_result]
        
        return genero
    except Exception as e:
        st.error(f"Erro ao classificar o texto: {str(e)}")
        return None, None


def plot_classification(genero):
    """Cria visualização apenas para os resultados da classificação de gênero textual."""

    plt.style.use('ggplot')
    colors = sns.color_palette("viridis", 2)

    if genero:
        genero_label = genero[0]['label']
        genero_score = genero[0]['score']

        gen_scores = [genero_score, 1 - genero_score]
        gen_labels = [genero_label, 'Outros']

        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(['Gênero Detectado', 'Outros Gêneros'], gen_scores, color=colors)
        ax.set_xlim(0, 1)
        ax.set_title('Classificação por Gênero Textual', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confiança')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{gen_labels[i]}: {gen_scores[i]:.2%}',
                    va='center', fontsize=10)

        plt.tight_layout()
        plt.show()
        return fig
    else:
        return None




