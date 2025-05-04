import os
import streamlit as st
import traceback
from transformers import pipeline

# carregamento com cache para economizar recursos
@st.cache_resource
def load_summarizer(texto):
    try:
        summarizer = pipeline("summarization", model="recogna-nlp/ptt5-base-summ-cstnews")

        summary = summarizer(texto)
        return summary[0]['summary_text']
    except Exception as e:
      
        return f"Desculpe, ocorreu um erro ao gerar o resumo: {str(e)}"