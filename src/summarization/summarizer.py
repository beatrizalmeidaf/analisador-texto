import streamlit as st
from transformers import pipeline

# carregamento com cache para economizar recursos
@st.cache_resource
def load_summarizer_pipeline():
    summarizer_pipeline = pipeline("text-generation", 
                                 model="cnmoro/TeenyTinyLlama-460m-Summarizer-PTBR", 
                                 tokenizer="cnmoro/TeenyTinyLlama-460m-Summarizer-PTBR")
    return summarizer_pipeline