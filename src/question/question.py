import streamlit as st
from transformers import pipeline

# carregamento com cache para economizar recursos
@st.cache_resource
def load_qa_pipeline():
    qa_pipeline = pipeline("text2text-generation", model="unicamp-dl/mt5-base-mmarco-v2", tokenizer="unicamp-dl/mt5-base-mmarco-v2")
    return qa_pipeline
