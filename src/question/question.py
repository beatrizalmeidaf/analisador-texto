import streamlit as st
from transformers import pipeline

# carregamento com cache para economizar recursos
@st.cache_resource
def load_qa_pipeline():
    qa_pipeline = pipeline("text2text-generation", model="unicamp-dl/ptt5-v2-base", tokenizer="unicamp-dl/ptt5-v2-base")
    return qa_pipeline
