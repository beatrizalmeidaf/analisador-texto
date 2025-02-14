import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import tokenizer, word_counter

def save_temp_text(text):
    """ Salva o texto em um arquivo temporário para processamento 
    """
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'teste.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return file_path

def main():
    st.title('Analisador de Texto')
    
    # opções de entrada: texto ou arquivo
    input_method = st.radio(
        "Escolha o método de entrada:",
        ("Digitar texto", "Upload de arquivo .txt")
    )
    
    text_input = ""
    if input_method == "Digitar texto":
        text_input = st.text_area("Digite ou cole seu texto aqui:", height=200)
    else:
        uploaded_file = st.file_uploader("Escolha um arquivo .txt", type="txt")
        if uploaded_file is not None:
            # lidar com diferentes encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_input = uploaded_file.read().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            # mostrar preview
            if text_input:
                st.subheader("Preview do arquivo:")
                st.write(text_input[:500] + "..." if len(text_input) > 500 else text_input)
            else:
                st.error("Não foi possível ler o arquivo. Verifique se ele está em um formato de texto válido.")
    
    process_button = st.button('Analisar Texto')
    
    if process_button:
        if text_input:
            save_temp_text(text_input)
            
            with st.spinner('Processando texto...'):
                try:
                    # processa o texto
                    cleaned_text = clean_text()
                    tokens = tokenizer(cleaned_text)
                    word_counts = word_counter(tokens)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader('Texto Original')
                        st.write(text_input[:1000] + "..." if len(text_input) > 1000 else text_input)
                    
                    with col2:
                        st.subheader('Texto Limpo')
                        st.write(cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text)
                    
                    st.subheader('Tokens (primeiros 100)')
                    st.write(tokens[:100])
                    
                    # palavras mais frequentes
                    st.subheader('Top 15 Palavras Mais Frequentes')
                    top_words = word_counts.most_common(15)
                    df = pd.DataFrame(top_words, columns=['Palavra', 'Frequência'])
                    
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.table(df)
                    
                    with col2:
                        # visualização - gráfico de barras
                        if len(top_words) > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(df['Palavra'], df['Frequência'], color='blue')
                            ax.set_title('Frequência das Palavras')
                            ax.set_xlabel('Frequência')
                            ax.invert_yaxis()  
                            
                            # adiciona os valores ao lado das barras
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                                        f'{width:.0f}', ha='left', va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Ocorreu um erro durante o processamento: {str(e)}")
                    st.exception(e)  
                    
        else:
            st.warning('Por favor, insira um texto ou faça upload de um arquivo para análise.')
    
    # informações sobre o analisador
    with st.expander("Sobre o Analisador de Texto"):
        st.write("""
        Este analisador de texto realiza as seguintes operações:
        1. **Limpeza de texto**: Remove pontuações, caracteres especiais, números e converte para minúsculas.
        2. **Normalização**: Corrige erros ortográficos comuns.
        3. **Tokenização**: Divide o texto em unidades menores (tokens).
        4. **Remoção de stopwords**: Elimina palavras comuns sem valor semântico significativo.
        5. **Análise de frequência**: Calcula e exibe as palavras mais frequentes no texto.
        """)

if __name__ == '__main__':
    main()