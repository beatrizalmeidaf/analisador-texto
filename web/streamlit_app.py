import streamlit as st
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import tokenizer, word_counter
from src.analysis.statistics import calculate_metrics, calculate_tfidf
from src.analysis.visualization import plot_metrics, generate_wordcloud, plot_tfidf, plot_word_frequency

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
    
    # método de entrada
    input_method = st.radio(
        "Escolha o método de entrada:",
        ("Digitar texto", "Upload de arquivo .txt")
    )
    
    # obter texto de entrada
    text_input = ""
    if input_method == "Digitar texto":
        text_input = st.text_area("Digite ou cole seu texto aqui:", height=200)
    else:
        uploaded_file = st.file_uploader("Escolha um arquivo .txt", type="txt")
        if uploaded_file:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_input = uploaded_file.read().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_input:
                st.subheader("Preview do arquivo:")
                st.write(text_input[:500] + "..." if len(text_input) > 500 else text_input)
    
    # botão principal de análise
    if st.button('Analisar Texto'):
        if text_input:
            with st.spinner('Processando texto...'):
                try:
                    # processamento inicial
                    save_temp_text(text_input)
                    cleaned_text = clean_text()
                    tokens = tokenizer(cleaned_text)
                    word_counts = word_counter(tokens)
                    
                    # diferentes tipos de análise
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Estatísticas", 
                        "Visualizações", 
                        "Busca e Informações",
                        "Classificação e Sumarização"
                    ])
                    
                    with tab1:
                        st.subheader("Análise Estatística")
                        
                        # métricas básicas
                        metrics = calculate_metrics(tokens)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Média Tamanho de Palavras", f"{metrics['media']:.2f}")
                        with col2:
                            st.metric("Mediana Tamanho de Palavras", f"{metrics['mediana']:.2f}")
                        with col3:
                            st.metric("Desvio Padrão Tamanho de Palavras", f"{metrics['desvio_padrao']:.2f}")
                        
                        # TF-IDF
                        st.subheader("Análise TF-IDF")
                        df_tfidf = calculate_tfidf(tokens)
                        
                        # tabela TF-IDF
                        st.write("Top 15 termos por TF-IDF:")
                        st.dataframe(df_tfidf.head(15))
                        
                        # gráfico TF-IDF
                        st.write("Visualização TF-IDF:")
                        fig_tfidf = plot_tfidf(df_tfidf)
                        st.pyplot(fig_tfidf)
                    
                    with tab2:
                        st.subheader("Visualizações")
                        
                        # gráfico de métricas
                        st.write("Gráfico de Métricas")
                        fig_metrics = plot_metrics(
                            metrics['media'],
                            metrics['mediana'],
                            metrics['desvio_padrao']
                        )
                        st.pyplot(fig_metrics)
                        
                        # gráfico de frequência de palavras
                        st.write("Palavras Mais Frequentes")
                        fig_freq = plot_word_frequency(word_counts)
                        st.pyplot(fig_freq)
                        
                        # nuvem de palavras
                        st.write("Nuvem de Palavras")
                        fig_wordcloud = generate_wordcloud(tokens)
                        if fig_wordcloud:
                            st.pyplot(fig_wordcloud)
                    
                    with tab3:
                        st.info("Funcionalidade de busca e informações em desenvolvimento")
                    
                    with tab4:
                        st.info("Funcionalidade de classificação e sumarização em desenvolvimento")
                
                except Exception as e:
                    st.error(f"Erro durante o processamento: {str(e)}")
                    st.exception(e)
        else:
            st.warning('Por favor, insira um texto ou faça upload de um arquivo para análise.')
    
    # informações sobre o analisador
    with st.expander("Sobre o Analisador de Texto"):
        st.write("""
        Este analisador de texto realiza as seguintes operações:
        1. **Limpeza e Preprocessamento**: Remove pontuações, caracteres especiais e normaliza o texto
        2. **Análise Estatística**: Calcula métricas sobre o comprimento das palavras
        3. **Análise TF-IDF**: Identifica termos mais relevantes no texto
        4. **Visualizações**: Gera gráficos e nuvem de palavras
        5. **Busca e Informações**: *(em desenvolvimento)*
        6. **Classificação e Sumarização**: *(em desenvolvimento)*
        """)

if __name__ == '__main__':
    main()