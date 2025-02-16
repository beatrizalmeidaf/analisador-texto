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
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'teste.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return file_path

def process_text(text):
    save_temp_text(text)
    cleaned_text = clean_text()
    tokens = tokenizer(cleaned_text)
    word_counts = word_counter(tokens)
    metrics = calculate_metrics(tokens)
    df_tfidf = calculate_tfidf(tokens)
    return cleaned_text, tokens, word_counts, metrics, df_tfidf

def render_visualization(viz_type, tokens, metrics, word_counts, df_tfidf):
    if viz_type == "Nuvem de Palavras":
        st.subheader("Nuvem de Palavras")
        fig_wordcloud = generate_wordcloud(tokens)
        if fig_wordcloud:
            st.pyplot(fig_wordcloud)
    
    elif viz_type == "Métricas":
        st.subheader("Gráfico de Métricas")
        fig_metrics = plot_metrics(
            metrics['media'],
            metrics['mediana'],
            metrics['desvio_padrao']
        )
        st.pyplot(fig_metrics)
    
    elif viz_type == "Frequência de Palavras":
        st.subheader("Palavras Mais Frequentes")
        fig_freq = plot_word_frequency(word_counts)
        st.pyplot(fig_freq)
    
    elif viz_type == "Análise TF-IDF":
        st.subheader("Visualização TF-IDF")
        fig_tfidf = plot_tfidf(df_tfidf)
        st.pyplot(fig_tfidf)

def initialize_session_state():
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'viz_type' not in st.session_state:
        st.session_state.viz_type = "Nuvem de Palavras"
    if 'tab' not in st.session_state:
        st.session_state.tab = "Estatísticas"

def handle_viz_selection(viz_name):
    st.session_state.viz_type = viz_name
    st.session_state.tab = "Visualizações"


def main():
    initialize_session_state()

    st.title('Analisador de Texto')
    
    input_method = st.radio(
        "Escolha o método de entrada:",
        ("Digitar texto", "Upload de arquivo .txt")
    )
    
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
                with st.expander("Mostrar Preview do Arquivo"):
                    st.write(text_input[:1000] + "..." if len(text_input) > 1000 else text_input)
    
    analyze_button = st.button('Analisar Texto')
    if analyze_button:
        if text_input:
            with st.spinner('Processando texto...'):
                try:
                    st.session_state.processed_data = process_text(text_input)
                    st.session_state.tab = "Estatísticas"
                except Exception as e:
                    st.error(f"Erro durante o processamento: {str(e)}")
                    st.exception(e)
        else:
            st.warning('Por favor, insira um texto ou faça upload de um arquivo para análise.')
    
    if st.session_state.processed_data is not None:
        cleaned_text, tokens, word_counts, metrics, df_tfidf = st.session_state.processed_data
        
        st.session_state.tab = st.radio(
            "",
            ["Estatísticas", "Visualizações", "Busca e Informações", "Classificação e Sumarização"],
            horizontal=True,
            label_visibility="hidden",
            index=["Estatísticas", "Visualizações", "Busca e Informações", "Classificação e Sumarização"].index(st.session_state.tab)
        )
        
        st.divider()
        
        if st.session_state.tab == "Estatísticas":
            st.subheader("Análise Estatística")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Média Tamanho de Palavras", f"{metrics['media']:.2f}")
            with col2:
                st.metric("Mediana Tamanho de Palavras", f"{metrics['mediana']:.2f}")
            with col3:
                st.metric("Desvio Padrão Tamanho de Palavras", f"{metrics['desvio_padrao']:.2f}")
            
            st.divider()

            st.subheader("Análise TF-IDF")
            
            with st.container():
                st.markdown("""
                    A análise TF-IDF é uma métrica estatística que avalia a importância de uma palavra em um texto. 
                    Ela combina dois fatores:
                    
                    * **TF (Term Frequency)**: Frequência com que uma palavra aparece no texto
                    * **IDF (Inverse Document Frequency)**: Quão única ou rara é essa palavra
                    
                    Um valor TF-IDF alto indica que a palavra é:
                    * Muito frequente neste texto específico
                    * Relativamente rara em textos em geral
                    
                    Isso ajuda a identificar as palavras mais relevantes e características do seu texto.
                """)
            
            st.divider()
            
            st.subheader("Top 15 Termos Mais Relevantes")
            
            with st.container():
                if not df_tfidf.empty:
                    df_tfidf = df_tfidf.reset_index().iloc[:, 1:]
                    styled_df = df_tfidf.head(15).style.format(precision=4)
            
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.warning("Não há dados TF-IDF disponíveis para exibição.")
        
        elif st.session_state.tab == "Visualizações":
            st.container()
            render_visualization(st.session_state.viz_type, tokens, metrics, word_counts, df_tfidf)

            st.divider()

            st.markdown("### Selecione a Visualização")
            
            viz_options = {
                "Nuvem de Palavras": "Visualização das palavras mais frequentes em forma de nuvem",
                "Métricas": "Gráfico comparativo das métricas estatísticas",
                "Frequência de Palavras": "Distribuição das palavras mais frequentes",
                "Análise TF-IDF": "Visualização dos termos mais relevantes"
            }

            cols = st.columns(4)
            for idx, (viz_name, viz_desc) in enumerate(viz_options.items()):
                with cols[idx]:
                    if st.button(
                        viz_name,
                        key=f"viz_btn_{idx}",
                        on_click=handle_viz_selection,
                        args=(viz_name,),
                        use_container_width=True,
                        type="primary" if viz_name == st.session_state.viz_type else "secondary"
                    ):
                        pass
                    st.markdown(f"<small>{viz_desc}</small>", unsafe_allow_html=True)
        
        elif st.session_state.tab == "Busca e Informações":
            st.info("Funcionalidade de busca e informações em desenvolvimento")
        
        elif st.session_state.tab == "Classificação e Sumarização":
            st.info("Funcionalidade de classificação e sumarização em desenvolvimento")
    
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