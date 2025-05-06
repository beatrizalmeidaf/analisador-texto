import streamlit as st
import sys
import os
from io import StringIO
import gc

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from src.extract_pdf.extractor import extract_pdf_to_text
from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import tokenizer, word_counter
from src.analysis.statistics import calculate_metrics, calculate_tfidf
from src.classification.model import classificador, create_plotly_visualization
from src.analysis.visualization import plot_metrics, generate_wordcloud, plot_tfidf, plot_word_frequency
from src.question.question import answer_question
from src.summarization.summarizer import resumir_texto
# from src.representation import bow, cooccurrence

st.set_page_config(layout="wide")

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

def process_large_file(uploaded_file, chunk_size=1024*1024):  # 1MB chunks
    """ Processa arquivos grandes em chunks"""
    text_chunks = []
    
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            uploaded_file.seek(0)
            
            while True:
                chunk = uploaded_file.read(chunk_size).decode(encoding)
                if not chunk:
                    break
                text_chunks.append(chunk)
            
                gc.collect()
            
            return ''.join(text_chunks)
            
        except UnicodeDecodeError:
            continue
    
    raise ValueError("Não foi possível decodificar o arquivo com nenhuma codificação suportada.")

@st.cache_data(max_entries=3)  # cache apenas os 3 últimos resultados
def process_text_chunked(text):
    """ Processa com memória eficiente """
    chunk_size = 1024 * 1024  # 1MB chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    all_tokens = []
    all_word_counts = {}
    
    for chunk in chunks:

        cleaned_chunk = clean_text(chunk)
        chunk_tokens = tokenizer(cleaned_chunk)
        
        all_tokens.extend(chunk_tokens)
        
        chunk_counts = word_counter(chunk_tokens)
        for word, count in chunk_counts.items():
            all_word_counts[word] = all_word_counts.get(word, 0) + count
        
        gc.collect()
    
    metrics = calculate_metrics(all_tokens)
    df_tfidf = calculate_tfidf(all_tokens)
    
    return ' '.join(all_tokens), all_tokens, all_word_counts, metrics, df_tfidf

def save_temp_text(text):
    """ Salva o arquivo de forma temporária """
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'teste.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return file_path

def process_text(text):
    """ Faz os processamentos necessários no texto """
    save_temp_text(text)
    cleaned_text = clean_text()
    tokens = tokenizer(cleaned_text)
    word_counts = word_counter(tokens)
    metrics = calculate_metrics(tokens)
    df_tfidf = calculate_tfidf(tokens)

    return cleaned_text, tokens, word_counts, metrics, df_tfidf

def render_visualization(viz_type, tokens, metrics, word_counts, df_tfidf):
    """ Carrega as visualizações necessárias para cada tipo de gráfico """
    if viz_type == "Nuvem de Palavras":
        st.subheader("Nuvem de Palavras")
        col1, col2, col3 = st.columns([1,5,1])  
        with col2:  
            fig_wordcloud = generate_wordcloud(tokens)
            if fig_wordcloud:
                st.pyplot(fig_wordcloud, use_container_width=True)
    
    elif viz_type == "Métricas":
        st.subheader("Gráfico de Métricas")
        col1, col2, col3 = st.columns([1,5,1])
        with col2:
            fig_metrics = plot_metrics(
                metrics['media'],
                metrics['mediana'],
                metrics['desvio_padrao']
            )
            st.pyplot(fig_metrics, use_container_width=True)
    
    elif viz_type == "Frequência de Palavras":
        st.subheader("Palavras Mais Frequentes")
        col1, col2, col3 = st.columns([1,5,1])
        with col2:
            fig_freq = plot_word_frequency(word_counts)
            st.pyplot(fig_freq, use_container_width=True)
    
    elif viz_type == "Análise TF-IDF":
        st.subheader("Visualização TF-IDF")
        col1, col2, col3 = st.columns([1,5,1])
        with col2:
            fig_tfidf = plot_tfidf(df_tfidf)
            st.pyplot(fig_tfidf, use_container_width=True)

def initialize_session_state():
    """ Inicialização de estados """
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'viz_type' not in st.session_state:
        st.session_state.viz_type = "Nuvem de Palavras"
    if 'tab' not in st.session_state:
        st.session_state.tab = "Estatísticas"

def handle_viz_selection(viz_name):
    """ Seleção das abas para cada tipo de gráficos """
    st.session_state.viz_type = viz_name
    st.session_state.tab = "Visualizações"

def get_metrics_interpretation(media, mediana, desvio_padrao):
    """Retorna interpretações concisas baseadas nos valores específicos das métricas"""
    interpretacao = ""
    
    # interpretação da média
    if media < 4:
        interpretacao += "**Média baixa**: Texto com predominância de palavras curtas, sugerindo linguagem mais informal ou simples.\n\n"
    elif media >= 4 and media <= 6:
        interpretacao += "**Média equilibrada**: Texto com boa distribuição entre palavras curtas e longas, comum em textos jornalísticos ou relatórios.\n\n"
    else:
        interpretacao += "**Média alta**: Texto com palavras mais longas e complexas, indicando possível natureza técnica ou acadêmica.\n\n"
    
    # interpretação do desvio padrão
    if desvio_padrao < 1.5:
        interpretacao += "**Baixa variação**: Palavras com tamanhos semelhantes, sugerindo estilo uniforme e direto."
    elif desvio_padrao >= 1.5 and desvio_padrao <= 2.5:
        interpretacao += "**Variação moderada**: Vocabulário diversificado com bom equilíbrio, típico de textos formais bem estruturados."
    elif desvio_padrao > 2.5 and desvio_padrao <= 3.0:
        interpretacao += "**Alta variação**: Mistura frequente de palavras curtas e longas, indicando linguagem expressiva ou criativa."
    else:
        interpretacao += "**Variação muito alta**: Vocabulário extremamente diverso, podendo refletir informalidade, uso de termos técnicos ou transcrições de fala."
    
    return interpretacao

def render_classification_section(tokens):
    """Renderiza a seção de classificação no Streamlit com visualizações aprimoradas."""
    st.subheader("Classificação do Texto")
    
    st.markdown("""
    Essa seção utiliza modelos de aprendizado profundo (XLM-RoBERTa) para classificar o texto quanto ao seu gênero textual.
    """)
    
    with st.spinner('Realizando classificação do texto... Isso pode levar alguns segundos.'):
        texto_completo = " ".join(tokens)
        texto_para_classificacao = texto_completo[:min(len(texto_completo), 512)]
        
        genero_results = classificador(texto_para_classificacao)
        
        if genero_results and isinstance(genero_results, list) and len(genero_results) > 0:
    
            genero_principal = genero_results[0]
            genero_nome_pt = LABELS_MAP_PT.get(genero_principal['label'], genero_principal['label'])
            
            st.success(f"**Gênero Textual Principal:** {genero_nome_pt}")
            st.progress(genero_principal['score'])
            st.caption(f"Confiança: {genero_principal['score']:.1%}")
            
            # adicionar descrição do gênero detectado
            st.markdown(f"""
            ### Sobre o gênero "{genero_nome_pt}":
            {get_genre_description(genero_nome_pt)}
            """)
            
            # visualização em barras horizontais com Plotly
            st.subheader("Distribuição de Confiança por Gênero")
            bar_fig = create_plotly_visualization(genero_results)
            if bar_fig:
                st.plotly_chart(bar_fig, use_container_width=True)
            
            with st.expander("Sobre os gêneros textuais"):
                st.markdown("""
                ### Descrição dos Gêneros Textuais

                - **Informativo/Explicativo**: Textos que transmitem conhecimento ou explicam conceitos, processos, fenômenos.
                - **Notícia**: Conteúdos jornalísticos objetivos que relatam fatos recentes de interesse público.
                - **Instrução**: Textos que ensinam procedimentos, como manuais, tutoriais e receitas.
                - **Opinião/Argumentação**: Textos que apresentam posicionamentos e argumentos para defender uma ideia.
                - **Fórum**: Conteúdos de discussão, típicos de ambientes online com interação entre vários participantes.
                - **Prosa/Lírico**: Textos literários ou poéticos, com linguagem elaborada e função estética.
                - **Jurídico**: Documentos legais como contratos, leis, regulamentos e termos de uso.
                - **Promocional**: Textos publicitários ou de marketing que visam promover produtos, serviços ou ideias.
                - **Outro**: Textos que não se encaixam claramente nas categorias acima.
                """)
        else:
            st.warning("Não foi possível classificar o texto. Verifique se o texto possui conteúdo suficiente ou tente novamente.")

def render_summarization_section(texto_completo):
    """Renderiza a seção de sumarização separadamente"""
    st.subheader("Sumarização do Texto")
    
    st.markdown("""
    Essa ferramenta utiliza o modelo do Recogna NLP para criar um resumo conciso 
    do texto analisado, mantendo os pontos principais.
    """)
    
    if st.button("Gerar Resumo"):
        st.info("O processamento pode levar alguns segundos. Por favor, aguarde...")
        with st.spinner("Gerando resumo..."):
            try:
                resumo = resumir_texto(texto_completo)
                st.write(resumo)
            except Exception as e:
                st.error(f"Erro ao gerar resumo: {str(e)}")

def get_genre_description(genre_name):
    """Retorna uma descrição detalhada para o gênero textual específico."""
    descriptions = {
        'Informativo/Explicativo': """
        Textos informativos ou explicativos têm como objetivo principal transmitir conhecimento de forma clara e objetiva. 
        Esses textos geralmente apresentam fatos, conceitos, definições ou explicações sobre determinado assunto, 
        utilizando linguagem objetiva e precisão técnica. Exemplos incluem artigos científicos, enciclopédias, 
        textos didáticos e reportagens aprofundadas.
        """,
        
        'Notícia': """
        Textos jornalísticos que relatam acontecimentos recentes considerados relevantes para a sociedade. 
        São caracterizados pela objetividade, concisão e estrutura que privilegia as informações mais importantes 
        no início (pirâmide invertida). Respondem às perguntas essenciais: o quê, quem, quando, onde, como e por quê.
        """,
        
        'Instrução': """
        Textos que orientam o leitor sobre como realizar uma tarefa ou procedimento específico. 
        Apresentam linguagem direta, uso de verbos no imperativo ou infinitivo, e frequentemente incluem 
        elementos visuais como diagramas ou imagens. Exemplos incluem manuais de instruções, receitas culinárias, 
        tutoriais e guias passo a passo.
        """,
        
        'Opinião/Argumentação': """
        Textos que apresentam um ponto de vista sobre determinado tema e desenvolvem argumentos para sustentá-lo. 
        Utilizam técnicas de persuasão, evidências, exemplos e raciocínio lógico para convencer o leitor. 
        Incluem artigos de opinião, editoriais, resenhas críticas, ensaios argumentativos e manifestos.
        """,
        
        'Fórum': """
        Textos produzidos em ambientes de discussão coletiva, caracterizados pela interação entre múltiplos 
        participantes. Apresentam linguagem mais informal, referências diretas a outros comentários, 
        e estrutura não-linear. Comuns em plataformas online como Reddit, Quora e grupos de discussão.
        """,
        
        'Prosa/Lírico': """
        Textos literários que privilegiam a função estética da linguagem. A prosa narrativa conta histórias 
        por meio de personagens, enredo, tempo e espaço. Já os textos líricos expressam emoções e sentimentos 
        com linguagem subjetiva e recursos como metáforas, rimas e ritmo. Inclui romances, contos, poemas e crônicas.
        """,
        
        'Jurídico': """
        Textos que tratam de questões legais e normativas, caracterizados pelo uso de linguagem técnica, 
        terminologia específica e estrutura formal. São altamente padronizados e precisos. 
        Exemplos incluem contratos, leis, regimentos, estatutos, processos judiciais e pareceres jurídicos.
        """,
        
        'Promocional': """
        Textos com finalidade persuasiva comercial, que visam promover produtos, serviços, marcas ou ideias. 
        Utilizam linguagem apelativa, recursos persuasivos e chamadas à ação. Exemplos incluem anúncios publicitários, 
        material de marketing, descrições de produtos e publicações promocionais em redes sociais.
        """,
        
        'Outro': """
        Textos que não se enquadram claramente nas categorias principais ou que combinam características 
        de múltiplos gêneros. Podem ser textos experimentais, híbridos ou de natureza específica para contextos particulares.
        """
    }
    
    return descriptions.get(genre_name, "Não há descrição detalhada disponível para esse gênero.")

def render_question(text_input):
    """ Carrega a função que chama o modelo para perguntas sobre o texto """
    st.title("Perguntas Interpretativas sobre o Texto")

    question = st.text_input("Faça uma pergunta sobre o texto:")

    if question:
        st.info("O processamento pode levar alguns segundos. Por favor, aguarde...")
        try:
            answer = answer_question(text_input[:512])
            st.subheader("Resposta:")
            st.write(answer)
        except Exception as e:
            st.error(f"Erro ao processar pergunta: {str(e)}")


def main():

    initialize_session_state()

    st.title('Analisador de Texto')
    
    input_method = st.radio(
        "Escolha o método de entrada:",
        ("Digitar texto", "Upload de arquivo .txt", "Upload de arquivo PDF")
    )
    
    text_input = ""

    if input_method == "Digitar texto":
        # área de texto para entrada manual
        text_input = st.text_area(
            "Digite ou cole seu texto aqui:",
            height=250,
            help="Digite ou cole o texto que deseja analisar."
        )
        
    elif input_method == "Upload de arquivo .txt":
        uploaded_file = st.file_uploader(
            "Escolha um arquivo .txt", 
            type="txt",
            help="Suporta arquivos grandes (>200MB). O processamento pode levar alguns minutos."
        )
        
        if uploaded_file:
            file_size = uploaded_file.size / (1024 * 1024)  # tamanho em MB
            st.info(f"Tamanho do arquivo: {file_size:.2f}MB")
            
            if file_size > 200:
                st.warning("Arquivo grande detectado. O processamento pode levar mais tempo.")
            
            try:
                with st.spinner('Carregando arquivo...'):
                    text_input = process_large_file(uploaded_file)
                    st.session_state.text_input = text_input
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                with st.expander("Mostrar Preview do Arquivo"):
                    st.write(text_input[:1000] + "..." if len(text_input) > 1000 else text_input)
                    
            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {str(e)}")
                return
    
    elif input_method == "Upload de arquivo PDF":
        uploaded_pdf = st.file_uploader(
            "Escolha um arquivo PDF", 
            type="pdf",
            help="Suporta arquivos PDF. O tempo de processamento depende do tamanho e complexidade do PDF."
        )
        
        if uploaded_pdf:
            file_size = uploaded_pdf.size / (1024 * 1024)  # tamanho em MB
            st.info(f"Tamanho do arquivo PDF: {file_size:.2f}MB")
            
            try:
                with st.spinner('Extraindo texto do PDF...'):
                    text_input = extract_pdf_to_text(uploaded_pdf)
                    st.session_state.text_input = text_input
                    st.session_state.uploaded_file_name = uploaded_pdf.name
                    
                if text_input:
                    st.success(f"Texto extraído com sucesso do arquivo: {uploaded_pdf.name}")
                    
                    with st.expander("Mostrar Preview do Texto Extraído"):
                        st.write(text_input[:1000] + "..." if len(text_input) > 1000 else text_input)

                        
            except Exception as e:
                st.error(f"Erro ao processar o arquivo PDF: {str(e)}")
                return
    
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
            ["Estatísticas", "Visualizações", "Busca e Informações", "Classificação", "Sumarização"],
            horizontal=True,
            label_visibility="hidden",
            index=["Estatísticas", "Visualizações", "Busca e Informações", "Classificação", "Sumarização"].index(st.session_state.tab) 
            if st.session_state.tab in ["Estatísticas", "Visualizações", "Busca e Informações", "Classificação", "Sumarização"] else 0
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

            interpretacao = get_metrics_interpretation(metrics['media'], metrics['mediana'], metrics['desvio_padrao'])
            st.info(interpretacao)
            
            with st.expander("Saiba mais sobre as métricas de tamanho de palavras"):
                st.markdown("""
                ### Interpretação Detalhada das Métricas

                A métrica de **comprimento de palavras** analisa a distribuição do tamanho das palavras em um texto.  
                Em português, a **média do comprimento das palavras** geralmente varia entre **4 e 6 caracteres**.

                - **Média menor que 4** → O texto contém muitas palavras curtas, como artigos, pronomes e preposições.  
                Pode indicar um estilo mais informal, como mensagens rápidas, redes sociais ou diálogos.
                - **Média entre 4 e 6** → Representa um equilíbrio entre palavras curtas e longas,  
                comum em textos jornalísticos, textos acadêmicos introdutórios e relatórios técnicos.
                - **Média maior que 6** → Indica presença de palavras mais longas e complexas,  
                geralmente associadas a vocabulário técnico, jurídico ou acadêmico avançado.

                O **desvio padrão** mede a dispersão do comprimento das palavras:
                - **Desvio padrão menor que 1.5** → Baixa variação: o texto utiliza palavras com tamanhos semelhantes,  
                sugerindo estilo uniforme e direto (ex: manuais, instruções técnicas).
                - **Desvio padrão entre 1.5 e 2.5** → Variação controlada: o texto apresenta vocabulário diversificado com equilíbrio,  
                típico de textos formais bem estruturados.
                - **Desvio padrão entre 2.5 e 3.0** → Alta variação: o texto mistura palavras curtas e longas com frequência,  
                o que pode indicar criatividade ou linguagem mais expressiva e dinâmica.
                - **Desvio padrão acima de 3.0** → Variação muito alta: textos assim costumam ter vocabulário extremamente diverso,  
                podendo refletir informalidade excessiva, uso de gírias, transcrições de fala espontânea ou áreas altamente técnicas.
                """)

            st.divider()
            
            with st.expander("Sobre análise TF-IDF"):
                st.markdown("""
                        A análise **TF-IDF** é uma métrica estatística que avalia a importância de uma palavra em um texto.  
                        Ela combina dois fatores:

                        * **TF (Term Frequency)**: Frequência com que uma palavra aparece no texto  
                        * **IDF (Inverse Document Frequency)**: Quão única ou rara é essa palavra

                        Um valor **TF-IDF alto** (ex: acima de 0.3) indica que a palavra é:
                        * Muito frequente neste texto específico
                        * Relativamente rara em textos em geral  
                        Portanto, é uma **palavra característica e relevante** do texto.

                        Um valor **TF-IDF baixo** (ex: abaixo de 0.1) sugere que a palavra:
                        * É comum em muitos textos ou
                        * Aparece poucas vezes no texto analisado  
                        Logo, é **menos útil para identificar o tema central** do texto.

                        Essa métrica ajuda a destacar as palavras mais importantes para a compreensão do conteúdo.
                        """)

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
                "Nuvem de Palavras": "Visualização das palavras mais frequentes em forma de nuvem. Útil para identificar rapidamente os termos mais recorrentes no texto e ter uma visão geral do seu conteúdo.",
                "Métricas": "Gráfico comparativo das métricas de comprimento de palavras no texto. Indicado para analisar a complexidade do vocabulário utilizado, identificando se o texto tende a ser mais informal ou técnico.",
                "Frequência de Palavras": "Distribuição das palavras mais frequentes. Permite entender quais palavras aparecem com maior frequência e detectar possíveis padrões ou repetições excessivas.",
                "Análise TF-IDF": "Visualização dos termos mais relevantes com base na métrica TF-IDF. Essencial para identificar palavras-chave que diferenciam o texto e são importantes para sua compreensão."
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
    
        
        elif st.session_state.tab == "Classificação":
            render_classification_section(tokens)
            
        elif st.session_state.tab == "Sumarização":
            texto_completo = " ".join(tokens)
            render_summarization_section(texto_completo)

        elif st.session_state.tab == "Busca e Informações":
            render_question(text_input)
    
    with st.expander("Sobre o Analisador de Texto"):
        st.write("""
        Esse analisador de texto realiza as seguintes operações:
        1. **Limpeza e Preprocessamento**: Remove pontuações, caracteres especiais e normaliza o texto
        2. **Análise Estatística**: Calcula métricas sobre o comprimento das palavras
        3. **Análise TF-IDF**: Identifica termos mais relevantes no texto
        4. **Visualizações**: Gera gráficos e nuvem de palavras
        5. **Buscas e Informações**: Permite fazer perguntas interpretativas sobre o texto
        6. **Classificação**: Identifica o gênero textual usando um modelo de aprendizado profundo
        7. **Sumarização**: Cria um resumo automático do texto
        """)

if __name__ == '__main__':
    main()