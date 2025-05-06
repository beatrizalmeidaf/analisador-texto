import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# carregamento com cache para economizar recursos
@st.cache_resource
def load_model():
    """Carrega o modelo usado e personaliza o prompt."""

    template = """Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

    ### Instrução:
    Resuma o seguinte texto de forma clara e objetiva:

    "{instruction}"

    ### Resumo:"""

    prompt = PromptTemplate(template=template, input_variables=["instruction"])
    llm = CTransformers(
        model="recogna-nlp/bode-7b-alpaca-pt-br-gguf",
        model_file="bode-7b-alpaca-q8_0.gguf",
        model_type='llama'
    )
    return LLMChain(prompt=prompt, llm=llm)

def resumir_texto(texto):
    """Função de resumir o texto enviado pelo usuário."""
    try:
        llm_chain = load_model()
        resumo = llm_chain.run(texto)
        return resumo
    except Exception as e:
        return f"Desculpe, ocorreu um erro ao gerar o resumo: {str(e)}"
