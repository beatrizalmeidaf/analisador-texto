import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2") # conta os tokens

# carregamento com cache para economizar recursos
@st.cache_resource
def load_model():
    """Carrega o modelo usado e personaliza o prompt."""

    template = """Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

    ### Instrução:
    De acordo com o texto fornecido, responda de forma direta à pergunta especificada:

    "{instruction}"

    ### Resumo:"""

    prompt = PromptTemplate(template=template, input_variables=["instruction"])
    llm = CTransformers(
        model="recogna-nlp/bode-7b-alpaca-pt-br-gguf",
        model_file="bode-7b-alpaca-q8_0.gguf",
        model_type='llama'
    )
    return LLMChain(prompt=prompt, llm=llm)

def truncate_text(text, max_tokens=512):
    """Trunca o texto para não exceder o número máximo de tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # trunca os tokens e converte de volta para texto
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    return truncated_text

def answer_question(text_input):
    """Função de responder perguntas específicas sobre o texto enviado pelo usuário."""

    truncated_text = truncate_text(text_input, max_tokens=512)

    try:
        llm_chain = load_model()
        answer = llm_chain.run(truncated_text)
        return answer
    except Exception as e:
        return f"Desculpe, ocorreu um erro ao gerar o resumo: {str(e)}"