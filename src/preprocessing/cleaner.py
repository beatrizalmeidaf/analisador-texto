# Funções para limpeza de texto

import os
import re
from enelvo.normaliser import Normaliser
from collections import Counter


# teste de leitura de arquivo
def read_txt():
    data = os.path.join(os.getcwd(), "data/teste.txt")

    with open(data, encoding='utf-8') as f:
        text = f.read()
    
    return text
    

def clean_text():
    """ Limpeza de texto, incluindo remoção de pontuações,
    caracteres especiais, números, conversão de letras de maiúsculas para minúsculas,
    correções ortográficas
    """
    txt = read_txt()
    txt = txt.lower() # conversão para minúsculas
    txt = re.sub(r'[^\w\s]', '', txt) # remove pontuações e caracteres especiais
    txt = re.sub(r'\d', '', txt) # remove números
    
    norm = Normaliser(sanitize=True, tokenizer='readable')
    txt = norm.normalise(txt)

    return txt

def word_counter():
   """ Contagem de frequência de palavras"""
   txt = clean_text()
   words = re.findall(r'\b\w+\b', txt) # encontrar todas as palavras sendo \b a borda da palavra
   words = Counter(words)
   return words


if __name__ =='__main__':

    text = clean_text()
    words = word_counter()
    print(text)
    print(words)