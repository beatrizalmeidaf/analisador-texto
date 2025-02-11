# Funções para limpeza de texto

import os
import re

# teste de leitura de arquivo
data = os.path.join(os.getcwd(), "data/teste.txt")

with open(data, encoding='utf-8') as f:
    text = f.read()
    

def clean_text(txt):
    """ Limpeza de texto, incluindo remoção de pontuações,
    caracteres especiais, números, conversão de letras
    """
    txt = txt.lower() # conversão para minúsculas
    txt = re.sub(r'[^\w\s]', '', txt) # remove pontuações e caracteres especiais
    return txt

texto = clean_text(text)
print(texto)