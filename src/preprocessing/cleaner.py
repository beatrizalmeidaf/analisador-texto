import os
import re

def read_txt():
    """ Teste de leitura de arquivo txt
    """
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
    
    return txt


if __name__ =='__main__':

    text = clean_text()
    print(text)
 