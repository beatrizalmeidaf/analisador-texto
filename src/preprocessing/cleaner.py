import re


def clean_text(text):
    """ Limpeza de texto, incluindo remoção de pontuações,
    caracteres especiais, números e conversão de letras de maiúsculas para minúsculas
    """
    txt = text.lower() # conversão para minúsculas
    txt = re.sub(r'[^\w\s]', '', txt) # remove pontuações e caracteres especiais
    txt = re.sub(r'\d', '', txt) # remove números

    return txt


if __name__ =='__main__':

    with open('data/teste.txt', encoding='utf-8') as f:
        print(clean_text(f.read()))
