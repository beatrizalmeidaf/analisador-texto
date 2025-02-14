# Funções para tokenização

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from cleaner import clean_text
stopwords = nltk.corpus.stopwords.words('portuguese')

text = clean_text()

def tokenizer(text):
    """ Tokenização para dividir um texto em unidades menores e remoção
     de palavras de parada """
    
    tokens = word_tokenize(text)
    tokens = [x for x in tokens if x not in stopwords]
    
    return tokens

if __name__ == '__main__':

    print(tokenizer(text))
    

