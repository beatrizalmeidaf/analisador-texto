import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from .cleaner import clean_text
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

text = clean_text()

def tokenizer(text):
    """ Tokenização para dividir um texto em unidades menores e remoção
     de palavras de parada """
    
    tokens = word_tokenize(text)
    tokens = [x for x in tokens if x not in stopwords]
    
    return tokens

def word_counter(tokens):
   """ Contagem de frequência de palavras importantes
   """
   return Counter(tokens)


if __name__ == '__main__':
    
    print(tokenizer(text))
    

