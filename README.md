# **Analisador Inteligente de Texto**

<p align="center">
  <img src="https://github.com/user-attachments/assets/0eac93fb-f5ff-4af3-b4cc-476aeb6acb28" alt="Logo" width="300"/>
</p>

O **Analisador Inteligente de Texto** é uma ferramenta robusta para processamento e análise textual. Utilizando técnicas de Processamento de Linguagem Natural (NLP), estatística e aprendizado de máquina, a aplicação permite extrair informações relevantes de textos em diversos formatos e contextos.


---

## **Objetivo do Projeto**

Desenvolver um analisador textual completo que combine métodos analíticos tradicionais e inteligência artificial para atender a diferentes necessidades de análise textual, como:

* **Classificação automática de documentos**: Categorizar e organizar textos como e-mails, artigos ou relatórios.
* **Extração de palavras-chave**: Identificar os termos mais relevantes de documentos extensos.
* **Resumo automático de textos**: Destacar as principais ideias de textos longos.
* **Visualização de dados textuais**: Gerar insights claros por meio de gráficos e representações matemáticas.

---

## **Diferenciais da Ferramenta**

* **Abordagem Modular**: Cada componente pode ser desenvolvido, testado e mantido de forma independente.
* **Técnicas Tradicionais e Modernas**: Combinação de estatísticas clássicas e modelos de inteligência artificial.
* **Interface Intuitiva**: Acesso via interface web com foco na experiência do usuário.
* **Escalabilidade**: Pode ser executada localmente ou implantada como serviço web.
* **Flexibilidade de Aplicação**: Útil em contextos acadêmicos, corporativos, jornalísticos e mais.

---

## **Funcionalidades Planejadas**

A ferramenta está composta pelos seguintes módulos:  

### 1. Leitura e Pré-processamento de Texto  
- **Objetivo**: Preparar o texto para análises posteriores, garantindo limpeza e consistência.  
- **Funcionalidades**:  
  - Leitura de arquivos `.txt` ou entrada de texto manual.  
  - Normalização do texto: remoção de caracteres especiais, acentos e conversão para letras minúsculas.  
  - Tokenização: segmentação em palavras ou frases.  
  - Contagem e frequência de palavras.  
- **Tecnologias Usadas**: Python (`open`, `re`, `collections.Counter`).  

### 2. Estatísticas e Análise de Texto  
- **Objetivo**: Oferecer insights quantitativos e visuais sobre o texto analisado.  
- **Funcionalidades**:  
  - Cálculo de métricas estatísticas: média, mediana e desvio padrão do comprimento das palavras.  
  - Geração de gráficos (histogramas, nuvem de palavras).  
  - Cálculo de TF-IDF para avaliar a relevância de termos específicos.  
- **Tecnologias Usadas**: `pandas`, `word_cloud`, `matplotlib`.  

### 3. Representação Matemática do Texto  (em desenvolvimento)
- **Objetivo**: Transformar o texto em formatos matemáticos para análises avançadas.  
- **Funcionalidades**:  
  - Criação de matrizes de coocorrência.  
  - Implementação do modelo Bag of Words (BoW).  
- **Tecnologias Usadas**: `numpy`.  

### 4. Classificação Automática de Texto com Inteligência Artificial  
- **Objetivo**: Automatizar a categorização de textos.
- **Funcionalidades**:  
  - Treinamento de modelos de aprendizado supervisionado para classificação de textos.   
- **Tecnologias Usadas**: `scikit-learn`, `PyTorch` ou `TensorFlow`.  

### 5. Resumo Automático de Textos  
- **Objetivo**: Extrair os principais pontos de textos longos.  
- **Funcionalidades**:  
  - Processamento de textos longos (notícias, relatórios).  
  - Geração de resumos utilizando técnicas baseadas em NLP modernas.  
- **Tecnologias Usadas**: `sumy`, `transformers`.  
- 
---

## **Casos de Uso**

* **Empresas de tecnologia**: Análise de feedbacks e dados de usuários.
* **Jornalismo**: Geração automática de resumos de notícias.
* **Academia**: Apoio a estudos quantitativos e qualitativos com grandes volumes de texto.
* **Marketing**: Identificação de temas e palavras-chave em campanhas.

---

## **Tecnologias Utilizadas**

* **Linguagem**: Python
* **NLP**: `nltk`, `spaCy`, `transformers`, `tika`
* **Visualização**: `matplotlib`, `wordcloud`
* **Estatística e Vetorização**: `numpy`, `pandas`, `scikit-learn`
* **Interface Web**: `Streamlit`
* **Sumarização**: `transformers`
* **Classificação e ML**: `scikit-learn`

---

## **Como Rodar o Projeto Localmente**

### 1. Clone o repositório

```bash
git clone https://github.com/beatrizalmeidaf/analisador-texto.git
cd analisador-texto
```

### 2. Crie um Ambiente Virtual

#### Usando `venv`:

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

#### Usando `conda`:

```bash
conda create -n analisador-texto python=3.10
conda activate analisador-texto
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

> **Atenção:** Para realizar o carregamento de arquivos PDF com a biblioteca `tika`, é necessário ter o Java instalado no sistema.
> O projeto foi testado com o **OpenJDK 24**, mas versões anteriores acima da versão 8 também podem funcionar.

### 4. Execute a Aplicação Web

```bash
streamlit run web/streamlit_app.py
```
