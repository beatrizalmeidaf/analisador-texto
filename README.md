<p align="center">
  <img src="web/assets/logo.png" alt="Analisador de Texto Logo" width="420"/>
</p>

<p align="center">
  <strong>Análise textual completa com NLP, IA e estatística — direto no navegador ou via Python.</strong>
</p>

<p align="center">
  <a href="https://beatrizalmeidaf.github.io/analisador-texto/">
    <img alt="Live Demo" src="https://img.shields.io/badge/demo-GitHub%20Pages-4f46e5?style=for-the-badge&logo=github"/>
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-Vite-06b6d4?style=for-the-badge&logo=typescript&logoColor=white"/>
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/github/license/beatrizalmeidaf/analisador-texto?style=for-the-badge"/>
</p>

---

## Sobre o projeto

O **Analisador de Texto** é uma ferramenta de Processamento de Linguagem Natural (NLP) que combina estatísticas clássicas e modelos de IA modernos para extrair insights de textos em português.

O projeto possui **duas versões complementares**:

| Versão | Stack | Deploy | Quando usar |
|---|---|---|---|
| **Web** (`site/`) | TypeScript + Vite + Transformers.js | GitHub Pages — zero servidor | Análise rápida no navegador, sem instalar nada |
| **Python** (`web/`) | Streamlit + HuggingFace + spaCy | Local / servidor com Python | Análises mais pesadas, PDFs, relatórios |

---

## Versão Web — GitHub Pages

> **Acesse agora:** [beatrizalmeidaf.github.io/analisador-texto](https://beatrizalmeidaf.github.io/analisador-texto/)

Roda **inteiramente no navegador** — nenhum dado sai do seu computador. A análise acontece em tempo real enquanto você digita.

### Funcionalidades disponíveis online

| Funcionalidade | Como funciona | Latência |
|---|---|---|
| Estatísticas, legibilidade (Flesch PT), histograma | TypeScript puro via Web Worker | ~0,2 s / 90 KB de texto |
| Frequência, TF-IDF entre parágrafos, n-gramas, exportação CSV | idem | idem |
| Nuvem de palavras interativa (clique → busca) | `wordcloud2.js` em canvas | Instantâneo |
| Resumo extrativo (TextRank + remoção de redundância) | TypeScript puro | Instantâneo |
| Busca exata tolerante a acentos com destaque visual | idem | Instantâneo |
| Leitura de PDF | `pdf.js` (sem Java) | Depende do arquivo |
| Busca semântica, resumo semântico, gênero textual (zero-shot) | Transformers.js + `paraphrase-multilingual-MiniLM-L12-v2` (~118 MB, cache local) | Alguns segundos (1ª vez) |
| Análise de sentimento por frase | `distilbert-base-multilingual-cased-sentiments-student` (~136 MB, sob demanda) | Alguns segundos |

> Os modelos de IA são baixados apenas quando você usa o recurso que depende deles e ficam em cache local.

### Rodar a versão web localmente

```bash
cd site
npm install --ignore-scripts   # binários nativos do onnxruntime-node não são necessários
npm run dev                    # servidor de desenvolvimento
npm test                       # testes do núcleo de análise
npm run build                  # gera site/dist para produção
```

O deploy no GitHub Pages é automático via [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) a cada push na `main` que altere `site/`.  
Para ativar no seu fork: **Settings → Pages → Build and deployment → GitHub Actions**.

---

## Versão Python — Streamlit

A versão Python oferece funcionalidades adicionais que exigem processamento server-side, como modelos maiores e geração de relatórios.

> **Importante:** Esta versão **não funciona no GitHub Pages** (requer servidor Python). Para hospedagem gratuita, use [Streamlit Community Cloud](https://streamlit.io/cloud) ou [Hugging Face Spaces](https://huggingface.co/spaces).

### Funcionalidades

| Funcionalidade | Modelo / Técnica | Latência |
|---|---|---|
| Limpeza e tokenização | `nltk` + regex | Instantâneo |
| Métricas estatísticas e TF-IDF | `scikit-learn` + `pandas` | Instantâneo |
| Nuvem de palavras e gráficos | `wordcloud` + `matplotlib` + `plotly` | < 1 s |
| Classificação de gênero textual | `XLM-RoBERTa` (zero-shot) | ~2–5 s |
| **Resumo extrativo** (TextRank) | TF-IDF (instantâneo) ou `MiniLM` (semântico) | < 1 s / ~2 s |
| **Análise de sentimento** por trecho | `distilbert-base-multilingual-cased-sentiments-student` | < 1 s/trecho |
| **Extração de frases-chave** | `KeyBERT` + `all-MiniLM-L6-v2` + MMR | ~1 s |
| Perguntas interpretativas | Modelo de Q&A | ~3–5 s |
| **Exportação de relatório** | HTML + PDF (pdfkit / weasyprint) | < 2 s |
| Leitura de PDF | `PyMuPDF` | Depende do arquivo |

### Como rodar localmente

#### 1. Clone o repositório

```bash
git clone https://github.com/beatrizalmeidaf/analisador-texto.git
cd analisador-texto
```

#### 2. Crie um ambiente virtual

```bash
# venv
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# ou conda
conda create -n analisador-texto python=3.10
conda activate analisador-texto
```

#### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

#### 4. Execute a aplicação

```bash
streamlit run web/streamlit_app.py
```

A aplicação abrirá em `http://localhost:8501`.

---

## Estrutura do Projeto

```
analisador-texto/
├── site/                   # Versão web (TypeScript + Vite → GitHub Pages)
│   ├── src/
│   │   ├── core/           # Lógica de análise em TypeScript
│   │   ├── workers/        # Web Workers (processamento sem travar a UI)
│   │   └── main.ts         # Entrypoint da aplicação web
│   └── index.html
│
├── web/
│   └── streamlit_app.py    # Versão Python (Streamlit)
│
├── src/                    # Módulos Python compartilhados
│   ├── preprocessing/      # Limpeza e tokenização
│   ├── analysis/           # Estatísticas, TF-IDF, sentimento, frases-chave, relatório
│   ├── summarization/      # TextRank (estatístico e semântico)
│   ├── classification/     # Classificação de gênero textual
│   ├── question/           # Q&A sobre o texto
│   └── extract_pdf/        # Extração de texto de PDFs
│
├── data/                   # Dados de teste
├── requirements.txt        # Dependências Python
└── README.md
```

---

## Tecnologias

### Versão Web
| Camada | Tecnologia |
|---|---|
| Linguagem | TypeScript |
| Build | Vite |
| NLP no browser | Transformers.js (ONNX Runtime) |
| Visualização | wordcloud2.js, canvas 2D |
| Deploy | GitHub Pages + GitHub Actions |

### Versão Python
| Camada | Tecnologia |
|---|---|
| Interface | Streamlit |
| NLP | HuggingFace Transformers, nltk |
| Vetorização | scikit-learn (TF-IDF) |
| Frases-chave | KeyBERT + sentence-transformers |
| Visualização | matplotlib, plotly, wordcloud |
| PDF | PyMuPDF |
| Relatórios | pdfkit / weasyprint |

---

## Casos de Uso

- **Jornalismo** — resumos automáticos de notícias e análise de sentimento
- **Academia** — estudo quantitativo de grandes corpora textuais
- **Empresas** — categorização de feedbacks, e-mails e documentos
- **Marketing** — identificação de temas e palavras-chave em campanhas

