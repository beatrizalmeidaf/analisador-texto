from tika import parser
import os
import re
import streamlit as st
import tempfile

# os.environ['TIKA_SERVER_JAR'] = 'tika-server.jar'

def is_page_number_line(line: str, max_page_num: int = 1000) -> bool:
    """
    Determina se uma linha contém apenas um número de página.
    Considera números alinhados à direita como números de página.
    """
    # remove espaços no início e fim
    stripped_line = line.strip()
    
    # se a linha está vazia não é número de página
    if not stripped_line:
        return False
        
    # se é apenas um número e está dentro do limite razoável de páginas
    if stripped_line.isdigit() and int(stripped_line) <= max_page_num:
        # verifica se o número está alinhado à direita na linha original
        if line.rstrip() == line.rstrip().rjust(len(line)):
            return True
    
    return False

def clean_page_numbers(text: str) -> str:
    """
    Remove números de página e limpa formatação do texto.
    """
    lines = text.split('\n')
    cleaned_lines = []
    previous_line = ''
    
    for i, line in enumerate(lines):
        # pula a linha se for um número de página isolado
        if is_page_number_line(line):
            continue
            
        # preserva números que fazem parte da estrutura do documento
        # remove apenas números que parecem ser números de página no final da linha
        cleaned_line = re.sub(r'\s+\d+\s*$', '', line)
            
        # se a linha anterior termina com hífen e esta linha começa com espaços,
        # mantém a formatação original
        if previous_line.rstrip().endswith('-'):
            cleaned_lines.append(cleaned_line)
        else:
            # remove qualquer ponto sozinho no início da linha
            cleaned_line = re.sub(r'^\s*\.\s*', '', cleaned_line)
            cleaned_lines.append(cleaned_line)
            
        previous_line = cleaned_line
    
    return '\n'.join(cleaned_lines)

def extract_pdf_to_text(pdf_file):
    """Extrai texto de um arquivo PDF usando o Apache Tika e limpa formatação."""
    try:
        # cria um arquivo temporário para salvar o PDF carregado
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_file.getvalue())
            temp_pdf_path = temp_pdf.name
        
        # extrai o conteúdo do PDF
        parsed_pdf = parser.from_file(temp_pdf_path)
        
        # remove o arquivo temporário
        os.unlink(temp_pdf_path)
        
        if 'content' in parsed_pdf:
            text_content = parsed_pdf['content']
        else:
            text_content = parsed_pdf.get('text', '')
            
        if not text_content:
            st.warning("Aviso: Nenhum texto foi extraído do PDF.")
            return ""
            
        # dividir o conteúdo em páginas
        pages = text_content.split('\f')
        
        processed_text = ""
        cleaned_pages = []

        for page in pages:
            cleaned_page = clean_page_numbers(page)
            if cleaned_page.strip():
                cleaned_pages.append(cleaned_page.strip())

        processed_text = '\n'.join(cleaned_pages)

        return processed_text
        
    except Exception as e:
        st.error(f"Erro ao processar o PDF: {str(e)}")
        return ""