from tika import parser
import os
import re
import streamlit as st
import tempfile

os.environ['TIKA_SERVER_JAR'] = 'tika-server.jar'

def extract_pdf_to_text(pdf_file):
    """Extrai texto de um arquivo PDF usando o Apache Tika."""
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
        page_number = 1
        
        for page in pages:
            page = page.strip()
            if page and not page.isspace():
                processed_page = re.sub(
                    r'^\s*(\d+)\s*$',
                    lambda m: f"\n{'='*10} PÁGINA {m.group(1)} {'='*10}\n",
                    page,
                    flags=re.MULTILINE
                )
                processed_text += processed_page.strip() + "\n\n"
                page_number += 1
                
        return processed_text
        
    except Exception as e:
        st.error(f"Erro ao processar o PDF: {str(e)}")
        return ""