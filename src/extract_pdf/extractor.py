from tika import parser
import os
import re

def extract_pdf_to_txt(pdf_path, output_txt_path):

    try:

        if not os.path.exists(pdf_path):
            print(f"Erro: O arquivo PDF '{pdf_path}' não foi encontrado.")
            return False
     
        print(f"Extraindo texto do arquivo: {pdf_path}")
        parsed_pdf = parser.from_file(pdf_path)
        
        if 'content' in parsed_pdf:
            text_content = parsed_pdf['content']
        else:
            text_content = parsed_pdf.get('text', '')
            
        if not text_content:
            print("Aviso: Nenhum texto foi extraído do PDF.")
            return False
        
        # dividir o conteúdo em páginas
        pages = text_content.split('\f')
        
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            page_number = 1
            
            for page in pages:
                
                page = page.strip()

                if page and not page.isspace():
                    processed_page = re.sub(r'^\s*(\d+)\s*$', 
                                         lambda m: f"\n{'='*10} PÁGINA {m.group(1)} {'='*10}\n", 
                                         page,
                                         flags=re.MULTILINE)
                    
                    txt_file.write(processed_page.strip())
                    txt_file.write("\n\n")
                    page_number += 1
            
        print(f"Texto extraído com sucesso e salvo em: {output_txt_path}")
        print(f"Total de páginas com conteúdo: {page_number - 1}")
        return True
        
    except Exception as e:
        print(f"Erro ao processar o PDF: {str(e)}")
        return False

if __name__ == "__main__":
    pdf_path = "../ebook.pdf" 
    output_txt_path = "ebook.txt"
    
    success = extract_pdf_to_txt(pdf_path, output_txt_path)
    
    if success:
        print("\nConteúdo do arquivo TXT:")
        with open(output_txt_path, 'r', encoding='utf-8') as txt_file:
            print(txt_file.read())