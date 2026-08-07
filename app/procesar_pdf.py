from app.parser import procesar_documento

def procesar_pdf(pdf_path: str):
    # Wrapper simple: delega todo el procesamiento a parser.py
    return procesar_documento(pdf_path)
