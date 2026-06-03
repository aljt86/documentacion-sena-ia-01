from app.parser import procesar_documento
from utils import extraer_texto
import fitz
    
def procesar_pdf(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
        resultados = []

        for pagina in doc:
            texto = pagina.get_text()
            resultados.append(texto)

        texto_final = "\n".join(resultados)

        return procesar_documento(texto_final)

    except Exception as e:
        return f"Error al procesar PDF: {e}"