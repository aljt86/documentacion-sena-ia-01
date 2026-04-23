import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from app.parser import extraer_campos

# Ruta exlicita al ejecutable de Tesseract en Windows

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def es_borroso(img):
    """Valida si la imagen está borrosa usando Laplacian Variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < 3  # Umbral ajustable según pruebas

def procesar_pdf(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
        resultados = []
        for pagina in doc:
            pix = pagina.get_pixmap()
            img = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            if es_borroso(img):
                return "Documento borroso, por favor suba una imagen más legible."
            texto = pytesseract.image_to_string(img, lang="spa")
            resultados.append(texto)
        
        texto_final = "\n".join(resultados)
        campos = extraer_campos(texto_final)
        return campos
    except Exception as e:
        return f"Error al procesar el PDF: {e}"
