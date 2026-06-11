import pytesseract
from pdf2image import convert_from_path
import cv2, numpy as np, os

# Configuración de Tesseract y Poppler
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pdf_path = r"F:\Detección documentos Sena 2.0\data\prueba14.pdf"
poppler_path = r"C:\poppler"

print("DEBUG: archivo PDF:", pdf_path)
print("DEBUG: existe archivo:", os.path.exists(pdf_path))

try:
    paginas = convert_from_path(pdf_path, poppler_path=poppler_path)
    print("DEBUG: páginas encontradas:", len(paginas))
    for i, pagina in enumerate(paginas, start=1):
        img = cv2.cvtColor(np.array(pagina), cv2.COLOR_RGB2BGR)
        texto = pytesseract.image_to_string(img, lang="spa")
        print(f"\n--- PÁGINA {i} ---\n")
        print(texto[:1000])  # muestra los primeros 1000 caracteres
except Exception as e:
    print("ERROR convert_from_path:", repr(e))
