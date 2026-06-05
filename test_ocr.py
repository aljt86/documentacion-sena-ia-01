import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
import cv2
import numpy as np
import os
from app.ocr_template import extract_fields

# Ruta al PDF que quieres probar
pdf_path = "data/prueba14.pdf"   # cámbiala según tu archivo

# Modelo: "digital" o "hologramas"
resultado = extract_fields(pdf_path, modelo="hologramas")

print("Resultado OCR:")
for campo, valor in resultado.items():
    print(f"{campo}: {valor}")
# Ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Ruta al PDF que quieres procesar
pdf_path = os.getenv("OCR_TEST_PDF", r"F:\Detección documentos Sena 2.0\data\prueba1.pdf")

# Ruta a la carpeta bin de Poppler
poppler_path = os.getenv("POPPLER_PATH", r"C:\poppler")

print("DEBUG: archivo PDF:", pdf_path)
print("DEBUG: existe archivo:", os.path.exists(pdf_path))

try:
    info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
    print("DEBUG: pdfinfo OK:", info)
except Exception as e:
    print("ERROR pdfinfo:", repr(e))

try:
    paginas = convert_from_path(pdf_path, poppler_path=poppler_path)
    print("DEBUG: páginas encontradas:", len(paginas))
    for i, pagina in enumerate(paginas, start=1):
        # Convertir a formato OpenCV
        img = cv2.cvtColor(np.array(pagina), cv2.COLOR_RGB2BGR)

        # OCR directo (color)
        texto_color = pytesseract.image_to_string(img, lang="spa")

        # Preprocesamiento: escala de grises + umbral fijo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        texto_bn = pytesseract.image_to_string(thresh, lang="spa")

        # Preprocesamiento avanzado: umbral adaptativo + escalado
        adapt = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )
        scaled = cv2.resize(adapt, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        texto_adapt = pytesseract.image_to_string(scaled, lang="spa")

        print(f"\n--- PÁGINA {i} ---\n")
        print(">>> OCR en color <<<")
        print(texto_color[:1000])
        print("\n>>> OCR en blanco y negro (umbral fijo) <<<")
        print(texto_bn[:1000])
        print("\n>>> OCR con umbral adaptativo + escalado <<<")
        print(texto_adapt[:1000])
        print("\n--- FIN PÁGINA ---\n")

except Exception as e:
    print("ERROR convert_from_path:", repr(e))