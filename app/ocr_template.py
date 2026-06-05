# ocr_template.py
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Función de preprocesamiento 
def preprocess_image(pil_img):
    img = np.array(pil_img.convert("L"))  # Convertir a escala de grises
    coords = np.column_stack(np.where(img > 0))  # Encontr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    denoised = cv2.medianBlur(thresh, 3)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return Image.fromarray(sharpened) 
 

# Plantilla para cédula digital (policarbonato)
zones_digital = {
    "numero_documento": (0.65, 0.08, 0.95, 0.13),   # arriba derecha
    "nombre_completo": (0.15, 0.25, 0.85, 0.32),    # parte central
    "fecha_nacimiento": (0.15, 0.35, 0.40, 0.40),   # debajo del nombre
    "sexo":            (0.45, 0.35, 0.55, 0.40),    # junto a fecha
    "lugar_nacimiento":(0.15, 0.45, 0.55, 0.50),    # línea siguiente
    "nacionalidad":    (0.60, 0.45, 0.85, 0.50),    # misma línea
    "tipo_sangre":     (0.75, 0.55, 0.95, 0.60),    # parte inferior derecha
}

# Plantilla para cédula amarilla con hologramas
zones_hologramas = {
    "numero_documento": (0.4/8.5, 1.6/5.4, 4.5/8.5, 3.4/5.4),
    "apellidos":        (0.4/8.5, 2.0/5.4, 4.5/8.5, 3.1/5.4),
    "nombres":          (0.4/8.5, 2.8/5.4, 4.5/8.5, 2.4/5.4),
    "fecha_nacimiento": (3.3/8.5, 0.5/5.4, 1.2/8.5, 4.5/5.4),
    "lugar_nacimiento": (3.3/8.5, 0.8/5.4, 3.1/8.5, 3.8/5.4),
    "tipo_sangre":      (4.9/8.5, 1.6/5.4, 3.0/8.5, 3.1/5.4),
    "sexo":             (6.3/8.5, 1.6/5.4, 1.6/8.5, 3.1/5.4),
}

def extract_fields(file_path, modelo="hologramas"):
    """
    Extrae los campos de la cédula según el modelo.
    modelo puede ser "digital" o "hologramas".
    """
    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[0]
        img = page.to_image(resolution=300).original

    width, height = img.size
    zones = zones_digital if modelo == "digital" else zones_hologramas

    results = {}
    for field, (x1, y1, x2, y2) in zones.items():
        box = (
            int(x1 * width), int(y1 * height),
            int(x2 * width), int(y2 * height)
        )
        crop = img.crop(box)
        crop = preprocess_image(crop)
        text = pytesseract.image_to_string(crop, lang="spa")
        results[field] = text.strip()
    
    return results
