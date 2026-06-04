# ocr_template.py
import pdfplumber
import pytesseract
from PIL import Image

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
    "numero_documento": (0.65, 0.05, 0.95, 0.12),   # arriba derecha
    "apellidos":        (0.20, 0.18, 0.80, 0.24),   # línea de apellidos
    "nombres":          (0.20, 0.25, 0.80, 0.31),   # línea de nombres
    "lugar_nacimiento": (0.20, 0.40, 0.60, 0.46),   # POPAYÁN (CAUCA)
    "fecha_nacimiento": (0.20, 0.47, 0.45, 0.53),   # 23-MAY-2004
    "sexo":             (0.65, 0.65, 0.75, 0.70),   # M/F
    "tipo_sangre":      (0.75, 0.65, 0.95, 0.70),   # O+, A+, etc.
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
        text = pytesseract.image_to_string(crop, lang="spa")
        results[field] = text.strip()

    # Selección de plantilla
    if modelo == "digital":
        zones = zones_digital
    else:
        zones = zones_hologramas

    results = {}
    for field, (x1, y1, x2, y2) in zones.items():
        box = (
            int(x1 * width), int(y1 * height),
            int(x2 * width), int(y2 * height)
        )
        crop = img.crop(box)
        text = pytesseract.image_to_string(crop, lang="spa")
        results[field] = text.strip()

    return results
