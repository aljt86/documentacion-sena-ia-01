import pdfplumber
import pytesseract
from PIL import Image

def extract_fields(file_path):
    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[0]
        img = page.to_image(resolution=300).original

    width, height = img.size

    # Zonas relativas (ejemplo para cédula)
    zones = {
        "numero_documento": (0.65, 0.08, 0.95, 0.13),   # arriba a la derecha
        "nombre_completo": (0.15, 0.25, 0.85, 0.32),    # parte central
        "fecha_nacimiento": (0.15, 0.35, 0.40, 0.40),   # debajo del nombre
        "sexo":            (0.45, 0.35, 0.55, 0.40),    # junto a fecha
        "lugar_nacimiento":(0.15, 0.45, 0.55, 0.50),    # línea siguiente
        "nacionalidad":    (0.60, 0.45, 0.85, 0.50),    # misma línea
        "tipo_sangre":     (0.75, 0.55, 0.95, 0.60),    # parte inferior derecha
    }

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
