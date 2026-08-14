import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
import re
from datetime import datetime

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# DETECCIÓN DE BORDES Y RECORTE DE CÉDULA
# ============================================
def detectar_y_recortar_cedula(imagen):
    """
    Detecta los bordes de la cédula dentro de una imagen (hoja tamaño carta)
    y devuelve la cédula recortada.
    """
    gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        min_area = 5000
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        if large_contours:
            large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)
            largest_contour = large_contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(imagen.shape[1] - x, w + 2 * margin)
            h = min(imagen.shape[0] - y, h + 2 * margin)
            cedula_recortada = imagen[y:y+h, x:x+w]
            logger.info(f"✅ Cédula recortada: {w}x{h} px")
            return cedula_recortada

    logger.warning("⚠️ No se detectó la cédula, usando imagen original")
    return imagen

# ============================================
# FUNCIONES DE LIMPIEZA (integradas)
# ============================================
def limpiar_numero(raw: str) -> str:
    """Elimina todo excepto dígitos."""
    if not raw:
        return None
    digits = re.sub(r'\D', '', raw)
    return digits if digits else None

def limpiar_texto(raw: str) -> str:
    """Limpia caracteres especiales, espacios múltiples y capitaliza."""
    if not raw:
        return None
    texto = re.sub(r'[^A-Za-zÁÉÍÓÚÑáéíóúñ\s]', '', raw)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto.title() if texto else None

def limpiar_fecha(raw: str) -> str:
    """Convierte fechas tipo '22-FEB-1986' a '1986-02-22'."""
    if not raw:
        return None
    meses = {
        "ENE":"01","FEB":"02","MAR":"03","ABR":"04","MAY":"05","JUN":"06",
        "JUL":"07","AGO":"08","SEP":"09","OCT":"10","NOV":"11","DIC":"12"
    }
    m = re.match(r'(\d{1,2})[-\s]([A-ZÁÉÍÓÚ]{3})[-\s](\d{4})', raw.upper())
    if m:
        d, mes, y = m.groups()
        mes_num = meses.get(mes[:3], "01")
        return f"{y}-{mes_num}-{d.zfill(2)}"
    return raw.strip() if raw.strip() else None

def limpiar_sexo(raw: str) -> str:
    """Normaliza M/F a 'Masculino'/'Femenino'."""
    if not raw:
        return None
    rv = raw.strip().upper()
    if rv.startswith("M"):
        return "Masculino"
    elif rv.startswith("F"):
        return "Femenino"
    return None

def limpiar_rh(raw: str) -> str:
    """Extrae tipo de sangre (O+, A-, etc.)."""
    if not raw:
        return None
    m = re.match(r'([ABO]{1,2}[+-])', raw.upper())
    return m.group(1) if m else None

# ============================================
# PREPROCESAMIENTO DE IMÁGENES
# ============================================
def preprocesa_image(pil_img):
    """Optimizado para documentos escaneados de baja calidad."""
    try:
        img = np.array(pil_img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(thresh)
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        return pil_img

# ============================================
# ZONAS PARA CÉDULA AMARILLA CON HOLOGRAMAS (MODELO VIEJO) - COORDENADAS ORIGINALES
# ============================================
zones_hologramas = {
    "numero_documento": (0.04, 0.17, 0.45, 0.34),    # Número (parte superior izquierda)
    "apellidos": (0.04, 0.45, 0.20, 0.31),           # Apellidos (debajo del número)
    "nombres": (0.04, 0.28, 0.45, 0.24),             # Nombres (debajo de apellidos)
    "fecha_nacimiento": (0.33, 0.05, 0.12, 0.45),    # Fecha de nacimiento (parte inferior izquierda)
    "lugar_nacimiento": (0.33, 0.08, 0.31, 0.38),    # Lugar de nacimiento (parte inferior derecha)
    "sexo": (0.63, 0.16, 0.16, 0.31),                # Sexo (debajo de lugar)
    "tipo_sangre": (0.49, 0.16, 0.30, 0.31),         # RH (debajo de fecha)
    "nacionalidad": (0.45, 0.10, 0.80, 0.18),        # Nacionalidad (parte superior derecha)
}

# ============================================
# ZONAS PARA CÉDULA DIGITAL (POLICARBONATO) (MODELO NUEVO)
# ============================================
zones_digital = {
    "numero_documento": (0.65, 0.08, 0.95, 0.13),
    "nombre_completo": (0.15, 0.25, 0.85, 0.32),
    "fecha_nacimiento": (0.15, 0.35, 0.40, 0.40),
    "sexo": (0.45, 0.35, 0.55, 0.40),
    "lugar_nacimiento": (0.15, 0.45, 0.55, 0.50),
    "nacionalidad": (0.60, 0.45, 0.85, 0.50),
    "tipo_sangre": (0.75, 0.55, 0.95, 0.60),
}

# ============================================
# EXTRACCIÓN DE CAMPOS DESDE TEXTO (PDF DIGITAL)
# ============================================
def extract_fields_from_text(text):
    """Extrae datos usando regex cuando el PDF tiene texto digital."""
    results = {}
    match = re.search(r'(\d{6,15})', text)
    results['numero_documento'] = match.group(1) if match else None
    match = re.search(r'(?:NOMBRE|NOMBRES?)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
    results['nombre_completo'] = limpiar_texto(match.group(1).strip()) if match else None
    if not results['nombre_completo']:
        match = re.search(r'(?:APELLIDOS?)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
        if match:
            results['apellidos'] = limpiar_texto(match.group(1).strip())
        match = re.search(r'(?:NOMBRES?)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
        if match:
            results['nombres'] = limpiar_texto(match.group(1).strip())
    match = re.search(r'(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})', text)
    results['fecha_nacimiento'] = match.group(1) if match else None
    match = re.search(r'(SEXO|GENERO)\s*:?\s*([MF])', text, re.IGNORECASE)
    results['sexo'] = match.group(2) if match else None
    match = re.search(r'(?:LUGAR|CIUDAD)\s*(?:DE)?\s*NACIMIENTO\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
    results['lugar_nacimiento'] = limpiar_texto(match.group(1).strip()) if match else None
    match = re.search(r'(NACIONALIDAD)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
    results['nacionalidad'] = limpiar_texto(match.group(2).strip()) if match else None
    match = re.search(r'(TIPO\s*(?:DE)?\s*SANGRE|RH)\s*:?\s*([A-Z0-9+-]+)', text, re.IGNORECASE)
    results['tipo_sangre'] = match.group(2) if match else None
    return results

# ============================================
# FUNCIÓN PRINCIPAL (SOPORTA AMBOS FORMATOS)
# ============================================
def extract_fields(file_path, modelo="hologramas"):
    results = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                logger.error("El PDF no contiene páginas.")
                return {}
            page = pdf.pages[0]
            text = page.extract_text() or ""

            if text.strip():
                results = extract_fields_from_text(text)
                if any(results.values()):
                    logger.info("✅ Datos extraídos del texto del PDF")
                    return results
                else:
                    logger.warning("⚠️ El texto no contenía datos válidos, usando OCR...")
            else:
                logger.warning("⚠️ El PDF no tiene texto extraíble, usando OCR...")

            img = page.to_image(resolution=300).original

            # ✅ DETECTAR Y RECORTAR LA CÉDULA
            img_np = np.array(img)
            cedula_np = detectar_y_recortar_cedula(img_np)
            img = Image.fromarray(cedula_np)

            width, height = img.size
            logger.info(f"📐 Imagen recortada (cédula): {width}x{height} px")

            zones = zones_hologramas if modelo == "hologramas" else zones_digital

            for field, (x1, y1, x2, y2) in zones.items():
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                box = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

                try:
                    crop = img.crop(box)
                    crop = preprocesa_image(crop)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    crop.save(f"/tmp/crop_{field}_{timestamp}.png")
                    logger.info(f"🖼️ Imagen recortada guardada: /tmp/crop_{field}_{timestamp}.png")

                    raw = pytesseract.image_to_string(crop, lang="spa", config="--psm 8 --oem 3").strip()
                    logger.info(f"OCR raw para {field}: {raw!r}")

                    # Limpiar según el campo
                    if field == "numero_documento":
                        clean = limpiar_numero(raw)
                    elif field == "fecha_nacimiento":
                        clean = limpiar_fecha(raw)
                    elif field in ("apellidos", "nombres", "nombre_completo", "lugar_nacimiento", "nacionalidad", "tipo_sangre"):
                        clean = limpiar_texto(raw)
                    elif field == "sexo":
                        clean = limpiar_sexo(raw)
                    else:
                        clean = raw

                    # Permitir que el campo esté vacío (None) sin marcar error
                    if field == "numero_documento" and not clean:
                        logger.warning("⚠️ Número de documento vacío para box %s", box)
                        clean = ""

                    results[field] = clean
                    logger.info(f"✅ {field}: {raw!r} → {clean!r}")

                except Exception as e:
                    logger.error(f"❌ Error en campo {field}: {e}")
                    results[field] = None

    except Exception as e:
        logger.error(f"❌ Error en extract_fields: {e}")
        return {}

    return results