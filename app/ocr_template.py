import logging
import re
from datetime import datetime

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

from app.parser import detectar_y_recortar_cedula

logger = logging.getLogger(__name__)

# ============================================================
# PLANTILLA NORMALIZADA
# ============================================================
# Estas zonas siguen siendo las zonas actuales del proyecto.
# En el siguiente paso las sustituiremos por las 4 medidas físicas
# que tomaste sobre la cédula original.
zones_hologramas_anverso = {
    "numero_documento": (0.04, 0.17, 0.45, 0.34),
    "apellidos": (0.04, 0.45, 0.55, 0.58),
    "nombres": (0.04, 0.60, 0.55, 0.72),
}

zones_hologramas_reverso = {
    "fecha_nacimiento": (0.05, 0.10, 0.45, 0.25),
    "lugar_nacimiento": (0.45, 0.10, 0.85, 0.25),
    "sexo": (0.70, 0.25, 0.90, 0.35),
    "tipo_sangre": (0.55, 0.25, 0.75, 0.35),
    "nacionalidad": (0.45, 0.02, 0.80, 0.10),
}

zones_digital = {
    "numero_documento": (0.65, 0.08, 0.95, 0.13),
    "nombre_completo": (0.15, 0.25, 0.85, 0.32),
    "fecha_nacimiento": (0.15, 0.35, 0.40, 0.40),
    "sexo": (0.45, 0.35, 0.55, 0.40),
    "lugar_nacimiento": (0.15, 0.45, 0.55, 0.50),
    "nacionalidad": (0.60, 0.45, 0.85, 0.50),
    "tipo_sangre": (0.75, 0.55, 0.95, 0.60),
}


# ============================================================
# LIMPIEZA
# ============================================================

def limpiar_numero(raw):
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    return digits or None


def limpiar_texto(raw):
    if not raw:
        return None
    texto = re.sub(r"[^A-Za-zÁÉÍÓÚÑáéíóúñ\s]", "", raw)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto.title() if texto else None


def limpiar_fecha(raw):
    if not raw:
        return None

    raw = raw.strip().upper()

    meses = {
        "ENE": "01", "FEB": "02", "MAR": "03", "ABR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AGO": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DIC": "12",
    }

    m = re.search(r"(\d{1,2})[-/\s]([A-ZÁÉÍÓÚ]{3})[-/\s](\d{4})", raw)
    if m:
        d, mes, year = m.groups()
        mes_num = meses.get(mes[:3], None)
        if mes_num:
            return f"{year}-{mes_num}-{d.zfill(2)}"

    m = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", raw)
    if m:
        d, mes, year = m.groups()
        return f"{year}-{mes.zfill(2)}-{d.zfill(2)}"

    return raw


def limpiar_sexo(raw):
    if not raw:
        return None
    rv = raw.strip().upper()
    if rv.startswith("M"):
        return "Masculino"
    if rv.startswith("F"):
        return "Femenino"
    return None


def limpiar_rh(raw):
    if not raw:
        return None

    raw = raw.upper().replace(" ", "")
    m = re.search(r"\b([ABO]{1,2}[+-])\b", raw)
    if m:
        return m.group(1)

    # OCR suele confundir O con 0.
    raw = raw.replace("0", "O")
    m = re.search(r"([ABO]{1,2}[+-])", raw)
    return m.group(1) if m else None


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preprocesa_image(pil_img):
    try:
        img = np.array(pil_img.convert("L"))

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        img = clahe.apply(img)

        img = cv2.GaussianBlur(img, (3, 3), 0)

        thresh = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7
        )

        return Image.fromarray(thresh)

    except Exception as e:
        logger.warning("Preprocesamiento falló: %s", e)
        return pil_img


# ============================================================
# OCR POR CAMPO
# ============================================================

def _ocr_field(crop, field):
    """
    OCR específico por tipo de dato.
    Probamos dos configuraciones cuando el primer resultado
    está vacío o no cumple una validación básica.
    """
    configs = {
        "numero_documento": [
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
        ],
        "fecha_nacimiento": [
            "--oem 3 --psm 7",
            "--oem 3 --psm 8",
        ],
        "sexo": [
            "--oem 3 --psm 10 -c tessedit_char_whitelist=MF",
            "--oem 3 --psm 8 -c tessedit_char_whitelist=MF",
        ],
        "tipo_sangre": [
            "--oem 3 --psm 10 -c tessedit_char_whitelist=ABO+−-",
            "--oem 3 --psm 8 -c tessedit_char_whitelist=ABO+−-",
        ],
    }

    field_configs = configs.get(
        field,
        ["--oem 3 --psm 7", "--oem 3 --psm 8"]
    )

    best = ""

    for config in field_configs:
        raw = pytesseract.image_to_string(
            crop,
            lang="spa",
            config=config
        ).strip()

        if raw:
            best = raw
            break

    return best


def _limpiar_campo(field, raw):
    if field == "numero_documento":
        return limpiar_numero(raw)

    if field == "fecha_nacimiento":
        return limpiar_fecha(raw)

    if field == "sexo":
        return limpiar_sexo(raw)

    if field == "tipo_sangre":
        return limpiar_rh(raw)

    if field in (
        "apellidos",
        "nombres",
        "nombre_completo",
        "lugar_nacimiento",
        "nacionalidad",
    ):
        return limpiar_texto(raw)

    return raw.strip() if raw else None


# ============================================================
# PDF DIGITAL
# ============================================================

def extract_fields_from_text(text):
    results = {}

    match = re.search(r"\b(\d{6,15})\b", text)
    results["numero_documento"] = match.group(1) if match else None

    match = re.search(
        r"(?:NOMBRE|NOMBRES?)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)",
        text,
        re.IGNORECASE,
    )
    results["nombre_completo"] = (
        limpiar_texto(match.group(1).strip()) if match else None
    )

    match = re.search(
        r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        text
    )
    results["fecha_nacimiento"] = match.group(1) if match else None

    match = re.search(
        r"(SEXO|GENERO)\s*:?\s*([MF])",
        text,
        re.IGNORECASE
    )
    results["sexo"] = match.group(2) if match else None

    match = re.search(
        r"(?:LUGAR|CIUDAD)\s*(?:DE)?\s*NACIMIENTO\s*:?\s*"
        r"([A-ZÁÉÍÓÚÑ\s]+)",
        text,
        re.IGNORECASE,
    )
    results["lugar_nacimiento"] = (
        limpiar_texto(match.group(1).strip()) if match else None
    )

    match = re.search(
        r"(NACIONALIDAD)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)",
        text,
        re.IGNORECASE,
    )
    results["nacionalidad"] = (
        limpiar_texto(match.group(2).strip()) if match else None
    )

    match = re.search(
        r"(TIPO\s*(?:DE)?\s*SANGRE|RH)\s*:?\s*([A-Z0-9+-]+)",
        text,
        re.IGNORECASE,
    )
    results["tipo_sangre"] = (
        limpiar_rh(match.group(2)) if match else None
    )

    return results


# ============================================================
# VALIDACIÓN
# ============================================================

def _resultado_util(results):
    principales = [
        "numero_documento",
        "apellidos",
        "nombres",
        "nombre_completo",
        "fecha_nacimiento",
        "sexo",
        "tipo_sangre",
    ]

    return sum(bool(results.get(k)) for k in principales) >= 2


# ============================================================
# EXTRACCIÓN PRINCIPAL
# ============================================================

def extract_fields(file_path, modelo="hologramas"):
    """
    Procesa hasta las dos primeras páginas.

    Página 1:
        número + apellidos + nombres

    Página 2:
        fecha + lugar + sexo + RH + nacionalidad

    La imagen de cada página se normaliza antes de aplicar
    las zonas. Esto desacopla las coordenadas de la resolución
    del PDF generado por el celular.
    """
    results = {}

    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                logger.error("El PDF no contiene páginas.")
                return {}

            for idx, page in enumerate(pdf.pages[:2]):

                # ------------------------------------------------
                # 1. Texto embebido
                # ------------------------------------------------
                text = page.extract_text() or ""

                if text.strip():
                    text_results = extract_fields_from_text(text)

                    for key, value in text_results.items():
                        if value and not results.get(key):
                            results[key] = value

                    if _resultado_util(text_results):
                        logger.info(
                            "Página %s: datos obtenidos desde texto PDF.",
                            idx + 1
                        )
                        continue

                # ------------------------------------------------
                # 2. Renderizar página
                # ------------------------------------------------
                logger.info(
                    "Página %s: entrando a OCR de imagen.",
                    idx + 1
                )

                img = page.to_image(resolution=300).original
                img_np = np.array(img)

                # ------------------------------------------------
                # 3. DETECTAR Y NORMALIZAR CÉDULA
                # ------------------------------------------------
                cedula_np, metadata = detectar_y_recortar_cedula(
                    img_np,
                    return_metadata=True
                )

                logger.info(
                    "Página %s | detección=%s | método=%s | score=%s | "
                    "aspect=%s | tamaño=%s",
                    idx + 1,
                    metadata["detected"],
                    metadata["method"],
                    metadata["score"],
                    metadata["aspect"],
                    metadata["normalized_size"],
                )

                img = Image.fromarray(cedula_np)

                width, height = img.size

                # ------------------------------------------------
                # 4. Seleccionar plantilla
                # ------------------------------------------------
                if modelo == "digital":
                    zones = zones_digital
                elif idx == 0:
                    zones = zones_hologramas_anverso
                else:
                    zones = zones_hologramas_reverso

                # ------------------------------------------------
                # 5. OCR por zona
                # ------------------------------------------------
                for field, coords in zones.items():

                    if results.get(field):
                        continue

                    x1, y1, x2, y2 = coords

                    x1, x2 = sorted((x1, x2))
                    y1, y2 = sorted((y1, y2))

                    # Seguridad ante coordenadas inválidas.
                    x1 = max(0.0, min(1.0, x1))
                    x2 = max(0.0, min(1.0, x2))
                    y1 = max(0.0, min(1.0, y1))
                    y2 = max(0.0, min(1.0, y2))

                    box = (
                        int(x1 * width),
                        int(y1 * height),
                        int(x2 * width),
                        int(y2 * height),
                    )

                    if box[2] <= box[0] or box[3] <= box[1]:
                        results[field] = None
                        continue

                    try:
                        crop = img.crop(box)

                        # Agrandar campos pequeños antes del OCR.
                        crop = crop.resize(
                            (crop.width * 2, crop.height * 2),
                            Image.Resampling.LANCZOS
                        )

                        crop = preprocesa_image(crop)

                        raw = _ocr_field(crop, field)
                        clean = _limpiar_campo(field, raw)

                        logger.info(
                            "Página %s | %s | raw=%r | clean=%r | box=%s",
                            idx + 1,
                            field,
                            raw,
                            clean,
                            box,
                        )

                        if clean:
                            results[field] = clean
                        elif field not in results:
                            results[field] = None

                    except Exception as e:
                        logger.exception(
                            "Error OCR campo %s página %s: %s",
                            field,
                            idx + 1,
                            e
                        )
                        results[field] = None

    except Exception as e:
        logger.exception("Error en extract_fields: %s", e)
        return {}

    return results
