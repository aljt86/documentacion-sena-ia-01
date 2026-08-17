import os
import logging
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
from datetime import datetime
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN
# ============================================================

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
elif os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER_PATH = os.getenv("POPPLER_PATH")

# Proporción física aproximada de una tarjeta ID-1:
# 85.60 x 53.98 mm = 1.586
CARD_ASPECT = 85.60 / 53.98


# ============================================================
# UTILIDADES
# ============================================================

def _ordenar_esquinas(pts):
    """Devuelve puntos en orden TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32)
    suma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    return np.array([
        pts[np.argmin(suma)],   # top-left
        pts[np.argmin(diff)],   # top-right
        pts[np.argmax(suma)],   # bottom-right
        pts[np.argmax(diff)],   # bottom-left
    ], dtype=np.float32)


def _perspectiva_rectificada(imagen, corners, width=1712, height=1080):
    """Rectifica la tarjeta y la lleva a una resolución fija."""
    src = _ordenar_esquinas(corners)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(imagen, matrix, (width, height))


def _score_quadrilateral(contour, image_shape):
    """Puntúa un contorno según área, forma rectangular y proporción ID-1."""
    area = cv2.contourArea(contour)
    img_h, img_w = image_shape[:2]
    img_area = img_w * img_h

    if area < img_area * 0.02:
        return None

    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None

    pts = approx.reshape(4, 2).astype(np.float32)

    ordered = _ordenar_esquinas(pts)
    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])

    width = (top + bottom) / 2
    height = (left + right) / 2

    if height <= 0:
        return None

    aspect = width / height
    aspect_error = abs(aspect - CARD_ASPECT) / CARD_ASPECT

    # Permitimos perspectiva moderada y fotografías algo deformadas.
    if aspect < 1.15 or aspect > 2.10:
        return None

    rectangularity = area / max(width * height, 1)
    area_ratio = area / img_area

    # Preferimos tarjetas grandes, rectangulares y cercanas a ID-1.
    score = (
        min(area_ratio / 0.25, 1.0) * 0.35
        + max(0.0, 1.0 - aspect_error / 0.35) * 0.40
        + min(rectangularity / 0.90, 1.0) * 0.25
    )

    return score, ordered, area, aspect


def detectar_y_recortar_cedula(imagen, return_metadata=False):
    """
    Detecta una cédula dentro de la hoja escaneada.

    Estrategia:
    1. Canny + morfología.
    2. Umbral adaptativo + morfología.
    3. Busca cuadriláteros con proporción de tarjeta ID-1.
    4. Si encuentra uno, corrige perspectiva.
    5. Si no encuentra uno, usa un fallback por bounding box.
    6. Si todo falla, devuelve la imagen original.

    La salida normalizada es 1712x1080, independiente de la
    resolución del móvil/escáner.
    """
    if imagen is None:
        raise ValueError("La imagen recibida es None")

    if imagen.ndim == 2:
        rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    else:
        rgb = imagen.copy()

    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Reducimos temporalmente imágenes enormes para que la detección
    # sea estable y rápida. Las coordenadas se escalan después.
    scale = min(1.0, 1800.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    blurred = cv2.GaussianBlur(small, (5, 5), 0)

    candidate_masks = []

    edges = cv2.Canny(blurred, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate_masks.append(edges)

    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )
    adaptive = cv2.bitwise_not(adaptive)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate_masks.append(adaptive)

    best = None

    for mask in candidate_masks:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            result = _score_quadrilateral(contour, small.shape)
            if result is None:
                continue

            score, ordered, area, aspect = result

            # Volvemos las esquinas a coordenadas de la imagen original.
            ordered_original = ordered / scale

            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "corners": ordered_original,
                    "area": float(area / (scale ** 2)),
                    "aspect": float(aspect),
                    "method": "quadrilateral",
                }

    if best is not None and best["score"] >= 0.45:
        rectified = _perspectiva_rectificada(
            rgb,
            best["corners"],
            width=1712,
            height=1080,
        )

        metadata = {
            "detected": True,
            "method": best["method"],
            "score": round(best["score"], 4),
            "aspect": round(best["aspect"], 4),
            "corners": best["corners"].round(2).tolist(),
            "original_size": [w, h],
            "normalized_size": [1712, 1080],
        }

        if return_metadata:
            return rectified, metadata
        return rectified

    # ========================================================
    # FALLBACK
    # ========================================================
    # Cuando el escaneo no conserva un borde rectangular claro,
    # intentamos localizar la región de contenido más grande.
    thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    fallback = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < small.shape[0] * small.shape[1] * 0.05:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        aspect = cw / max(ch, 1)

        if 1.10 <= aspect <= 2.20:
            score = min(area / (small.shape[0] * small.shape[1] * 0.50), 1.0)
            if fallback is None or score > fallback[0]:
                fallback = (score, x, y, cw, ch)

    if fallback is not None:
        _, x, y, cw, ch = fallback
        x = int(x / scale)
        y = int(y / scale)
        cw = int(cw / scale)
        ch = int(ch / scale)

        pad = int(min(w, h) * 0.01)
        x = max(0, x - pad)
        y = max(0, y - pad)
        x2 = min(w, x + cw + 2 * pad)
        y2 = min(h, y + ch + 2 * pad)

        crop = rgb[y:y2, x:x2]

        # El fallback no conoce las cuatro esquinas, por lo que no
        # inventamos una perspectiva. Solo normalizamos manteniendo
        # la imagen disponible.
        normalized = cv2.resize(
            crop, (1712, 1080), interpolation=cv2.INTER_CUBIC
        )

        metadata = {
            "detected": True,
            "method": "bounding_box_fallback",
            "score": round(float(fallback[0]), 4),
            "corners": None,
            "original_size": [w, h],
            "normalized_size": [1712, 1080],
        }

        if return_metadata:
            return normalized, metadata
        return normalized

    metadata = {
        "detected": False,
        "method": "original_image",
        "score": 0.0,
        "corners": None,
        "original_size": [w, h],
        "normalized_size": [w, h],
    }

    if return_metadata:
        return rgb, metadata

    return rgb


# ============================================================
# EXTRACCIÓN LEGACY / PDF DIGITAL
# ============================================================

def calcular_edad(fecha_str: str):
    meses = {
        "ENE": "Jan", "FEB": "Feb", "MAR": "Mar", "ABR": "Apr",
        "MAY": "May", "JUN": "Jun", "JUL": "Jul", "AGO": "Aug",
        "SEP": "Sep", "OCT": "Oct", "NOV": "Nov", "DIC": "Dec"
    }

    if not fecha_str:
        return None

    try:
        partes = fecha_str.replace(",", "").split("-")
        if len(partes) == 3:
            mes_eng = meses.get(partes[1].upper(), partes[1])
            fecha_dt = datetime.strptime(
                f"{partes[0]}-{mes_eng}-{partes[2]}", "%d-%b-%Y"
            )
        else:
            partes = fecha_str.split()
            dia, mes, anio = partes
            mes_eng = meses.get(mes.upper(), mes)
            fecha_dt = datetime.strptime(
                f"{dia}-{mes_eng}-{anio}", "%d-%b-%Y"
            )

        hoy = datetime.today()
        return hoy.year - fecha_dt.year - (
            (hoy.month, hoy.day) < (fecha_dt.month, fecha_dt.day)
        )
    except Exception:
        return None


def preprocesar_imagen(pagina):
    img = np.array(pagina)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    return cv2.medianBlur(thresh, 3)


def ocr_pagina(pagina):
    procesada = preprocesar_imagen(pagina)
    return pytesseract.image_to_string(
        procesada,
        lang="spa",
        config="--psm 6"
    )


def normalizar(texto: str) -> str:
    reemplazos = {"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U"}
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)
    return texto.upper()


def limpiar_numero(texto: str):
    if not texto:
        return None
    digits = re.sub(r"\D", "", texto)
    return digits if digits else None


def limpiar_texto(texto: str):
    if not texto:
        return None
    return re.sub(r"[^A-Za-zÁÉÍÓÚÑ ]", "", texto).strip()


def limpiar_fecha(texto: str):
    if not texto:
        return None
    match = re.search(
        r"\d{2}[-/][A-Za-z]{3}[-/]\d{4}|\d{2}/\d{2}/\d{4}",
        texto
    )
    return match.group(0) if match else None


def extraer_campos_por_lineas(texto: str):
    datos = {
        "nombre_completo": None,
        "numero_documento": None,
        "fecha_nacimiento": None,
        "sexo": None,
        "lugar_nacimiento": None,
        "fecha_expedicion": None,
        "nacionalidad": None,
        "rh": None,
        "edad": None,
    }

    lineas = texto.splitlines()
    tipo_doc = "desconocido"

    encabezados = {
        "REPUBLICA", "REPUBLICA DE COLOMBIA", "COLOMBIA",
        "IDENTIFICACION", "IDENTIFICACIÓN", "IDENTIFICACIÓN PERSONAL",
        "CEDULA DE CIUDADANÍA", "CEDULA DE CIUDADANIA", "CEDULA"
    }

    n = 0
    while n < 3 and lineas:
        primera = normalizar(lineas[0].strip())
        if any(k in primera for k in encabezados):
            lineas.pop(0)
            n += 1
        else:
            break

    for i, linea in enumerate(lineas):
        l = normalizar(linea.strip())

        if "NUIP" in l:
            tipo_doc = "nueva"
        if re.search(r"(APELLIDOS?|NOMBRES?)", l):
            tipo_doc = "antigua"

        if re.search(r"NUMER[O0]", l) and i + 1 < len(lineas):
            datos["numero_documento"] = re.sub(r"\D", "", lineas[i + 1].strip())

        if "NUIP" in l and i + 1 < len(lineas):
            datos["numero_documento"] = re.sub(r"\D", "", lineas[i + 1].strip())

        if "FECHA DE NACIMIENTO" in l and i + 1 < len(lineas):
            fnac = lineas[i + 1].strip()
            datos["fecha_nacimiento"] = fnac
            datos["edad"] = calcular_edad(fnac)

        if "LUGAR DE NACIMIENTO" in l and i - 2 >= 0 and i - 1 >= 0:
            ciudad = lineas[i - 2].strip()
            depto = lineas[i - 1].strip()
            datos["lugar_nacimiento"] = f"{ciudad} {depto}".replace("!", "P")

        if "SEXO" in l and i + 1 < len(lineas):
            sexo = lineas[i + 1].strip().upper()
            if sexo.startswith("M"):
                datos["sexo"] = "Masculino"
            elif sexo.startswith("F"):
                datos["sexo"] = "Femenino"

        if "EXPEDICION" in l and i - 1 >= 0:
            datos["fecha_expedicion"] = lineas[i - 1].strip()

        if tipo_doc == "antigua":
            apellidos, nombres = "", ""
            if re.search(r"APELLIDOS?", l) and i - 1 >= 0:
                apellidos = lineas[i - 1].strip()
            if re.search(r"NOMBRES?", l) and i - 1 >= 0:
                nombres = lineas[i - 1].strip()
            if nombres or apellidos:
                datos["nombre_completo"] = f"{nombres} {apellidos}".strip()

        if tipo_doc == "nueva":
            apellidos, nombres = "", ""
            if re.search(r"APELLIDOS?", l) and i + 1 < len(lineas):
                apellidos = lineas[i + 1].strip()
            if re.search(r"NOMBRES?", l) and i + 1 < len(lineas):
                nombres = lineas[i + 1].strip()
            if nombres or apellidos:
                datos["nombre_completo"] = f"{nombres} {apellidos}".strip()

        if "RH" in l and i - 1 >= 0:
            rh_line = lineas[i - 1].strip()
            rh_match = re.search(r"\b[OAB][+-]\b", rh_line)
            if rh_match:
                datos["rh"] = rh_match.group()

        if tipo_doc == "nueva":
            if "NACIONALIDAD" in l and i + 1 < len(lineas):
                datos["nacionalidad"] = lineas[i + 1].strip()
            if re.search(r"\b[OAB][+-]\b", l):
                datos["rh"] = l.strip()

    if not datos["numero_documento"]:
        doc_match = re.findall(r"\d{8,10}", texto)
        if doc_match:
            datos["numero_documento"] = max(doc_match, key=len)

    if not datos["nombre_completo"]:
        candidatos = re.findall(
            r"[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{3,}){1,3}",
            texto
        )
        candidatos = [
            c for c in candidatos
            if "COLOMBIA" not in c and "IDENTIFICACION" not in c
        ]
        if candidatos:
            datos["nombre_completo"] = candidatos[0].title()

    return datos


# ============================================================
# ORQUESTADOR
# ============================================================

def _resultado_util(datos):
    if not datos:
        return False

    campos = [
        "numero_documento",
        "nombre_completo",
        "fecha_nacimiento",
        "sexo",
        "rh",
    ]
    return sum(bool(datos.get(c)) for c in campos) >= 2


def procesar_documento(pdf_path):
    """
    Único punto de entrada del procesamiento documental.

    1. Intenta extracción de texto del PDF.
    2. Si no es suficiente, delega el escaneo a ocr_template.py.
    """
    try:
        texto_total = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texto_total += page_text + "\n"

        if texto_total.strip():
            datos_texto = extraer_campos_por_lineas(texto_total)
            if _resultado_util(datos_texto):
                return {"resultado": datos_texto, "metodo": "pdf_texto"}

        # Importación tardía para evitar dependencias circulares.
        from app.ocr_template import extract_fields

        datos_ocr = extract_fields(pdf_path)

        if datos_ocr:
            return {"resultado": datos_ocr, "metodo": "ocr_template"}

        return {"error": "El documento no se pudo leer correctamente."}

    except Exception as e:
        return {"error": f"Error procesando documento: {e}"}


def extraer_campos(texto: str):
    return extraer_campos_por_lineas(texto)
