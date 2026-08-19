import logging
import re
import os
from datetime import datetime

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

from app.parser import detectar_y_recortar_cedula


logger = logging.getLogger(__name__)

# ============================================================
# URL PÚBLICA PARA CROPS DE DEBUG
# ============================================================

def _obtener_url_crop(file_path):
    """
    Convierte la ruta interna del crop en una URL pública
    accesible desde Render.

    Render proporciona RENDER_EXTERNAL_URL automáticamente.
    También permite definir PUBLIC_BASE_URL manualmente.
    """

    base_url = (
        os.getenv("PUBLIC_BASE_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or ""
    ).rstrip("/")

    if not base_url:
        return None

    # Normalizar separadores de Windows/Linux
    normalized = file_path.replace("\\", "/")

    marcador = "/documentos/"

    if marcador not in normalized:
        return None

    parte = normalized.split(marcador, 1)[1]

    # parte:
    # desarrollador_software/ocr_debug/pagina_01_apellidos_original.png

    return f"{base_url}/ocr/debug/{parte}"

# ============================================================
# PLANTILLA NORMALIZADA
# ============================================================

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

    digits = re.sub(r"\D", "", str(raw))

    return digits or None


def limpiar_texto(raw):
    if not raw:
        return None

    texto = re.sub(
        r"[^A-Za-zÁÉÍÓÚÑáéíóúñ\s]",
        "",
        str(raw)
    )

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto.title() if texto else None


def limpiar_fecha(raw):
    if not raw:
        return None

    raw = str(raw).strip().upper()

    meses = {
        "ENE": "01",
        "FEB": "02",
        "MAR": "03",
        "ABR": "04",
        "MAY": "05",
        "JUN": "06",
        "JUL": "07",
        "AGO": "08",
        "SEP": "09",
        "OCT": "10",
        "NOV": "11",
        "DIC": "12",
    }

    # Formato: 17-AGO-1990 / 17 AGO 1990 / 17/AGO/1990
    m = re.search(
        r"(\d{1,2})[-/\s]([A-ZÁÉÍÓÚ]{3})[-/\s](\d{4})",
        raw
    )

    if m:
        d, mes, year = m.groups()

        mes_num = meses.get(mes[:3])

        if mes_num:
            try:
                fecha = datetime.strptime(
                    f"{year}-{mes_num}-{d.zfill(2)}",
                    "%Y-%m-%d"
                )

                return fecha.strftime("%Y-%m-%d")

            except ValueError:
                return None

    # Formato: 17/08/1990
    m = re.search(
        r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})",
        raw
    )

    if m:
        d, mes, year = m.groups()

        try:
            fecha = datetime.strptime(
                f"{year}-{mes.zfill(2)}-{d.zfill(2)}",
                "%Y-%m-%d"
            )

            return fecha.strftime("%Y-%m-%d")

        except ValueError:
            return None

    return None


def limpiar_sexo(raw):
    if not raw:
        return None

    rv = str(raw).strip().upper()

    if rv.startswith("M"):
        return "Masculino"

    if rv.startswith("F"):
        return "Femenino"

    return None


def limpiar_rh(raw):
    if not raw:
        return None

    raw = str(raw).upper().replace(" ", "")

    # OCR suele confundir O con 0.
    raw = raw.replace("0", "O")

    m = re.search(
        r"(AB|A|B|O)[+-]",
        raw
    )

    return m.group(0) if m else None


# ============================================================
# VALIDACIÓN DE RESULTADOS OCR
# ============================================================

def validar_campo_ocr(field, value):
    """
    Evita aceptar resultados OCR claramente inválidos.

    Ejemplos:
        fecha_nacimiento = "A"      -> inválido
        nacionalidad = "O"          -> inválido
        nombres = "O"               -> inválido
        numero_documento = "A123"   -> inválido
        sexo = "A"                  -> inválido
        RH = "A"                    -> inválido
    """

    if value is None:
        return False

    value = str(value).strip()

    if not value:
        return False

    # --------------------------------------------------------
    # NÚMERO DE DOCUMENTO
    # --------------------------------------------------------

    if field == "numero_documento":
        return bool(
            re.fullmatch(r"\d{6,12}", value)
        )

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    if field == "fecha_nacimiento":
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            value
        ):
            return False

        try:
            datetime.strptime(
                value,
                "%Y-%m-%d"
            )
            return True

        except ValueError:
            return False

    # --------------------------------------------------------
    # SEXO
    # --------------------------------------------------------

    if field == "sexo":
        return value in (
            "Masculino",
            "Femenino",
        )

    # --------------------------------------------------------
    # RH
    # --------------------------------------------------------

    if field == "tipo_sangre":
        return bool(
            re.fullmatch(
                r"(AB|A|B|O)[+-]",
                value
            )
        )

    # --------------------------------------------------------
    # NOMBRES / APELLIDOS
    # --------------------------------------------------------

    if field in (
        "apellidos",
        "nombres",
        "nombre_completo",
    ):
        palabras = value.split()

        if len(value) < 3:
            return False

        if not palabras:
            return False

        # Evita aceptar una única letra como "O" o "A".
        if len(value.replace(" ", "")) < 3:
            return False

        return True

    # --------------------------------------------------------
    # LUGAR / NACIONALIDAD
    # --------------------------------------------------------

    if field in (
        "lugar_nacimiento",
        "nacionalidad",
    ):
        if len(value) < 3:
            return False

        if len(value.replace(" ", "")) < 3:
            return False

        return True

    return True


# ============================================================
# DEBUG DE CROPS
# ============================================================

def guardar_crop_debug(
    crop,
    field,
    page_number,
    debug_dir
):
    """
    Guarda físicamente el crop utilizado durante el OCR.

    Se guardan tres versiones:

        original
        ampliado
        ocr

    Esto permite comprobar exactamente qué está viendo
    Tesseract.
    """

    try:
        os.makedirs(
            debug_dir,
            exist_ok=True
        )

        filename = (
            f"pagina_{page_number:02d}_"
            f"{field}.png"
        )

        path = os.path.join(
            debug_dir,
            filename
        )

        crop.save(path)

        url = _obtener_url_crop(path)

        logger.info(
            "OCR_CROP_GUARDADO | pagina=%s | campo=%s | archivo=%s",
            page_number,
            field,
            path
        )

        if url:
            logger.info(
                "OCR_CROP_URL | pagina=%s | campo=%s | url=%s",
                page_number,
                field,
                url
            )

        return path

    except Exception as e:
        logger.warning(
            "OCR_CROP_ERROR | "
            "pagina=%s | campo=%s | error=%s",
            page_number,
            field,
            e
        )

        return None


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preprocesa_image(pil_img):

    try:

        img = np.array(
            pil_img.convert("L")
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        img = clahe.apply(img)

        img = cv2.GaussianBlur(
            img,
            (3, 3),
            0
        )

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

        logger.warning(
            "Preprocesamiento falló: %s",
            e
        )

        return pil_img


# ============================================================
# OCR POR CAMPO
# ============================================================

def _ocr_field(crop, field):

    configs = {

        "numero_documento": [
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=0123456789",

            "--oem 3 --psm 8 "
            "-c tessedit_char_whitelist=0123456789",
        ],

        "fecha_nacimiento": [
            "--oem 3 --psm 7",
            "--oem 3 --psm 8",
        ],

        "sexo": [
            "--oem 3 --psm 10 "
            "-c tessedit_char_whitelist=MF",

            "--oem 3 --psm 8 "
            "-c tessedit_char_whitelist=MF",
        ],

        "tipo_sangre": [
            "--oem 3 --psm 10 "
            "-c tessedit_char_whitelist=ABO+−-",

            "--oem 3 --psm 8 "
            "-c tessedit_char_whitelist=ABO+−-",
        ],
    }

    field_configs = configs.get(
        field,
        [
            "--oem 3 --psm 7",
            "--oem 3 --psm 8",
        ]
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

    if not raw:
        return None

    if field == "numero_documento":
        value = limpiar_numero(raw)

    elif field == "fecha_nacimiento":
        value = limpiar_fecha(raw)

    elif field == "sexo":
        value = limpiar_sexo(raw)

    elif field == "tipo_sangre":
        value = limpiar_rh(raw)

    elif field in (
        "apellidos",
        "nombres",
        "nombre_completo",
        "lugar_nacimiento",
        "nacionalidad",
    ):
        value = limpiar_texto(raw)

    else:
        value = str(raw).strip() if raw else None

    if not validar_campo_ocr(
        field,
        value
    ):
        logger.warning(
            "OCR_VALOR_RECHAZADO | "
            "campo=%s | raw=%r | clean=%r",
            field,
            raw,
            value
        )

        return None

    return value


# ============================================================
# PDF DIGITAL
# ============================================================

def extract_fields_from_text(text):

    results = {}

    # --------------------------------------------------------
    # DOCUMENTO
    # --------------------------------------------------------

    match = re.search(
        r"\b(\d{6,15})\b",
        text
    )

    results["numero_documento"] = (
        match.group(1)
        if match
        else None
    )

    # --------------------------------------------------------
    # NOMBRE
    # --------------------------------------------------------

    match = re.search(
        r"(?:NOMBRE|NOMBRES?)\s*:?\s*"
        r"([A-ZÁÉÍÓÚÑ\s]+)",
        text,
        re.IGNORECASE,
    )

    results["nombre_completo"] = (
        limpiar_texto(
            match.group(1).strip()
        )
        if match
        else None
    )

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    match = re.search(
        r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        text
    )

    results["fecha_nacimiento"] = (
        limpiar_fecha(
            match.group(1)
        )
        if match
        else None
    )

    # --------------------------------------------------------
    # SEXO
    # --------------------------------------------------------

    match = re.search(
        r"(SEXO|GENERO)\s*:?\s*([MF])",
        text,
        re.IGNORECASE
    )

    results["sexo"] = (
        limpiar_sexo(
            match.group(2)
        )
        if match
        else None
    )

    # --------------------------------------------------------
    # LUGAR
    # --------------------------------------------------------

    match = re.search(
        r"(?:LUGAR|CIUDAD)\s*(?:DE)?\s*"
        r"NACIMIENTO\s*:?\s*"
        r"([A-ZÁÉÍÓÚÑ\s]+)",
        text,
        re.IGNORECASE,
    )

    results["lugar_nacimiento"] = (
        limpiar_texto(
            match.group(1).strip()
        )
        if match
        else None
    )

    # --------------------------------------------------------
    # NACIONALIDAD
    # --------------------------------------------------------

    match = re.search(
        r"(NACIONALIDAD)\s*:?\s*"
        r"([A-ZÁÉÍÓÚÑ\s]+)",
        text,
        re.IGNORECASE
    )

    results["nacionalidad"] = (
        limpiar_texto(
            match.group(2).strip()
        )
        if match
        else None
    )

    # --------------------------------------------------------
    # RH
    # --------------------------------------------------------

    match = re.search(
        r"(TIPO\s*(?:DE)?\s*SANGRE|RH)"
        r"\s*:?\s*([A-Z0-9+-]+)",
        text,
        re.IGNORECASE
    )

    results["tipo_sangre"] = (
        limpiar_rh(
            match.group(2)
        )
        if match
        else None
    )

    return results


# ============================================================
# VALIDACIÓN GENERAL
# ============================================================

def _resultado_util(results):

    if not results:
        return False

    principales = [
        "numero_documento",
        "apellidos",
        "nombres",
        "nombre_completo",
        "fecha_nacimiento",
        "sexo",
        "tipo_sangre",
    ]

    return (
        sum(
            bool(results.get(k))
            for k in principales
        )
        >= 2
    )


# ============================================================
# RESULTADO SEGURO
# ============================================================

def normalizar_resultado_final(results):

    campos = [
        "numero_documento",
        "apellidos",
        "nombres",
        "nombre_completo",
        "fecha_nacimiento",
        "sexo",
        "lugar_nacimiento",
        "nacionalidad",
        "tipo_sangre",
    ]

    resultado = {}

    for campo in campos:

        valor = results.get(campo)

        if valor is None:
            resultado[campo] = ""
        else:
            resultado[campo] = str(valor).strip()

    return resultado


# ============================================================
# EXTRACCIÓN PRINCIPAL
# ============================================================

def extract_fields(
    file_path,
    modelo="hologramas"
):
    """
    Procesa hasta las dos primeras páginas.

    Página 1:
        número + apellidos + nombres

    Página 2:
        fecha + lugar + sexo + RH + nacionalidad

    Durante esta etapa de depuración se guardan los crops
    utilizados por OCR dentro de:

        <carpeta_del_pdf>/ocr_debug/
    """

    results = {}

    # --------------------------------------------------------
    # DIRECTORIO DE DEBUG
    # --------------------------------------------------------

    debug_dir = os.path.join(
        os.path.dirname(file_path),
        "ocr_debug"
    )

    logger.info(
        "OCR_DEBUG_DIR: %s",
        debug_dir
    )

    try:

        with pdfplumber.open(file_path) as pdf:

            if not pdf.pages:

                logger.error(
                    "El PDF no contiene páginas."
                )

                return {}

            for idx, page in enumerate(
                pdf.pages[:2]
            ):

                page_number = idx + 1

                # ====================================================
                # 1. TEXTO EMBEBIDO
                # ====================================================

                text = (
                    page.extract_text()
                    or ""
                )

                if text.strip():

                    text_results = (
                        extract_fields_from_text(
                            text
                        )
                    )

                    for key, value in text_results.items():

                        if (
                            value
                            and not results.get(key)
                            and validar_campo_ocr(
                                key,
                                value
                            )
                        ):
                            results[key] = value

                    if _resultado_util(
                        text_results
                    ):

                        logger.info(
                            "Página %s: datos obtenidos "
                            "desde texto PDF.",
                            page_number
                        )

                        continue

                # ====================================================
                # 2. RENDERIZAR PÁGINA
                # ====================================================

                logger.info(
                    "Página %s: entrando a OCR "
                    "de imagen.",
                    page_number
                )

                img = page.to_image(
                    resolution=300
                ).original

                img_np = np.array(img)

                logger.info(
                    "Página %s | imagen original=%s",
                    page_number,
                    img_np.shape
                )

                # ====================================================
                # 3. DETECTAR Y NORMALIZAR CÉDULA
                # ====================================================

                cedula_np, metadata = (
                    detectar_y_recortar_cedula(
                        img_np,
                        return_metadata=True
                    )
                )

                logger.info(
                    "Página %s | detección=%s | "
                    "método=%s | score=%s | "
                    "aspect=%s | tamaño=%s",
                    page_number,
                    metadata.get(
                        "detected"
                    ),
                    metadata.get(
                        "method"
                    ),
                    metadata.get(
                        "score"
                    ),
                    metadata.get(
                        "aspect"
                    ),
                    metadata.get(
                        "normalized_size"
                    ),
                )

                img = Image.fromarray(
                    cedula_np
                )

                width, height = img.size

                logger.info(
                    "Página %s | imagen OCR "
                    "final=%sx%s",
                    page_number,
                    width,
                    height
                )

                # ====================================================
                # 4. SELECCIONAR PLANTILLA
                # ====================================================

                if modelo == "digital":

                    zones = zones_digital

                elif idx == 0:

                    zones = (
                        zones_hologramas_anverso
                    )

                else:

                    zones = (
                        zones_hologramas_reverso
                    )

                # ====================================================
                # 5. OCR POR ZONA
                # ====================================================

                for field, coords in zones.items():

                    if results.get(field):
                        continue

                    x1, y1, x2, y2 = coords

                    x1, x2 = sorted(
                        (x1, x2)
                    )

                    y1, y2 = sorted(
                        (y1, y2)
                    )

                    # ------------------------------------------------
                    # Seguridad
                    # ------------------------------------------------

                    x1 = max(
                        0.0,
                        min(1.0, x1)
                    )

                    x2 = max(
                        0.0,
                        min(1.0, x2)
                    )

                    y1 = max(
                        0.0,
                        min(1.0, y1)
                    )

                    y2 = max(
                        0.0,
                        min(1.0, y2)
                    )

                    # ------------------------------------------------
                    # Coordenadas reales
                    # ------------------------------------------------

                    box = (
                        int(x1 * width),
                        int(y1 * height),
                        int(x2 * width),
                        int(y2 * height),
                    )

                    logger.info(
                        "OCR_COORDENADAS | "
                        "pagina=%s | campo=%s | "
                        "normalizadas=(%.4f, %.4f, %.4f, %.4f) | "
                        "pixeles=%s",
                        page_number,
                        field,
                        x1,
                        y1,
                        x2,
                        y2,
                        box
                    )

                    if (
                        box[2] <= box[0]
                        or box[3] <= box[1]
                    ):

                        logger.warning(
                            "OCR_BOX_INVALIDO | "
                            "pagina=%s | campo=%s | "
                            "box=%s",
                            page_number,
                            field,
                            box
                        )

                        results[field] = None

                        continue

                    # ====================================================
                    # CROP
                    # ====================================================

                    try:

                        crop = img.crop(
                            box
                        )

                        logger.info(
                            "OCR_CROP | "
                            "pagina=%s | campo=%s | "
                            "tamaño_original=%sx%s",
                            page_number,
                            field,
                            crop.width,
                            crop.height
                        )

                        # ------------------------------------------------
                        # Guardar ORIGINAL
                        # ------------------------------------------------

                        guardar_crop_debug(
                            crop,
                            f"{field}_original",
                            page_number,
                            debug_dir
                        )

                        # ====================================================
                        # AMPLIAR
                        # ====================================================

                        crop = crop.resize(
                            (
                                crop.width * 2,
                                crop.height * 2
                            ),
                            Image.Resampling.LANCZOS
                        )

                        logger.info(
                            "OCR_CROP_AMPLIADO | "
                            "pagina=%s | campo=%s | "
                            "tamaño=%sx%s",
                            page_number,
                            field,
                            crop.width,
                            crop.height
                        )

                        guardar_crop_debug(
                            crop,
                            f"{field}_ampliado",
                            page_number,
                            debug_dir
                        )

                        # ====================================================
                        # PREPROCESAR
                        # ====================================================

                        crop = preprocesa_image(
                            crop
                        )

                        logger.info(
                            "OCR_CROP_PREPROCESADO | "
                            "pagina=%s | campo=%s | "
                            "tamaño=%sx%s",
                            page_number,
                            field,
                            crop.width,
                            crop.height
                        )

                        # ------------------------------------------------
                        # Guardar exactamente lo que recibe Tesseract
                        # ------------------------------------------------

                        guardar_crop_debug(
                            crop,
                            f"{field}_ocr",
                            page_number,
                            debug_dir
                        )

                        # ====================================================
                        # TESSERACT
                        # ====================================================

                        raw = _ocr_field(
                            crop,
                            field
                        )

                        clean = _limpiar_campo(
                            field,
                            raw
                        )

                        logger.info(
                            "OCR_RESULTADO_CAMPO | "
                            "pagina=%s | campo=%s | "
                            "raw=%r | clean=%r | "
                            "box=%s",
                            page_number,
                            field,
                            raw,
                            clean,
                            box
                        )

                        # ====================================================
                        # GUARDAR SOLO RESULTADOS VÁLIDOS
                        # ====================================================

                        if clean:

                            results[field] = clean

                        elif field not in results:

                            results[field] = None

                    except Exception as e:

                        logger.exception(
                            "Error OCR campo %s "
                            "página %s: %s",
                            field,
                            page_number,
                            e
                        )

                        results[field] = None

    except Exception as e:

        logger.exception(
            "Error en extract_fields: %s",
            e
        )

        return {}

    # ============================================================
    # NORMALIZAR RESULTADO FINAL
    # ============================================================

    resultado_final = (
        normalizar_resultado_final(
            results
        )
    )

    logger.info(
        "OCR_RESULTADO_FINAL: %s",
        resultado_final
    )

    return resultado_final