import logging
import re
import os
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

from app.parser import detectar_y_recortar_cedula
from app.limpieza_campos import (
    limpiar_numero,
    limpiar_texto,
    limpiar_fecha,
    limpiar_sexo,
    limpiar_rh
)

logger = logging.getLogger(__name__)


# ============================================================
# URL PÚBLICA PARA CROPS DE DEBUG
# ============================================================

def _obtener_url_crop(file_path):
    """
    Convierte la ruta física del crop en una URL pública.

    Ejemplo de archivo:
        /app/api/documentos/desarrollador_software/ocr_debug/pagina_01_nombres_ocr.png

    URL generada:
        https://TU-DOMINIO/ocr/debug/desarrollador_software/ocr_debug/pagina_01_nombres_ocr.png
    """

    base_url = (
        os.getenv("PUBLIC_BASE_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or ""
    ).rstrip("/")

    if not base_url:
        logger.warning(
            "OCR_CROP_URL_ERROR | "
            "No existe PUBLIC_BASE_URL ni RENDER_EXTERNAL_URL"
        )
        return None

    normalized = os.path.abspath(
        file_path
    ).replace("\\", "/")

    marcador = "/documentos/"

    if marcador not in normalized:

        logger.warning(
            "OCR_CROP_URL_ERROR | "
            "ruta fuera de documentos | archivo=%s",
            normalized
        )

        return None

    parte = normalized.split(
        marcador,
        1
    )[1]

    return (
        f"{base_url}/ocr/debug/"
        f"{parte}"
    )


# ============================================================
# PLANTILLA NORMALIZADA
# ============================================================

zones_hologramas_anverso = {
    "numero_documento": (0.04, 0.26, 0.85, 0.36),
    "apellidos":        (0.04, 0.34, 0.50, 0.44),
    "nombres":          (0.04, 0.46, 0.99, 0.60),
}


zones_hologramas_reverso = {
    "fecha_nacimiento": (0.70, 0.07, 0.94, 0.16),
    "lugar_nacimiento": (0.10, 0.07, 0.43, 0.16),
    "tipo_sangre":      (0.47, 0.16, 0.62, 0.23),
    "sexo":             (0.68, 0.16, 0.80, 0.23),
}


zones_digital = {
    "numero_documento": (0.65, 0.08, 0.95, 0.13),
    "nombre_completo":  (0.15, 0.25, 0.85, 0.32),
    "fecha_nacimiento": (0.15, 0.35, 0.40, 0.40),
    "sexo":             (0.45, 0.35, 0.55, 0.40),
    "lugar_nacimiento": (0.15, 0.45, 0.55, 0.50),
    "nacionalidad":     (0.60, 0.45, 0.85, 0.50),
    "tipo_sangre":      (0.75, 0.55, 0.95, 0.60),
}


# ============================================================
# ETIQUETAS QUE EL SISTEMA BUSCARÁ
# ============================================================

LABELS = {

    "numero_documento": [
        "NUMERO DE DOCUMENTO",
        "NÚMERO DE DOCUMENTO",
        "NUMERO",
        "NÚMERO",
        "NUM",
        "NUM.",
        "IDENTIFICACION",
        "IDENTIFICACIÓN",
    ],

    "apellidos": [
        "APELLIDOS",
        "APELLIDO",
        "APEL",
    ],

    "nombres": [
        "NOMBRES",
        "NOMBRE",
        "NOMBRES:",
        "NOMBRE:",
        "NOMB",
        "NOM",
    ],

    "fecha_nacimiento": [
        "FECHA DE NACIMIENTO",
        "FECHA DE NACIMIENTO:",
        "FECHA NACIMIENTO",
        "NACIMIENTO",
    ],

    "lugar_nacimiento": [
        "LUGAR DE NACIMIENTO",
        "LUGAR DE NACIMIETO",
        "LUGAR NACIMIENTO",
    ],

    "nacionalidad": [
        "NACIONALIDAD",
        "NACIONAL",
    ],

    "tipo_sangre": [
        "G.S. RH",
        "G. S. RH",
        "GS RH",
        "G S RH",
        "TIPO DE SANGRE",
        "GRUPO SANGUINEO",
        "GRUPO SANGUÍNEO",
        "RH",
    ],

    "sexo": [
        "SEXO",
    ],

    "estatura": [
        "ESTATURA",
    ],

    "fecha_expedicion": [
        "FECHA Y LUGAR DE EXPEDICION",
        "FECHA Y LUGAR DE EXPEDICIÓN",
        "FECHA EXPEDICION",
        "FECHA EXPEDICIÓN",
    ],
}

# ============================================================
# NORMALIZACIÓN DE TEXTO
# ============================================================

def normalizar_ocr_texto(texto):
    """
    Normaliza texto OCR para comparar etiquetas.

    Ejemplo:

        "NACIMIÉNTO" -> "NACIMIENTO"
        "NÚMERO"     -> "NUMERO"
    """

    if not texto:
        return ""

    texto = str(texto).upper().strip()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        c
        for c in texto
        if unicodedata.category(c) != "Mn"
    )

    texto = re.sub(
        r"[^A-Z0-9+#.\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


# ============================================================
# SIMILITUD DE ETIQUETAS
# ============================================================

def similitud_texto(a, b):

    a = normalizar_ocr_texto(a)
    b = normalizar_ocr_texto(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.95

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()

# ============================================================
# VALIDACIÓN DE RESULTADOS OCR
# ============================================================

def validar_campo_ocr(
    field,
    value
):
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
            re.fullmatch(
                r"\d{6,12}",
                value
            )
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

        # Debe contener únicamente letras, espacios,
        # guiones o apóstrofes.
        if not re.fullmatch(
            r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’-]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’-]+)*",
            value
        ):
            return False

        # Evitar resultados OCR absurdamente cortos.
        if len(
            value.replace(" ", "")
        ) < 3:
            return False

        return True 
    
    # --------------------------------------------------------
    # LUGAR / NACIONALIDAD
    # --------------------------------------------------------

    if field == "lugar_nacimiento":

        valor = normalizar(value)

        # ------------------------------------------------------
        # DESCARTAR VALORES DEMASIADO CORTOS
        # ------------------------------------------------------

        if len(value) < 3:
            return False

        # ------------------------------------------------------
        # PALABRAS QUE PERTENECEN A OTROS CAMPOS
        # ------------------------------------------------------
        
        palabras_invalidas = (
            "FECHA",
            "NACIMIENTO",
            "LUGAR",
            "ESTATURA",
            "SEXO",
            "RH",
            "REGISTRADOR",
            "INDICE",
            "DERECHO",
            "NACIONAL",
            "EXPEDICION",
            "IDENTIFICACION",
            "APELLIDOS",
            "NOMBRES",
            "FIRMA"
        )

        # ------------------------------------------------------
        # RECHAZAR SI EL VALOR CONTIENE INFORMACIÓN DE OTRO CAMPO
        # ------------------------------------------------------

        if any(
            palabra in valor
            for palabra in palabras_invalidas
        ):
            logger.warning(
                "OCR_VALOR_RECHAZADO | "
                "campo=lugar_nacimiento | "
                "motivo=CONTAMINACION_OTRO_CAMPO | "
                "valor=%r",
                value
            )
            return False

        # ------------------------------------------------------
        # DEBE CONTENER LETRAS
        # ------------------------------------------------------
        
        if not re.search(r"[A-ZÁÉÍÓÚ´Ñ]", valor):
            return False
        
        return True

    # ==========================================================
    # NACIONALIDAD
    # ==========================================================

    if field == "nacionaldad":
        valor = normalizar(value)

        # ------------------------------------------------------
        # DESCARTAR VALORES DEMASIADO CORTOS
        # ------------------------------------------------------
        
        if len(valor) < 4:
            return False

        # ------------------------------------------------------
        # NUNCA ACEPTAR TEXTO DEL REGISTRADOR
        # ------------------------------------------------------
        
        patrones_invalidos = (
            "REGISTRADOR",
            "REGISTRADOR NACIONAL",
            "INDICE DERECHO",
            "INDICE",
            "DERECHO",
            "REGISTRADOR NACIONAL DEL ESTADO CIVIL"
        )

        if any(
            patron in valor
            for patron in patrones_invalidos
        ):
            logger.warning(
                "OCR_NACIONALIDAD_RECHAZADA | "
                "motivo=REGISTRADOR_O_INSTITUCION | "
                "valor=r%"
                value
            )
            return False

        # ------------------------------------------------------
        # DESCARTAR PALABRAS QUE PERTENECEN A OTROS CAMPOS
        # ------------------------------------------------------

        palabras_invalidas = (
            "FECHA",
            "NACIMIENTO",
            "LUGAR",
            "ESTATURA",
            "SEXO",
            "EXPEDICION",
            "IDENTIFICACION",
            "APELLIDOS",
            "NOMBRES",
            "FIRMA"
        )

        if any(
            palabra in valor
            for palabra in palabras_invalidas
        ):
            logger.warning(
                "OCR_NACIONALIDAD_RECHAZADA | "
                "motivo=CONTAMINACION_OTRO_CAMPO | "
                "valor=%r",
                value
            )
            return False

        # ------------------------------------------------------
        # NACIONALIDADES VÁLIDAS
        # ------------------------------------------------------   

        nacionalidades_validas = (
            "COLOMBIANO",
            "COLOMBIANA",
            "VENEZOLANO",
            "VENEZOLANA",
            "ECUATORIANO",
            "ECUATORIANA",
            "PERUANO",
            "PERUANA",
            "BOLIVIANO",
            "BOLIVIANA",
            "CHILENO",
            "CHILENA",
            "ARGENTINO",
            "ARGENTINA",
            "BRASILEÑO",
            "BRASILEÑA",
            "PARAGUAYO",
            "PARAGUAYA",
            "URUGUAYO",
            "URUGUAYA",
            "MEXICANO",
            "MEXICANA",
            "PANAMEÑO",
            "PANAMEÑA",
            "COSTARRICENSE",
            "NICARAGUENSE",
            "HONDUREÑO",
            "HONDUREÑA",
            "SALVADOREÑO",
            "SALVADOREÑA",
            "GUATEMALTECO",
            "GUATEMALTECA",
            "ESTADOUNIDENSE",
            "CANADIENSE",
            "ESPAÑOL",
            "ESPAÑOLA",
            "FRANCES",
            "FRANCESA",
            "ITALIANO",
            "ITALIANA",
            "ALEMAN",
            "ALEMANA",
            "INGLES",
            "INGLESA",
            "PORTUGUES",
            "PORTUGUESA",
        )

        # ------------------------------------------------------
        # LA NACIONALIDAD DEBE CORRESPONDER A UNA NACIONALIDAD
        # CONOCIDA
        # ------------------------------------------------------

        if not any(
            nacionalidad in valor
            for nacionalidad in nacionalidades_validas
        ):
            logger.warning(
                "OCR_NACIONALIDAD_RECHAZADA | "
                "motivo=NACIONALIDAD_NO_RECONOCIDA | "
                "valor=%r",
                value
            )
            return False

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

        crop.save(
            path
        )

        url = _obtener_url_crop(
            path
        )

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

def preprocesa_image(
    pil_img
):

    try:

        img = np.array(
            pil_img.convert("L")
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        img = clahe.apply(
            img
        )

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

        return Image.fromarray(
            thresh
        )

    except Exception as e:

        logger.warning(
            "Preprocesamiento falló: %s",
            e
        )

        return pil_img


# ============================================================
# OCR POR CAMPO
# ============================================================

def _ocr_field(
    crop,
    field
):
    """
    Ejecuta varias configuraciones de Tesseract y selecciona
    el mejor resultado válido para el campo.

    Antes:
        devolvía siempre resultados[0].

    Ahora:
        evalúa todos los resultados y prioriza el que pueda
        limpiarse y validarse correctamente.
    """

    configs = {

        "numero_documento": [

            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=0123456789",

            "--oem 3 --psm 8 "
            "-c tessedit_char_whitelist=0123456789",

            "--oem 3 --psm 6 "
            "-c tessedit_char_whitelist=0123456789",
        ],

        "fecha_nacimiento": [

            "--oem 3 --psm 7",

            "--oem 3 --psm 8",

            "--oem 3 --psm 6",
        ],

        "sexo": [

            "--oem 3 --psm 10 "
            "-c tessedit_char_whitelist=MF",

            "--oem 3 --psm 8 "
            "-c tessedit_char_whitelist=MF",

            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=MF",
        ],

        "tipo_sangre": [

            "--oem 3 --psm 10 "
            "-c tessedit_char_whitelist=ABO+-",

            "--oem 3 --psm 8 "
            "-c tessedit_char_whitelist=ABO+-",

            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=ABO+-",
        ],
    }

    field_configs = configs.get(
        field,
        [
            "--oem 3 --psm 7",
            "--oem 3 --psm 8",
            "--oem 3 --psm 6",
        ]
    )

    candidatos = []

    for indice, config in enumerate(field_configs):

        try:

            raw = pytesseract.image_to_string(
                crop,
                lang="spa",
                config=config
            ).strip()

            if not raw:
                continue

            clean = _limpiar_campo(
                field,
                raw
            )

            logger.info(
                "OCR_CONFIG_RESULTADO | "
                "campo=%s | config=%s | raw=%r | clean=%r",
                field,
                config,
                raw,
                clean
            )

            if clean:

                candidatos.append({
                    "raw": raw,
                    "clean": clean,
                    "config": config,
                    "indice": indice,
                })

        except Exception as e:

            logger.warning(
                "OCR_ERROR | "
                "campo=%s | config=%r | error=%s",
                field,
                config,
                e
            )

    if not candidatos:

        return None

    # ========================================================
    # PRIORIZAR RESULTADOS REPETIDOS
    # ========================================================

    agrupados = {}

    for candidato in candidatos:

        clave = normalizar_ocr_texto(
            candidato["clean"]
        )

        if clave not in agrupados:

            agrupados[clave] = []

        agrupados[clave].append(
            candidato
        )

    # Si dos configuraciones producen el mismo resultado,
    # consideramos ese resultado más confiable.

    mejor_grupo = max(
        agrupados.values(),
        key=lambda grupo: (
            len(grupo),
            len(str(grupo[0]["clean"]))
        )
    )

    mejor = mejor_grupo[0]

    logger.info(
        "OCR_CONFIG_ELEGIDA | "
        "campo=%s | raw=%r | clean=%r | "
        "config=%r | coincidencias=%s",
        field,
        mejor["raw"],
        mejor["clean"],
        mejor["config"],
        len(mejor_grupo)
    )

    return mejor["clean"]


# ============================================================
# LIMPIAR CAMPO
# ============================================================

def _limpiar_campo(
    field,
    raw
):

    if not raw:
        return None

    if field == "numero_documento":

        value = limpiar_numero(
            raw
        )

    elif field == "fecha_nacimiento":

        value = limpiar_fecha(
            raw
        )

    elif field == "sexo":

        value = limpiar_sexo(
            raw
        )

    elif field == "tipo_sangre":

        value = limpiar_rh(
            raw
        )

    elif field in (
        "apellidos",
        "nombres",
        "nombre_completo",
        "lugar_nacimiento",
        "nacionalidad",
    ):

        value = limpiar_texto(
            raw
        )

    else:

        value = (
            str(raw).strip()
            if raw
            else None
        )

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
# OCR GENERAL CON COORDENADAS DE PALABRAS
# ============================================================

def obtener_datos_ocr_pagina(
    imagen
):

    """
    Lee toda la página y devuelve las palabras
    junto con sus coordenadas.

    Esto es independiente de las zonas de coordenadas.

    Es la segunda capa del sistema:
        COORDENADAS + ETIQUETAS
    """

    datos = []

    configuraciones = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
    ]

    for config in configuraciones:

        try:

            data = pytesseract.image_to_data(
                imagen,
                lang="spa",
                config=config,
                output_type=pytesseract.Output.DICT
            )

            cantidad = len(
                data.get(
                    "text",
                    []
                )
            )

            for i in range(
                cantidad
            ):

                texto = (
                    data["text"][i]
                    or ""
                ).strip()

                if not texto:
                    continue

                try:

                    conf = float(
                        data["conf"][i]
                    )

                except Exception:

                    conf = 0

                if conf < 15:
                    continue

                x = int(
                    data["left"][i]
                )

                y = int(
                    data["top"][i]
                )

                w = int(
                    data["width"][i]
                )

                h = int(
                    data["height"][i]
                )

                datos.append({

                    "text": texto,

                    "norm": normalizar_ocr_texto(
                        texto
                    ),

                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,

                    "cx": x + (w / 2),
                    "cy": y + (h / 2),

                    "conf": conf,

                    "block": data["block_num"][i],

                    "par": data["par_num"][i],

                    "line": data["line_num"][i],
                })

        except Exception as e:

            logger.warning(
                "OCR_DATA_ERROR | "
                "config=%s | error=%s",
                config,
                e
            )

    # ========================================================
    # ELIMINAR DUPLICADOS
    # ========================================================

    unicos = []

    vistos = set()

    for item in datos:

        clave = (
            item["norm"],
            round(item["x"] / 10),
            round(item["y"] / 10),
        )

        if clave in vistos:
            continue

        vistos.add(
            clave
        )

        unicos.append(
            item
        )

    return unicos


# ============================================================
# AGRUPAR PALABRAS EN LÍNEAS
# ============================================================

def agrupar_lineas(
    datos
):

    """
    Agrupa las palabras por proximidad vertical.

    No depende exclusivamente de line_num de Tesseract,
    porque OCR puede equivocarse con ese identificador.
    """

    if not datos:
        return []

    datos = sorted(
        datos,
        key=lambda d: (
            d["cy"],
            d["x"]
        )
    )

    lineas = []

    tolerancia_base = 0.6

    for palabra in datos:

        agregada = False

        for linea in lineas:

            promedio_y = linea["cy"]

            altura = max(
                linea["altura"],
                palabra["h"]
            )

            tolerancia = max(
                12,
                altura * tolerancia_base
            )

            if abs(
                palabra["cy"] - promedio_y
            ) <= tolerancia:

                linea["items"].append(
                    palabra
                )

                ys = [
                    x["cy"]
                    for x in linea["items"]
                ]

                linea["cy"] = (
                    sum(ys)
                    /
                    len(ys)
                )

                linea["altura"] = max(
                    x["h"]
                    for x in linea["items"]
                )

                agregada = True

                break

        if not agregada:

            lineas.append({

                "cy": palabra["cy"],

                "altura": palabra["h"],

                "items": [
                    palabra
                ],
            })

    resultado = []

    for linea in lineas:

        items = sorted(
            linea["items"],
            key=lambda d: d["x"]
        )

        texto = " ".join(
            item["text"]
            for item in items
        )

        x1 = min(
            item["x"]
            for item in items
        )

        y1 = min(
            item["y"]
            for item in items
        )

        x2 = max(
            item["x"] + item["w"]
            for item in items
        )

        y2 = max(
            item["y"] + item["h"]
            for item in items
        )

        resultado.append({

            "text": texto,

            "norm": normalizar_ocr_texto(
                texto
            ),

            "items": items,

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,

            "cx": (
                x1 + x2
            ) / 2,

            "cy": (
                y1 + y2
            ) / 2,
        })

    return sorted(
        resultado,
        key=lambda l: l["cy"]
    )


# ============================================================
# BUSCAR ETIQUETA
# ============================================================

def encontrar_etiqueta(
    lineas,
    field
):
    """
    Busca la etiqueta del campo dentro de las líneas OCR.

    Prioridad:

        1. coincidencia exacta
        2. etiqueta contenida dentro de la línea
        3. coincidencia aproximada de la línea
        4. coincidencia aproximada por palabra

    Las etiquetas genéricas no deben tener la misma fuerza
    que una etiqueta específica.
    """

    etiquetas = LABELS.get(
        field,
        []
    )

    mejor = None

    for linea_index, linea in enumerate(lineas):

        texto = linea["norm"]

        if not texto:
            continue

        for etiqueta in etiquetas:

            etiqueta_norm = normalizar_ocr_texto(
                etiqueta
            )

            if not etiqueta_norm:
                continue

            score = 0.0

            # ====================================================
            # 1. COINCIDENCIA EXACTA
            # ====================================================

            if texto == etiqueta_norm:

                score = 1.00

            # ====================================================
            # 2. ETIQUETA DENTRO DE LA LÍNEA
            # ====================================================

            elif etiqueta_norm in texto:

                score = 0.96

            # ====================================================
            # 3. LÍNEA COMPLETA SIMILAR
            # ====================================================

            else:

                similitud = similitud_texto(
                    texto,
                    etiqueta_norm
                )

                if similitud >= 0.82:

                    score = similitud

            # ====================================================
            # 4. PALABRA INDIVIDUAL
            # ====================================================

            palabras_linea = texto.split()

            for palabra in palabras_linea:

                if len(palabra) < 4:
                    continue

                similitud = similitud_texto(
                    palabra,
                    etiqueta_norm
                )

                # Mucho más estricto que antes.
                if similitud >= 0.88:

                    score = max(
                        score,
                        similitud * 0.90
                    )

            # ====================================================
            # ACEPTAR CANDIDATO
            # ====================================================

            if score < 0.82:
                continue

            candidato = {

                "field": field,

                "line_index": linea_index,

                "line": linea,

                "score": score,

                "etiqueta": etiqueta,
            }

            if (
                mejor is None
                or score > mejor["score"]
            ):

                mejor = candidato

    if mejor:

        logger.info(
            "OCR_ETIQUETA_ENCONTRADA | "
            "campo=%s | etiqueta=%s | "
            "score=%.3f | texto_linea=%r",
            field,
            mejor["etiqueta"],
            mejor["score"],
            mejor["line"]["text"]
        )

    else:

        logger.info(
            "OCR_ETIQUETA_NO_ENCONTRADA | "
            "campo=%s",
            field
        )

    return mejor

# ============================================================
# TEXTO DE LÍNEA
# ============================================================

def texto_de_linea(
    linea
):

    if not linea:
        return ""

    return " ".join(
        item["text"]
        for item in sorted(
            linea["items"],
            key=lambda x: x["x"]
        )
    ).strip()


# ============================================================
# CAPTURAR LÍNEA SUPERIOR
# ============================================================

def obtener_linea_superior(
    lineas,
    indice,
    cantidad=1
):

    resultado = []

    inicio = max(
        0,
        indice - cantidad
    )

    for i in range(
        inicio,
        indice
    ):

        resultado.append(
            lineas[i]
        )

    return resultado


# ============================================================
# CAPTURAR LÍNEA INFERIOR
# ============================================================

def obtener_linea_inferior(
    lineas,
    indice,
    cantidad=1
):

    resultado = []

    fin = min(
        len(lineas),
        indice + cantidad + 1
    )

    for i in range(
        indice + 1,
        fin
    ):

        resultado.append(
            lineas[i]
        )

    return resultado


# ============================================================
# EXTRAER TEXTO ALREDEDOR DE ETIQUETA
# ============================================================

def extraer_por_etiqueta(
    lineas,
    field
):

    etiqueta = encontrar_etiqueta(
        lineas,
        field
    )

    if not etiqueta:
        return None

    indice = etiqueta[
        "line_index"
    ]

    linea = etiqueta[
        "line"
    ]

    texto_linea = texto_de_linea(
        linea
    )

    etiqueta_norm = normalizar_ocr_texto(
        etiqueta["etiqueta"]
    )

    texto_norm = normalizar_ocr_texto(
        texto_linea
    )

    # ========================================================
    # NUMERO
    # ========================================================

    if field == "numero_documento":

        numeros = re.findall(
            r"\d{6,12}",
            texto_linea
        )

        if numeros:
            return numeros[0]

        for siguiente in obtener_linea_inferior(
            lineas,
            indice,
            2
        ):

            texto = texto_de_linea(
                siguiente
            )

            numeros = re.findall(
                r"\d{6,12}",
                texto
            )

            if numeros:
                return numeros[0]

        return None

    # ========================================================
    # FECHA
    # ========================================================

    if field == "fecha_nacimiento":

        fecha = limpiar_fecha(
            texto_linea
        )

        if fecha:
            return fecha

        for siguiente in obtener_linea_inferior(
            lineas,
            indice,
            2
        ):

            fecha = limpiar_fecha(
                texto_de_linea(
                    siguiente
                )
            )

            if fecha:
                return fecha

        return None

    # ========================================================
    # SEXO
    # ========================================================

    if field == "sexo":

        candidatos = []

        for item in linea["items"]:

            texto = normalizar_ocr_texto(
                item["text"]
            )

            if texto in (
                "M",
                "F"
            ):

                candidatos.append(
                    texto
                )

        if candidatos:

            return limpiar_sexo(
                candidatos[-1]
            )

        for siguiente in obtener_linea_inferior(
            lineas,
            indice,
            2
        ):

            for item in siguiente["items"]:

                texto = normalizar_ocr_texto(
                    item["text"]
                )

                if texto in (
                    "M",
                    "F"
                ):

                    return limpiar_sexo(
                        texto
                    )

        return None

    # ========================================================
    # RH
    # ========================================================

    if field == "tipo_sangre":

        rh = limpiar_rh(
            texto_linea
        )

        if rh:
            return rh

        for siguiente in obtener_linea_inferior(
            lineas,
            indice,
            2
        ):

            rh = limpiar_rh(
                texto_de_linea(
                    siguiente
                )
            )

            if rh:
                return rh

        return None

    # ========================================================
    # APELLIDOS
    # ========================================================

    if field == "apellidos":

        superiores = obtener_linea_superior(
            lineas,
            indice,
            2
        )

        for superior in reversed(
            superiores
        ):

            texto = texto_de_linea(
                superior
            )

            if not texto:
                continue

            if encontrar_etiqueta_en_texto(
                texto,
                LABELS["apellidos"]
            ):

                continue

            if encontrar_cualquier_etiqueta(
                texto
            ):

                continue

            value = limpiar_texto(
                texto
            )

            if validar_campo_ocr(
                field,
                value
            ):

                return value

        return None

    # ========================================================
    # NOMBRES
    # ========================================================

    if field == "nombres":

        superiores = obtener_linea_superior(
            lineas,
            indice,
            2
        )

        for superior in reversed(
            superiores
        ):

            texto = texto_de_linea(
                superior
            )

            if not texto:
                continue

            if encontrar_cualquier_etiqueta(
                texto
            ):

                continue

            value = limpiar_texto(
                texto
            )

            if validar_campo_ocr(
                field,
                value
            ):

                return value

        return None

    # ========================================================
    # LUGAR DE NACIMIENTO
    # ========================================================

    if field == "lugar_nacimiento":

        superiores = obtener_linea_superior(
            lineas,
            indice,
            3
        )

        candidatos = []

        for superior in reversed(
            superiores
        ):

            texto = texto_de_linea(
                superior
            )

            if not texto:
                continue

            if encontrar_cualquier_etiqueta(
                texto
            ):

                continue

            if limpiar_fecha(
                texto
            ):

                continue

            candidatos.append(
                texto
            )

            if len(candidatos) >= 2:
                break

        if candidatos:

            candidatos.reverse()

            value = limpiar_texto(
                " ".join(
                    candidatos
                )
            )

            if validar_campo_ocr(
                field,
                value
            ):

                return value

        return None

    # ========================================================
    # NACIONALIDAD
    # ========================================================

    if field == "nacionalidad":

        texto = texto_linea

        texto_limpio = re.sub(
            re.escape(etiqueta_norm),
            "",
            normalizar_ocr_texto(
                texto
            ),
            flags=re.IGNORECASE
        ).strip()

        if len(texto_limpio) >= 3:

            value = limpiar_texto(
                texto_limpio
            )

            if validar_campo_ocr(
                field,
                value
            ):

                return value

        for siguiente in obtener_linea_inferior(
            lineas,
            indice,
            2
        ):

            texto = texto_de_linea(
                siguiente
            )

            if not texto:
                continue

            if encontrar_cualquier_etiqueta(
                texto
            ):

                continue

            value = limpiar_texto(
                texto
            )

            if validar_campo_ocr(
                field,
                value
            ):

                return value

        return None

    return None


# ============================================================
# AYUDAS PARA ETIQUETAS
# ============================================================

def encontrar_etiqueta_en_texto(
    texto,
    etiquetas
):

    texto_norm = normalizar_ocr_texto(
        texto
    )

    for etiqueta in etiquetas:

        etiqueta_norm = normalizar_ocr_texto(
            etiqueta
        )

        if etiqueta_norm in texto_norm:
            return True

        if (
            similitud_texto(
                texto_norm,
                etiqueta_norm
            ) >= 0.75
        ):

            return True

    return False


def encontrar_cualquier_etiqueta(
    texto
):

    for etiquetas in LABELS.values():

        if encontrar_etiqueta_en_texto(
            texto,
            etiquetas
        ):

            return True

    return False


# ============================================================
# DETECCIÓN DE LADO DE CÉDULA
# ============================================================

def detectar_lado_cedula(
    lineas
):

    """
    Determina si la página corresponde a:

        ANVERSO
        REVERSO

    No depende del número de página.
    """

    texto_total = " ".join(
        linea["norm"]
        for linea in lineas
    )

    texto_total = normalizar_ocr_texto(
        texto_total
    )

    score_anverso = 0
    score_reverso = 0

    # --------------------------------------------------------
    # ANVERSO
    # --------------------------------------------------------

    etiquetas_anverso = [
        "NUMERO",
        "NÚMERO",
        "APELLIDOS",
        "NOMBRES",
    ]

    for etiqueta in etiquetas_anverso:

        etiqueta_norm = normalizar_ocr_texto(
            etiqueta
        )

        if etiqueta_norm in texto_total:

            score_anverso += 2

        else:

            for palabra in texto_total.split():

                if similitud_texto(
                    palabra,
                    etiqueta_norm
                ) >= 0.70:

                    score_anverso += 1

                    break

    # --------------------------------------------------------
    # REVERSO
    # --------------------------------------------------------

    etiquetas_reverso = [
        "FECHA DE NACIMIENTO",
        "LUGAR DE NACIMIENTO",
        "NACIONALIDAD",
        "ESTATURA",
        "SEXO",
        "RH",
        "G.S. RH",
        "FECHA Y LUGAR DE EXPEDICION",
    ]

    for etiqueta in etiquetas_reverso:

        etiqueta_norm = normalizar_ocr_texto(
            etiqueta
        )

        if etiqueta_norm in texto_total:

            score_reverso += 2

        else:

            for palabra in texto_total.split():

                if similitud_texto(
                    palabra,
                    etiqueta_norm
                ) >= 0.70:

                    score_reverso += 1

                    break

    if score_anverso > score_reverso:

        lado = "anverso"

    elif score_reverso > score_anverso:

        lado = "reverso"

    else:

        lado = "desconocido"

    logger.info(
        "OCR_LADO_DETECTADO | "
        "lado=%s | "
        "score_anverso=%s | "
        "score_reverso=%s",
        lado,
        score_anverso,
        score_reverso
    )

    return lado


# ============================================================
# OCR POR COORDENADAS
# ============================================================

def procesar_por_coordenadas(
    img,
    page_number,
    zones,
    debug_dir
):

    resultados = {}

    width, height = img.size

    logger.info(
        "OCR_COORDENADAS_INICIO | "
        "pagina=%s | tamaño=%sx%s",
        page_number,
        width,
        height
    )

    for field, coords in zones.items():

        x1, y1, x2, y2 = coords

        x1, x2 = sorted(
            (x1, x2)
        )

        y1, y2 = sorted(
            (y1, y2)
        )

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

            resultados[field] = None

            continue

        try:

            crop = img.crop(box)

            guardar_crop_debug(
                crop,
                f"{field}_coordenadas_original",
                page_number,
                debug_dir
            )

            crop = crop.resize(
                (
                    crop.width * 2,
                    crop.height * 2
                ),
                Image.Resampling.LANCZOS
            )

            guardar_crop_debug(
                crop,
                f"{field}_coordenadas_ampliado",
                page_number,
                debug_dir
            )

            crop_ocr = preprocesa_image(
                crop
            )

            guardar_crop_debug(
                crop_ocr,
                f"{field}_coordenadas_ocr",
                page_number,
                debug_dir
            )

            raw = _ocr_field(
                crop_ocr,
                field
            )

            clean = _limpiar_campo(
                field,
                raw
            )

            logger.info(
                "OCR_RESULTADO_COORDENADAS | "
                "pagina=%s | campo=%s | "
                "raw=%r | clean=%r",
                page_number,
                field,
                raw,
                clean
            )

            resultados[field] = clean

        except Exception as e:

            logger.exception(
                "OCR_COORDENADAS_ERROR | "
                "pagina=%s | campo=%s | error=%s",
                page_number,
                field,
                e
            )

            resultados[field] = None

    return resultados


# ============================================================
# OCR POR ETIQUETAS
# ============================================================

def procesar_por_etiquetas(
    img,
    page_number,
    debug_dir,
    lado
):

    resultados = {}

    datos = obtener_datos_ocr_pagina(
        img
    )

    logger.info(
        "OCR_ETIQUETAS_DATA | "
        "pagina=%s | palabras=%s",
        page_number,
        len(datos)
    )

    lineas = agrupar_lineas(
        datos
    )

    logger.info(
        "OCR_ETIQUETAS_LINEAS | "
        "pagina=%s | lineas=%s",
        page_number,
        len(lineas)
    )

    for i, linea in enumerate(
        lineas
    ):

        logger.info(
            "OCR_LINEA | "
            "pagina=%s | linea=%s | "
            "texto=%r | box=(%s,%s,%s,%s)",
            page_number,
            i,
            linea["text"],
            linea["x1"],
            linea["y1"],
            linea["x2"],
            linea["y2"]
        )

    if lado == "anverso":

        campos = [
            "numero_documento",
            "apellidos",
            "nombres",
        ]

    elif lado == "reverso":

        campos = [
            "fecha_nacimiento",
            "lugar_nacimiento",
            "nacionalidad",
            "tipo_sangre",
            "sexo",
        ]

    else:

        campos = [
            "numero_documento",
            "apellidos",
            "nombres",
            "fecha_nacimiento",
            "lugar_nacimiento",
            "nacionalidad",
            "tipo_sangre",
            "sexo",
        ]

    for field in campos:

        try:

            valor = extraer_por_etiqueta(
                lineas,
                field
            )

            if valor:

                valor_limpio = _limpiar_campo(
                    field,
                    valor
                )

            else:

                valor_limpio = None

            logger.info(
                "OCR_RESULTADO_ETIQUETA | "
                "pagina=%s | campo=%s | "
                "valor=%r | limpio=%r",
                page_number,
                field,
                valor,
                valor_limpio
            )

            resultados[field] = (
                valor_limpio
            )

        except Exception as e:

            logger.exception(
                "OCR_ETIQUETA_ERROR | "
                "pagina=%s | campo=%s | error=%s",
                page_number,
                field,
                e
            )

            resultados[field] = None

    return resultados


# ============================================================
# PUNTAJE DE CONFIANZA
# ============================================================

def puntuar_resultado(
    field,
    value,
    origen
):

    if not value:
        return 0

    if not validar_campo_ocr(
        field,
        value
    ):
        return 0

    score = 50

    if origen == "etiqueta":
        score += 10

    if field == "numero_documento":

        if re.fullmatch(
            r"\d{8,10}",
            str(value)
        ):

            score += 25

    elif field in (
        "apellidos",
        "nombres",
    ):

        palabras = str(
            value
        ).split()

        if len(palabras) >= 2:

            score += 20

    elif field == "tipo_sangre":

        score += 30

    elif field == "sexo":

        score += 30

    elif field == "fecha_nacimiento":

        score += 30

    return score


# ============================================================
# COMPARAR OCR GENERAL VS OCR POR CROP
# ============================================================

def comparar_resultados_ocr(
    general,
    crop,
    field,
    lado
):
    """
    Compara el resultado obtenido por OCR general
    contra el resultado obtenido mediante CROP.

    REGLA:

    1. General válido + CROP válido:
       - Si coinciden -> resultado confirmado.
       - Si difieren -> se conserva GENERAL.

    2. General válido + CROP inválido:
       - Se conserva GENERAL.

    3. General inválido + CROP válido:
       - Se utiliza CROP.

    4. Ninguno válido:
       - Resultado None.
    """

    general_valido = validar_campo_ocr(
        field,
        general
    )

    crop_valido = validar_campo_ocr(
        field,
        crop
    )

    # ========================================================
    # AMBOS VÁLIDOS
    # ========================================================

    if general_valido and crop_valido:

        general_norm = normalizar_ocr_texto(
            general
        )

        crop_norm = normalizar_ocr_texto(
            crop
        )

        coinciden = (
            general_norm == crop_norm
        )

        # ========================================================
        # COINCIDEN
        # ========================================================

        if coinciden:

            logger.info(
                "OCR_COMPARACION_OK | "
                "lado=%s | campo=%s | "
                "general=%r | crop=%r | "
                "resultado=%r",
                lado,
                field,
                general,
                crop,
                crop
            )

            return {
                "value": crop,
                "origen": "OCR_COP_CONFIRMADO",
                "general_valido": True,
                "crop_valido": True,
                "coinciden": True,
            }

        # ========================================================
        # DIFIEREN
        # ========================================================


        logger.warning(
            "OCR_COMPARACION_CONFLICTO | "
            "lado=%s | campo=%s | "
            "general=%r | crop=%r | "
            "SE_CONSERVA_GENERAL=%r",
            lado,
            field,
            general,
            crop,
            general
        )

        return {
            "value": general,
            "origen": "OCR_GENERAL",
            "general_valido": True,
            "crop_valido": True,
            "coinciden": coinciden,
        }

    # ========================================================
    # SOLO GENERAL VÁLIDO
    # ========================================================

    if general_valido:

        logger.info(
            "OCR_GENERAL_VALIDO | "
            "lado=%s | campo=%s | "
            "general=%r | crop=%r",
            lado,
            field,
            general,
            crop
        )

        return {
            "value": general,
            "origen": "OCR_GENERAL",
            "general_valido": True,
            "crop_valido": False,
            "coinciden": False,
        }

    # ========================================================
    # SOLO CROP VÁLIDO
    # ========================================================

    if crop_valido and not general_valido:

        logger.warning(
            "OCR_CROP_RECUPERACION | "
            "lado=%s | campo=%s | "
            "general=%r | crop=%r",
            lado,
            field,
            general,
            crop
        )

        return {
            "value": crop,
            "origen": "OCR_CROP",
            "general_valido": False,
            "crop_valido": True,
            "coinciden": False,
        }

    # ========================================================
    # NINGUNO VÁLIDO
    # ========================================================

    logger.warning(
        "OCR_SIN_RESULTADO_VALIDO | "
        "lado=%s | campo=%s | "
        "general=%r | crop=%r",
        lado,
        field,
        general,
        crop
    )

    return {
        "value": None,
        "origen": "NINGUNO",
        "general_valido": False,
        "crop_valido": False,
        "coinciden": False,
    }


# ============================================================
# ELEGIR ENTRE COORDENADAS Y ETIQUETAS
# ============================================================

def combinar_resultados(
    coordenadas,
    etiquetas,
    lado
):

    resultado = {}

    campos = set(
        list(coordenadas.keys())
        +
        list(etiquetas.keys())
    )

    for field in campos:

        valor_coord = coordenadas.get(
            field
        )

        valor_etiqueta = etiquetas.get(
            field
        )

        score_coord = puntuar_resultado(
            field,
            valor_coord,
            "coordenadas"
        )

        score_etiqueta = puntuar_resultado(
            field,
            valor_etiqueta,
            "etiqueta"
        )

        if (
            score_etiqueta
            >
            score_coord
        ):

            elegido = valor_etiqueta
            origen = "ETIQUETA"

        else:

            elegido = valor_coord
            origen = "COORDENADAS"

        if (
            not valor_coord
            and valor_etiqueta
        ):

            elegido = valor_etiqueta
            origen = "ETIQUETA"

        elif (
            not valor_etiqueta
            and valor_coord
        ):

            elegido = valor_coord
            origen = "COORDENADAS"

        logger.info(
            "OCR_COMBINACION | "
            "lado=%s | campo=%s | "
            "coordenadas=%r | score_coord=%s | "
            "etiqueta=%r | score_etiqueta=%s | "
            "ELEGIDO=%r | origen=%s",
            lado,
            field,
            valor_coord,
            score_coord,
            valor_etiqueta,
            score_etiqueta,
            elegido,
            origen
        )

        resultado[field] = elegido

    # ========================================================
    # NOMBRE COMPLETO
    # ========================================================

    apellidos = resultado.get(
        "apellidos"
    )

    nombres = resultado.get(
        "nombres"
    )

    if apellidos and nombres:

        resultado[
            "nombre_completo"
        ] = (
            f"{nombres} {apellidos}"
        )

        logger.info(
            "OCR_NOMBRE_COMPLETO | "
            "nombres=%r | apellidos=%r | "
            "nombre_completo=%r",
            nombres,
            apellidos,
            resultado[
                "nombre_completo"
            ]
        )

    return resultado


# ============================================================
# PDF DIGITAL
# ============================================================

def extract_fields_from_text(
    text
):

    results = {}

    if not text:
        return results

    # --------------------------------------------------------
    # DOCUMENTO
    # --------------------------------------------------------

    match = re.search(
        r"\b(\d{6,15})\b",
        text
    )

    if match:

        results[
            "numero_documento"
        ] = match.group(1)

    # --------------------------------------------------------
    # NOMBRE
    # --------------------------------------------------------

    match = re.search(
        r"(?:NOMBRE|NOMBRES?)\s*:?\s*"
        r"([A-ZÁÉÍÓÚÑ\s]+)",
        text,
        re.IGNORECASE,
    )

    if match:

        results[
            "nombre_completo"
        ] = limpiar_texto(
            match.group(1).strip()
        )

    else:

        results[
            "nombre_completo"
        ] = None

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    match = re.search(
        r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        text
    )

    if match:

        results[
            "fecha_nacimiento"
        ] = limpiar_fecha(
            match.group(1)
        )

    else:

        results[
            "fecha_nacimiento"
        ] = None

    # --------------------------------------------------------
    # SEXO
    # --------------------------------------------------------

    match = re.search(
        r"(SEXO|GENERO)\s*:?\s*([MF])",
        text,
        re.IGNORECASE
    )

    if match:

        results[
            "sexo"
        ] = limpiar_sexo(
            match.group(2)
        )

    else:

        results[
            "sexo"
        ] = None

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

    if match:

        results[
            "lugar_nacimiento"
        ] = limpiar_texto(
            match.group(1).strip()
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

    if match:

        results[
            "nacionalidad"
        ] = limpiar_texto(
            match.group(2).strip()
        )

    else:

        results[
            "nacionalidad"
        ] = None

    # --------------------------------------------------------
    # RH
    # --------------------------------------------------------

    match = re.search(
        r"(TIPO\s*(?:DE)?\s*SANGRE|RH)"
        r"\s*:?\s*([A-Z0-9+-]+)",
        text,
        re.IGNORECASE
    )

    if match:

        results[
            "tipo_sangre"
        ] = limpiar_rh(
            match.group(2)
        )

    return results


# ============================================================
# RESULTADO ÚTIL
# ============================================================

def _resultado_util(
    results
):

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
            bool(
                results.get(k)
            )
            for k in principales
        )
        >= 2
    )


# ============================================================
# RESULTADO FINAL
# ============================================================

def normalizar_resultado_final(
    results
):

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

        valor = results.get(
            campo
        )

        if valor is None:

            resultado[campo] = ""

        else:

            resultado[campo] = (
                str(valor).strip()
            )

    return resultado


# ============================================================
# EXTRACCIÓN PRINCIPAL
# ============================================================

def extract_fields(
    file_path,
    modelo="hologramas"
):
    """
    Procesa las dos primeras páginas utilizando DOS mecanismos:

    1. Lectura estructural por texto / etiquetas / líneas / columnas.
    2. OCR por coordenadas.

    Los dos resultados se conservan y posteriormente se combinan.

    Además mantiene:
        - crops originales
        - crops ampliados
        - crops preprocesados
        - URLs públicas de debug
        - logs de coordenadas
        - detección de cédula
        - validación OCR
    """

    results = {}

    # Resultados obtenidos por OCR GENERAL.
    # Esta será nuestra fuente principal.
   
    # --------------------------------------------------------
    # DIRECTORIO DEBUG
    # --------------------------------------------------------

    debug_dir = os.path.join(
        os.path.dirname(file_path),
        "ocr_debug"
    )

    logger.info(
        "OCR_DEBUG_DIR: %s",
        debug_dir
    )

    OCR_RENDER_DPI = int(os.getenv("OCR_RENDER_DPI", 300))

    try:

        with pdfplumber.open(
            file_path
        ) as pdf:

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
                # RESULTADOS AISLADOS POR PÁGINA
                # ==================================================== 
                general_results = {}
                crop_results = {}

                # ====================================================
                # 1. TEXTO EMBEBIDO
                # ====================================================

                text = (
                    page.extract_text()
                    or ""
                )

                text_results = {}

                if text.strip():

                    # ------------------------------------------------
                    # LECTURA TRADICIONAL
                    # ------------------------------------------------

                    text_results = (
                        extract_fields_from_text(
                            text
                        )
                    )

                    # ------------------------------------------------
                    # COMBINAR RESULTADOS DE TEXTO
                    # ------------------------------------------------

                    for key, value in (
                        text_results.items()
                    ):

                        if (
                            value
                            and not results.get(key)
                            and validar_campo_ocr(
                                key,
                                value
                            )
                        ):

                            results[key] = value

                    logger.info(
                        "Página %s: resultados obtenidos "
                        "por lectura estructurada. "
                        "Continuando con OCR por coordenadas.",
                        page_number
                    )

                # ====================================================
                # 2. RENDERIZAR PÁGINA
                # ====================================================

                logger.info(
                    "Página %s: entrando a OCR "
                    "de imagen.",
                    page_number
                )

                img = page.to_image(
                    resolution=OCR_RENDER_DPI
                ).original

                img_np = np.array(
                    img
                )

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
                # 3.5. OCR POR ETIQUETAS Y LÍNEAS
                # ====================================================

                lado = detectar_lado_cedula(
                    agrupar_lineas(
                        obtener_datos_ocr_pagina(
                            img
                        )
                    )
                )

                logger.info(
                    "OCR_LADO_FINAL | "
                    "pagina=%s | lado=%s",
                    page_number,
                    lado
                )

                etiqueta_results = (
                    procesar_por_etiquetas(
                        img,
                        page_number,
                        debug_dir,
                        lado
                    )
                )

                for key, value in (
                    etiqueta_results.items()
                ):

                    if (
                        value
                        and validar_campo_ocr(
                            key,
                            value
                        )
                    ):

                        general_results[key] = value

                        logger.info(
                            "OCR_GENERAL_RESULTADO | "
                            "pagina=%s | campo=%s | valor=%s",
                            page_number,
                            key,
                            value
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

                    # ------------------------------------------------
                    # NO saltamos el campo aunque ya exista.
                    #
                    # Lo procesamos porque necesitamos los crops
                    # para depuración y comparación.
                    # ------------------------------------------------

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

                        if field in (
                            "apellidos",
                            "nombres",
                            "lugar_nacimiento",
                        ):
                            crop_ocr = crop

                        else:
                            
                            crop_ocr = preprocesa_image(
                                crop
                            )

                        logger.info(
                            "OCR_CROP_PREPROCESADO | "
                            "pagina=%s | campo=%s | "
                            "tamaño=%sx%s",
                            page_number,
                            field,
                            crop.width,
                            crop.height,
                        )

                        # ------------------------------------------------
                        # Guardar exactamente lo que recibe Tesseract
                        # ------------------------------------------------

                        guardar_crop_debug(
                            crop_ocr,
                            f"{field}_ocr",
                            page_number,
                            debug_dir
                        )

                        # ====================================================
                        # TESSERACT
                        # ====================================================

                        raw = _ocr_field(
                            crop_ocr,
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
                        # LIMPIEZA CROP
                        # ====================================================

                        if clean:

                            crop_results[field] = clean

                            logger.info(
                                "OCR_CROP_RESULTADO | "
                                "pagina=%s | campo=%s | valor=%s",
                                page_number,
                                field,
                                clean
                            )

                        else:

                            logger.info(
                                "OCR_CROP_SIN_RESULTADO | "
                                "pagina=%s | campo=%s",
                                page_number,
                                field
                            )

                    except Exception as e:

                        logger.exception(
                            "OCR_CROP_ERROR | "
                            "pagina=%s | campo=%s | error=%s",
                            page_number,
                            field,
                            e
                        )

                        # El OCR por CROP falló.
                        # NO modificamos el resultado obtenido
                        # por OCR general / etiquetas.

                        logger.info(
                            "OCR_CROP_ERROR | "
                            "pagina=%s | campo=%s | "
                            "se conserva OCR general",
                            page_number,
                            field
                        )

                # ====================================================
                # 5.5. COMPARAR OCR GENERAL VS OCR POR CROP
                #
                # IMPORTANTE:
                #
                # Este bloque está FUERA del:
                #
                #     for field, coords in zones.items():
                #
                # Primero terminamos TODOS los crops.
                # Después comparamos los resultados.
                # ====================================================

                campos_comparacion = set(
                    list(general_results.keys())
                    +
                    list(crop_results.keys())
                )

                for field in campos_comparacion:

                    valor_general = general_results.get(
                        field
                    )

                    valor_crop = crop_results.get(
                        field
                    )

                    comparacion = comparar_resultados_ocr(
                        valor_general,
                        valor_crop,
                        field,
                        lado
                    )

                    resultado = comparacion.get(
                        "value"
                    )

                    if resultado:

                        results[field] = resultado

                    else:

                        results[field] = None

                    logger.info(
                        "OCR_DECISION_FINAL | "
                        "pagina=%s | lado=%s | campo=%s | "
                        "general=%r | crop=%r | "
                        "elegido=%r | origen=%s | "
                        "general_valido=%s | "
                        "crop_valido=%s | "
                        "coinciden=%s",
                        page_number,
                        lado,
                        field,
                        valor_general,
                        valor_crop,
                        resultado,
                        comparacion.get(
                            "origen"
                        ),
                        comparacion.get(
                            "general_valido"
                        ),
                        comparacion.get(
                            "crop_valido"
                        ),
                        comparacion.get(
                            "coinciden"
                        )
                    )

                # ====================================================
                # 6. RECONSTRUIR NOMBRE COMPLETO
                # ====================================================
                
                nombre_completo = ""

                nombres = (results.get("nombres") or "").strip()
                apellidos = (results.get("apellidos") or "").strip()

                if nombres or apellidos:

                    nombre_completo = limpiar_texto(f"{nombres} {apellidos}".strip())                          

                    if validar_campo_ocr(
                        "nombre_completo",
                        nombre_completo
                    ):

                        results["nombre_completo"] = (
                            nombre_completo
                        )

                logger.info(
                    "OCR_NOMBRE_COMPLETO_FINAL | "
                    "nombres=%r | apellidos=%r | "
                    "nombre_completo=%r",
                    results["nombres"],
                    results["apellidos"],
                    nombre_completo
                )
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