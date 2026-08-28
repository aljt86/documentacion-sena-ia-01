import re
import unicodedata
from datetime import datetime


# ============================================================
# LIMPIEZA DE CAMPOS OCR
# ============================================================

def limpiar_numero(raw):
    """
    Limpia un número de documento.
    Conserva únicamente dígitos.

    Ejemplos:
       '1.061.686.689' -> '1061686689'
        '1,061.686.689' -> '1061686689'
        ' 1.061.686.689 ' -> '1061686689'

    No determina por sí sola si el número es correcto.
    Solamente normaliza el valor recibido.
    """ 
    
    if raw is None:
        return ""

    texto = str(raw).strip()

    if not texto:
        return ""

    # Eliminar separadores habituales de miles
    texto = re.sub(r"[.\s,-]", "", texto)

    # Mantener únicamente números 
    texto = re.sub(r"\D", "", texto)

    # Longitud razonable para docuemnto de identidad
    if not 6 <= len(texto) <= 12:
        return ""

    return texto 

def limpiar_texto(raw: str) -> str:
    """
    Limpia nombres, apellidos, nacionalidad y lugares.
    """
    if not raw:
        return ""

    texto = unicodedata.normalize("NFKC", str(raw))

    texto = re.sub(
        r"[^A-Za-zÁÉÍÓÚÑáéíóúñ\s]",
        "",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto.title() if texto else ""


def limpiar_fecha(raw: str) -> str:
    """
    Normaliza fechas OCR a YYYY-MM-DD.

    Soporta:

        22-FEB-1986
        22 FEB 1986
        22/FEB/1986
        22-02-1986
        22/02/1986
        YYYY-MM-DD
    """

    if not raw:
        return ""

    raw = str(raw).strip().upper()
    raw = re.sub(r"\s+", " ", raw) 
    
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

    # ========================================================
    # FECHA YA NORMALIZADA: YYYY-MM-DD
    # ========================================================

    match = re.fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})", 
        raw
    )

    if match:
        try:
            fecha = datetime.strptime(
                raw,
                "%Y-%m-%d"
            )

            return fecha.strftime("%Y-%m-%d")
        
        except ValueError:
            return ""  

    # ========================================================
    # FECHA CON MES EN TEXTO
    # ========================================================

    match = re.search(
        r"(\d{1,2})[-/\s]+([A-ZÁÉÍÓÚ]{3})[-/\s]+(\d{4})",
        raw
    )

    if match:
        dia, mes, anio = match.groups()

        mes_numero = meses.get(mes[:3])

        if mes_numero:
            try:
                fecha = datetime.strptime(
                    f"{anio}-{mes_numero}-{dia.zfill(2)}",
                    "%Y-%m-%d"
                )

                return fecha.strftime("%Y-%m-%d")
            
            except ValueError:
                return ""

    # ========================================================
    # FECHA NUMÉRICA
    # ========================================================

    match = re.search(
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
        raw
    )

    if match:
        dia, mes, anio = match.groups()

        try:

            fecha = datetime.strptime(
                f"{anio}-{mes.zfill(2)}-{dia.zfill(2)}",
                "%Y-%m-%d"
            )

            return fecha.strftime("%Y-%m-%d")
        except ValueError:
            return ""

    return ""


def limpiar_sexo(raw: str) -> str:
    """
    Normaliza M/F.
    """

    if not raw:
        return ""

    valor = str(raw).strip().upper()

    if valor.startswith("M"):
        return "Masculino"

    if valor.startswith("F"):
        return "Femenino"

    return ""

def limpiar_rh(raw: str) -> str:
    """
    Normaliza grupos sanguíneos.
    """

    if not raw:
        return ""

    valor = str(raw).upper()

    valor = valor.replace(" ", "")
    valor = valor.replace("0", "O")

    # Normalizar símbolos OCR frecuentes
    valor = (
        valor
        .replace("-", "-")
        .replace("-", "-")
        .replace("-", "-")
    )

    match = re.search(
        r"(AB|A|B|O)[+-]",
        valor
    )

    return match.group(0) if match else ""