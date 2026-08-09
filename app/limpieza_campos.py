import re
import logging

# ============================================
# LIMPIEZA DE CAMPOS OCR
# ============================================

def limpiar_numero(raw: str) -> str:
    """
    Limpia el número de documento: elimina puntos, espacios y caracteres no numéricos.
    """
    if not raw:
        return ""
    numero = re.sub(r'\D', '', raw)
    return numero.strip() if numero else ""

def limpiar_texto(raw: str) -> str:
    """
    Limpia nombres, apellidos, nacionalidad, lugar de nacimiento:
    - Elimina caracteres raros
    - Convierte múltiples espacios en uno
    - Capitaliza cada palabra
    """
    if not raw:
        return ""
    texto = re.sub(r'[^A-Za-zÁÉÍÓÚÑáéíóúñ\s]', '', raw)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto.title()

def limpiar_fecha(raw: str) -> str:
    """
    Convierte fechas OCR como '22-FEB-1986' o '15 AGO 1993' a formato ISO '1986-02-22'.
    """
    if not raw:
        return ""
    meses = {
        "ENE":"01","FEB":"02","MAR":"03","ABR":"04","MAY":"05","JUN":"06",
        "JUL":"07","AGO":"08","SEP":"09","OCT":"10","NOV":"11","DIC":"12"
    }
    # Formato con guiones
    m = re.match(r'(\d{1,2})[-\s]([A-ZÁÉÍÓÚ]{3})[-\s](\d{4})', raw.upper())
    if m:
        d, mes, y = m.groups()
        mes_num = meses.get(mes[:3], "01")
        return f"{y}-{mes_num}-{d.zfill(2)}"
    return raw.strip()

def limpiar_sexo(raw: str) -> str:
    """
    Normaliza sexo: M/F → Masculino/Femenino
    """
    if not raw:
        return ""
    rv = raw.strip().upper()
    if rv.startswith("M"):
        return "Masculino"
    elif rv.startswith("F"):
        return "Femenino"
    return ""

def limpiar_rh(raw: str) -> str:
    """
    Normaliza tipo de sangre: O+, A-, etc.
    """
    if not raw:
        return ""
    m = re.match(r'([ABO]{1,2}[+-])', raw.upper())
    return m.group(1) if m else ""
