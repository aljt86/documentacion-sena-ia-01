import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
from datetime import datetime

# Configuración de Tesseract y Poppler para entornos Windows y Docker/Linux
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
elif os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

poppler_path = os.getenv("POPPLER_PATH")

def calcular_edad(fecha_str: str):
    meses = {"ENE":"Jan","FEB":"Feb","MAR":"Mar","ABR":"Apr","MAY":"May","JUN":"Jun",
             "JUL":"Jul","AGO":"Aug","SEP":"Sep","OCT":"Oct","NOV":"Nov","DIC":"Dec"}
    try:
        partes = fecha_str.replace(",", "").split("-")
        if len(partes) == 3:
            mes_eng = meses.get(partes[1].upper(), partes[1])
            fecha_dt = datetime.strptime(f"{partes[0]}-{mes_eng}-{partes[2]}", "%d-%b-%Y")
        else:
            partes = fecha_str.split()
            dia, mes, anio = partes
            mes_eng = meses.get(mes.upper(), mes)
            fecha_dt = datetime.strptime(f"{dia}-{mes_eng}-{anio}", "%d-%b-%Y")
        hoy = datetime.today()
        return hoy.year - fecha_dt.year - ((hoy.month, hoy.day) < (fecha_dt.month, fecha_dt.day))
    except:
        return None

def preprocesar_imagen(pagina):
    img = np.array(pagina)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    blur = cv2.medianBlur(thresh, 3)
    return blur

def ocr_pagina(pagina):
    procesada = preprocesar_imagen(pagina)
    texto = pytesseract.image_to_string(procesada, lang="spa", config="--psm 6")
    return texto

def normalizar(texto: str) -> str:
    reemplazos = {"Á":"A","É":"E","Í":"I","Ó":"O","Ú":"U"}
    for k,v in reemplazos.items():
        texto = texto.replace(k,v)
    return texto.upper()

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
        "edad": None
    }

    lineas = texto.splitlines()
    tipo_doc = "desconocido"

    # patrones de encabezado
    encabezados = {"REPUBLICA","REPUBLICA DE COLOMBIA","COLOMBIA","IDENTIFICACION",
                   "IDENTIFICACIÓN","IDENTIFICACIÓN PERSONAL","CEDULA DE CIUDADANÍA",
                   "CEDULA DE CIUDADANIA","CEDULA"}

    # eliminar encabezados iniciales
    n = 0
    while n < 3 and lineas:
        primera = normalizar(lineas[0].strip()).upper()
        if any(k in primera for k in encabezados):
            lineas.pop(0)
            n += 1
        else:
            break

    for i, linea in enumerate(lineas):
        l = normalizar(linea.strip())

        # Detectar tipo de documento
        if "NUIP" in l:
            tipo_doc = "nueva"
        if re.search(r"(APELLIDOS?|NOMBRES?)", l):
            tipo_doc = "antigua"

        # Documento
        if re.search(r"NUMER[O0]", l) and i+1 < len(lineas):
            datos["numero_documento"] = re.sub(r"\D", "", lineas[i+1].strip())
        if "NUIP" in l and i+1 < len(lineas):
            datos["numero_documento"] = re.sub(r"\D", "", lineas[i+1].strip())

        # Fecha nacimiento
        if "FECHA DE NACIMIENTO" in l and i+1 < len(lineas):
            fnac = lineas[i+1].strip()
            datos["fecha_nacimiento"] = fnac
            datos["edad"] = calcular_edad(fnac)

        # Lugar nacimiento
        if "LUGAR DE NACIMIENTO" in l and i-2 >= 0 and i-1 >= 0:
            ciudad = lineas[i-2].strip()
            depto = lineas[i-1].strip()
            lugar = f"{ciudad} {depto}".replace("!", "P")
            datos["lugar_nacimiento"] = lugar

        # Sexo
        if "SEXO" in l and i+1 < len(lineas):
            sexo = lineas[i+1].strip().upper()
            if sexo.startswith("M"):
                datos["sexo"] = "Masculino"
            elif sexo.startswith("F"):
                datos["sexo"] = "Femenino"

        # Fecha expedición
        if "EXPEDICION" in l and i-1 >= 0:
            datos["fecha_expedicion"] = lineas[i-1].strip()

        # Antigua: apellidos y nombres (línea anterior)
        if tipo_doc == "antigua":
            apellidos, nombres = "", ""
            if re.search(r"APELLIDOS?", l) and i-1 >= 0:
                apellidos = lineas[i-1].strip()
            if re.search(r"NOMBRES?", l) and i-1 >= 0:
                nombres = lineas[i-1].strip()
            if nombres or apellidos:
                datos["nombre_completo"] = f"{nombres} {apellidos}".strip()

        # Nueva: apellidos y nombres (línea siguiente)
        if tipo_doc == "nueva":
            apellidos, nombres = "", ""
            if re.search(r"APELLIDOS?", l) and i+1 < len(lineas):
                apellidos = lineas[i+1].strip()
            if re.search(r"NOMBRES?", l) and i+1 < len(lineas):
                nombres = lineas[i+1].strip()
            if nombres or apellidos:
                datos["nombre_completo"] = f"{nombres} {apellidos}".strip()

        # RH
        if "RH" in l and i-1 >= 0:
            rh_line = lineas[i-1].strip()
            rh_match = re.search(r"\b[OAB][+-]\b", rh_line)
            if rh_match:
                datos["rh"] = rh_match.group()

        if tipo_doc == "nueva":
            if "NACIONALIDAD" in l and i+1 < len(lineas):
                datos["nacionalidad"] = lineas[i+1].strip()
            if re.search(r"\b[OAB][+-]\b", l):
                datos["rh"] = l.strip()

    # --- Fallback adicional ---
    if not datos["numero_documento"]:
        doc_match = re.findall(r"\d{8,10}", texto)
        if doc_match:
            datos["numero_documento"] = max(doc_match, key=len)

    if not datos["nombre_completo"]:
        candidatos = re.findall(r"[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{3,}){1,3}", texto)
        candidatos = [c for c in candidatos if "COLOMBIA" not in c and "IDENTIFICACION" not in c]
        if candidatos:
            datos["nombre_completo"] = candidatos[0].title()

    return datos

def procesar_documento(pdf_path):
    texto_total = ""
    datos = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    texto_total += page_text + "\n"
            if texto_total.strip():
                datos = extraer_campos_por_lineas(texto_total)
    except Exception:
        pass

    if not datos or all(v is None or v == "" for v in datos.values()):
        try:
            paginas = convert_from_path(pdf_path, poppler_path=poppler_path)
            for pagina in paginas:
                texto_total += ocr_pagina(pagina) + "\n"

            with open("ocr_debug.txt", "w", encoding="utf-8") as f:
                f.write(texto_total)

            datos = extraer_campos_por_lineas(texto_total)
        except Exception as e:
            return {"error": f"OCR falló: {e}"}

    if not datos or all(v is None or v == "" for v in datos.values()):
        return {"error": "El documento no se pudo leer correctamente."}

    return {"resultado": datos}

# Wrapper para compatibilidad
def extraer_campos(texto: str):
    return extraer_campos_por_lineas(texto)
