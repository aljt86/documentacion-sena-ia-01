import pdfplumber
import pytesseract
import cv2
import numpy as np
import re
import logging
from pytesseract import Output
from app.limpieza_campos import limpiar_numero, limpiar_texto, limpiar_fecha, limpiar_sexo, limpiar_rh

def procesar_pdf_hibrido(file_path: str):
    datos = {
        "numero_documento": "",
        "apellidos": "",
        "nombres": "",
        "nombre_completo": "",
        "fecha_nacimiento": "",
        "sexo": "",
        "lugar_nacimiento": "",
        "nacionalidad": "",
        "tipo_sangre": "",
    }

    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""

            if text.strip():
                # Documento
                doc_match = re.search(r"\d{6,15}", text)
                datos["numero_documento"] = limpiar_numero(doc_match.group(0)) if doc_match else ""

                # Fecha nacimiento
                fecha_match = re.search(r"\d{2}-[A-Z]{3}-\d{4}", text)
                datos["fecha_nacimiento"] = limpiar_fecha(fecha_match.group(0)) if fecha_match else ""

                # RH
                rh_match = re.search(r"(O|A|B|AB)[+-]", text)
                datos["tipo_sangre"] = limpiar_rh(rh_match.group(0)) if rh_match else ""

                # Sexo
                sexo_match = re.search(r"\b(M|F)\b", text.upper())
                datos["sexo"] = limpiar_sexo(sexo_match.group(0)) if sexo_match else ""

                # Apellidos y nombres
                apellidos_match = re.search(r"APELLIDOS\s*\n([A-ZÁÉÍÓÚÑ\s]+)", text, re.IGNORECASE)
                nombres_match = re.search(r"NOMBRES\s*\n([A-ZÁÉÍÓÚÑ\s]+)", text, re.IGNORECASE)
                if apellidos_match:
                    datos["apellidos"] = limpiar_texto(apellidos_match.group(1).strip())
                if nombres_match:
                    datos["nombres"] = limpiar_texto(nombres_match.group(1).strip())

                if datos["apellidos"] or datos["nombres"]:
                    datos["nombre_completo"] = f"{datos.get('nombres','')} {datos.get('apellidos','')}".strip()

                # Lugar nacimiento
                lugar_match = re.search(r"LUGAR DE NACIMIENTO\s*\n([A-ZÁÉÍÓÚÑ\s\(\)]+)", text, re.IGNORECASE)
                if lugar_match:
                    datos["lugar_nacimiento"] = limpiar_texto(lugar_match.group(1).strip())

                # Nacionalidad
                nac_match = re.search(r"NACIONALIDAD\s*\n([A-ZÁÉÍÓÚÑ\s]+)", text, re.IGNORECASE)
                if nac_match:
                    datos["nacionalidad"] = limpiar_texto(nac_match.group(1).strip())

        # Si no hubo texto embebido, usar OCR con data
        if not datos["numero_documento"]:
            img = pdf.pages[0].to_image(resolution=300).original
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

            # Procesamiento para baja calidad
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
            kermel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            gray = cv2.filter2D(img, -1, kermel)

            data = pytesseract.image_to_data(img, lang="spa", config="--psm 6", output_type=Output.DICT)
            texto_total = " ".join(data["text"])

            doc_match = re.search(r"\d{6,15}", texto_total)
            fecha_match = re.search(r"\d{2}-[A-Z]{3}-\d{4}", texto_total)
            rh_match = re.search(r"(O|A|B|AB)[+-]", texto_total)
            sexo_match = re.search(r"\b(M|F)\b", texto_total)

            datos["numero_documento"] = limpiar_numero(doc_match.group(0)) if doc_match else ""
            datos["fecha_nacimiento"] = limpiar_fecha(fecha_match.group(0)) if fecha_match else ""
            datos["tipo_sangre"] = limpiar_rh(rh_match.group(0)) if rh_match else ""
            datos["sexo"] = limpiar_sexo(sexo_match.group(0)) if sexo_match else ""

    except Exception as e:
        logging.error(f"OCR falló: {e}")
        return {"error": str(e)}

    return datos
