import re
import cv2
import pytesseract

def detectar_tipo_documento(datos):
    numero = datos.get("numero_documento", "")
    if re.match(r"^\d{8,10}$", numero):
        return "colombia_cedula"
    elif re.match(r"^\d{6,8}$", numero):
        return "colombia_tarjeta"
    elif re.match(r"^\d{7,9}$", numero):
        return "venezuela"
    else:
        return "extranjero"

def validar_datos(datos, tipo_doc="colombia_cedula"):
    validaciones = {}

    # Nombre completo: solo letras y espacios
    validaciones["nombre_completo"] = bool(
        re.match(r"^[A-Za-zÁÉÍÓÚÑ ]+$", datos.get("nombre_completo", ""))
    )

    # Número de documento: depende del tipo
    numero = datos.get("numero_documento", "")
    if tipo_doc == "colombia_cedula":
        validaciones["numero_documento"] = bool(re.match(r"^\d{8,10}$", numero))
    elif tipo_doc == "colombia_tarjeta":
        validaciones["numero_documento"] = bool(re.match(r"^\d{6,8}$", numero))
    elif tipo_doc == "venezuela":
        validaciones["numero_documento"] = bool(re.match(r"^\d{7,9}$", numero))
    else:  # genérico
        validaciones["numero_documento"] = bool(re.match(r"^\d{6,12}$", numero))

    # Sexo: M/F
    validaciones["sexo"] = datos.get("sexo", "").upper() in ["M", "F"]

    # Tipo de sangre: A/B/O con +/-
    validaciones["tipo_sangre"] = bool(
        re.match(r"^(A|B|O)[+-]$", datos.get("tipo_sangre", ""))
    )

    return validaciones


# --- NUEVO BLOQUE DE PREPROCESAMIENTO Y OCR ---

def preprocess_image(path):
    """Preprocesa la imagen para mejorar la lectura OCR"""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extraer_texto(path):
    """Aplica OCR sobre la imagen ya preprocesada"""
    processed = preprocess_image(path)
    texto = pytesseract.image_to_string(processed, lang="spa")
    return texto
