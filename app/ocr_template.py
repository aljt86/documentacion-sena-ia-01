import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
import re

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(level=logging.INFO)

# ============================================
# PREPROCESAMIENTO DE IMÁGENES
# ============================================
def preprocess_image(pil_img):
    """
    Preprocesa la imagen para mejorar la precisión del OCR.
    Optimizado para documentos escaneados de baja calidad.
    """
    try:
        # Convertir a escala de grises
        img = np.array(pil_img.convert("L"))
        
        # Ecualización de histograma (mejora contraste en imágenes oscuras)
        img = cv2.equalizeHist(img)
        
        # Filtro bilateral (reduce ruido sin perder bordes)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Umbralización adaptativa (mejor para diferentes condiciones de luz)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Eliminar ruido pequeño con operaciones morfológicas
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(thresh)
    except Exception as e:
        logging.error(f"Error en preprocesamiento: {e}")
        return pil_img  # Devolver la imagen original si falla

# ============================================
# ZONAS PARA CÉDULA DIGITAL (POLICARBONATO)
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
# ZONAS PARA CÉDULA AMARILLA CON HOLOGRAMAS
# ============================================
zones_hologramas = {
    "numero_documento": (0.4/8.5, 1.6/5.4, 4.5/8.5, 3.4/5.4),
    "apellidos": (0.4/8.5, 2.0/5.4, 4.5/8.5, 3.1/5.4),
    "nombres": (0.4/8.5, 2.8/5.4, 4.5/8.5, 2.4/5.4),
    "fecha_nacimiento": (3.3/8.5, 0.5/5.4, 1.2/8.5, 4.5/5.4),
    "lugar_nacimiento": (3.3/8.5, 0.8/5.4, 3.1/8.5, 3.8/5.4),
    "tipo_sangre": (4.9/8.5, 1.6/5.4, 3.0/8.5, 3.1/5.4),
    "sexo": (6.3/8.5, 1.6/5.4, 1.6/8.5, 3.1/5.4),
}

# ============================================
# EXTRACCIÓN DE CAMPOS DESDE TEXTO (PDF DIGITAL)
# ============================================
def extract_fields_from_text(text):
    """
    Extrae campos usando expresiones regulares del texto del PDF.
    """
    results = {}

    # Número de documento
    match = re.search(r'(\d{6,15})', text)
    results['numero_documento'] = match.group(1) if match else ""

    # Nombre completo
    match = re.search(r'(?:NOMBRE|NOMBRES?)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
    results['nombre_completo'] = match.group(1).strip() if match else ""

    # Apellidos y nombres (fallback si no encuentra nombre completo)
    if not results['nombre_completo']:
        match = re.search(r'(?:APELLIDOS?)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
        if match:
            results['apellidos'] = match.group(1).strip()
        match = re.search(r'(?:NOMBRES?)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
        if match:
            results['nombres'] = match.group(1).strip()

    # Fecha de nacimiento
    match = re.search(r'(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})', text)
    results['fecha_nacimiento'] = match.group(1) if match else ""

    # Sexo
    match = re.search(r'(SEXO|GENERO)\s*:?\s*([MF])', text, re.IGNORECASE)
    results['sexo'] = match.group(2) if match else ""

    # Lugar de nacimiento
    match = re.search(r'(?:LUGAR|CIUDAD)\s*(?:DE)?\s*NACIMIENTO\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
    results['lugar_nacimiento'] = match.group(1).strip() if match else ""

    # Nacionalidad
    match = re.search(r'(NACIONALIDAD)\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+)', text, re.IGNORECASE)
    results['nacionalidad'] = match.group(2).strip() if match else ""

    # Tipo de sangre
    match = re.search(r'(TIPO\s*(?:DE)?\s*SANGRE|RH)\s*:?\s*([A-Z0-9+-]+)', text, re.IGNORECASE)
    results['tipo_sangre'] = match.group(2) if match else ""

    return results

# ============================================
# FUNCIÓN PRINCIPAL (SOPORTA AMBOS FORMATOS)
# ============================================
def extract_fields(file_path, modelo="hologramas"):
    """
    Extrae los campos de la cédula según el modelo.
    
    Args:
        file_path (str): Ruta al archivo PDF.
        modelo (str): "digital" (cédula policarbonato) o "hologramas" (cédula amarilla)
    
    Returns:
        dict: Diccionario con los campos extraídos.
    """
    results = {}

    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                logging.error("El PDF no contiene páginas.")
                return {}

            page = pdf.pages[0]
            text = page.extract_text() or ""

            # ✅ Intentar extraer texto directamente (PDF digital)
            if text.strip():
                results = extract_fields_from_text(text)
                if any(results.values()):
                    logging.info("✅ Datos extraídos exitosamente del texto del PDF")
                    return results
                else:
                    logging.warning("⚠️ El texto extraído no contiene datos válidos, usando OCR...")
            else:
                logging.warning("⚠️ El PDF no tiene texto extraíble, usando OCR...")

            # ✅ Si no hay texto o no se encontraron datos, usar OCR con imágenes
            img = page.to_image(resolution=300).original
            width, height = img.size
            logging.info(f"📐 Tamaño de imagen: {width}x{height} px")

            # Seleccionar zonas según el modelo
            zones = zones_digital if modelo == "digital" else zones_hologramas

            for field, (x1, y1, x2, y2) in zones.items():
                # Normalizar coordenadas (entre 0 y 1)
                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))

                # Asegurar que x1 < x2 y y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                box = (
                    int(x1 * width), int(y1 * height),
                    int(x2 * width), int(y2 * height)
                )

                logging.info(f"🔍 Recortando '{field}' con box: {box}")

                try:
                    # Recortar la región
                    crop = img.crop(box)

                    # Preprocesar la imagen
                    crop = preprocess_image(crop)

                    # Aplicar OCR con Tesseract
                    text_ocr = pytesseract.image_to_string(
                        crop,
                        lang="spa",
                        config="--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
                    )

                    results[field] = text_ocr.strip()
                    logging.warning(f"OCR {field}: {results[field]}")

                except Exception as e:
                    logging.error(f"❌ Error al procesar '{field}': {e}")
                    results[field] = ""

    except Exception as e:
        logging.error(f"❌ Error al procesar el PDF: {e}")
        return {}

    return results


                


            
   
