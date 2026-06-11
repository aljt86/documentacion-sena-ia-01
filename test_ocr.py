from app.ocr_template import extract_fields

# Ruta al PDF que quieres probar
pdf_path = "data/prueba14.pdf"   # cámbiala según tu archivo

# Modelo: "digital" o "hologramas"
resultado = extract_fields(pdf_path, modelo="hologramas")

print("Resultado OCR con plantilla:")
for campo, valor in resultado.items():
    print(f"{campo}: {valor}")
