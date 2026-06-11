from app.ocr_template import extract_fields

# Ruta al PDF que quieres probar
pdf_path = "data/prueba14.pdf"   # cámbiala según tu archivo

# Probar modelo "digital"
resultado_digital = extract_fields(pdf_path, modelo="digital")

# Probar modelo "hologramas"
resultado_hologramas = extract_fields(pdf_path, modelo="hologramas")

print("\n=== RESULTADOS COMPARATIVOS ===\n")

print(">>> Modelo DIGITAL <<<")
for campo, valor in resultado_digital.items():
    print(f"{campo}: {valor}")

print("\n>>> Modelo HOLOGRAMAS <<<")
for campo, valor in resultado_hologramas.items():
    print(f"{campo}: {valor}")
