
from fastapi import FastAPI, UploadFile, File, Query
from app.ocr import procesar_pdf

app = FastAPI(title="OCR Documentos Identidad 2.0")

@app.get("/")
def home():
    return {"mensaje": "API OCR 2.0 funcionando correctamente"}

# Endpoint por subida de archivo
@app.post("/ocr/upload/")
async def ocr_upload(file: UploadFile = File(...)):
    # Guardar temporalmente el archivo subido
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Procesar el PDF
    resultado = procesar_pdf(temp_path)
    return {"resultado": resultado}
        