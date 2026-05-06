
from fastapi import FastAPI, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db import get_db
from app.ocr import procesar_pdf

app = FastAPI(title="OCR Documentos Identidad 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes poner aquí el dominio de tu frontend en lugar de "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    

@app.get("/")
def home():
    return {"mensaje": "API OCR 2.0 funcionando correctamente"}

# Endpoint por subida de archivo
@app.post("/ocr/upload/")
async def ocr_upload(file: UploadFile = File(...)):
    # Guardar temporalmente el archivo subido
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Procesar el PDF
    resultado = procesar_pdf(temp_path)
    return {"resultado": resultado}

# Endpoint para crear usuario
@app.post("/usuarios/")
def crear_usuario(nombre: str, apellido: str, email: str, db: Session = Depends(get_db)):
    nuevo = Usuario(Nombre=nombre, Apellido=apellido, Email=email)
    db.add(nuevo)
    db.commit()
    db.refresh(nuevo)
    return {"mensaje": "Usuario creado", "usuario_id": nuevo.Id}

# Endpoint para listar usuarios
@app.get("/usuarios/")
def listar_usuarios(db: Session = Depends(get_db)):
    return db.query(Usuario).all()
