
import sys
import os 
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import engine, Base, get_db
from api.models import Usuario, Documento 
from app.ocr import procesar_pdf

Base.metadata.create_all(bind=engine)

app = FastAPI(title="OCR Documentos Identidad 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sna-2-0-3.onrender.com"],  # puedes poner aquí el dominio de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"mensaje": "API OCR 2.0 funcionando correctamente"}

# ---------------------------
# 📌 OCR Upload
# ---------------------------
@app.post("/ocr/upload/")
async def ocr_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    resultado = procesar_pdf(temp_path)

    if "error" in resultado:
        raise HTTPException(status_code=400, detail=resultado["error"])
    datos = resultado["resultado"]

    nuevo_doc = Documento(
        NumeroDocumento=datos.get("numero_documento", ""),
        NombreCompleto=datos.get("nombre_completo", ""),
        FechaNacimiento=datos.get("fecha_nacimiento", ""),
        Sexo=datos.get("sexo", ""),
        LugarNacimiento=datos.get("lugar_nacimiento", ""),
        Nacionalidad=datos.get("nacionalidad", ""),
        TipoSangre=datos.get("tipo_sangre", "")
    )
    db.add(nuevo_doc)
    db.commit()
    db.refresh(nuevo_doc)

    return {
        "mensaje": "Documento guardado desde OCR",
        "documento_id": nuevo_doc.Id,
        "datos_extraidos": datos
    }

# ---------------------------
# 📌 Registro de usuarios
# ---------------------------
class UserRegister(BaseModel):
    nombre: str
    email: str
    password: str

@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(Usuario).filter(Usuario.Email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="El correo ya está registrado")

    nuevo = Usuario(Nombre=user.nombre, Apellido="", Email=user.email, Password=user.password)
    db.add(nuevo)
    db.commit()
    db.refresh(nuevo)
    return {"mensaje": "Usuario registrado correctamente", "usuario_id": nuevo.Id}

# ---------------------------
# 📌 Login de usuarios
# ---------------------------
class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    usuario = db.query(Usuario).filter(
        Usuario.Email == user.email, 
        Usuario.Password == user.password
    ).first()
    if not usuario:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    # ⚠️ Aquí deberías generar un JWT en producción
    token = f"fake-token-{usuario.Id}"
    return {"mensaje": "Login exitoso", "token": token}
