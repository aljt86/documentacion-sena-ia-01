from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db import get_db
from app.ocr import procesar_pdf
from models import Usuario  # asegúrate de tener este modelo definido
from pydantic import BaseModel

app = FastAPI(title="OCR Documentos Identidad 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes poner aquí el dominio de tu frontend
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
async def ocr_upload(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    resultado = procesar_pdf(temp_path)
    return {"resultado": resultado}

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
    usuario = db.query(Usuario).filter(Usuario.Email == user.email, Usuario.Password == user.password).first()
    if not usuario:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    # ⚠️ Aquí deberías generar un JWT en producción
    token = f"fake-token-{usuario.Id}"
    return {"mensaje": "Login exitoso", "token": token}
