
import sys
import os 
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from utils import extraer_texto, detectar_tipo_documento, validar_datos
from app.ocr import procesar_pdf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import engine, Base, get_db
from api.models import Usuario, Documento 
from app.ocr_template import extract_fields
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

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
async def ocr_upload(
    file: UploadFile = File(...),
    programa: str = Form(...),
    modelo: str = Form("hologramas"),
    usuario_id: int = Form(...), 
    db: Session = Depends(get_db)
):
    temp_path = os.path.join("data", "tmp", file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    datos = extract_fields(temp_path, modelo=modelo)
    print("DEBUG OCR", datos)

    tipo_doc = detectar_tipo_documento(datos)
    validaciones = validar_datos(datos, tipo_doc)

    if not all(validaciones.values()):
        raise HTTPException(
            status_code=400,
            detail={"error": "Datos extraídos no válidos", "validaciones": validaciones}
        )

    nuevo_doc = Documento(
        UsuarioId=usuario_id,
        NumeroDocumento=datos.get("numero_documento", ""),
        NombreCompleto=datos.get("nombre_completo", datos.get("apellidos", "") + " " + datos.get("nombres", "")),
        FechaNacimiento=datos.get("fecha_nacimiento", ""),
        Sexo=datos.get("sexo", ""),
        LugarNacimiento=datos.get("lugar_nacimiento", ""),
        Nacionalidad=datos.get("nacionalidad", ""),
        TipoSangre=datos.get("tipo_sangre", ""),
        Programa=programa
    )
    db.add(nuevo_doc)
    db.commit()
    db.refresh(nuevo_doc)

    return {
        "mensaje": "Documento guardado desde OCR",
        "documento_id": nuevo_doc.Id,
        "datos_extraidos": datos,
        "tipo_documento": tipo_doc,
        "programa": programa
    }

# ---------------------------
# 📌 Registro de usuarios
# ---------------------------
class UserRegister(BaseModel):
    nombre: str
    email: str
    password: str
    apellido: str | None = None   # opcional

@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(Usuario).filter(Usuario.Email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="El correo ya está registrado")

    nuevo = Usuario(
        Nombre=user.nombre, 
        Apellido=user.apellido if user.apellido else "", 
        Email=user.email, 
        Password=hash_password(user.password),
        ConteoIngresos=0
    )
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
        Usuario.Password == hash_password(user.password)
    ).first()
    if not usuario:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    usuario.ConteoIngresos += 1
    db.commit()
    db.refresh(usuario)
    
    # ⚠️ Aquí deberías generar un JWT en producción
    token = f"fake-token-{usuario.Id}"
    return {"mensaje": "Login exitoso", 
            "token": token,
            "usuario_id": usuario.Id,
            "conteo_ingresos": usuario.ConteoIngresos
            }


