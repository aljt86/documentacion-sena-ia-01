
import sys
import os
import logging  
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

logging.basicConfig(level=logging.INFO) 
Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _truncate_password(password: str) -> str:
    encoded = password.encode('utf-8')
    if len(encoded) > 72:
        return encoded[:72].decode('utf-8')
    return password

def hash_password(password: str) -> str:
    return pwd_context.hash(_truncate_password(password))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(_truncate_password(plain_password), hashed_password)


app = FastAPI(title="OCR Documentos Identidad 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sna-2-0-3.onrender.com",
                   "https://localhost:3000",
                   "http://127.0.0.1:3000"
    ],  # puedes poner aquí el dominio de tu frontend
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
    logging.info("DEBUG OCR: %s", datos)

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
    logging.info(f"📝 Registro intentado para: {user.email}")
    try:
        # Verificar si el correo ya existe
        existing = db.query(Usuario).filter(Usuario.Email == user.email).first()
        if existing:
            logging.warning(f"⚠️ Correo ya registrado: {user.email}")
            raise HTTPException(status_code=400, detail="El correo ya está registrado")

        # Crear nuevo usuario
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
        logging.info(f"✅ Usuario registrado: ID {nuevo.Id}, Email {nuevo.Email}")
        return {"mensaje": "Usuario registrado correctamente", "usuario_id": nuevo.Id}

    except HTTPException as e:
        # Relanzar excepciones HTTP (400, 401, etc.)
        raise e
    except Exception as e:
        # Capturar cualquier otro error (500)
        logging.error(f"❌ Error en registro: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ---------------------------
# 📌 Login de usuarios
# ---------------------------
class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    logging.info(f"🔐 Login intentado: {user.email}")
    usuario = db.query(Usuario).filter(Usuario.Email == user.email).first()
    
    if not usuario or not verify_password(user.password, usuario.Password):
        logging.warning(f"⚠️ Credenciales inválidas para: {user.email}")
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


