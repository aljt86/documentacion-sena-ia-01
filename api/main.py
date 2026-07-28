import sys
import os
import logging
import bcrypt
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, EmailStr, root_validator
from utils import extraer_texto, detectar_tipo_documento, validar_datos
from app.ocr import procesar_pdf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import engine, Base, get_db
from api.models import Usuario, Documento 
from app.ocr_template import extract_fields

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(level=logging.INFO)

# ============================================
# HASH DE CONTRASEÑAS (bcrypt directo)
# ============================================
def hash_password(password: str) -> str:
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)

Base.metadata.create_all(bind=engine)

# ============================================
# APLICACIÓN FASTAPI
# ============================================
app = FastAPI(title="OCR Documentos Identidad 2.0")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

allow_origin_regex = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"https://.*\.onrender\.com|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?",
)

# Si se necesita abrir a cualquier frontend en un despliegue de prueba,
# puede utilizar CORS_ALLOW_ALL=true y la app responderá dinámicamente a todos los orígenes.
allow_all_origins = os.getenv("CORS_ALLOW_ALL", "false").lower() in {"1", "true", "yes"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_origins,
    allow_origin_regex=None if allow_all_origins else allow_origin_regex,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"mensaje": "API OCR 2.0 funcionando correctamente"}

# ============================================
# FUNCIÓN DE OCR EN SEGUNDO PLANO
# ============================================
def procesar_ocr_en_segundo_plano(file_path: str, programa: str, usuario_id: int, db: Session):
    """
    Función que se ejecuta en segundo plano para procesar el OCR.
    """
    try:
        logging.info(f"📂 Procesando OCR para: {file_path}")
        
        # Extraer datos del PDF
        datos = extract_fields(file_path, modelo="hologramas")
        logging.info(f"🔍 Datos extraídos: {datos}")
        
        # Construir nombre completo
        nombre_completo = datos.get("nombre_completo", "")
        if not nombre_completo:
            apellidos = datos.get("apellidos", "")
            nombres = datos.get("nombres", "")
            nombre_completo = f"{apellidos} {nombres}".strip()
        
        # Guardar en la base de datos
        nuevo_doc = Documento(
            UsuarioId=usuario_id,
            NumeroDocumento=datos.get("numero_documento", ""),
            NombreCompleto=nombre_completo,
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
        
        logging.info(f"✅ Documento guardado en BD con ID: {nuevo_doc.Id}")
        
    except Exception as e:
        logging.error(f"❌ Error en OCR en segundo plano: {e}")
        db.rollback()

# ============================================
# OCR UPLOAD (PROCESAMIENTO ASÍNCRONO)
# ============================================
@app.post("/ocr/upload/")
async def ocr_upload(
    file: UploadFile = File(...),
    programa: str = Form(...),
    modelo: str = Form("hologramas"),
    usuario_id: int = Form(...), 
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Endpoint para subir documentos.
    - Guarda el archivo inmediatamente.
    - Devuelve confirmación al frontend.
    - Procesa el OCR en segundo plano.
    """
    try:
        # 1. Guardar el archivo en disco
        base_dir = os.path.dirname(os.path.abspath(__file__))
        programa_dir = os.path.join(base_dir, "documentos", programa.replace(" ", "_"))
        os.makedirs(programa_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{usuario_id}_{timestamp}_{file.filename}"
        file_path = os.path.join(programa_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        logging.info(f"📂 Archivo guardado en: {file_path}")
        
        # 2. Agregar el OCR a tarea en segundo plano
        background_tasks.add_task(procesar_ocr_en_segundo_plano, file_path, programa, usuario_id, db)
        
        # 3. Devolver confirmación INMEDIATA
        return {
            "mensaje": "Documento recibido y guardado correctamente. El procesamiento continuará en segundo plano.",
            "archivo": file_path,
            "status": "processing"
        }
        
    except Exception as e:
        logging.error(f"❌ Error al subir documento: {e}")
        raise HTTPException(status_code=500, detail=f"Error al subir documento: {str(e)}")

# ============================================
# REGISTRO DE USUARIOS
# ============================================
class UserRegister(BaseModel):
    nombre: str | None = Field(None, alias="name")
    apellido: str | None = Field(None, alias="lastName")
    email: EmailStr
    password: str

    @root_validator(pre=True)
    def normalize_fields(cls, values):
        # Acepta payloads del frontend con name/lastName o nombre/apellido
        if "name" in values and "nombre" not in values:
            values["nombre"] = values.pop("name")
        if "lastName" in values and "apellido" not in values:
            values["apellido"] = values.pop("lastName")
        return values

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    logging.info(f"📝 Registro intentado para: {user.email}")
    try:
        existing = db.query(Usuario).filter(Usuario.Email == user.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="El correo ya está registrado")
        
        nuevo = Usuario(
            Nombre=user.nombre or "",
            Apellido=user.apellido or "",
            Email=user.email,
            Password=hash_password(user.password),
            ConteoIngresos=0
        )
        db.add(nuevo)
        db.commit()
        db.refresh(nuevo)
        logging.info(f"✅ Usuario registrado: ID {nuevo.Id}, Email {nuevo.Email}")
        return {"mensaje": "Usuario registrado correctamente", "usuario_id": nuevo.Id}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Error en registro: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ============================================
# LOGIN DE USUARIOS
# ============================================
class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    logging.info(f"🔐 Login intentado: {user.email}")
    try:
        usuario = db.query(Usuario).filter(Usuario.Email == user.email).first()
        if not usuario:
            logging.warning(f"⚠️ Usuario no encontrado: {user.email}")
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        
        if not verify_password(user.password, usuario.Password):
            logging.warning(f"⚠️ Contraseña incorrecta para: {user.email}")
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        
        usuario.ConteoIngresos += 1
        db.commit()
        db.refresh(usuario)
        
        token = f"fake-token-{usuario.Id}"
        return {
            "mensaje": "Login exitoso",
            "token": token,
            "usuario_id": usuario.Id,
            "conteo_ingresos": usuario.ConteoIngresos
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Error en login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

