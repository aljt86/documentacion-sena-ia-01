import sys
import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field, EmailStr, root_validator
from dotenv import load_dotenv
from passlib.context import CryptContext
from app.extractor import procesar_pdf_hibrido

# cargar .env
load_dotenv()

# módulos del proyecto
from app.ocr_template import extract_fields
from utils import extraer_texto, detectar_tipo_documento, validar_datos
from app.ocr import procesar_pdf

# asegurar que el path al paquete api esté en sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import engine, Base, get_db, SessionLocal
from api.models import Usuario, Documento

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CREAR TABLAS (si no existen)
# ============================================
Base.metadata.create_all(bind=engine)

# ============================================
# VERIFICAR Y AGREGAR COLUMNA UsuarioId SI NO EXISTE
# ============================================
try:
    with engine.connect() as conn:
        # Verificar si la columna UsuarioId existe en la tabla documentos
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'documentos' AND column_name = 'UsuarioId'"
        ))
        if not result.fetchone():
            logger.warning("⚠️ Columna UsuarioId no encontrada en 'documentos'. Agregando...")
            conn.execute(text('ALTER TABLE documentos ADD COLUMN "UsuarioId" INTEGER'))
            conn.execute(text('ALTER TABLE documentos ALTER COLUMN "UsuarioId" SET NOT NULL'))
            conn.execute(text(
                'ALTER TABLE documentos ADD CONSTRAINT fk_documentos_usuarios '
                'FOREIGN KEY ("UsuarioId") REFERENCES usuarios("Id")'
            ))
            conn.commit()
            logger.info("✅ Columna UsuarioId agregada correctamente")
        else:
            logger.info("✅ Columna UsuarioId ya existe en 'documentos'")
except Exception as e:
    logger.error(f"⚠️ Error al verificar/agregar UsuarioId: {e}")

# ============================================
# APLICACIÓN FASTAPI
# ============================================
app = FastAPI(title="OCR Documentos Identidad 2.0")

# ============================================
# CONFIGURACIÓN CORS
# ============================================
allow_all_origins = os.getenv("CORS_ALLOW_ALL", "false").lower() in {"1", "true", "yes"}
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_origins,
    allow_origin_regex=None if allow_all_origins else allow_origin_regex,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# HASH DE CONTRASEÑAS
# ============================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ============================================
# ENDPOINT: HOME
# ============================================
@app.get("/")
def home():
    return {"mensaje": "API OCR 2.0 funcionando correctamente"}

# ============================================
# BACKGROUND OCR (crea su propia sesión DB)
# ============================================
def procesar_ocr_en_segundo_plano(file_path: str, programa: str, usuario_id: int):
    db = SessionLocal()
    try:
        logger.info(f"🔍 Procesando OCR para: {file_path}")
        datos = procesar_pdf_hibrido(file_path)
        logger.info(f"📊 Datos extraídos: {datos}")

        numero_doc = datos.get("numero_documento", "").strip()
        if not numero_doc:
            logger.warning("⚠️ OCR no capturó número de documento, no se insertará en BD")
            return

        nombre_completo = datos.get("nombre_completo") or f"{datos.get('apellidos','')} {datos.get('nombres','')}".strip()

        # Buscar estudiante existente
        estudiante = db.query(Estudiante).filter(Estudiante.NumeroDocumento == numero_doc).first()
        if estudiante:
            estudiante.NombreCompleto = nombre_completo
            estudiante.FechaNacimiento = datos.get("fecha_nacimiento", "")
            estudiante.Sexo = datos.get("sexo", "")
            estudiante.LugarNacimiento = datos.get("lugar_nacimiento", "")
            estudiante.Nacionalidad = datos.get("nacionalidad", "")
            estudiante.TipoSangre = datos.get("tipo_sangre", "")
            estudiante.Programa = programa
            db.commit()
            db.refresh(estudiante)
            logger.info(f"🔄 Estudiante actualizado en BD con ID: {estudiante.Id}")
        else:
            estudiante = Estudiante(
                NumeroDocumento=numero_doc,
                NombreCompleto=nombre_completo,
                FechaNacimiento=datos.get("fecha_nacimiento", ""),
                Sexo=datos.get("sexo", ""),
                LugarNacimiento=datos.get("lugar_nacimiento", ""),
                Nacionalidad=datos.get("nacionalidad", ""),
                TipoSangre=datos.get("tipo_sangre", ""),
                Programa=programa,
                UsuarioId=usuario_id
            )
            db.add(estudiante)
            db.commit()
            db.refresh(estudiante)
            logger.info(f"✅ Estudiante guardado en BD con ID: {estudiante.Id}")

        # Crear documento asociado
        nuevo_doc = Documento(
            EstudianteId=estudiante.Id,
            ProgramaId=estudiante.ProgramaId if hasattr(estudiante, "ProgramaId") else 1,  # ajusta según tu lógica
            UsuarioId=usuario_id,
            TipoDocumento="PDF",
            Archivo=file_path
        )
        db.add(nuevo_doc)
        db.commit()
        db.refresh(nuevo_doc)
        logger.info(f"✅ Documento guardado en BD con ID: {nuevo_doc.Id}")

    except Exception as e:
        logger.error(f"❌ Error en OCR en segundo plano: {e}")
        db.rollback()
    finally:
        db.close()


# ============================================
# MODELOS PYDANTIC PARA REGISTRO Y LOGIN
# ============================================
class UserRegister(BaseModel):
    nombre: Optional[str] = Field(None, alias="name")
    apellido: Optional[str] = Field(None, alias="lastName")
    email: EmailStr
    password: str

    @root_validator(pre=True)
    def normalize_fields(cls, values):
        if "name" in values and "nombre" not in values:
            values["nombre"] = values.pop("name")
        if "lastName" in values and "apellido" not in values:
            values["apellido"] = values.pop("lastName")
        return values

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# ============================================
# ENDPOINT: REGISTRO
# ============================================
@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    logger.info(f"📝 Registro intentado para: {user.email}")
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
        logger.info(f"✅ Usuario registrado: ID {nuevo.Id}, Email {nuevo.Email}")
        return {"mensaje": "Usuario registrado correctamente", "usuario_id": nuevo.Id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en registro: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ============================================
# ENDPOINT: LOGIN
# ============================================
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    logger.info(f"🔐 Login intentado: {user.email}")
    try:
        usuario = db.query(Usuario).filter(Usuario.Email == user.email).first()
        if not usuario:
            logger.warning(f"⚠️ Usuario no encontrado: {user.email}")
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

        if not verify_password(user.password, usuario.Password):
            logger.warning(f"⚠️ Contraseña incorrecta para: {user.email}")
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

        usuario.ConteoIngresos = (usuario.ConteoIngresos or 0) + 1
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
        logger.error(f"❌ Error en login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ============================================
# ENDPOINT: SUBIR DOCUMENTO (OCR)
# ============================================
@app.post("/ocr/upload/")
async def ocr_upload(
    file: UploadFile = File(...),
    programa: str = Form(...),
    modelo: str = Form("hologramas"),
    usuario_id: int = Form(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()  # ✅ CORREGIDO
):
    try:
        # Validar usuario
        usuario = db.query(Usuario).filter(Usuario.Id == usuario_id).first()
        if not usuario:
            raise HTTPException(status_code=400, detail="Usuario no existe")

        # Guardar archivo en disco
        base_dir = os.path.dirname(os.path.abspath(__file__))
        programa_dir = os.path.join(base_dir, "documentos", programa.replace(" ", "_"))
        os.makedirs(programa_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{usuario_id}_{timestamp}_{file.filename}"
        file_path = os.path.join(programa_dir, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"📂 Archivo guardado en: {file_path}")

        # Lanzar OCR en segundo plano (sin pasar db)
        background_tasks.add_task(procesar_ocr_en_segundo_plano, file_path, programa, usuario_id)

        return {
            "mensaje": "Documento recibido y guardado correctamente. El procesamiento continuará en segundo plano.",
            "archivo": file_path,
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"❌ Error al subir documento: {e}")
        raise HTTPException(status_code=500, detail=f"Error al subir documento: {str(e)}")