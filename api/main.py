import sys
import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field, EmailStr, root_validator
from dotenv import load_dotenv
from passlib.context import CryptContext
from app.parser import procesar_documento 
# from app.extractor import procesar_pdf_hibrido #

# cargar .env
load_dotenv()

# módulos del proyecto
from app.ocr_template import extract_fields
from utils import extraer_texto, detectar_tipo_documento, validar_datos
from app.ocr import procesar_pdf 

# asegurar que el path al paquete api esté en sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import engine, Base, get_db, SessionLocal
from api.models import (
    Usuario, 
    Documento, 
    Estudiante,
    Programa
) 

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
# APLICACIГ“N FASTAPI
# ============================================
app = FastAPI(title="OCR Documentos Identidad 2.0")

# ============================================
# CONFIGURACIГ“N CORS
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
# HASH DE CONTRASEГ‘AS
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
# ENDPOINT: DESCARGAR CROPS DEL OCR
# ============================================
@app.get("/ocr/debug/{programa}/{filename:path}")
def descargar_crop(programa: str, filename: str):
    """
    Permite acceder directamente desde el navegador
    a los crops generados por el OCR.

    Acepta rutas como:

    /ocr/debug/desarrollador_software/pagina_01_apellidos_original.png

    y también:

    /ocr/debug/desarrollador_software/ocr_debug/pagina_01_apellidos_original.png
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --------------------------------------------------------
    # Carpeta raíz permitida
    # --------------------------------------------------------

    programa_dir = os.path.join(
        base_dir,
        "documentos",
        programa
    )

    programa_real = os.path.realpath(programa_dir)

    # --------------------------------------------------------
    # Normalizar la ruta recibida
    # --------------------------------------------------------

    filename = filename.replace("\\", "/").lstrip("/")

    # Si la URL contiene "ocr_debug/", lo eliminamos porque
    # físicamente esa carpeta ya se agrega abajo.
    if filename.startswith("ocr_debug/"):
        filename = filename[len("ocr_debug/"):]

    # --------------------------------------------------------
    # Directorio real de crops
    # --------------------------------------------------------

    debug_dir = os.path.join(
        programa_dir,
        "ocr_debug"
    )

    debug_real = os.path.realpath(debug_dir)

    # --------------------------------------------------------
    # Archivo solicitado
    # --------------------------------------------------------

    file_path = os.path.join(
        debug_dir,
        filename
    )

    file_real = os.path.realpath(file_path)

    # --------------------------------------------------------
    # SEGURIDAD
    # Impedir salir de ocr_debug mediante ../
    # --------------------------------------------------------

    if not file_real.startswith(
        debug_real + os.sep
    ):
        logger.warning(
            "OCR_DEBUG_ACCESO_DENEGADO | "
            "programa=%s | archivo=%s",
            programa,
            filename
        )

        raise HTTPException(
            status_code=403,
            detail="Ruta no permitida"
        )

    # --------------------------------------------------------
    # Verificar existencia
    # --------------------------------------------------------

    if not os.path.isfile(file_real):

        logger.warning(
            "OCR_DEBUG_NO_ENCONTRADO | "
            "programa=%s | archivo=%s | ruta=%s",
            programa,
            filename,
            file_real
        )

        raise HTTPException(
            status_code=404,
            detail="Crop no encontrado"
        )

    # --------------------------------------------------------
    # LOG DE DESCARGA
    # --------------------------------------------------------

    logger.info(
        "OCR_DEBUG_DESCARGA | "
        "programa=%s | archivo=%s | ruta=%s",
        programa,
        os.path.basename(file_real),
        file_real
    )

    # --------------------------------------------------------
    # ENTREGAR IMAGEN
    # --------------------------------------------------------

    return FileResponse(
        file_real,
        media_type="image/png",
        filename=os.path.basename(file_real),
        headers={
            "Content-Disposition": (
                f'attachment; '
                f'filename="{os.path.basename(file_real)}"'
            )
        }
    )

import re

def normalizar_programa(programa: str) -> str:
    programa = programa.strip()

    # Espacios -> _
    programa = re.sub(r"\s+", "_", programa)

    # Solo letras, números, _, - y.
    programa = re.sub(r"[^a-zA-Z0-9_.-]", "", programa)

    # Evitar nombres problemáticos
    programa = programa.strip(".")

    if not programa:
        raise ValueError("Nombre de programa invalido")
    
    return programa

# ============================================
# BACKGROUND OCR (crea su propia sesiГіn DB)
# ============================================
def procesar_ocr_en_segundo_plano(file_path: str, programa: str, usuario_id: int):
    db = SessionLocal()

    try:
        logger.info("🔍 Procesando OCR para: %s", file_path)
        logger.info("=== OCR PRODUCCION: INICIO ===")
        logger.info("Archivo: %s", file_path)
        logger.info("Programa: %s | UsuarioId: %s", programa, usuario_id)

        # ==================================================
        # OCR
        # ==================================================

        resultado_ocr = procesar_documento(file_path)
        logger.info("OCR_RESULTADO: %s", resultado_ocr)

        if resultado_ocr.get("error"):
            logger.error("OCR_ERROR: %s", resultado_ocr["error"])
            return

        datos = resultado_ocr.get("resultado", {})
        logger.info("OCR_METODO: %s", resultado_ocr.get("metodo"))
        logger.info("OCR_DATOS: %s", datos)

        # ==================================================
        # NORMALIZAR TIPO DE SANGRE
        # ==================================================

        if (
            not datos.get("tipo_sangre") and datos.get("rh")
        ):  
            datos["tipo_sangre"] = datos["rh"]

        # ==================================================
        # DATOS OCR
        # ==================================================

        numero_doc = (datos.get("numero_documento") or "").strip()
        apellidos = (datos.get("apellidos") or "").strip()
        nombres = (datos.get("nombres") or "").strip()
        nombre_completo = (datos.get("nombre_completo") or f"{apellidos} {nombres}".strip())
        fecha_nacimiento = (datos.get("fecha_nacimiento") or "")
        sexo = (datos.get("sexo") or "")
        lugar_nacimiento = (datos.get("lugar_nacimiento") or "")
        nacionalidad = (datos.get("nacionalidad") or "")
        tipo_sangre = (datos.get("tipo_sangre") or "")

        logger.info(
            "OCR_CAMPOS | numero=%r | nombre=%r | fecha=%r | sexo=%r | lugar=%r | nacionalidad=%r | rh=%r",
            numero_doc,
            nombre_completo,
            fecha_nacimiento,
            sexo,
            lugar_nacimiento,
            nacionalidad,
            tipo_sangre
        )

        # ==================================================
        # VALIDAR NUMERO DE DOCUMENTO
        # ==================================================
                
        if numero_doc:

            numero_doc = "".join(
                caracter 
                for caracter in numero_doc
                if caracter.isdigit()
            )

            if not (
                6 <= len(numero_doc) <= 10
            ):

                logger.warning(
                    "OCR_NUMERO_DOCUMENTO_INVALIDO | "
                    "valor=%s | longitud=%s",
                    numero_doc,
                    len(numero_doc)
                )

                numero_doc = None

        else:

            logger.warning(
                "OCR_NUMERO_DOCUMENTO_NO_CAPTURADO | "
                "Se continuará procesando los demás campos"
            )

            numero_doc = None 
          
        # ==================================================
        # VALIDAR USUARIO
        # ==================================================

        usuario = (db.query(Usuario).filter(Usuario.Id == usuario_id).first())

        if not usuario:
            logger.error(
                "USUARIO_NO_EXISTE | UsuarioId=%s",
                usuario_id
            )
            return

        # ==================================================
        # BUSCAR ESTUDIANTE
        # ==================================================

        estudiante = None

        if numero_doc:
            estudiante = (db.query(Estudiante).filter(Estudiante.NumeroDocumento == numero_doc).first())

        # ==================================================
        # CREAR O ACTUALIZAR ESTUDIANTE
        # ==================================================

        if estudiante:

            logger.info(
                "ESTUDIANTE_EXISTENTE | "
                "Id=%s | NumeroDocumento=%s",
                estudiante.Id,
                numero_doc
            )

            estudiante.NombreCompleto = (
                nombre_completo
                or estudiante.NombreCompleto
            )

            estudiante.FechaNacimiento = (
                fecha_nacimiento
                or estudiante.FechaNacimiento
            )

            estudiante.Sexo = (
                sexo
                or estudiante.Sexo
            )

            estudiante.LugarNacimiento = (
                lugar_nacimiento
                or estudiante.LugarNacimiento
            )

            estudiante.Nacionalidad = (
                nacionalidad
                or estudiante.Nacionalidad
            )

            estudiante.TipoSangre = (
                tipo_sangre
                or estudiante.TipoSangre
            )

            estudiante.Programa = programa

            # IMPORTANTE:
            # UsuarioId aquí representa quién registró/
            # procesó originalmente al estudiante.
            # No modificamos el UsuarioId de un estudiante
            # existente solamente porque otro usuario
            # suba nuevamente su documento.

        else:

            # ==================================================
            # NO CREAR ESTUDIANTE SIN NUMERO DE DOCUMENTO
            # ==================================================

            if not numero_doc:

                logger.error(
                    "ESTUDIANTE_NO_CREADO | "
                    "OCR no obtuvo un numero de documento valido | "
                    "Nombre=%r | Programa=%r",
                    nombre_completo,
                    programa 
                )

                return

            estudiante = Estudiante(
                NumeroDocumento=numero_doc,
                NombreCompleto=nombre_completo,
                FechaNacimiento=fecha_nacimiento,
                Sexo=sexo,
                LugarNacimiento=lugar_nacimiento,
                Nacionalidad=nacionalidad,
                TipoSangre=tipo_sangre,
                Programa=programa,
                UsuarioId=usuario_id
            )

            db.add(estudiante)

            db.flush()

            logger.info(
                "ESTUDIANTE_CREADO | "
                "Id=%s | NumeroDocumento=%s",
                estudiante.Id,
                numero_doc
            )

        # ==================================================
        # BUSCAR PROGRAMA
        # ==================================================

        programa_db = (db.query(Programa).filter(Programa.nombre == programa).first())

        # ==================================================
        # CREAR PROGRAMA SI NO EXISTE
        # ==================================================

        if not programa_db:

            programa_db = Programa(
                nombre=programa
            )

            db.add(programa_db)

            db.flush()

            logger.info(
                "PROGRAMA_CREADO | Id=%s | nombre=%s",
                programa_db.id,
                programa
            )

        # ==================================================
        # CREAR DOCUMENTO
        # ==================================================

        nuevo_documento = Documento(
            EstudianteId=estudiante.Id,
            ProgramaId=programa_db.id,
            UsuarioId=usuario_id,
            TipoDocumento="Cedula",
            Archivo=file_path,
            FechaSubida=datetime.now()
        )

        db.add(nuevo_documento)

        # ==================================================
        # GUARDAR TODO
        # ==================================================

        logger.info("== INICIANDO GUARDADOEN POSTGRESQL ==")

        # --------------------------------------------------
        # FLUSH
        # Envía los INSERT a PostgreSQL sin cerrar
        # todavía la transacción.
        # --------------------------------------------------

        db.flush()

        logger.info("POSTGRESQL_FLUSH_OK | EstudianteId=%s | ProgramaId=%s | UsuarioId=%s | DocumentoId=%s", estudiante.Id, programa_db.id, usuario_id, nuevo_documento.Id)

        # --------------------------------------------------
        # COMMIT
        # Confirma definitivamente la transacción.
        # --------------------------------------------------

        db.commit()

        logger.info(
            "POSTGRESQL_COMMIT_OK | "
            "Transacción confirmada correctamente"
        )

        # ==================================================
        # REFRESCAR OBJETOS DESDE POSTGRESQL
        # ==================================================

        db.refresh(estudiante)
        db.refresh(programa_db)
        db.refresh(nuevo_documento)

        logger.info(
            "POSTGRESQL_REFRESH_OK | "
            "Los objetos fueron actualizados desde PostgresSQL"
        )

        # ==================================================
        # VERIFICAR TABLA ESTUDIANTE
        # ==================================================

        estudiante_bd = (db.query(Estudiante).filter(Estudiante.Id == estudiante.Id).first())

        if estudiante_bd:
            logger.info(
                "ESTUDIANTE_VERIFICADO | "
                "Id=%s | " 
                "NumeroDocumento=%s | "
                "NombreCompleto=%s | "
                "FechaNacimiento=%s | "
                "Sexo=%s | "
                "LugarNacimiento=%s | "
                "Nacionalidad=%s | "
                "TipoSangre=%s | "
                "Programa=%s | "
                "UsuarioId=%s",
                estudiante_bd.Id,
                estudiante_bd.NumeroDocumento,
                estudiante_bd.NombreCompleto,
                estudiante_bd.FechaNacimiento,
                estudiante_bd.Sexo,
                estudiante_bd.LugarNacimiento,
                estudiante_bd.Nacionalidad,
                estudiante_bd.TipoSangre,
                estudiante_bd.Programa,
                estudiante_bd.UsuarioId
            )
        else:
            logger.error(
                "POSTGRESQL_ERROR_ESTUDIANTE_NO_ENCONTRADO | "
                "EstudianteId=%s",
                estudiante.Id,
            )

        # ==================================================
        # VERIFICAR TABLA PROGRAMA
        # ==================================================

        programa_bd = (
            db.query(Programa)
            .filter(
                Programa.id == programa_db.id
            )
            .first()
        )

        if programa_bd:

            logger.info(
                "POSTGRESQL_PROGRAMA_INSERTADO | "
                "Id=%s | "
                "Nombre=%r",
                programa_bd.id,
                programa_bd.nombre
            )

        else:

            logger.error(
                "POSTGRESQL_ERROR_PROGRAMA_NO_ENCONTRADO | "
                "ProgramaId=%s",
                programa_db.id
            )

        # ==================================================
        # VERIFICAR TABLA DOCUMENTO
        # ==================================================

        documento_bd = (
            db.query(Documento)
            .filter(
                Documento.Id == nuevo_documento.Id
            )
            .first()
        )

        if documento_bd:

            logger.info(
                "POSTGRESQL_DOCUMENTO_INSERTADO | "
                "DocumentoId=%s | "
                "EstudianteId=%s | "
                "ProgramaId=%s | "
                "UsuarioId=%s | "
                "TipoDocumento=%r | "
                "Archivo=%r",
                documento_bd.Id,
                documento_bd.EstudianteId,
                documento_bd.ProgramaId,
                documento_bd.UsuarioId,
                documento_bd.TipoDocumento,
                documento_bd.Archivo
            )

        else:

            logger.error(
                "POSTGRESQL_ERROR_DOCUMENTO_NO_ENCONTRADO | "
                "DocumentoId=%s",
                nuevo_documento.Id
            )

        # ==================================================
        # CONFIRMACIÓN FINAL
        # ==================================================

        if (
            estudiante_bd
            and programa_bd
            and documento_bd
        ):

            logger.info(
                "=================================================="
            )

            logger.info(
                "POSTGRESQL_GUARDADO_OK | "
                "LOS DATOS FUERON GUARDADOS Y RECUPERADOS"
            )

            logger.info(
                "ESTUDIANTE | "
                "Id=%s | "
                "NumeroDocumento=%r | "
                "Nombre=%r | "
                "Sexo=%r | "
                "TipoSangre=%r",
                estudiante_bd.Id,
                estudiante_bd.NumeroDocumento,
                estudiante_bd.NombreCompleto,
                estudiante_bd.Sexo,
                estudiante_bd.TipoSangre
            )

            logger.info(
                "PROGRAMA | "
                "Id=%s | "
                "Nombre=%r",
                programa_bd.id,
                programa_bd.nombre
            )

            logger.info(
                "DOCUMENTO | "
                "Id=%s | "
                "EstudianteId=%s | "
                "ProgramaId=%s | "
                "UsuarioId=%s",
                documento_bd.Id,
                documento_bd.EstudianteId,
                documento_bd.ProgramaId,
                documento_bd.UsuarioId
            )

            logger.info(
                "=================================================="
            )

        else:

            logger.error(
                "POSTGRESQL_GUARDADO_ERROR | "
                "No fue posible verificar uno o más registros"
            )

        # ==================================================
        # FIN DEL PROCESAMIENTO
        # ==================================================
     
        logger.info(
            "=== OCR PRODUCCION: FIN ==="
        )
    
    except Exception as e:

        db.rollback()

        logger.exception(
            "❌ ERROR_OCR_POSTGRESQL | "
            "Error en OCR en segundo plano: %s",
            e
        )

    finally:

        db.close()

        logger.info(
            "POSTGRESQL_SESION_CERRADA | "
            "UsuarioId=%s | Programa=%s",
            usuario_id,
            programa
        )

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
    logger.info(f"рџ“ќ Registro intentado para: {user.email}")
    try:
        existing = db.query(Usuario).filter(Usuario.Email == user.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="El correo ya estГЎ registrado")

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
       logger.error(f"вќЊ Error en registro: {str(e)}")
       raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ============================================
# ENDPOINT: LOGIN
# ============================================
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    logger.info(f"рџ”ђ Login intentado: {user.email}")
    try:
        usuario = db.query(Usuario).filter(Usuario.Email == user.email).first()
        if not usuario:
            logger.warning("Usuario no encontrado: %s, user.email")
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

        if not verify_password(user.password, usuario.Password):
           logger.warning("ContraseГ±a incorrecta para: %s", user.email)
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

        db.roLlback()
        logger.exception("ERROR_LOGIN | %s", e)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor")

# ============================================
# ENDPOINT: SUBIR DOCUMENTO (OCR)
# ============================================
@app.post("/ocr/upload/")
async def ocr_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    programa: str = Form(...),
    modelo: str = Form("hologramas"),
    usuario_id: int = Form(...),
    db: Session = Depends(get_db),
   ):
    try:
        # ==================================================
        # Validar usuario
        # ==================================================

        if not file.filename:
            raise HTTPException(status_code=400, detail="No se recibio un nombre de archivo")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
        # ==================================================
        # Validar usuario
        # ==================================================

        usuario = (db.query(Usuario).filter(Usuario.Id == usuario_id).first())
        if not usuario:
            raise HTTPException(status_code=400, detail="Usuario no existe")

        # ==================================================
        # Normalizar programa
        # ==================================================

        programa_nombre = normalizar_programa(programa)

        # ==================================================
        # Guardar en disco
        # ==================================================

        base_dir = os.path.dirname(os.path.abspath(__file__))
        programa_dir = os.path.join(base_dir, "documentos", programa_nombre)
        os.makedirs(programa_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_original = os.path.basename(file.filename)
        filename = f"{usuario_id}_{timestamp}_{nombre_original}"
        file_path = os.path.join(programa_dir, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(
            "ARCHIVO_GUARDADO | ruta=%s",
            file_path
        )

        # ==================================================
        # PROGRAMAR OCR 
        # ==================================================

        background_tasks.add_task(
           procesar_ocr_en_segundo_plano,
           file_path,
           programa_nombre,
           usuario_id
        )
        logger.info(
            "OCR_BACKGROUND_SCHEDULED | archivo=%s", 
            file_path
        )

        return {
           "mensaje": (
               "Documento recibido y guardado correctamente. "
               "El procesamiento continuarГЎ en segundo plano."
            ),
            "archivo": file_path,
            "status": "processing"
        }
    except HTTPException:
        raise

    except Exception as e:

        logger.exception("ERRO_UPLOAD | %s", e)
        raise HTTPException(status_code=500, detail="Error al subir documento")