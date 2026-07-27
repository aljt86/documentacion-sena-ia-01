from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# 🔹 Usa la variable de entorno DATABASE_URL
DATABASE_URL =  DATABASE_URL = "postgresql://postgres:Alejo_1986@localhost:5432/documentos_sena_ia3_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 🔹 Crea las tablas en la base de datos
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
