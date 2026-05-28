import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 🔹 Usa la variable de entorno DATABASE_URL
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg2://usuario:clave@host.docker.internal:5432/mi_db"
    )

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
