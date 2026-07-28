from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import os


def build_database_url(raw_url: str | None) -> str:
    if not raw_url:
        raise RuntimeError("DATABASE_URL no está configurada")

    url = raw_url.replace("postgres://", "postgresql://", 1)
    parsed = urlparse(url)

    if parsed.scheme in {"postgres", "postgresql"}:
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        host = parsed.hostname or ""
        is_local = host in {"localhost", "127.0.0.1", "::1"}

        if not any(key == "sslmode" for key, _ in query_items):
            query_items.append(("sslmode", "disable" if is_local else "require"))

        parsed = parsed._replace(query=urlencode(query_items))
        url = urlunparse(parsed)

    return url


# 🔹 Usa la variable de entorno DATABASE_URL
DATABASE_URL = build_database_url(os.getenv("DATABASE_URL"))

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 🔹 Crea las tablas en la base de datos si la conexión está disponible
try:
    Base.metadata.create_all(bind=engine)
except Exception as exc:
    print(f"Advertencia: no se pudieron crear las tablas automáticamente: {exc}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
