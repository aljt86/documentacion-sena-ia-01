from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

server = "localhost"
database = "documentos_sena"
username = "tu_usuario"
password = "tu_contraseña"

connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"

engine = create_engine(conection_string)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally: 
        db.close()
