from sqlalchemy import Column, Integer, String
from api.db import Base

class Documento(Base):
    __tablename__ = "documentos"
    Id = Column(Integer, primary_key=True, index=True)
    NumeroDocumento = Column(String, unique=True, index=True)
    NombreCompleto = Column(String)
    FechaNacimiento = Column(String)
    Sexo = Column(String)
    LugarNacimiento = Column(String)
    Nacionalidad = Column(String)
    TipoSangre = Column(String)