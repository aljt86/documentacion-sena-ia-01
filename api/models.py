from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship 
from api.db import Base

class Usuario(Base):
    __tablename__ = "usuarios"
    Id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    Nombre = Column(String)
    Apellido = Column(String, nullable=True)
    Email = Column(String, unique=True, index=True)
    Password = Column(String)
    ConteoIngresos = Column(Integer, default=0)

    documentos = relationship("Documento", back_populates="usuario")

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
    Programa = Column(String)

    usuario = relationship("Usuario", back_populates="documentos")
    
