from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from api.db import Base

class Usuario(Base):
    __tablename__ = "usuarios"
    Id = Column(Integer, primary_key=True, index=True)  # coincide con Postgres
    Nombre = Column(String, nullable=True)
    Apellido = Column(String, nullable=True)
    Email = Column(String, unique=True, index=True)
    Password = Column(String)
    ConteoIngresos = Column(Integer, default=0)

    documentos = relationship("Documento", back_populates="usuario")


class Estudiante(Base):
    __tablename__ = "estudiante"
    Id = Column(Integer, primary_key=True, index=True)  # coincide con Postgres
    NumeroDocumento = Column(String, index=True)
    NombreCompleto = Column(String)
    FechaNacimiento = Column(String)
    Sexo = Column(String)
    LugarNacimiento = Column(String)
    Nacionalidad = Column(String)
    TipoSangre = Column(String)
    Programa = Column(String)
    UsuarioId = Column(Integer, ForeignKey("usuarios.Id"), nullable=False)
    documentos = relationship("Documento", back_populates="estudiante")


class Programa(Base):
    __tablename__ = "programa"
    id = Column(Integer, primary_key=True, index=True)  # minúscula en BD
    nombre = Column(String(200), nullable=False)
    documentos = relationship("Documento", back_populates="programa")


class Documento(Base):
    __tablename__ = "documento"
    Id = Column(Integer, primary_key=True, index=True)
    EstudianteId = Column(Integer, ForeignKey("estudiante.Id"), nullable=False)
    ProgramaId = Column(Integer, ForeignKey("programa.id"), nullable=False)
    UsuarioId = Column(Integer, ForeignKey("usuarios.Id"), nullable=False)
    TipoDocumento = Column(String(50))
    Archivo = Column(String(255), nullable=False)
    FechaSubida = Column(TIMESTAMP)
    usuario = relationship("Usuario", back_populates="documentos")
    estudiante = relationship("Estudiante", back_populates="documentos")
    programa = relationship("Programa", back_populates="documentos")
