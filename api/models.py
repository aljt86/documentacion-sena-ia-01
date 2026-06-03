from sqlalchemy import Column, Integer, String, ForeignKey
from api.db import Base

class Usuario(Base):
    __tablename__ = "usuarios"
    Id = Column(Integer, primary_key=True, index=True)
    Nombre = Column(String)
    Apellido = Column(String)
    Email = Column(String, unique=True, index=True)
    Password = Column(String)

class Documento(Base):
    __tablename__ = "documentos"
    Id = Column(Integer, primary_key=True, index=True)
    UsuarioId = Column(Integer, ForeignKey("usuarios.Id"))
    NumeroDocumento = Column(String, unique=True, index=True)
    NombreCompleto = Column(String)
    FechaNacimiento = Column(String)
    Sexo = Column(String)
    LugarNacimiento = Column(String)
    Nacionalidad = Column(String)
    TipoSangre = Column(String)
    Programa = Column(String)
 