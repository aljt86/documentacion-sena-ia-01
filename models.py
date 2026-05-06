from sqlalchemy import Column, Integer, String, DateTime, func
from db import Base

class Usuario(Base):
    __tablename__ = "Usuarios"

    Id = Column(Integer, primary_key=True, index=True)
    Nombre = Column(String(100), nullable=False)
    Apellido = Column(String(100), nullable=False)
    Email = Column(String(150), unique=True, nullable=False)
    FechaRegistro = Column(DateTime, server_default=func.now())
