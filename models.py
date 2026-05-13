from sqlalchemy import Column, Integer, String
from sqalchemy.ext.declarative import declarative_base

Base = declarative_base()
class Usuario(Base):
    __tablename__ = "usuarios"

    Id = Column(Integer, primary_key=True, index=True)
    Nombre = Column(String, nullable=False)
    Apellido = Column(String, nullable=True)
    Email = Column(String, unique=True, index=True, nullable=False)
    Password = Column(String, nullable=False)
