from pydinamic import BaseModel

class DocumentoEscaneado(BaseModel):
    numero_documento: str
    nombre_completo: str
    fecha_nacimiento: str
    sexo: str
    lugar_nacimiento: str
    nacionalidad: str
    tipo_sangre: str
    programa: str
    