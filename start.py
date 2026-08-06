import subprocess
import os
import sys

# Aplicar migraciones de Alembic
subprocess.run(["alembic", "upgrade", "head"])

# Ejecutar Uvicorn
os.execvp("uvicorn", ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", os.getenv("PORT", "8000")])