from api.db import engine

# Probar conexión
try:
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM usuarios LIMIT 1")
        print("Usuario de prueba:", result.fetchone())
except Exception as e:
    print("Error de conexión:", e)
