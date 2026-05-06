from db import engine

# Probar conexión
try:
    with engine.connect() as conn:
        result = conn.execute("SELECT TOP 1 * FROM Usuarios")  # Reemplaza 'Usuarios' con el nombre de una tabla en tu base de datos
        print("Usuario de prueba:", result.fetchone())
except Exception as e:
    print("Error de conexión:", e)
