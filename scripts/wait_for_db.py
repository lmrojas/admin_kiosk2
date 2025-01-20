"""
Script de utilidad para esperar a que la base de datos esté disponible.

Funcionalidad:
- Intenta conectarse a la base de datos repetidamente
- Útil para contenedores Docker y despliegues
- Evita errores de inicio por base de datos no disponible
- Configurable con variables de entorno

Uso:
python scripts/wait_for_db.py

Variables de entorno:
- DB_HOST: Host de la base de datos
- DB_PORT: Puerto de la base de datos
- MAX_RETRIES: Número máximo de intentos
"""

import os
import time
import psycopg2
from psycopg2 import OperationalError

def wait_for_database():
    """Espera hasta que la base de datos esté disponible."""
    max_retries = 30
    retry_interval = 2

    db_params = {
        'dbname': os.getenv('POSTGRES_DB', 'kiosk'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432')
    }

    for attempt in range(max_retries):
        try:
            print(f"Intento {attempt + 1} de {max_retries} para conectar a la base de datos...")
            conn = psycopg2.connect(**db_params)
            conn.close()
            print("¡Base de datos lista!")
            return True
        except OperationalError as e:
            print(f"La base de datos no está lista (Error: {e})")
            if attempt < max_retries - 1:
                print(f"Esperando {retry_interval} segundos antes del siguiente intento...")
                time.sleep(retry_interval)
            else:
                print("Se alcanzó el número máximo de intentos")
                raise

if __name__ == '__main__':
    wait_for_database() 