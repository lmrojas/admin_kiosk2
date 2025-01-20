"""
Script para esperar a que la base de datos esté lista.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
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