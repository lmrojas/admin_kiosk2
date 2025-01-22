import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def clean_database():
    # Conexión directa a PostgreSQL
    connection = psycopg2.connect(
        dbname='admin_kiosk2',
        user='postgres',
        password='postgres',
        host='localhost',
        port='5432'
    )
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
        with connection.cursor() as cursor:
            # Desactivar restricciones de clave foránea temporalmente
            cursor.execute("SET session_replication_role = 'replica';")
            
            # Limpiar todas las tablas excepto users
            cursor.execute("TRUNCATE TABLE drift_metrics CASCADE;")
            cursor.execute("TRUNCATE TABLE model_metrics CASCADE;")
            cursor.execute("TRUNCATE TABLE prediction_logs CASCADE;")
            cursor.execute("TRUNCATE TABLE sensor_data CASCADE;")
            cursor.execute("TRUNCATE TABLE kiosks CASCADE;")
            
            # Borrar todos los usuarios excepto admin
            cursor.execute("DELETE FROM users WHERE username != 'admin';")
            
            # Reactivar restricciones de clave foránea
            cursor.execute("SET session_replication_role = 'origin';")
            
            print("Base de datos limpiada exitosamente. Solo se mantuvo el usuario admin.")
            
    except Exception as e:
        print(f"Error al limpiar la base de datos: {e}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == '__main__':
    clean_database() 