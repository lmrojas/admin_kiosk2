# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import os
import sys
from flask import Flask
from flask_migrate import Migrate
from app import create_app, db

def initialize_migrations():
    """Inicializar y aplicar migraciones de base de datos"""
    # Configurar la aplicación
    app = create_app()
    
    # Inicializar extensión de migración
    migrate = Migrate(app, db)
    
    # Cambiar al contexto de la aplicación
    with app.app_context():
        try:
            # Inicializar directorio de migraciones si no existe
            if not os.path.exists('migrations'):
                os.system('flask db init')
                print("✅ Directorio de migraciones inicializado")
            
            # Generar migración inicial
            os.system('flask db migrate -m "Migración inicial de modelos"')
            print("✅ Migración generada exitosamente")
            
            # Aplicar migración
            os.system('flask db upgrade')
            print("✅ Migración aplicada exitosamente")
            
        except Exception as e:
            print(f"❌ Error en la inicialización de migraciones: {e}")
            sys.exit(1)

if __name__ == '__main__':
    initialize_migrations() 