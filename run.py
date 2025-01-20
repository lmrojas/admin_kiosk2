# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import create_app, db
import os
from config.logging_config import LoggingConfig

# Crear aplicación
app = create_app()

def init_db():
    """Inicializar base de datos"""
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    # Configurar logging
    LoggingConfig.configure_logging(app)
    
    # Inicializar base de datos
    init_db()
    
    # Obtener configuración del entorno
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Iniciar servidor
    app.run(
        host=host,
        port=port,
        debug=True
    ) 