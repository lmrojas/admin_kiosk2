"""
Script principal para ejecutar la aplicación Flask.
Configura el servidor eventlet para WebSocket y ejecuta la aplicación.
"""

# Monkey patch debe ser lo primero
import eventlet
eventlet.monkey_patch()  # Debe ejecutarse antes de importar otros módulos

# Importaciones después del monkey patch
import os
from app import create_app, socketio, db

# Crear la aplicación
app = create_app()

def init_app():
    """Inicializa la aplicación y la base de datos"""
    try:
        with app.app_context():
            # Asegurar que las tablas existan
            db.create_all()
            app.logger.info("Base de datos inicializada correctamente")
            
            # Inicializar otros servicios que requieran contexto
            from app.services.kiosk_ai_service import KioskAIService
            KioskAIService()  # Inicializar el servicio de IA
            
    except Exception as e:
        app.logger.error(f"Error al inicializar la base de datos: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Inicializar la aplicación
        init_app()
        
        # Configurar y ejecutar el servidor
        socketio.run(
            app,
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=False,  # Desactivar debug para producción
            use_reloader=False,  # Evitar problemas con eventlet
            log_output=True  # Habilitar logging
        )
    except Exception as e:
        app.logger.error(f"Error al iniciar el servidor: {str(e)}")
        raise 