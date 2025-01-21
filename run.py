"""
Script principal para ejecutar la aplicación Flask.
Configura el servidor eventlet para WebSocket y ejecuta la aplicación.
Este código solo puede ser modificado siguiendo lo establecido en 'cura.md' y 'project_custom_structure.txt'
"""

# Monkey patch debe ser lo primero
import eventlet
eventlet.monkey_patch(socket=True, select=True, thread=True)

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
            
            # Inicializar servicios
            from app.services.kiosk_ai_service import KioskAIService
            from app.services.ai_metrics import AIMetricsService
            
            # Inicializar servicios (usando patrón Singleton)
            app.ai_service = KioskAIService()
            app.metrics_service = AIMetricsService()
            
            app.logger.info("Servicios inicializados correctamente")
            
    except Exception as e:
        app.logger.error(f"Error al inicializar la aplicación: {str(e)}")
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
            debug=True,  # Activar debug para desarrollo
            use_reloader=False  # Evitar problemas con eventlet
        )
    except Exception as e:
        app.logger.error(f"Error al iniciar el servidor: {str(e)}")
        raise 