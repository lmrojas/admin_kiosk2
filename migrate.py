"""Script temporal para ejecutar migraciones dentro del contexto de la aplicación."""

import eventlet
eventlet.monkey_patch(socket=True, select=True, thread=True)

from app import create_app, db
from flask_migrate import upgrade

app = create_app()

with app.app_context():
    # Ejecutar migraciones
    upgrade()
    
    # Verificar estructura de la tabla
    from app.services.kiosk_service import KioskService
    service = KioskService()
    service.verify_tables()
