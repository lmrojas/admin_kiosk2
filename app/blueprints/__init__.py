"""
Registro de blueprints.
Sigue el patrón MVT + S.
"""

from flask import Blueprint
from app.blueprints.main import main_bp
from app.blueprints.auth import auth_bp
from app.blueprints.kiosk import kiosk_bp
from app.blueprints.api.kiosk import kiosk_api

def register_blueprints(app):
    """Registra todos los blueprints de la aplicación."""
    app.register_blueprint(main_bp)  # Blueprint principal para la ruta raíz
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(kiosk_bp, url_prefix='/kiosk')
    app.register_blueprint(kiosk_api, url_prefix='/api') 