"""
Inicialización de blueprints API.
Registra todos los blueprints relacionados con la API REST.
"""

from flask import Blueprint
from .kiosk import kiosk_api

# Crear blueprint principal para la API
api = Blueprint('api', __name__, url_prefix='/api')

# Registrar sub-blueprints
api.register_blueprint(kiosk_api) 