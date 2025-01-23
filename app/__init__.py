"""
Módulo principal de la aplicación.
Sigue el patrón MVT + S, centralizando la configuración.
"""

from flask import Flask
from flask_socketio import SocketIO
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from config import config
from app.models.base import db, init_db
from app.services.websocket_service import WebSocketService

# Inicializar extensiones
migrate = Migrate()
login_manager = LoginManager()
socketio = SocketIO(cors_allowed_origins="*")
session = Session()
limiter = Limiter(key_func=get_remote_address)
csrf = CSRFProtect()

# Configurar login manager
@login_manager.user_loader
def load_user(user_id):
    """Carga un usuario por su ID."""
    from app.models.user import User
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        return None

def create_app(config_name='default'):
    """Crea y configura la aplicación Flask."""
    app = Flask(__name__)
    
    # Cargar configuración
    app.config.from_object(config[config_name])
    
    # Inicializar extensiones
    init_db(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    socketio.init_app(app)
    session.init_app(app)
    limiter.init_app(app)
    csrf.init_app(app)
    
    # Configurar login
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Por favor inicia sesión para acceder a esta página.'
    login_manager.login_message_category = 'info'
    
    # Registrar blueprints
    register_blueprints(app)
    
    # Inicializar WebSockets usando el método estático
    WebSocketService.init_websockets()
    
    return app 

def register_blueprints(app):
    """Registra los blueprints de la aplicación."""
    
    # Importar blueprints
    from app.blueprints.main import main_bp
    from app.blueprints.auth import auth_bp
    from app.blueprints.kiosk import kiosk_bp
    from app.blueprints.api.kiosk import kiosk_api
    from app.blueprints.api.ai import ai_api
    
    # Registrar blueprints principales
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(kiosk_bp, url_prefix='/kiosk')
    
    # Registrar blueprints de API bajo /api/v1
    app.register_blueprint(kiosk_api, url_prefix='/api/v1/kiosks')
    app.register_blueprint(ai_api, url_prefix='/api/v1/ai') 