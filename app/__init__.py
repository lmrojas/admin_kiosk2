# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_socketio import SocketIO
from config.default import config
from config.logging_config import LoggingConfig
import logging

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'
socketio = SocketIO(cors_allowed_origins="*", async_mode='eventlet')

def create_app(config_name='development'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Inicializar extensiones
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Configurar logging
    LoggingConfig.configure_logging(app)
    logger = logging.getLogger(__name__)
    
    # Importar modelos
    from .models.user import User
    from .models.kiosk import Kiosk
    from .models.ai import ModelMetrics, PredictionLog, DriftMetrics
    
    # Importar e inicializar servicios
    from .services.auth_service import AuthService
    auth_service = AuthService()
    
    # Inicializar login después del servicio de auth
    login_manager.init_app(app)
    
    # Importar e inicializar websockets
    from .websockets import init_websockets
    socketio.init_app(app)
    init_websockets(socketio)
    
    # Registrar blueprints
    from .blueprints.auth import auth_bp
    from .blueprints.kiosk import kiosk_bp
    from .blueprints.ai import ai_bp
    from .blueprints.monitor import monitor_bp
    from .blueprints.main import main_bp
    
    # Registrar blueprint main primero (sin prefijo)
    app.register_blueprint(main_bp)
    
    # Registrar otros blueprints con sus prefijos
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(kiosk_bp, url_prefix='/kiosk')
    app.register_blueprint(ai_bp, url_prefix='/ai')
    app.register_blueprint(monitor_bp, url_prefix='/monitor')
    
    # Manejadores de error
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f'Page not found: {request.url}')
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'Server Error: {error}')
        db.session.rollback()
        return render_template('errors/500.html'), 500
        
    return app 