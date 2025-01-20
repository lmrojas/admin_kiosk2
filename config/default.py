# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import os
from datetime import timedelta

class Config:
    """Clase base de configuración."""
    
    # Configuración base
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-highly-secret'
    
    # Configuración de base de datos PostgreSQL
    DB_USER = os.environ.get('DB_USER', 'postgres')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', 'admin_kiosk2')
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Configuración de Redis
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
    
    # Configuración de sesión
    SESSION_TYPE = 'redis'
    SESSION_REDIS = None  # Se configura en la aplicación
    SESSION_COOKIE_NAME = 'kiosk_session'
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_REFRESH_EACH_REQUEST = True
    
    # Configuración JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Configuración de logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Configuración de rate limiting
    RATELIMIT_DEFAULT = "100/hour"
    RATELIMIT_STORAGE_URL = "redis://localhost:6379/0"
    RATELIMIT_STRATEGY = "fixed-window"
    
    # Configuración de notificaciones - Email
    SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_USER = os.environ.get('SMTP_USER', 'alerts@yourdomain.com')
    SMTP_PASS = os.environ.get('SMTP_PASS', 'your-app-specific-password')
    ALERT_FROM_EMAIL = os.environ.get('ALERT_FROM_EMAIL', 'alerts@yourdomain.com')
    ALERT_TO_EMAILS = os.environ.get('ALERT_TO_EMAILS', 'admin@yourdomain.com').split(',')
    
    # Configuración de notificaciones - Slack
    SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/services/your/webhook/url')
    SLACK_CHANNEL = os.environ.get('SLACK_CHANNEL', '#alerts')
    
    # Configuración de notificaciones - SMS (Twilio)
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
    TWILIO_FROM_NUMBER = os.environ.get('TWILIO_FROM_NUMBER')
    ALERT_TO_NUMBERS = os.environ.get('ALERT_TO_NUMBERS', '+1234567890').split(',')
    
    # Configuración de seguridad
    SECURITY_PASSWORD_SALT = 'secure-salt'
    SECURITY_PASSWORD_HASH = 'pbkdf2_sha512'
    SECURITY_TRACKABLE = True
    SECURITY_REGISTERABLE = True
    SECURITY_SEND_REGISTER_EMAIL = True
    SECURITY_RECOVERABLE = True
    SECURITY_CHANGEABLE = True
    
    # Headers de seguridad
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'"
    }
    
    # Configuración de migraciones
    FLASK_APP = 'app'
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    MIGRATIONS_DIR = os.path.join(BASE_DIR, '../migrations')
    
    # Configuración de 2FA
    TWO_FACTOR_ISSUER = 'Admin Kiosk'
    TWO_FACTOR_DIGITS = 6
    TWO_FACTOR_INTERVAL = 30
    TWO_FACTOR_BACKUP_CODES = 10
    TWO_FACTOR_TEMP_CODE_EXPIRY = timedelta(minutes=10)

class DevelopmentConfig(Config):
    """Configuración de desarrollo."""
    DEBUG = True
    # Usar la misma configuración PostgreSQL que la clase base
    pass

class TestingConfig(Config):
    """Configuración de pruebas."""
    TESTING = True
    # Base de datos PostgreSQL para pruebas
    DB_NAME = os.environ.get('DB_NAME', 'admin_kiosk2_test')
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or \
        f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{DB_NAME}"
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Configuración de producción."""
    DEBUG = False
    # Usar la URL de base de datos de producción
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 