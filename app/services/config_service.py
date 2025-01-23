"""
Servicio para manejar las configuraciones de la aplicación.
Siguiendo el patrón MVT+S, este servicio centraliza toda la lógica de configuración.
"""

from typing import Dict, Any
from datetime import timedelta

class ConfigService:
    """Servicio para manejar las configuraciones de la aplicación."""
    
    @staticmethod
    def get_security_config() -> Dict[str, Any]:
        """Retorna la configuración de seguridad."""
        return {
            'CSRF_CONFIG': {
                'WTF_CSRF_ENABLED': True,
                'WTF_CSRF_SECRET_KEY': 'csrf-secret-key',
                'WTF_CSRF_TIME_LIMIT': 3600,
                'WTF_CSRF_SSL_STRICT': True
            },
            'INPUT_SANITIZATION': {
                'ALLOWED_TAGS': ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3'],
                'ALLOWED_ATTRIBUTES': {'*': ['class']},
                'STRIP': True,
                'MAX_LENGTH': 10000
            },
            'SESSION': {
                'PERMANENT_SESSION_LIFETIME': timedelta(minutes=30),
                'SESSION_COOKIE_SECURE': True,
                'SESSION_COOKIE_HTTPONLY': True,
                'SESSION_COOKIE_SAMESITE': 'Lax'
            },
            'HEADERS': {
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'SAMEORIGIN',
                'X-XSS-Protection': '1; mode=block',
                'Content-Security-Policy': "default-src 'self'"
            }
        }

config_service = ConfigService() 