"""
Configuraciones de hardening de seguridad para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from typing import Dict, List
import os
from datetime import timedelta

# Configuraciones de Seguridad
SECURITY_CONFIG = {
    # Headers de Seguridad
    'HSTS_MAX_AGE': 31536000,  # 1 año
    'HSTS_INCLUDE_SUBDOMAINS': True,
    'HSTS_PRELOAD': True,
    'SSL_REDIRECT': True,
    'SESSION_COOKIE_SECURE': True,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Strict',
    'PERMANENT_SESSION_LIFETIME': timedelta(hours=1),
    
    # Configuración de Contraseñas
    'PASSWORD_MIN_LENGTH': 12,
    'PASSWORD_COMPLEXITY': {
        'UPPER': 1,    # mínimo 1 mayúscula
        'LOWER': 1,    # mínimo 1 minúscula
        'DIGITS': 1,   # mínimo 1 número
        'SPECIAL': 1   # mínimo 1 carácter especial
    },
    
    # Configuración de JWT
    'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY', 'your-secret-key'),
    'JWT_ACCESS_TOKEN_EXPIRES': timedelta(hours=1),
    'JWT_REFRESH_TOKEN_EXPIRES': timedelta(days=30),
    'JWT_ALGORITHM': 'HS256'
}

# Configuración de Rate Limiting
RATE_LIMIT_CONFIG = {
    'limit': 100,           # número de peticiones
    'period': 3600,         # período en segundos (1 hora)
    'by_endpoint': {
        'login': {'limit': 5, 'period': 60},        # 5 intentos por minuto
        '2fa_verify': {'limit': 3, 'period': 60},   # 3 intentos por minuto
        'api_token': {'limit': 10, 'period': 60},   # 10 intentos por minuto
        'password_reset': {'limit': 3, 'period': 3600}  # 3 intentos por hora
    }
}

# Configuración de CORS
CORS_CONFIG = {
    'ORIGINS': [
        'https://admin.kiosk.com',
        'https://api.kiosk.com'
    ],
    'METHODS': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    'ALLOWED_HEADERS': [
        'Content-Type',
        'Authorization',
        'X-Request-With'
    ],
    'EXPOSE_HEADERS': [
        'Content-Range',
        'X-Total-Count'
    ],
    'MAX_AGE': 600  # 10 minutos
}

def get_security_headers() -> Dict[str, str]:
    """Retorna los headers de seguridad para las respuestas HTTP."""
    return {
        'Strict-Transport-Security': f"max-age={SECURITY_CONFIG['HSTS_MAX_AGE']}; includeSubDomains; preload",
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }

def validate_security_config() -> List[str]:
    """Valida la configuración de seguridad."""
    errors = []
    
    # Validar SECRET_KEY
    if not os.getenv('SECRET_KEY'):
        errors.append('SECRET_KEY no está configurada en variables de entorno')
    
    # Validar JWT_SECRET_KEY
    if SECURITY_CONFIG['JWT_SECRET_KEY'] == 'your-secret-key':
        errors.append('JWT_SECRET_KEY está usando el valor por defecto')
    
    # Validar configuración SSL
    if not SECURITY_CONFIG['SSL_REDIRECT']:
        errors.append('SSL_REDIRECT está desactivado')
    
    # Validar configuración de cookies
    if not SECURITY_CONFIG['SESSION_COOKIE_SECURE']:
        errors.append('SESSION_COOKIE_SECURE está desactivado')
    
    return errors

def apply_security_headers(response) -> None:
    """Aplica los headers de seguridad a una respuesta HTTP."""
    headers = get_security_headers()
    for header, value in headers.items():
        response[header] = value 