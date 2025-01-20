"""
Configuraciones de hardening de seguridad para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from typing import Dict, List
import os

# Configuraciones de Django Security
SECURITY_CONFIG = {
    # Headers de Seguridad
    'SECURE_HSTS_SECONDS': 31536000,  # 1 año
    'SECURE_HSTS_INCLUDE_SUBDOMAINS': True,
    'SECURE_HSTS_PRELOAD': True,
    'SECURE_SSL_REDIRECT': True,
    'SESSION_COOKIE_SECURE': True,
    'CSRF_COOKIE_SECURE': True,
    'SECURE_BROWSER_XSS_FILTER': True,
    'SECURE_CONTENT_TYPE_NOSNIFF': True,
    'X_FRAME_OPTIONS': 'DENY',
    'SECURE_REFERRER_POLICY': 'strict-origin-when-cross-origin',

    # Configuración de Contraseñas
    'AUTH_PASSWORD_VALIDATORS': [
        {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
        {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
         'OPTIONS': {'min_length': 12}},
        {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
        {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
    ],

    # Configuración de Sesión
    'SESSION_EXPIRE_AT_BROWSER_CLOSE': True,
    'SESSION_COOKIE_AGE': 3600,  # 1 hora
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Strict',
}

# Configuración de Rate Limiting
RATE_LIMIT_CONFIG = {
    'DEFAULT': '1000/hour',
    'LOGIN': '5/minute',
    '2FA_VERIFY': '3/minute',
    'API_TOKEN': '10/minute',
    'PASSWORD_RESET': '3/hour',
}

# Configuración de CORS
CORS_CONFIG = {
    'CORS_ALLOWED_ORIGINS': [
        'https://admin-kiosk.com',
        'https://staging.admin-kiosk.com',
    ],
    'CORS_ALLOW_METHODS': [
        'GET',
        'POST',
        'PUT',
        'DELETE',
        'OPTIONS',
    ],
    'CORS_ALLOW_HEADERS': [
        'accept',
        'accept-encoding',
        'authorization',
        'content-type',
        'origin',
        'user-agent',
        'x-csrftoken',
        'x-requested-with',
    ],
}

# Configuración de JWT
JWT_CONFIG = {
    'ACCESS_TOKEN_LIFETIME': 3600,  # 1 hora
    'REFRESH_TOKEN_LIFETIME': 86400,  # 24 horas
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
}

# Configuración de 2FA
TWO_FACTOR_CONFIG = {
    'BACKUP_CODES_COUNT': 5,
    'TOKEN_VALIDITY': 30,  # segundos
    'MAX_ATTEMPTS': 3,
    'LOCKOUT_TIME': 300,  # 5 minutos
}

# Configuración de Logs de Seguridad
SECURITY_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'security': {
            'format': '[%(asctime)s] [%(levelname)s] [%(ip)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'security_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/admin-kiosk/security.log',
            'formatter': 'security',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
        },
    },
    'loggers': {
        'security': {
            'handlers': ['security_file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

def get_security_headers() -> Dict[str, str]:
    """Retorna los headers de seguridad para las respuestas HTTP."""
    return {
        'Strict-Transport-Security': f'max-age={SECURITY_CONFIG["SECURE_HSTS_SECONDS"]}; includeSubDomains; preload',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': SECURITY_CONFIG['X_FRAME_OPTIONS'],
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': SECURITY_CONFIG['SECURE_REFERRER_POLICY'],
        'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:;",
    }

def validate_security_config() -> List[str]:
    """Valida la configuración de seguridad y retorna una lista de problemas encontrados."""
    issues = []
    
    # Validar configuración HTTPS
    if not SECURITY_CONFIG['SECURE_SSL_REDIRECT']:
        issues.append('SSL redirect no está habilitado')
    
    # Validar configuración de cookies
    if not SECURITY_CONFIG['SESSION_COOKIE_SECURE']:
        issues.append('Cookies de sesión no están configuradas como seguras')
    
    # Validar CORS
    if '*' in CORS_CONFIG['CORS_ALLOWED_ORIGINS']:
        issues.append('CORS permite todos los orígenes')
    
    # Validar JWT
    if JWT_CONFIG['ACCESS_TOKEN_LIFETIME'] > 3600:
        issues.append('Tiempo de vida del token de acceso es mayor a 1 hora')
    
    return issues

def apply_security_headers(response) -> None:
    """Aplica los headers de seguridad a una respuesta HTTP."""
    headers = get_security_headers()
    for header, value in headers.items():
        response[header] = value 