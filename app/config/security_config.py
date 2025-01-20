"""
Configuración de seguridad y hardening.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from datetime import timedelta
import secrets

# Configuración de sesiones
SESSION_CONFIG = {
    'PERMANENT_SESSION_LIFETIME': timedelta(hours=1),
    'SESSION_COOKIE_SECURE': True,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Strict',
    'REMEMBER_COOKIE_SECURE': True,
    'REMEMBER_COOKIE_HTTPONLY': True,
    'REMEMBER_COOKIE_DURATION': timedelta(days=7)
}

# Configuración de headers de seguridad
SECURITY_HEADERS = {
    'X-Frame-Options': 'SAMEORIGIN',
    'X-XSS-Protection': '1; mode=block',
    'X-Content-Type-Options': 'nosniff',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
}

# Configuración de rate limiting
RATE_LIMIT_CONFIG = {
    'DEFAULT': '100 per hour',
    'LOGIN_ATTEMPTS': '5 per minute',
    'API_CALLS': '1000 per day'
}

# Configuración de 2FA
TWO_FACTOR_CONFIG = {
    'ISSUER_NAME': 'Admin Kiosk',
    'CODE_TTL_SECONDS': 30,
    'BACKUP_CODES_COUNT': 8,
    'TEMP_CODE_TTL_MINUTES': 10,
    'MAX_ATTEMPTS': 3
}

# Configuración de contraseñas
PASSWORD_POLICY = {
    'MIN_LENGTH': 12,
    'REQUIRE_UPPERCASE': True,
    'REQUIRE_LOWERCASE': True,
    'REQUIRE_NUMBERS': True,
    'REQUIRE_SPECIAL': True,
    'MAX_AGE_DAYS': 90,
    'PREVENT_REUSE_COUNT': 5
}

# Configuración de JWT
JWT_CONFIG = {
    'ACCESS_TOKEN_EXPIRES': timedelta(minutes=15),
    'REFRESH_TOKEN_EXPIRES': timedelta(days=7),
    'ALGORITHM': 'HS256',
    'SECRET_KEY': secrets.token_urlsafe(32)
}

# Configuración de logging de seguridad
SECURITY_LOGGING = {
    'ENABLED': True,
    'LOG_LEVEL': 'INFO',
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_EVENTS': [
        'login_attempt',
        'password_change',
        '2fa_setup',
        'role_change',
        'permission_change'
    ]
}

# Lista de IPs bloqueadas
BLOCKED_IPS = set()

# Configuración de timeout de sesión por inactividad
INACTIVITY_TIMEOUT = timedelta(minutes=30)

# Configuración de bloqueo de cuenta
ACCOUNT_LOCKOUT = {
    'MAX_ATTEMPTS': 5,
    'LOCKOUT_DURATION': timedelta(minutes=15),
    'RESET_AFTER': timedelta(hours=24)
}

# Configuración de sanitización de entrada
INPUT_SANITIZATION = {
    'ALLOWED_HTML_TAGS': ['p', 'br', 'strong', 'em'],
    'ALLOWED_ATTRIBUTES': {'a': ['href', 'title']},
    'STRIP_COMMENTS': True
}

# Configuración de auditoría
AUDIT_CONFIG = {
    'ENABLED': True,
    'STORE_DURATION_DAYS': 90,
    'EVENTS_TO_TRACK': [
        'user_creation',
        'role_modification',
        'permission_changes',
        'login_attempts',
        'data_access',
        'configuration_changes'
    ]
} 