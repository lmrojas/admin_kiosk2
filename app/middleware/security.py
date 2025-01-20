"""
Middleware de seguridad para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import logging
from typing import Callable
from django.http import HttpRequest, HttpResponse
from django.conf import settings
from config.security.hardening import (
    get_security_headers,
    RATE_LIMIT_CONFIG,
    JWT_CONFIG
)
from app.services.rate_limit import RateLimitService
from app.services.security import SecurityAuditService

logger = logging.getLogger('security')

class SecurityMiddleware:
    """Middleware para aplicar configuraciones de seguridad."""

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response
        self.rate_limit_service = RateLimitService()
        self.audit_service = SecurityAuditService()

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Procesa la request y aplica medidas de seguridad."""
        try:
            # Verificar rate limiting
            if not self._check_rate_limit(request):
                return self._rate_limit_exceeded_response()

            # Verificar y actualizar token JWT si es necesario
            if hasattr(request, 'user') and request.user.is_authenticated:
                self._handle_jwt_token(request)

            # Registrar intento de acceso
            self._log_access_attempt(request)

            # Procesar la request
            response = self.get_response(request)

            # Aplicar headers de seguridad
            self._apply_security_headers(response)

            return response

        except Exception as e:
            logger.error(f"Error en SecurityMiddleware: {str(e)}", 
                        extra={'ip': self._get_client_ip(request)})
            raise

    def _check_rate_limit(self, request: HttpRequest) -> bool:
        """Verifica si la request excede los límites de rate limiting."""
        endpoint = self._get_endpoint_key(request)
        limit = RATE_LIMIT_CONFIG.get(endpoint, RATE_LIMIT_CONFIG['DEFAULT'])
        return self.rate_limit_service.check_limit(
            key=f"{self._get_client_ip(request)}:{endpoint}",
            limit=limit
        )

    def _handle_jwt_token(self, request: HttpRequest) -> None:
        """Maneja la rotación y validación de tokens JWT."""
        if hasattr(request, 'auth') and request.auth:
            token_age = self._get_token_age(request.auth)
            if token_age > JWT_CONFIG['ACCESS_TOKEN_LIFETIME'] * 0.8:
                self._rotate_token(request)

    def _apply_security_headers(self, response: HttpResponse) -> None:
        """Aplica los headers de seguridad a la respuesta."""
        headers = get_security_headers()
        for header, value in headers.items():
            response[header] = value

        # Agregar headers específicos para la respuesta
        response['X-Request-ID'] = self._generate_request_id()

    def _log_access_attempt(self, request: HttpRequest) -> None:
        """Registra los intentos de acceso para auditoría."""
        self.audit_service.log_access(
            ip=self._get_client_ip(request),
            method=request.method,
            path=request.path,
            user_id=getattr(request.user, 'id', None)
        )

    def _get_client_ip(self, request: HttpRequest) -> str:
        """Obtiene la IP real del cliente."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR')

    def _get_endpoint_key(self, request: HttpRequest) -> str:
        """Determina la clave de rate limiting basada en el endpoint."""
        if request.path.startswith('/api/auth/login'):
            return 'LOGIN'
        elif request.path.startswith('/api/auth/2fa'):
            return '2FA_VERIFY'
        elif request.path.startswith('/api/auth/token'):
            return 'API_TOKEN'
        elif request.path.startswith('/api/auth/password-reset'):
            return 'PASSWORD_RESET'
        return 'DEFAULT'

    def _rate_limit_exceeded_response(self) -> HttpResponse:
        """Genera una respuesta cuando se excede el rate limit."""
        return HttpResponse(
            status=429,
            content='Rate limit exceeded',
            content_type='text/plain'
        )

    def _generate_request_id(self) -> str:
        """Genera un ID único para la request."""
        import uuid
        return str(uuid.uuid4())

    def _get_token_age(self, token) -> int:
        """Calcula la edad del token JWT en segundos."""
        from datetime import datetime
        import jwt
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=['HS256']
            )
            issued_at = datetime.fromtimestamp(payload['iat'])
            return (datetime.now() - issued_at).seconds
        except jwt.InvalidTokenError:
            return 0

    def _rotate_token(self, request: HttpRequest) -> None:
        """Rota el token JWT si está próximo a expirar."""
        from app.services.auth import AuthService
        auth_service = AuthService()
        new_token = auth_service.rotate_token(request.auth)
        request.auth = new_token 