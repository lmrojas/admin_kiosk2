"""
Middleware de seguridad para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import logging
from typing import Callable
from flask import request, Response, current_app
from werkzeug.wrappers import Request, Response
from config.security.hardening import (
    get_security_headers,
    RATE_LIMIT_CONFIG,
    JWT_CONFIG
)
from app.services.security_service import SecurityService

logger = logging.getLogger('security')

class SecurityMiddleware:
    """Middleware para aplicar configuraciones de seguridad."""

    def __init__(self, app):
        self.app = app
        self.security_service = SecurityService()

    def __call__(self, environ, start_response):
        """Procesa la request y aplica medidas de seguridad."""
        request = Request(environ)
        
        try:
            # Verificar rate limiting
            if not self._check_rate_limit(request):
                response = self._rate_limit_exceeded_response()
                return response(environ, start_response)

            # Verificar y actualizar token JWT si es necesario
            if hasattr(request, 'user') and request.user.is_authenticated:
                self._handle_jwt_token(request)

            # Registrar intento de acceso
            self._log_access_attempt(request)

            # Procesar la request
            response = self.app(environ, start_response)

            # Aplicar headers de seguridad
            self._apply_security_headers(response)

            return response

        except Exception as e:
            logger.error(f"Error en middleware de seguridad: {str(e)}")
            response = Response('Error interno del servidor', status=500)
            return response(environ, start_response)

    def _check_rate_limit(self, request: Request) -> bool:
        """Verifica el rate limiting para la request."""
        key = self._get_endpoint_key(request)
        return self.security_service.rate_limit(
            key, 
            RATE_LIMIT_CONFIG['limit'], 
            RATE_LIMIT_CONFIG['period']
        )

    def _handle_jwt_token(self, request: Request) -> None:
        """Maneja la rotación de tokens JWT."""
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if token:
            token_age = self._get_token_age(token)
            if token_age > JWT_CONFIG['rotate_after_seconds']:
                self._rotate_token(request)

    def _apply_security_headers(self, response) -> None:
        """Aplica headers de seguridad a la respuesta."""
        headers = get_security_headers()
        for header, value in headers.items():
            response.headers[header] = value

    def _log_access_attempt(self, request: Request) -> None:
        """Registra el intento de acceso."""
        self.security_service.log_access(
            ip=self._get_client_ip(request),
            method=request.method,
            path=request.path,
            user_id=getattr(request.user, 'id', None)
        )

    def _get_client_ip(self, request: Request) -> str:
        """Obtiene la IP real del cliente."""
        if request.headers.get('X-Forwarded-For'):
            return request.headers['X-Forwarded-For'].split(',')[0].strip()
        return request.remote_addr

    def _get_endpoint_key(self, request: Request) -> str:
        """Genera una clave única para el rate limiting."""
        return f"rate_limit:{self._get_client_ip(request)}:{request.endpoint}"

    def _rate_limit_exceeded_response(self) -> Response:
        """Genera una respuesta cuando se excede el rate limit."""
        return Response(
            'Demasiadas solicitudes',
            status=429,
            headers={'Retry-After': str(RATE_LIMIT_CONFIG['period'])}
        )

    def _get_token_age(self, token: str) -> int:
        """Obtiene la edad del token JWT en segundos."""
        try:
            payload = self.security_service.verify_jwt(token)
            if payload and 'iat' in payload:
                from datetime import datetime, timezone
                iat = datetime.fromtimestamp(payload['iat'], tz=timezone.utc)
                now = datetime.now(timezone.utc)
                return int((now - iat).total_seconds())
        except Exception as e:
            logger.error(f"Error al obtener edad del token: {str(e)}")
        return 0

    def _rotate_token(self, request: Request) -> None:
        """Rota el token JWT."""
        try:
            old_token = request.headers['Authorization'].replace('Bearer ', '')
            payload = self.security_service.verify_jwt(old_token)
            if payload:
                new_token = self.security_service.generate_jwt(
                    user_id=payload['user_id'],
                    additional_claims={'rotated_from': old_token[:8]}
                )
                request.headers['Authorization'] = f'Bearer {new_token}'
        except Exception as e:
            logger.error(f"Error al rotar token: {str(e)}") 