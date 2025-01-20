"""
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'
"""

from functools import wraps
from flask import request, g, current_app, jsonify
from app.services.security_service import SecurityService
from app.models.user import UserRole, UserPermission, User
import logging

logger = logging.getLogger(__name__)
security_service = SecurityService()

def apply_security_middleware(app):
    """Aplica el middleware de seguridad a la aplicación Flask.
    
    Args:
        app: Instancia de Flask
    
    Returns:
        app: Instancia de Flask con middleware aplicado
    """
    
    @app.before_request
    def before_request():
        """Middleware que se ejecuta antes de cada request.
        Verifica JWT, rate limiting y registra accesos."""
        # Verificar JWT si existe
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            payload = security_service.verify_jwt(token)
            
            if payload:
                # Cargar usuario en g
                user = User.query.get(payload.get('user_id'))
                if user:
                    g.user = user
                else:
                    return jsonify({'error': 'Usuario no encontrado'}), 401
            else:
                return jsonify({'error': 'Token inválido'}), 401

        # Verificar rate limiting
        if not security_service.rate_limit(
            key=f"{request.remote_addr}:{request.endpoint}",
            limit=current_app.config.get('RATE_LIMIT', 60),
            period=current_app.config.get('RATE_LIMIT_PERIOD', 60)
        ):
            logger.warning(f"Rate limit excedido para {request.remote_addr}")
            return jsonify({'error': 'Rate limit excedido'}), 429

        # Registrar acceso en logs
        security_service.audit_log(
            user_id=getattr(g.get('user', None), 'id', None),
            action='request',
            resource=request.endpoint,
            details={
                'method': request.method,
                'path': request.path,
                'ip': request.remote_addr
            }
        )

    @app.after_request
    def after_request(response):
        """Middleware que se ejecuta después de cada request.
        Agrega headers de seguridad.
        
        Args:
            response: Objeto response de Flask
            
        Returns:
            response: Objeto response modificado
        """
        # Agregar headers de seguridad
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response

    return app 