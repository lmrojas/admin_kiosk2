from functools import wraps
from flask import abort, flash, redirect, url_for, request, jsonify, g
from flask_login import current_user
from app.services.security_service import SecurityService
from app.models.user import UserRole
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)
security_service = SecurityService()

def jwt_required():
    """
    Decorador que verifica que exista un JWT válido
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Token no proporcionado'}), 401
            
            token = auth_header.split(' ')[1]
            payload = security_service.verify_jwt(token)
            
            if not payload:
                return jsonify({'error': 'Token inválido o expirado'}), 401
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def permission_required(permissions: Union[str, List[str]], require_all: bool = False):
    """
    Decorador para verificar permisos de usuario
    
    Args:
        permissions: Permiso o lista de permisos requeridos
        require_all: Si es True, requiere todos los permisos. Si es False, requiere al menos uno
    """
    if isinstance(permissions, str):
        permissions = [permissions]
        
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                if request.is_json:
                    return jsonify({'error': 'Autenticación requerida'}), 401
                flash('Por favor inicia sesión para acceder a esta página.', 'warning')
                return redirect(url_for('auth.login'))
            
            if require_all:
                has_permissions = all(current_user.has_permission(p) for p in permissions)
            else:
                has_permissions = any(current_user.has_permission(p) for p in permissions)
            
            if not has_permissions:
                if request.is_json:
                    return jsonify({'error': 'Permisos insuficientes'}), 403
                flash('No tienes los permisos necesarios para acceder a esta página.', 'danger')
                return abort(403)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def role_required(roles: Union[str, List[str]], require_all: bool = False):
    """
    Decorador para verificar rol de usuario
    
    Args:
        roles: Rol o lista de roles requeridos
        require_all: Si es True, requiere todos los roles. Si es False, requiere al menos uno
    """
    if isinstance(roles, str):
        roles = [roles]
        
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                if request.is_json:
                    return jsonify({'error': 'Autenticación requerida'}), 401
                flash('Por favor inicia sesión para acceder a esta página.', 'warning')
                return redirect(url_for('auth.login'))
            
            if require_all:
                has_roles = all(current_user.has_role(r) for r in roles)
            else:
                has_roles = any(current_user.has_role(r) for r in roles)
            
            if not has_roles:
                if request.is_json:
                    return jsonify({'error': 'Rol insuficiente'}), 403
                flash('No tienes el rol necesario para acceder a esta página.', 'danger')
                return abort(403)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def admin_required(f):
    """
    Decorador que verifica que el usuario sea administrador
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            if request.is_json:
                return jsonify({'error': 'Autenticación requerida'}), 401
            flash('Por favor inicia sesión para acceder a esta página.', 'warning')
            return redirect(url_for('auth.login'))
        
        if not current_user.has_role(UserRole.ADMIN.value):
            if request.is_json:
                return jsonify({'error': 'Se requiere rol de administrador'}), 403
            flash('Se requiere rol de administrador para acceder a esta página.', 'danger')
            return abort(403)
        
        return f(*args, **kwargs)
    return decorated_function

def audit_action(action: str, resource: Optional[str] = None):
    """
    Decorador para registrar acciones en el log de auditoría
    
    Args:
        action: Tipo de acción realizada
        resource: Recurso afectado (opcional, se puede inferir de la ruta)
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
                
                # Registrar la acción exitosa
                security_service.audit_log(
                    user_id=getattr(current_user, 'id', None),
                    action=action,
                    resource=resource or request.endpoint,
                    details={
                        'method': request.method,
                        'path': request.path,
                        'args': kwargs,
                        'status': 'success'
                    }
                )
                
                return result
                
            except Exception as e:
                # Registrar la acción fallida
                security_service.audit_log(
                    user_id=getattr(current_user, 'id', None),
                    action=action,
                    resource=resource or request.endpoint,
                    details={
                        'method': request.method,
                        'path': request.path,
                        'args': kwargs,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise
                
        return decorated_function
    return decorator 