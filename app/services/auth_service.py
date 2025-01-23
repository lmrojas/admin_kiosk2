# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import db, login_manager
from app.models.user import User, UserRole
from email_validator import validate_email, EmailNotValidError
from config.logging_config import LoggingConfig
import logging
from functools import wraps
from flask import abort, flash, redirect, url_for, request, jsonify, g
from flask_login import current_user
from app.services.security_service import SecurityService
from typing import Union, List, Optional

# Registrar user_loader a nivel de módulo
@login_manager.user_loader
def load_user(user_id):
    """Carga un usuario por su ID para Flask-Login"""
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        LoggingConfig.log_system_error('User Loading', str(e))
        return None

logger = logging.getLogger(__name__)
security_service = SecurityService()

# Exponer decorador a nivel de módulo
def admin_required(f):
    """Decorador que verifica que el usuario sea administrador."""
    return AuthService.admin_required(f)

class AuthService:
    """Servicio para manejar autenticación y autorización."""
    
    @staticmethod
    def jwt_required(f):
        """Decorador que verifica que exista un JWT válido."""
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

    @staticmethod
    def permission_required(permissions: Union[str, List[str]], require_all: bool = False):
        """
        Decorador para verificar permisos de usuario.
        
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

    @staticmethod
    def role_required(roles: Union[str, List[str]], require_all: bool = False):
        """
        Decorador para verificar rol de usuario.
        
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

    @staticmethod
    def admin_required(f):
        """Decorador que verifica que el usuario sea administrador."""
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

    @staticmethod
    def audit_action(action: str, resource: Optional[str] = None):
        """
        Decorador para registrar acciones en el log de auditoría.
        
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

    def __init__(self):
        """Inicializar servicio"""
        pass

    @staticmethod
    def validate_email(email):
        """Validar formato de correo electrónico"""
        try:
            valid = validate_email(email)
            return valid.email
        except EmailNotValidError:
            return None

    @staticmethod
    def register_user(username, email, password, role_name=UserRole.VIEWER.value):
        """
        Registrar un nuevo usuario
        
        Args:
            username (str): Nombre de usuario
            email (str): Correo electrónico
            password (str): Contraseña
            role_name (str): Nombre del rol (por defecto: viewer)
        
        Returns:
            User: Usuario registrado
        
        Raises:
            ValueError: Si el nombre de usuario o correo ya existen
        """
        # Validar correo electrónico
        try:
            validate_email(email)
        except EmailNotValidError as e:
            LoggingConfig.log_system_error('Email Validation', str(e))
            raise ValueError("Correo electrónico inválido")
        
        # Verificar si el nombre de usuario ya existe
        if User.query.filter_by(username=username).first():
            LoggingConfig.log_system_error('User Registration', 'Nombre de usuario ya existe')
            raise ValueError("Nombre de usuario ya existe")
        
        # Verificar si el correo ya existe
        if User.query.filter_by(email=email).first():
            LoggingConfig.log_system_error('User Registration', 'Correo electrónico ya registrado')
            raise ValueError("Correo electrónico ya registrado")
        
        # Verificar si el rol es válido
        if role_name not in [role.value for role in UserRole]:
            LoggingConfig.log_system_error('User Registration', f'Rol {role_name} no existe')
            raise ValueError(f"Rol {role_name} no existe")
        
        # Crear nuevo usuario
        user = User(
            username=username,
            email=email,
            role_name=role_name
        )
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            
            # Registrar evento de registro
            LoggingConfig.log_auth_event('registro', username)
            
            return user
        except Exception as e:
            db.session.rollback()
            LoggingConfig.log_system_error('User Registration', str(e))
            raise

    @staticmethod
    def process_login(username: str, password: str, code_2fa: str = None) -> dict:
        """
        Procesa el intento de login completo incluyendo 2FA.
        Unifica la autenticación y verificación 2FA en un solo método.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            code_2fa: Código 2FA (opcional)
            
        Returns:
            dict: Resultado del proceso con claves:
                - success: bool
                - user: User object (si success es True)
                - user_id: int (si needs_2fa es True)
                - needs_2fa: bool
                - error: str (si hay error)
        """
        try:
            # Buscar usuario y verificar contraseña
            user = User.query.filter_by(username=username).first()
            if not user or not user.check_password(password):
                LoggingConfig.log_auth_event('login', username, success=False)
                return {'success': False, 'error': 'Credenciales inválidas'}

            if not user.is_active:
                LoggingConfig.log_auth_event('login', username, success=False)
                return {'success': False, 'error': 'Usuario inactivo'}
            
            # Verificar 2FA si está habilitado
            if user.two_factor_enabled:
                if not code_2fa:
                    return {
                        'success': False,
                        'needs_2fa': True,
                        'user_id': user.id
                    }
                
                if not two_factor_service.verify_code(user, code_2fa):
                    return {'success': False, 'error': 'Código 2FA inválido'}
            
            # Actualizar última fecha de login
            user.update_last_login()
            LoggingConfig.log_auth_event('login', username)
            return {'success': True, 'user': user}
            
        except Exception as e:
            LoggingConfig.log_system_error('Login Process', str(e))
            return {'success': False, 'error': 'Error en el proceso de login'}

    @staticmethod
    def verify_auth(user_id: int, code: str, verification_type: str = '2fa') -> dict:
        """
        Verifica la autenticación usando diferentes métodos.
        Unifica todas las verificaciones en un solo método.
        
        Args:
            user_id: ID del usuario
            code: Código a verificar
            verification_type: Tipo de verificación ('2fa', 'backup', 'temp')
            
        Returns:
            dict: Resultado del proceso con claves:
                - success: bool
                - user: User object (si success es True)
                - error: str (si hay error)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return {'success': False, 'error': 'Usuario no encontrado'}
            
            # Verificar según el tipo
            if verification_type == '2fa':
                success = two_factor_service.verify_code(user, code)
                event = '2fa_verify'
            elif verification_type == 'backup':
                success = two_factor_service.verify_backup_code(user, code)
                event = 'backup_verify'
            elif verification_type == 'temp':
                success = two_factor_service.verify_temp_code(user, code)
                event = 'temp_verify'
            else:
                return {'success': False, 'error': 'Tipo de verificación inválido'}
            
            if success:
                LoggingConfig.log_auth_event(event, user.username)
                return {'success': True, 'user': user}
            
            return {'success': False, 'error': 'Código inválido'}
            
        except Exception as e:
            LoggingConfig.log_system_error('Auth Verification', str(e))
            return {'success': False, 'error': 'Error en la verificación'}

    @staticmethod
    def process_registration(username: str, email: str, password: str) -> dict:
        """
        Procesa el registro completo de un usuario.
        
        Args:
            username: Nombre de usuario
            email: Correo electrónico
            password: Contraseña
            
        Returns:
            dict: Resultado del proceso con claves:
                - success: bool
                - error: str (si hay error)
        """
        try:
            user = AuthService.register_user(username, email, password)
            if user:
                LoggingConfig.log_auth_event('registro', username)
                return {'success': True}
            return {'success': False, 'error': 'Error al registrar usuario'}
        except ValueError as e:
            LoggingConfig.log_system_error('Registration Process', str(e))
            return {'success': False, 'error': str(e)}
        except Exception as e:
            LoggingConfig.log_system_error('Registration Process', str(e))
            return {'success': False, 'error': 'Error en el proceso de registro'}

    @staticmethod
    def change_password(user, old_password, new_password):
        """
        Cambiar contraseña de usuario
        
        Args:
            user (User): Usuario
            old_password (str): Contraseña actual
            new_password (str): Nueva contraseña
        
        Raises:
            ValueError: Si la contraseña actual es incorrecta
        """
        if not user.check_password(old_password):
            LoggingConfig.log_auth_event('cambio_password', user.username, success=False)
            raise ValueError("Contraseña actual incorrecta")
        
        user.set_password(new_password)
        
        try:
            db.session.commit()
            LoggingConfig.log_auth_event('cambio_password', user.username)
        except Exception as e:
            db.session.rollback()
            LoggingConfig.log_system_error('Password Change', str(e))
            raise

    @staticmethod
    def update_user_role(user, new_role_name):
        """
        Actualizar rol de usuario
        
        Args:
            user (User): Usuario a actualizar
            new_role_name (str): Nombre del nuevo rol
        
        Raises:
            ValueError: Si el rol no existe
        """
        if new_role_name not in [role.value for role in UserRole]:
            raise ValueError(f"Rol {new_role_name} no existe")
        
        user.role_name = new_role_name
        
        try:
            db.session.commit()
            LoggingConfig.log_auth_event('cambio_rol', user.username)
        except Exception as e:
            db.session.rollback()
            LoggingConfig.log_system_error('Role Update', str(e))
            raise

    @staticmethod
    def deactivate_user(user):
        """
        Desactivar cuenta de usuario
        
        Args:
            user (User): Usuario a desactivar
        """
        user.is_active = False
        
        try:
            db.session.commit()
            LoggingConfig.log_auth_event('desactivacion', user.username)
        except Exception as e:
            db.session.rollback()
            LoggingConfig.log_system_error('User Deactivation', str(e))
            raise

    @staticmethod
    def activate_user(user):
        """
        Activar cuenta de usuario
        
        Args:
            user (User): Usuario a activar
        """
        user.is_active = True
        
        try:
            db.session.commit()
            LoggingConfig.log_auth_event('activacion', user.username)
        except Exception as e:
            db.session.rollback()
            LoggingConfig.log_system_error('User Activation', str(e))
            raise

auth_service = AuthService() 