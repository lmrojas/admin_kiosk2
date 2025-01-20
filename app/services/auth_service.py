# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import db, login_manager
from app.models.user import User, UserRole
from email_validator import validate_email, EmailNotValidError
from config.logging_config import LoggingConfig
import logging

# Registrar user_loader a nivel de módulo
@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        LoggingConfig.log_system_error('User Loading', str(e))
        return None

class AuthService:
    """Servicio de autenticación y gestión de usuarios"""

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
    def authenticate(username, password):
        """
        Autenticar usuario
        
        Args:
            username (str): Nombre de usuario
            password (str): Contraseña
        
        Returns:
            User or None: Usuario autenticado o None si las credenciales son inválidas
        """
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                LoggingConfig.log_auth_event('login', username, success=False)
                return None
            
            # Actualizar última fecha de inicio de sesión
            user.update_last_login()
            
            # Registrar evento de inicio de sesión exitoso
            LoggingConfig.log_auth_event('login', username)
            
            return user
        else:
            # Registrar intento de inicio de sesión fallido
            LoggingConfig.log_auth_event('login', username, success=False)
            return None

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