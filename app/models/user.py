# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.postgresql import JSON
from enum import Enum

class UserRole(str, Enum):
    """Roles de usuario disponibles."""
    ADMIN = 'admin'
    MANAGER = 'manager'
    OPERATOR = 'operator'
    VIEWER = 'viewer'

class UserPermission(str, Enum):
    """Permisos disponibles en el sistema."""
    MANAGE_USERS = 'manage_users'
    MANAGE_KIOSKS = 'manage_kiosks'
    VIEW_DASHBOARD = 'view_dashboard'
    VIEW_LOGS = 'view_logs'
    MANAGE_BACKUPS = 'manage_backups'
    MANAGE_SETTINGS = 'manage_settings'
    UPDATE_KIOSK = 'update_kiosk'
    VIEW_KIOSK = 'view_kiosk'

class User(db.Model, UserMixin):
    """Modelo de usuario con soporte para autenticación de dos factores."""
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role_name = db.Column(db.String(20), nullable=False, default=UserRole.VIEWER.value)
    
    # Campos para 2FA
    two_factor_enabled = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))
    backup_codes = db.Column(JSON)  # Lista de códigos de respaldo
    temp_2fa_code = db.Column(JSON)  # Código temporal y fecha de expiración
    
    # Campos de estado y auditoría
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    failed_login_attempts = db.Column(db.Integer, default=0)
    account_locked = db.Column(db.Boolean, default=False)
    
    def set_password(self, password):
        """Establece el hash de la contraseña."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verifica la contraseña."""
        return check_password_hash(self.password_hash, password)

    def has_role(self, role_name):
        """Verifica si el usuario tiene un rol específico."""
        return self.role_name == role_name

    def has_permission(self, permission):
        """Verifica si el usuario tiene un permiso específico basado en su rol."""
        role_permissions = {
            UserRole.ADMIN.value: [p.value for p in UserPermission],
            UserRole.MANAGER.value: [
                UserPermission.MANAGE_KIOSKS.value,
                UserPermission.VIEW_DASHBOARD.value,
                UserPermission.VIEW_LOGS.value,
                UserPermission.UPDATE_KIOSK.value,
                UserPermission.VIEW_KIOSK.value
            ],
            UserRole.OPERATOR.value: [
                UserPermission.UPDATE_KIOSK.value,
                UserPermission.VIEW_KIOSK.value,
                UserPermission.VIEW_DASHBOARD.value
            ],
            UserRole.VIEWER.value: [
                UserPermission.VIEW_KIOSK.value,
                UserPermission.VIEW_DASHBOARD.value
            ]
        }
        return permission in role_permissions.get(self.role_name, [])

    def to_dict(self):
        """Convierte el usuario a diccionario."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role_name,
            'is_active': self.is_active,
            'two_factor_enabled': self.two_factor_enabled,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'account_locked': self.account_locked
        }

    def __repr__(self):
        return f'<User {self.username}>' 