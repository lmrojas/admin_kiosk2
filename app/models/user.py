# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from enum import Enum
from datetime import datetime
from app.models.base import db
from app import login_manager
from sqlalchemy.dialects.postgresql import JSON

class UserRole(Enum):
    """Roles de usuario."""
    ADMIN = 'admin'
    OPERATOR = 'operator'
    VIEWER = 'viewer'

class UserPermission(Enum):
    """Permisos de usuario."""
    VIEW_KIOSK = 'view_kiosk'
    CREATE_KIOSK = 'create_kiosk'
    UPDATE_KIOSK = 'update_kiosk'
    DELETE_KIOSK = 'delete_kiosk'
    VIEW_METRICS = 'view_metrics'
    VIEW_ALERTS = 'view_alerts'
    MANAGE_USERS = 'manage_users'

@login_manager.user_loader
def load_user(user_id):
    """Carga un usuario por su ID."""
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    """Modelo de usuario.
    Sigue el patrón MVT + S.
    """
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    role_name = db.Column(db.String(20), nullable=False, default=UserRole.VIEWER.value)
    
    # Campos para 2FA
    two_factor_enabled = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))
    backup_codes = db.Column(db.JSON)  # Lista de códigos de respaldo
    temp_2fa_code = db.Column(JSON)  # Código temporal y fecha de expiración
    
    # Campos de estado y auditoría
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    failed_login_attempts = db.Column(db.Integer, default=0)
    account_locked = db.Column(db.Boolean, default=False)
    
    # Relación con Kiosks
    owned_kiosks = db.relationship('Kiosk', back_populates='owner', lazy='dynamic')
    
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
        """Verifica si el usuario tiene un permiso específico."""
        role_permissions = {
            UserRole.ADMIN.value: [p.value for p in UserPermission],
            UserRole.OPERATOR.value: [
                UserPermission.UPDATE_KIOSK.value,
                UserPermission.VIEW_KIOSK.value,
                UserPermission.VIEW_METRICS.value,
                UserPermission.VIEW_ALERTS.value
            ],
            UserRole.VIEWER.value: [
                UserPermission.VIEW_KIOSK.value,
                UserPermission.VIEW_METRICS.value,
                UserPermission.VIEW_ALERTS.value
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

    def update_last_login(self):
        """Actualiza la fecha del último login y reinicia los intentos fallidos."""
        self.last_login = datetime.utcnow()
        self.failed_login_attempts = 0
        db.session.commit() 