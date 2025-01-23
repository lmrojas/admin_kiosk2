"""
Modelos de seguridad para Admin Kiosk.
Sigue el patrón MVT + S.
"""

from datetime import datetime
from app.models.base import db
from sqlalchemy.dialects.postgresql import JSONB
from enum import Enum

class EventType(str, Enum):
    """Tipos de eventos de seguridad."""
    ACCESS = 'ACCESS'
    LOGIN = 'LOGIN'
    LOGIN_FAILED = 'LOGIN_FAILED'
    LOGOUT = 'LOGOUT'
    PASSWORD_CHANGE = 'PASSWORD_CHANGE'
    PASSWORD_RESET = 'PASSWORD_RESET'
    TWO_FACTOR = 'TWO_FACTOR'
    TOKEN_REFRESH = 'TOKEN_REFRESH'

class SecurityAudit(db.Model):
    """Modelo para auditar accesos al sistema."""
    
    __tablename__ = 'security_audit'

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(45), nullable=False, 
                          comment='Dirección IP desde donde se realizó el acceso')
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True,
                       comment='ID del usuario que realizó la acción')
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow,
                         comment='Fecha y hora del evento')
    event_type = db.Column(db.String(20), nullable=False,
                          comment='Tipo de evento de seguridad')
    method = db.Column(db.String(10), nullable=True,
                      comment='Método HTTP utilizado')
    path = db.Column(db.String(255), nullable=True,
                    comment='Ruta accedida')

    def __repr__(self):
        return f'<SecurityAudit {self.event_type} {self.timestamp}>'

class SecurityEvent(db.Model):
    """Modelo para eventos de seguridad del sistema."""
    
    __tablename__ = 'security_event'

    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False,
                          comment='Tipo de evento de seguridad')
    description = db.Column(db.Text, nullable=False,
                          comment='Descripción del evento')
    severity = db.Column(db.String(20), nullable=False,
                        comment='Severidad del evento (LOW, MEDIUM, HIGH, CRITICAL)')
    event_metadata = db.Column(JSONB, nullable=True,
                        comment='Metadatos adicionales del evento')
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True,
                       comment='ID del usuario relacionado con el evento')
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow,
                         comment='Fecha y hora del evento')
    
    # Campos de resolución
    resolved = db.Column(db.Boolean, default=False,
                        comment='Indica si el evento ha sido resuelto')
    resolved_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True,
                           comment='ID del usuario que resolvió el evento')
    resolution_notes = db.Column(db.Text, nullable=True,
                                comment='Notas sobre la resolución del evento')
    resolved_at = db.Column(db.DateTime, nullable=True,
                           comment='Fecha y hora de resolución del evento')

    def __repr__(self):
        return f'<SecurityEvent {self.event_type} {self.severity} {self.timestamp}>'

    def resolve(self, user_id: int, notes: str) -> None:
        """
        Marca el evento como resuelto.
        
        Args:
            user_id: ID del usuario que resuelve el evento
            notes: Notas de resolución
        """
        self.resolved = True
        self.resolved_by = user_id
        self.resolution_notes = notes
        self.resolved_at = datetime.utcnow()
        db.session.add(self)
        db.session.commit() 