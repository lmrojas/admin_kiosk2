from datetime import datetime
from app.extensions import db
from sqlalchemy.dialects.postgresql import ENUM

class KioskLocationHistory(db.Model):
    """Modelo para almacenar el historial de ubicaciones de los kiosks."""
    
    __tablename__ = 'kiosk_location_history'
    
    id = db.Column(db.Integer, primary_key=True)
    kiosk_id = db.Column(db.Integer, db.ForeignKey('kiosks.id', ondelete='CASCADE'), nullable=False, index=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    location_type = db.Column(
        ENUM('assigned', 'reported', name='location_type_enum'),
        nullable=False,
        index=True
    )
    previous_latitude = db.Column(db.Float, nullable=True)
    previous_longitude = db.Column(db.Float, nullable=True)
    change_reason = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Relaciones
    kiosk = db.relationship('Kiosk', backref=db.backref('location_history', lazy='dynamic'))
    user = db.relationship('User', backref=db.backref('location_changes', lazy='dynamic'))
    
    def __repr__(self):
        return f'<KioskLocationHistory {self.kiosk_id} {self.location_type} {self.timestamp}>'
    
    @property
    def has_previous_location(self):
        """Indica si hay una ubicación anterior registrada."""
        return self.previous_latitude is not None and self.previous_longitude is not None
    
    def to_dict(self):
        """Convierte el registro a diccionario."""
        return {
            'id': self.id,
            'kiosk_id': self.kiosk_id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'accuracy': self.accuracy,
            'timestamp': self.timestamp.isoformat(),
            'location_type': self.location_type,
            'previous_latitude': self.previous_latitude,
            'previous_longitude': self.previous_longitude,
            'change_reason': self.change_reason,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by
        } 