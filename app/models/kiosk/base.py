"""
Modelo base para kiosks.
Sigue el patrón MVT + S.
"""

from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import relationship
from flask_login import current_user
from app.models.base import db

class Kiosk(db.Model):
    """Modelo base para kiosks."""
    
    __tablename__ = 'kiosks'

    # Identificadores
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Información básica
    name = db.Column(db.String(100), nullable=False)
    store_name = db.Column(db.String(200), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    
    # Sistema Operativo
    os_name = db.Column(db.String(50), nullable=True)
    os_version = db.Column(db.String(50), nullable=True)
    os_platform = db.Column(db.String(50), nullable=True)
    
    # Chromium
    chromium_status = db.Column(db.String(20), nullable=True)
    chromium_version = db.Column(db.String(50), nullable=True)
    
    # Información de Geolocalización
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    altitude = db.Column(db.Float, nullable=True)
    location_updated_at = db.Column(db.DateTime, nullable=True)
    location_accuracy = db.Column(db.Float, nullable=True)
    
    # Ubicación reportada
    reported_latitude = db.Column(db.Float, nullable=True)
    reported_longitude = db.Column(db.Float, nullable=True)
    reported_altitude = db.Column(db.Float, nullable=True)
    reported_location_updated_at = db.Column(db.DateTime, nullable=True)
    reported_location_accuracy = db.Column(db.Float, nullable=True)
    
    # Estado
    status = db.Column(db.String(20), default='inactive')
    last_online = db.Column(db.DateTime, nullable=True)
    last_telemetry = db.Column(db.DateTime, nullable=True)
    
    # Hardware
    cpu_model = db.Column(db.String(100), nullable=True)
    cpu_temperature = db.Column(db.Float, nullable=True)
    ram_total = db.Column(db.Float, nullable=True)
    storage_total = db.Column(db.Float, nullable=True)
    disk_usage = db.Column(db.Float, nullable=True)
    disk_free = db.Column(db.Float, nullable=True)
    fan_rpm = db.Column(db.Integer, nullable=True)
    
    # Red
    ip_address = db.Column(db.String(45), nullable=True)
    public_ip = db.Column(db.String(45), nullable=True)
    mac_address = db.Column(db.String(17), nullable=True)
    wifi_signal_strength = db.Column(db.Float, nullable=True)
    connection_speed = db.Column(db.Float, nullable=True)
    network_latency = db.Column(db.Float, nullable=True)
    packets_sent = db.Column(db.Integer, nullable=True)
    packets_received = db.Column(db.Integer, nullable=True)
    packets_lost = db.Column(db.Integer, nullable=True)
    
    # Sensores
    ambient_temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    voltage = db.Column(db.Float, nullable=True)
    door_status = db.Column(db.String(20), nullable=True)
    
    # Seguridad
    last_unauthorized_access = db.Column(db.DateTime, nullable=True)
    block_reason = db.Column(db.String(200), nullable=True)
    
    # Tiempo
    local_timezone = db.Column(db.String(50), nullable=True)
    utc_offset = db.Column(db.Integer, nullable=True)
    
    # Capacidades
    capabilities = db.Column(db.JSON, nullable=True)
    credentials_hash = db.Column(db.String(128), nullable=True)
    
    # Metadatos
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    owner = relationship('User', foreign_keys=[owner_id], back_populates='owned_kiosks')
    
    # Métricas
    health_score = db.Column(db.Float, default=100.0)
    anomaly_probability = db.Column(db.Float, default=0.0)
    
    # WebSocket
    socket_id = db.Column(db.String(50))

    # Relaciones
    location_history = relationship('KioskLocationHistory', back_populates='kiosk', lazy='dynamic', cascade='all, delete-orphan')
    sensor_data = relationship('SensorData', back_populates='kiosk', lazy='dynamic')

    def update_location(self, latitude, longitude, altitude=None, accuracy=None):
        """Actualiza la ubicación asignada del kiosk."""
        from .location_history import KioskLocationHistory
        
        prev_lat = self.latitude
        prev_lon = self.longitude
        
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.location_accuracy = accuracy
        self.location_updated_at = datetime.utcnow()
        
        history = KioskLocationHistory(
            kiosk_id=self.id,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            timestamp=self.location_updated_at,
            location_type='assigned',
            previous_latitude=prev_lat,
            previous_longitude=prev_lon,
            change_reason='Actualización manual de ubicación',
            created_by=current_user.id if current_user and not current_user.is_anonymous else None
        )
        db.session.add(history)

    def update_reported_location(self, latitude, longitude, altitude=None, accuracy=None):
        """Actualiza la ubicación reportada por el kiosk."""
        from .location_history import KioskLocationHistory
        
        prev_lat = self.reported_latitude
        prev_lon = self.reported_longitude
        
        self.reported_latitude = latitude
        self.reported_longitude = longitude
        self.reported_altitude = altitude
        self.reported_location_accuracy = accuracy
        self.reported_location_updated_at = datetime.utcnow()
        
        history = KioskLocationHistory(
            kiosk_id=self.id,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            timestamp=self.reported_location_updated_at,
            location_type='reported',
            previous_latitude=prev_lat,
            previous_longitude=prev_lon,
            change_reason='Ubicación reportada por el kiosk',
            created_by=None
        )
        db.session.add(history)

    def to_dict(self):
        """Convierte el kiosk a un diccionario para la API."""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'status': self.status,
            'last_online': self.last_online.isoformat() if self.last_online else None,
            'last_telemetry': self.last_telemetry.isoformat() if self.last_telemetry else None,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'location_accuracy': self.location_accuracy,
            'location_updated_at': self.location_updated_at.isoformat() if self.location_updated_at else None,
            'reported_latitude': self.reported_latitude,
            'reported_longitude': self.reported_longitude,
            'reported_altitude': self.reported_altitude,
            'reported_location_accuracy': self.reported_location_accuracy,
            'reported_location_updated_at': self.reported_location_updated_at.isoformat() if self.reported_location_updated_at else None,
            'socket_id': self.socket_id
        } 