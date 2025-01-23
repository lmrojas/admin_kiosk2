"""
Modelo para datos de sensores de kiosks.
Sigue el patrón MVT + S.
"""

from datetime import datetime
from app.models.base import db
from sqlalchemy.orm import relationship

class SensorData(db.Model):
    """Modelo para almacenar datos de sensores de un Kiosk."""
    
    __tablename__ = 'sensor_data'
    
    id = db.Column(db.Integer, primary_key=True)
    kiosk_id = db.Column(db.Integer, db.ForeignKey('kiosks.id'), nullable=False)
    
    # Métricas de sistema
    cpu_usage = db.Column(db.Float, nullable=False)
    cpu_temperature = db.Column(db.Float, nullable=True)
    memory_usage = db.Column(db.Float, nullable=False)
    network_latency = db.Column(db.Float, nullable=True)
    
    # Sensores ambientales
    ambient_temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    voltage = db.Column(db.Float, nullable=True)
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relaciones
    kiosk = relationship('Kiosk', back_populates='sensor_data')
    
    def __repr__(self):
        return f'<SensorData for Kiosk {self.kiosk_id} at {self.timestamp}>'
    
    def to_dict(self):
        """Convierte el registro a diccionario."""
        return {
            'id': self.id,
            'kiosk_id': self.kiosk_id,
            'cpu_usage': self.cpu_usage,
            'cpu_temperature': self.cpu_temperature,
            'memory_usage': self.memory_usage,
            'network_latency': self.network_latency,
            'ambient_temperature': self.ambient_temperature,
            'humidity': self.humidity,
            'voltage': self.voltage,
            'timestamp': self.timestamp.isoformat()
        } 