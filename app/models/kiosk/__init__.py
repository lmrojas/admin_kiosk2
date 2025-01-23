"""
Módulo de modelos relacionados con Kiosks.
Sigue el patrón MVT, centralizando todos los modelos relacionados.
"""

from .base import Kiosk
from .location_history import KioskLocationHistory
from .sensor_data import SensorData

__all__ = ['Kiosk', 'KioskLocationHistory', 'SensorData'] 