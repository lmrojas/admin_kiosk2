"""
Módulo de modelos de la aplicación.
Sigue el patrón MVT, centralizando todos los modelos.
"""

from .kiosk import Kiosk, KioskLocationHistory, SensorData

__all__ = [
    'Kiosk',
    'KioskLocationHistory', 
    'SensorData'
] 