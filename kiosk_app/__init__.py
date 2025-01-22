"""
Paquete kiosk_app para la simulación y gestión de kiosks.
"""

__version__ = '1.0.0'

# Importaciones necesarias para hacer el paquete funcional
from .kiosk_app import KioskApp
from .kiosk_spawner import KioskSpawner

__all__ = ['KioskApp', 'KioskSpawner'] 