"""
Servicio de geolocalización.
Sigue el patrón MVT + S.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import requests
from flask import current_app
from app.models.kiosk import Kiosk
from app.models.base import db

class GeolocationService:
    """Servicio para manejar la geolocalización de kiosks"""
    
    def __init__(self):
        """Inicializa el servicio de geolocalización"""
        self.logger = logging.getLogger(__name__)
        
    def update_kiosk_location(
        self, 
        kiosk: Kiosk, 
        latitude: float, 
        longitude: float,
        altitude: Optional[float] = None,
        accuracy: Optional[float] = None
    ) -> bool:
        """
        Actualiza la ubicación de un kiosk
        
        Args:
            kiosk: Instancia del kiosk a actualizar
            latitude: Latitud en grados decimales
            longitude: Longitud en grados decimales
            altitude: Altitud en metros (opcional)
            accuracy: Precisión en metros (opcional)
            
        Returns:
            bool: True si la actualización fue exitosa
        """
        try:
            kiosk.update_location(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                accuracy=accuracy
            )
            
            # Actualizar dirección basada en coordenadas
            address = self.get_address_from_coordinates(latitude, longitude)
            if address:
                kiosk.location = address
            
            db.session.commit()
            
            self.logger.info(
                f"Ubicación actualizada para kiosk {kiosk.uuid} en "
                f"({latitude}, {longitude})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando ubicación: {str(e)}")
            db.session.rollback()
            return False
    
    def get_address_from_coordinates(
        self, 
        latitude: float, 
        longitude: float
    ) -> Optional[str]:
        """
        Obtiene la dirección a partir de coordenadas usando un servicio de geocodificación
        
        Args:
            latitude: Latitud en grados decimales
            longitude: Longitud en grados decimales
            
        Returns:
            str: Dirección formateada o None si hay error
        """
        try:
            # Usar Nominatim (OpenStreetMap) como servicio de geocodificación
            url = (
                f"https://nominatim.openstreetmap.org/reverse"
                f"?format=json&lat={latitude}&lon={longitude}"
            )
            
            headers = {
                'User-Agent': current_app.config.get(
                    'GEOCODING_USER_AGENT', 
                    'KioskAdminSystem/1.0'
                )
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get('display_name')
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error en geocodificación inversa: {str(e)}")
            return None
    
    def get_nearby_kiosks(
        self, 
        latitude: float, 
        longitude: float, 
        radius_km: float = 5.0
    ) -> list:
        """
        Encuentra kiosks cercanos a una ubicación
        
        Args:
            latitude: Latitud en grados decimales
            longitude: Longitud en grados decimales
            radius_km: Radio de búsqueda en kilómetros
            
        Returns:
            list: Lista de kiosks dentro del radio especificado
        """
        try:
            # Consulta usando la fórmula del haversine
            # Nota: Esta es una aproximación, para mayor precisión usar PostGIS
            nearby_kiosks = Kiosk.query.filter(
                db.text(
                    """
                    6371 * 2 * ASIN(
                        SQRT(
                            POWER(SIN((RADIANS(:lat) - RADIANS(latitude)) / 2), 2) +
                            COS(RADIANS(:lat)) * COS(RADIANS(latitude)) *
                            POWER(SIN((RADIANS(:lon) - RADIANS(longitude)) / 2), 2)
                        )
                    ) <= :radius
                    """
                )
            ).params(lat=latitude, lon=longitude, radius=radius_km).all()
            
            return nearby_kiosks
            
        except Exception as e:
            self.logger.error(f"Error buscando kiosks cercanos: {str(e)}")
            return []
    
    def validate_coordinates(
        self, 
        latitude: float, 
        longitude: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Valida que las coordenadas sean válidas
        
        Args:
            latitude: Latitud a validar
            longitude: Longitud a validar
            
        Returns:
            Tuple[bool, str]: (es_válido, mensaje_error)
        """
        try:
            if not -90 <= latitude <= 90:
                return False, "Latitud debe estar entre -90 y 90 grados"
                
            if not -180 <= longitude <= 180:
                return False, "Longitud debe estar entre -180 y 180 grados"
                
            return True, None
            
        except Exception as e:
            return False, f"Error validando coordenadas: {str(e)}" 