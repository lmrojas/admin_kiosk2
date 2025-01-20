"""
Tests de integración para el sistema de caché.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from app.services.cache_service import CacheService
from app.services.kiosk_service import KioskService
from app.services.data_service import DataService
from datetime import datetime, timedelta
import json

class TestCacheFlow:
    """Suite de tests de integración para el sistema de caché."""
    
    @pytest.fixture
    def cache_service(self, app):
        """Fixture para el servicio de caché."""
        return CacheService()
        
    @pytest.fixture
    def kiosk_service(self, app):
        """Fixture para el servicio de kiosks."""
        return KioskService()
        
    @pytest.fixture
    def data_service(self, app):
        """Fixture para el servicio de datos."""
        return DataService()
        
    def test_kiosk_data_caching_flow(self, client, cache_service, kiosk_service):
        """Test del flujo de caché para datos de kiosks."""
        # 1. Crear kiosk de prueba
        kiosk = kiosk_service.create_kiosk(
            serial_number='CACHE-TEST-001',
            location='Cache Test Location'
        )
        
        # 2. Primera petición (sin caché)
        start_time = datetime.utcnow()
        response = client.get(f'/api/kiosks/{kiosk.id}')
        first_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        
        # 3. Segunda petición (con caché)
        start_time = datetime.utcnow()
        response = client.get(f'/api/kiosks/{kiosk.id}')
        second_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        
        # La segunda petición debería ser más rápida
        assert second_request_time < first_request_time
        
        # 4. Actualizar kiosk
        response = client.put(
            f'/api/kiosks/{kiosk.id}',
            json={'location': 'Updated Location'}
        )
        assert response.status_code == 200
        
        # 5. Verificar que el caché se invalidó
        cached_data = cache_service.get(f'kiosk:{kiosk.id}')
        assert cached_data is None
        
    def test_sensor_data_caching_flow(self, client, cache_service, data_service, kiosk_service):
        """Test del flujo de caché para datos de sensores."""
        # 1. Crear kiosk y datos de sensores
        kiosk = kiosk_service.create_kiosk(
            serial_number='CACHE-TEST-002',
            location='Cache Test Location 2'
        )
        
        sensor_data = []
        for i in range(100):
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            data = {
                'kiosk_id': kiosk.id,
                'cpu_usage': 50 + (i % 20),
                'ram_usage': 1000 + (i * 10),
                'temperature': 35 + (i % 5),
                'timestamp': timestamp.isoformat()
            }
            sensor_data.append(data)
            
        data_service.bulk_save_sensor_data(sensor_data)
        
        # 2. Primera petición de estadísticas (sin caché)
        start_time = datetime.utcnow()
        response = client.get(f'/api/kiosks/{kiosk.id}/stats')
        first_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        
        # 3. Segunda petición (con caché)
        start_time = datetime.utcnow()
        response = client.get(f'/api/kiosks/{kiosk.id}/stats')
        second_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        
        # Verificar mejora en tiempo de respuesta
        assert second_request_time < first_request_time
        
        # 4. Agregar nuevos datos
        new_data = {
            'kiosk_id': kiosk.id,
            'cpu_usage': 75,
            'ram_usage': 2000,
            'temperature': 40,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        response = client.post(
            f'/api/kiosks/{kiosk.id}/sensor-data',
            json=new_data
        )
        assert response.status_code == 201
        
        # 5. Verificar invalidación de caché
        cached_stats = cache_service.get(f'kiosk_stats:{kiosk.id}')
        assert cached_stats is None
        
    def test_dashboard_caching_flow(self, client, cache_service, kiosk_service):
        """Test del flujo de caché para el dashboard."""
        # 1. Crear varios kiosks
        kiosks = []
        for i in range(5):
            kiosk = kiosk_service.create_kiosk(
                serial_number=f'CACHE-TEST-{i+1:03d}',
                location=f'Location {i+1}'
            )
            kiosks.append(kiosk)
            
        # 2. Primera carga del dashboard (sin caché)
        start_time = datetime.utcnow()
        response = client.get('/api/dashboard/summary')
        first_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        
        # 3. Segunda carga (con caché)
        start_time = datetime.utcnow()
        response = client.get('/api/dashboard/summary')
        second_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        
        # Verificar mejora en tiempo de respuesta
        assert second_request_time < first_request_time
        
        # 4. Modificar un kiosk
        response = client.put(
            f'/api/kiosks/{kiosks[0].id}',
            json={'status': 'maintenance'}
        )
        assert response.status_code == 200
        
        # 5. Verificar invalidación del caché del dashboard
        cached_summary = cache_service.get('dashboard:summary')
        assert cached_summary is None
        
    def test_search_caching_flow(self, client, cache_service, kiosk_service):
        """Test del flujo de caché para búsquedas."""
        # 1. Crear kiosks con patrones similares
        locations = [
            'Shopping Mall North',
            'Shopping Mall South',
            'Shopping Mall East',
            'Airport Terminal 1',
            'Airport Terminal 2'
        ]
        
        for location in locations:
            kiosk_service.create_kiosk(
                serial_number=f'SEARCH-{location.replace(" ", "-")}',
                location=location
            )
            
        # 2. Primera búsqueda (sin caché)
        start_time = datetime.utcnow()
        response = client.get('/api/kiosks/search?q=Shopping')
        first_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        assert len(response.json) == 3
        
        # 3. Segunda búsqueda (con caché)
        start_time = datetime.utcnow()
        response = client.get('/api/kiosks/search?q=Shopping')
        second_request_time = datetime.utcnow() - start_time
        assert response.status_code == 200
        
        # Verificar mejora en tiempo de respuesta
        assert second_request_time < first_request_time
        
        # 4. Agregar nuevo kiosk que coincide
        new_kiosk = kiosk_service.create_kiosk(
            serial_number='SEARCH-Shopping-Mall-West',
            location='Shopping Mall West'
        )
        
        # 5. Verificar que el caché de búsqueda se invalidó
        cached_search = cache_service.get('search:Shopping')
        assert cached_search is None
        
        # 6. Nueva búsqueda debería incluir el nuevo kiosk
        response = client.get('/api/kiosks/search?q=Shopping')
        assert response.status_code == 200
        assert len(response.json) == 4 