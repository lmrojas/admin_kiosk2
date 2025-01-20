"""
Tests de integración para el flujo de kiosks.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from flask import url_for
from app.models.kiosk import Kiosk
from app.services.kiosk_service import KioskService
from app.services.websocket_service import WebSocketService
import json
from datetime import datetime, timedelta

class TestKioskFlow:
    """Suite de tests de integración para el flujo completo de kiosks."""
    
    @pytest.fixture
    def kiosk_service(self, app):
        """Fixture para el servicio de kiosks."""
        return KioskService()
        
    @pytest.fixture
    def ws_service(self, app):
        """Fixture para el servicio de WebSocket."""
        return WebSocketService()
        
    def test_registro_kiosk_flow(self, client, kiosk_service):
        """Test del flujo completo de registro y configuración de kiosk."""
        # 1. Registro de kiosk
        kiosk_data = {
            'serial_number': 'TEST-KIOSK-001',
            'location': 'Test Location',
            'ip_address': '192.168.1.100',
            'mac_address': '00:11:22:33:44:55'
        }
        
        response = client.post('/api/kiosks/register', json=kiosk_data)
        assert response.status_code == 201
        
        # Verificar datos del kiosk
        kiosk = Kiosk.query.filter_by(serial_number=kiosk_data['serial_number']).first()
        assert kiosk is not None
        assert kiosk.status == 'registered'
        
        # 2. Configuración inicial
        config_data = {
            'display_brightness': 80,
            'volume_level': 50,
            'auto_update': True,
            'maintenance_window': '02:00-04:00'
        }
        
        response = client.post(
            f'/api/kiosks/{kiosk.id}/configure',
            json=config_data
        )
        assert response.status_code == 200
        
        # 3. Verificar configuración
        response = client.get(f'/api/kiosks/{kiosk.id}/config')
        assert response.status_code == 200
        config = response.json
        assert config['display_brightness'] == 80
        
    def test_monitoreo_kiosk_flow(self, client, kiosk_service, ws_service):
        """Test del flujo de monitoreo de kiosk."""
        # 1. Crear kiosk de prueba
        kiosk = kiosk_service.create_kiosk(
            serial_number='TEST-KIOSK-002',
            location='Test Location 2'
        )
        
        # 2. Simular datos de monitoreo
        monitoring_data = {
            'cpu_usage': 45.5,
            'ram_usage': 1024,
            'disk_space': 5000,
            'temperature': 38.2,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Enviar datos vía WebSocket
        ws_service.process_monitoring_data(kiosk.id, monitoring_data)
        
        # 3. Verificar datos almacenados
        response = client.get(f'/api/kiosks/{kiosk.id}/monitoring')
        assert response.status_code == 200
        data = response.json
        
        assert abs(data['cpu_usage'] - monitoring_data['cpu_usage']) < 0.1
        assert data['ram_usage'] == monitoring_data['ram_usage']
        
    def test_mantenimiento_kiosk_flow(self, client, kiosk_service):
        """Test del flujo de mantenimiento de kiosk."""
        # 1. Crear kiosk de prueba
        kiosk = kiosk_service.create_kiosk(
            serial_number='TEST-KIOSK-003',
            location='Test Location 3'
        )
        
        # 2. Programar mantenimiento
        maintenance_data = {
            'type': 'software_update',
            'scheduled_time': (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            'estimated_duration': 30,  # minutos
            'description': 'Actualización de software programada'
        }
        
        response = client.post(
            f'/api/kiosks/{kiosk.id}/maintenance',
            json=maintenance_data
        )
        assert response.status_code == 201
        
        # 3. Verificar mantenimiento programado
        response = client.get(f'/api/kiosks/{kiosk.id}/maintenance')
        assert response.status_code == 200
        maintenance = response.json
        assert maintenance['type'] == maintenance_data['type']
        
        # 4. Simular inicio de mantenimiento
        response = client.post(
            f'/api/kiosks/{kiosk.id}/maintenance/start',
            json={'maintenance_id': maintenance['id']}
        )
        assert response.status_code == 200
        
        # Verificar estado del kiosk
        response = client.get(f'/api/kiosks/{kiosk.id}')
        assert response.status_code == 200
        assert response.json['status'] == 'maintenance'
        
    def test_alertas_kiosk_flow(self, client, kiosk_service):
        """Test del flujo de alertas de kiosk."""
        # 1. Crear kiosk de prueba
        kiosk = kiosk_service.create_kiosk(
            serial_number='TEST-KIOSK-004',
            location='Test Location 4'
        )
        
        # 2. Generar alerta
        alert_data = {
            'type': 'high_temperature',
            'severity': 'high',
            'value': 85.5,
            'threshold': 80.0,
            'message': 'Temperatura crítica detectada'
        }
        
        response = client.post(
            f'/api/kiosks/{kiosk.id}/alerts',
            json=alert_data
        )
        assert response.status_code == 201
        
        # 3. Verificar alerta generada
        response = client.get(f'/api/kiosks/{kiosk.id}/alerts')
        assert response.status_code == 200
        alerts = response.json
        assert len(alerts) > 0
        assert alerts[0]['type'] == alert_data['type']
        
        # 4. Resolver alerta
        response = client.post(
            f'/api/kiosks/{kiosk.id}/alerts/{alerts[0]["id"]}/resolve',
            json={'resolution': 'Ventilación activada manualmente'}
        )
        assert response.status_code == 200
        
        # Verificar estado de alerta
        response = client.get(f'/api/kiosks/{kiosk.id}/alerts/{alerts[0]["id"]}')
        assert response.status_code == 200
        assert response.json['status'] == 'resolved' 