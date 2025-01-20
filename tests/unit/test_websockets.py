# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from app.websockets import (
    socketio, format_timestamp, emit_current_metrics,
    emit_new_alert, emit_anomaly_detected
)
from app.models.kiosk import Kiosk, SensorData

@pytest.fixture
def mock_app():
    """Fixture para mock de la aplicación Flask"""
    app = MagicMock()
    app.logger = MagicMock()
    return app

@pytest.fixture
def mock_socketio():
    """Fixture para mock de SocketIO"""
    return MagicMock()

def test_format_timestamp():
    """Prueba el formateo de timestamps"""
    dt = datetime(2024, 1, 1, 12, 0, 0)
    formatted = format_timestamp(dt)
    assert formatted == '2024-01-01 12:00:00'

@patch('app.websockets.Kiosk')
@patch('app.websockets.SensorData')
def test_emit_current_metrics(mock_sensor_data, mock_kiosk, mock_socketio):
    """Prueba la emisión de métricas actuales"""
    # Configurar mocks
    mock_kiosk.query.count.return_value = 10
    mock_kiosk.query.filter_by().count.return_value = 8
    
    mock_data = [
        MagicMock(cpu_usage=80, memory_usage=70),
        MagicMock(cpu_usage=60, memory_usage=50)
    ]
    mock_sensor_data.query.order_by().limit().all.return_value = mock_data
    
    with patch('app.websockets.socketio', mock_socketio):
        emit_current_metrics()
    
    # Verificar que se emitieron las métricas correctas
    mock_socketio.emit.assert_called_once()
    args = mock_socketio.emit.call_args[0]
    assert args[0] == 'metrics_update'
    metrics = args[1]
    
    assert metrics['total_kiosks'] == 10
    assert metrics['online_kiosks'] == 8
    assert metrics['avg_cpu'] == 70  # (80 + 60) / 2
    assert metrics['avg_memory'] == 60  # (70 + 50) / 2

@patch('app.websockets.Kiosk')
def test_emit_new_alert(mock_kiosk, mock_socketio):
    """Prueba la emisión de nuevas alertas"""
    # Configurar mock
    mock_kiosk_instance = MagicMock()
    mock_kiosk_instance.name = 'Test Kiosk'
    mock_kiosk.query.get.return_value = mock_kiosk_instance
    
    with patch('app.websockets.socketio', mock_socketio):
        emit_new_alert(1, 'Test Alert')
    
    # Verificar que se emitió la alerta correcta
    mock_socketio.emit.assert_called_once()
    args = mock_socketio.emit.call_args[0]
    assert args[0] == 'new_alert'
    alert = args[1]
    
    assert alert['kiosk_id'] == 1
    assert alert['kiosk_name'] == 'Test Kiosk'
    assert alert['message'] == 'Test Alert'

@patch('app.websockets.Kiosk')
def test_emit_anomaly_detected(mock_kiosk, mock_socketio):
    """Prueba la emisión de anomalías detectadas"""
    # Configurar mock
    mock_kiosk_instance = MagicMock()
    mock_kiosk_instance.name = 'Test Kiosk'
    mock_kiosk.query.get.return_value = mock_kiosk_instance
    
    metrics = {
        'cpu_usage': 95,
        'memory_usage': 90,
        'network_latency': 300
    }
    
    with patch('app.websockets.socketio', mock_socketio):
        emit_anomaly_detected(1, 0.95, metrics)
    
    # Verificar que se emitió la anomalía correcta
    mock_socketio.emit.assert_called_once()
    args = mock_socketio.emit.call_args[0]
    assert args[0] == 'new_anomaly'
    anomaly = args[1]
    
    assert anomaly['kiosk_id'] == 1
    assert anomaly['kiosk_name'] == 'Test Kiosk'
    assert anomaly['probability'] == 0.95
    assert anomaly['metrics'] == metrics

def test_emit_current_metrics_no_data(mock_socketio):
    """Prueba la emisión de métricas cuando no hay datos"""
    with patch('app.websockets.SensorData') as mock_sensor_data:
        mock_sensor_data.query.order_by().limit().all.return_value = []
        
        with patch('app.websockets.socketio', mock_socketio):
            emit_current_metrics()
        
        # Verificar que se emitieron métricas con valores por defecto
        mock_socketio.emit.assert_called_once()
        args = mock_socketio.emit.call_args[0]
        metrics = args[1]
        
        assert metrics['avg_cpu'] == 0
        assert metrics['avg_memory'] == 0

def test_emit_new_alert_no_kiosk(mock_socketio):
    """Prueba la emisión de alertas cuando no existe el kiosk"""
    with patch('app.websockets.Kiosk') as mock_kiosk:
        mock_kiosk.query.get.return_value = None
        
        with patch('app.websockets.socketio', mock_socketio):
            emit_new_alert(999, 'Test Alert')
        
        # Verificar que no se emitió ninguna alerta
        mock_socketio.emit.assert_not_called()

@patch('app.websockets.current_app')
def test_error_handling(mock_current_app, mock_socketio):
    """Prueba el manejo de errores en las emisiones"""
    mock_socketio.emit.side_effect = Exception('Test error')
    
    with patch('app.websockets.socketio', mock_socketio):
        emit_current_metrics()
    
    # Verificar que se registró el error
    mock_current_app.logger.error.assert_called_once() 