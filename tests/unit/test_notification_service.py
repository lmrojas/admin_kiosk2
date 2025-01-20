# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from app.services.notification_service import (
    NotificationService,
    EmailChannel,
    SlackChannel,
    SMSChannel
)

@pytest.fixture
def notification_service(app):
    """Fixture que proporciona una instancia del servicio de notificaciones"""
    with app.test_request_context():
        return NotificationService()

@pytest.fixture
def sample_alert():
    """Fixture que proporciona una alerta de ejemplo"""
    return {
        'name': 'test_alert',
        'severity': 'high',
        'message': 'Test alert message',
        'timestamp': datetime.utcnow().isoformat(),
        'channels': ['email', 'slack', 'sms']
    }

def test_email_notification(app, notification_service, sample_alert):
    """Prueba el envío de notificaciones por email"""
    with patch('smtplib.SMTP') as mock_smtp:
        # Configurar el mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Enviar alerta
        results = notification_service.send_alert({
            **sample_alert,
            'channels': ['email']
        })
        
        # Verificar que se llamó al servidor SMTP
        assert mock_server.starttls.called
        assert mock_server.login.called
        assert mock_server.send_message.called
        assert results['email'] is True

def test_slack_notification(app, notification_service, sample_alert):
    """Prueba el envío de notificaciones por Slack"""
    with patch('requests.post') as mock_post:
        # Configurar el mock
        mock_post.return_value.status_code = 200
        
        # Enviar alerta
        results = notification_service.send_alert({
            **sample_alert,
            'channels': ['slack']
        })
        
        # Verificar que se hizo la llamada POST
        assert mock_post.called
        assert mock_post.call_args[1]['json']['username'] == 'Sistema de Alertas'
        assert results['slack'] is True

def test_sms_notification(app, notification_service, sample_alert):
    """Prueba el envío de notificaciones por SMS"""
    with patch('twilio.rest.Client') as mock_client:
        # Configurar el mock
        mock_messages = MagicMock()
        mock_client.return_value.messages = mock_messages
        
        # Enviar alerta
        results = notification_service.send_alert({
            **sample_alert,
            'channels': ['sms']
        })
        
        # Verificar que se llamó al cliente de Twilio
        assert mock_messages.create.called
        assert results['sms'] is True

def test_multiple_channels(app, notification_service, sample_alert):
    """Prueba el envío de notificaciones por múltiples canales"""
    with patch('smtplib.SMTP') as mock_smtp, \
         patch('requests.post') as mock_post, \
         patch('twilio.rest.Client') as mock_client:
        
        # Configurar mocks
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_client.return_value.messages = MagicMock()
        
        # Enviar alerta
        results = notification_service.send_alert(sample_alert)
        
        # Verificar que se usaron todos los canales
        assert all(channel in results for channel in ['email', 'slack', 'sms'])
        assert all(results.values())

def test_invalid_channel(app, notification_service, sample_alert):
    """Prueba el manejo de canales inválidos"""
    # Enviar alerta con canal inválido
    results = notification_service.send_alert({
        **sample_alert,
        'channels': ['invalid_channel']
    })
    
    # Verificar que se marcó como fallido
    assert results['invalid_channel'] is False

def test_channel_failure(app, notification_service, sample_alert):
    """Prueba el manejo de fallos en los canales"""
    with patch('smtplib.SMTP') as mock_smtp:
        # Simular fallo en SMTP
        mock_smtp.side_effect = Exception('SMTP error')
        
        # Enviar alerta
        results = notification_service.send_alert({
            **sample_alert,
            'channels': ['email']
        })
        
        # Verificar que se marcó como fallido
        assert results['email'] is False

def test_bulk_notifications(app, notification_service, sample_alert):
    """Prueba el envío de múltiples alertas"""
    with patch('smtplib.SMTP') as mock_smtp:
        # Configurar mock
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        
        # Crear múltiples alertas
        alerts = [
            {**sample_alert, 'name': f'alert_{i}'}
            for i in range(3)
        ]
        
        # Enviar alertas en bulk
        results = notification_service.send_bulk_alerts(alerts)
        
        # Verificar resultados
        assert len(results) == 3
        assert all('alert_' in r['alert'] for r in results)
        assert all(r['results']['email'] for r in results)

def test_notification_formatting(app, notification_service, sample_alert):
    """Prueba el formato de las notificaciones"""
    with patch('smtplib.SMTP') as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Enviar alerta
        notification_service.send_alert({
            **sample_alert,
            'channels': ['email']
        })
        
        # Verificar formato del mensaje
        send_message_call = mock_server.send_message.call_args
        message = send_message_call[0][0]
        
        assert f"[{sample_alert['severity'].upper()}]" in message['Subject']
        assert sample_alert['name'] in message['Subject']
        assert sample_alert['message'] in str(message.get_payload()[0])
``` 