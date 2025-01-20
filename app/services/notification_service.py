# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import smtplib
import requests
import logging
from typing import Dict, Any, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app
from abc import ABC, abstractmethod

class NotificationChannel(ABC):
    """Clase base abstracta para canales de notificación"""
    
    @abstractmethod
    def send_notification(self, message: Dict[str, Any]) -> bool:
        """
        Envía una notificación
        
        Args:
            message: Mensaje a enviar con formato y detalles
            
        Returns:
            bool: True si se envió correctamente
        """
        pass

class EmailChannel(NotificationChannel):
    """Canal de notificación por email"""
    
    def __init__(self):
        """Inicializa el canal de email"""
        self._smtp_host = None
        self._smtp_port = None
        self._smtp_user = None
        self._smtp_pass = None
        self._from_email = None
        self._to_emails = None
        
    @property
    def smtp_host(self):
        if self._smtp_host is None:
            self._smtp_host = current_app.config['SMTP_HOST']
        return self._smtp_host
        
    @property
    def smtp_port(self):
        if self._smtp_port is None:
            self._smtp_port = current_app.config['SMTP_PORT']
        return self._smtp_port
        
    @property
    def smtp_user(self):
        if self._smtp_user is None:
            self._smtp_user = current_app.config['SMTP_USER']
        return self._smtp_user
        
    @property
    def smtp_pass(self):
        if self._smtp_pass is None:
            self._smtp_pass = current_app.config['SMTP_PASS']
        return self._smtp_pass
        
    @property
    def from_email(self):
        if self._from_email is None:
            self._from_email = current_app.config['ALERT_FROM_EMAIL']
        return self._from_email
        
    @property
    def to_emails(self):
        if self._to_emails is None:
            self._to_emails = current_app.config['ALERT_TO_EMAILS']
        return self._to_emails
        
    def send_notification(self, message: Dict[str, Any]) -> bool:
        """Envía notificación por email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{message['severity'].upper()}] {message['name']}"
            
            body = f"""
            Alerta del Sistema
            ------------------
            Nombre: {message['name']}
            Severidad: {message['severity']}
            Mensaje: {message['message']}
            Timestamp: {message['timestamp']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logging.error(f"Error enviando email: {str(e)}")
            return False

class SlackChannel(NotificationChannel):
    """Canal de notificación por Slack"""
    
    def __init__(self):
        """Inicializa el canal de Slack"""
        self._webhook_url = None
        self._channel = None
        
    @property
    def webhook_url(self):
        if self._webhook_url is None:
            self._webhook_url = current_app.config['SLACK_WEBHOOK_URL']
        return self._webhook_url
        
    @property
    def channel(self):
        if self._channel is None:
            self._channel = current_app.config['SLACK_CHANNEL']
        return self._channel
        
    def send_notification(self, message: Dict[str, Any]) -> bool:
        """Envía notificación por Slack"""
        try:
            severity_emoji = {
                'low': ':information_source:',
                'medium': ':warning:',
                'high': ':rotating_light:',
                'critical': ':skull:'
            }
            
            payload = {
                'channel': self.channel,
                'username': 'Sistema de Alertas',
                'icon_emoji': severity_emoji.get(message['severity'], ':bell:'),
                'blocks': [
                    {
                        'type': 'header',
                        'text': {
                            'type': 'plain_text',
                            'text': f"Alerta: {message['name']}"
                        }
                    },
                    {
                        'type': 'section',
                        'fields': [
                            {
                                'type': 'mrkdwn',
                                'text': f"*Severidad:*\n{message['severity']}"
                            },
                            {
                                'type': 'mrkdwn',
                                'text': f"*Timestamp:*\n{message['timestamp']}"
                            }
                        ]
                    },
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f"*Mensaje:*\n{message['message']}"
                        }
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"Error enviando mensaje a Slack: {str(e)}")
            return False

class SMSChannel(NotificationChannel):
    """Canal de notificación por SMS (usando Twilio)"""
    
    def __init__(self):
        """Inicializa el canal de SMS"""
        self._account_sid = None
        self._auth_token = None
        self._from_number = None
        self._to_numbers = None
        
    @property
    def account_sid(self):
        if self._account_sid is None:
            self._account_sid = current_app.config['TWILIO_ACCOUNT_SID']
        return self._account_sid
        
    @property
    def auth_token(self):
        if self._auth_token is None:
            self._auth_token = current_app.config['TWILIO_AUTH_TOKEN']
        return self._auth_token
        
    @property
    def from_number(self):
        if self._from_number is None:
            self._from_number = current_app.config['TWILIO_FROM_NUMBER']
        return self._from_number
        
    @property
    def to_numbers(self):
        if self._to_numbers is None:
            self._to_numbers = current_app.config['ALERT_TO_NUMBERS']
        return self._to_numbers
        
    def send_notification(self, message: Dict[str, Any]) -> bool:
        """Envía notificación por SMS"""
        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            
            sms_body = (
                f"[{message['severity'].upper()}] {message['name']}\n"
                f"Mensaje: {message['message']}\n"
                f"Timestamp: {message['timestamp']}"
            )
            
            success = True
            for number in self.to_numbers:
                try:
                    client.messages.create(
                        body=sms_body,
                        from_=self.from_number,
                        to=number
                    )
                except Exception as e:
                    logging.error(f"Error enviando SMS a {number}: {str(e)}")
                    success = False
            
            return success
            
        except Exception as e:
            logging.error(f"Error en el servicio SMS: {str(e)}")
            return False

class NotificationService:
    """Servicio central de notificaciones"""
    
    def __init__(self):
        """Inicializa el servicio de notificaciones"""
        self.channels = {
            'email': EmailChannel(),
            'slack': SlackChannel(),
            'sms': SMSChannel()
        }
        self.logger = logging.getLogger('notifications')
    
    def send_alert(self, alert: Dict[str, Any]) -> Dict[str, bool]:
        """
        Envía una alerta a través de los canales especificados
        
        Args:
            alert: Datos de la alerta a enviar
            
        Returns:
            Dict[str, bool]: Estado de envío por cada canal
        """
        results = {}
        
        for channel in alert.get('channels', []):
            if channel in self.channels:
                success = self.channels[channel].send_notification(alert)
                results[channel] = success
                
                # Registrar resultado
                if success:
                    self.logger.info(
                        f"Alerta '{alert['name']}' enviada por {channel}"
                    )
                else:
                    self.logger.error(
                        f"Error enviando alerta '{alert['name']}' por {channel}"
                    )
            else:
                self.logger.warning(
                    f"Canal de notificación '{channel}' no implementado"
                )
                results[channel] = False
        
        return results
    
    def send_bulk_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Envía múltiples alertas
        
        Args:
            alerts: Lista de alertas a enviar
            
        Returns:
            List[Dict]: Resultados del envío de cada alerta
        """
        results = []
        
        for alert in alerts:
            channel_results = self.send_alert(alert)
            results.append({
                'alert': alert['name'],
                'results': channel_results
            })
        
        return results 