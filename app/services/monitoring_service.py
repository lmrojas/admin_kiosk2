# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import current_app
from dataclasses import dataclass
from app.services.logging_service import LoggingService

@dataclass
class AlertConfig:
    """Configuración de una alerta"""
    name: str
    log_type: str
    conditions: Dict[str, Any]
    threshold: int
    time_window: timedelta
    severity: str
    notification_channels: List[str]

class MonitoringService:
    """Servicio para monitoreo y alertas del sistema"""
    
    def __init__(self):
        """Inicializa el servicio de monitoreo"""
        self._logger = None
        self._logging_service = None
        self._alert_configs = None
        
    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger('monitoring')
        return self._logger
        
    @property
    def logging_service(self):
        if self._logging_service is None:
            self._logging_service = LoggingService()
        return self._logging_service
        
    @property
    def alert_configs(self):
        if self._alert_configs is None:
            self._alert_configs = self._load_alert_configs()
        return self._alert_configs
        
    def _load_alert_configs(self) -> List[AlertConfig]:
        """Carga las configuraciones de alertas desde el archivo de configuración"""
        try:
            configs_path = os.path.join(current_app.root_path, 'config', 'alerts.json')
            if not os.path.exists(configs_path):
                return self._get_default_alert_configs()
                
            with open(configs_path, 'r') as f:
                configs_data = json.load(f)
                
            return [
                AlertConfig(
                    name=config['name'],
                    log_type=config['log_type'],
                    conditions=config['conditions'],
                    threshold=config['threshold'],
                    time_window=timedelta(minutes=config['time_window_minutes']),
                    severity=config['severity'],
                    notification_channels=config['notification_channels']
                )
                for config in configs_data
            ]
        except Exception as e:
            self.logger.error(f"Error loading alert configs: {str(e)}")
            return self._get_default_alert_configs()
    
    def _get_default_alert_configs(self) -> List[AlertConfig]:
        """Retorna configuraciones de alerta por defecto"""
        return [
            AlertConfig(
                name='failed_login_attempts',
                log_type='security',
                conditions={
                    'event_type': 'login_attempt',
                    'details.success': False
                },
                threshold=5,
                time_window=timedelta(minutes=15),
                severity='high',
                notification_channels=['email', 'slack']
            ),
            AlertConfig(
                name='critical_errors',
                log_type='error',
                conditions={
                    'error_type': ['DatabaseError', 'ConnectionError']
                },
                threshold=1,
                time_window=timedelta(minutes=5),
                severity='critical',
                notification_channels=['email', 'slack', 'sms']
            )
        ]
    
    def check_alert_conditions(self, alert_config: AlertConfig) -> bool:
        """
        Verifica si se cumplen las condiciones para una alerta
        
        Args:
            alert_config: Configuración de la alerta a verificar
            
        Returns:
            bool: True si se deben disparar las alertas
        """
        end_date = datetime.utcnow()
        start_date = end_date - alert_config.time_window
        
        # Obtener logs del período
        logs = self.logging_service.get_logs(
            alert_config.log_type,
            start_date,
            end_date
        )
        
        # Filtrar logs que cumplen las condiciones
        matching_logs = []
        for log in logs:
            matches = True
            for key, value in alert_config.conditions.items():
                if '.' in key:
                    # Manejar condiciones anidadas (e.g., 'details.success')
                    parts = key.split('.')
                    log_value = log
                    for part in parts:
                        log_value = log_value.get(part, None)
                        if log_value is None:
                            matches = False
                            break
                    if isinstance(value, list):
                        matches = matches and log_value in value
                    else:
                        matches = matches and log_value == value
                else:
                    if isinstance(value, list):
                        matches = matches and log.get(key) in value
                    else:
                        matches = matches and log.get(key) == value
            
            if matches:
                matching_logs.append(log)
        
        return len(matching_logs) >= alert_config.threshold
    
    def monitor_system(self) -> List[Dict[str, Any]]:
        """
        Monitorea el sistema y genera alertas si es necesario
        
        Returns:
            List[Dict]: Lista de alertas generadas
        """
        alerts = []
        for config in self.alert_configs:
            if self.check_alert_conditions(config):
                alert = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'name': config.name,
                    'severity': config.severity,
                    'message': f'Alert condition met for {config.name}',
                    'channels': config.notification_channels
                }
                alerts.append(alert)
                
                # Registrar la alerta
                self.logging_service.log_security_event(
                    event_type='alert_triggered',
                    details=alert,
                    level='WARNING'
                )
        
        return alerts
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del sistema basadas en los logs
        
        Returns:
            Dict: Métricas del sistema
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=1)
        
        # Obtener logs de la última hora
        security_logs = self.logging_service.get_logs('security', start_date, end_date)
        error_logs = self.logging_service.get_logs('error', start_date, end_date)
        audit_logs = self.logging_service.get_logs('audit', start_date, end_date)
        
        # Calcular métricas
        metrics = {
            'timestamp': end_date.isoformat(),
            'last_hour': {
                'security_events': len(security_logs),
                'errors': len(error_logs),
                'audit_events': len(audit_logs),
                'failed_logins': sum(
                    1 for log in security_logs
                    if log.get('event_type') == 'login_attempt' 
                    and not log.get('details', {}).get('success', True)
                ),
                'critical_errors': sum(
                    1 for log in error_logs
                    if log.get('error_type') in ['DatabaseError', 'ConnectionError']
                )
            }
        }
        
        return metrics
    
    def get_alerts_history(
        self,
        start_date: datetime,
        end_date: datetime,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de alertas
        
        Args:
            start_date: Fecha inicial
            end_date: Fecha final
            severity: Filtrar por severidad
            
        Returns:
            List[Dict]: Lista de alertas en el período
        """
        logs = self.logging_service.get_logs('security', start_date, end_date)
        alerts = [
            log for log in logs
            if log.get('event_type') == 'alert_triggered'
            and (severity is None or log.get('details', {}).get('severity') == severity)
        ]
        
        return [log.get('details', {}) for log in alerts] 