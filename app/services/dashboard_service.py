# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from flask import current_app
from .logging_service import LoggingService
from .backup_service import BackupService
from .monitoring_service import MonitoringService

class DashboardService:
    """Servicio para gestionar el dashboard de monitoreo"""
    
    def __init__(self):
        """Inicializa el servicio de dashboard"""
        self.logger = logging.getLogger('dashboard')
        self.logging_service = LoggingService()
        self.backup_service = BackupService()
        self.monitoring_service = MonitoringService()
    
    def get_system_status(self) -> Dict:
        """
        Obtiene el estado general del sistema
        
        Returns:
            Dict: Estado del sistema incluyendo métricas clave
        """
        try:
            return {
                'status': 'healthy',
                'last_check': datetime.utcnow().isoformat(),
                'metrics': self.get_system_metrics(),
                'alerts': self.get_recent_alerts(),
                'backups': self.get_backup_status()
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo estado del sistema: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_system_metrics(self) -> Dict:
        """
        Obtiene métricas del sistema
        
        Returns:
            Dict: Métricas del sistema
        """
        try:
            return {
                'logs': {
                    'total': self.logging_service.get_log_count(),
                    'errors': self.logging_service.get_error_count(),
                    'warnings': self.logging_service.get_warning_count()
                },
                'security': {
                    'failed_logins': self.logging_service.get_failed_login_count(),
                    'suspicious_activity': self.logging_service.get_suspicious_activity_count()
                },
                'performance': self.monitoring_service.get_performance_metrics()
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo métricas: {str(e)}")
            return {}
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """
        Obtiene alertas recientes
        
        Args:
            hours: Número de horas hacia atrás para buscar
            
        Returns:
            List[Dict]: Lista de alertas recientes
        """
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            return self.monitoring_service.get_alerts(since=since)
        except Exception as e:
            self.logger.error(f"Error obteniendo alertas: {str(e)}")
            return []
    
    def get_backup_status(self) -> Dict:
        """
        Obtiene estado de los backups
        
        Returns:
            Dict: Estado de los backups
        """
        try:
            backups = self.backup_service.get_backup_info()
            last_backup = backups[0] if backups else None
            
            return {
                'total_backups': len(backups),
                'last_backup': last_backup,
                'backup_size': sum(b['size'] for b in backups),
                'status': 'ok' if last_backup else 'warning'
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo estado de backups: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_log_summary(self, days: int = 7) -> List[Dict]:
        """
        Obtiene resumen de logs por día
        
        Args:
            days: Número de días para el resumen
            
        Returns:
            List[Dict]: Resumen diario de logs
        """
        try:
            since = datetime.utcnow() - timedelta(days=days)
            return self.logging_service.get_daily_summary(since=since)
        except Exception as e:
            self.logger.error(f"Error obteniendo resumen de logs: {str(e)}")
            return []
    
    def get_security_events(self, days: int = 7) -> List[Dict]:
        """
        Obtiene eventos de seguridad recientes
        
        Args:
            days: Número de días hacia atrás
            
        Returns:
            List[Dict]: Lista de eventos de seguridad
        """
        try:
            since = datetime.utcnow() - timedelta(days=days)
            return self.logging_service.get_security_events(since=since)
        except Exception as e:
            self.logger.error(f"Error obteniendo eventos de seguridad: {str(e)}")
            return [] 