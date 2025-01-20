import logging
import logging.handlers
import os
from datetime import datetime
from flask import current_app, request
from typing import Optional, Dict, Any
import json

class LoggingService:
    """Servicio para gestionar el logging centralizado"""
    
    def __init__(self):
        """Inicializa el servicio de logging"""
        self._logger = None
        
    @property
    def logger(self):
        if self._logger is None:
            self._setup_logging()
        return self._logger
        
    def _setup_logging(self):
        """Configura los handlers y formatos de logging"""
        log_dir = os.path.join(current_app.root_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Logger principal
        self._logger = logging.getLogger('kiosk')
        self._logger.setLevel(current_app.config.get('LOG_LEVEL', 'INFO'))
        
        # Formato común para todos los logs
        formatter = logging.Formatter(
            fmt=current_app.config.get('LOG_FORMAT'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para archivo de logs general
        general_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'kiosk.log'),
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        general_handler.setFormatter(formatter)
        self._logger.addHandler(general_handler)
        
        # Handler para logs de seguridad
        security_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'security.log'),
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        security_handler.setFormatter(formatter)
        security_handler.addFilter(lambda record: record.name == 'security')
        self._logger.addHandler(security_handler)
        
        # Handler para logs de errores
        error_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'error.log'),
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        self._logger.addHandler(error_handler)
        
        # Handler para logs de auditoría
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'audit.log'),
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        audit_handler.setFormatter(formatter)
        audit_handler.addFilter(lambda record: record.name == 'audit')
        self._logger.addHandler(audit_handler)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], level: str = 'INFO'):
        """
        Registra un evento de seguridad
        
        Args:
            event_type: Tipo de evento de seguridad
            details: Detalles del evento
            level: Nivel de logging (INFO, WARNING, ERROR, etc.)
        """
        logger = logging.getLogger('security')
        log_func = getattr(logger, level.lower())
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'ip': request.remote_addr,
            'user_agent': request.user_agent.string,
            'details': details
        }
        
        log_func(json.dumps(log_data))
    
    def log_audit_event(self, user_id: Optional[int], action: str, resource: str, 
                       status: str, details: Dict[str, Any]):
        """
        Registra un evento de auditoría
        
        Args:
            user_id: ID del usuario que realiza la acción
            action: Tipo de acción realizada
            resource: Recurso afectado
            status: Estado de la acción (success/error)
            details: Detalles adicionales
        """
        logger = logging.getLogger('audit')
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'status': status,
            'ip': request.remote_addr,
            'user_agent': request.user_agent.string,
            'details': details
        }
        
        logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """
        Registra un error con contexto
        
        Args:
            error: Excepción ocurrida
            context: Contexto del error
        """
        logger = logging.getLogger('kiosk')
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'ip': request.remote_addr,
            'endpoint': request.endpoint,
            'method': request.method,
            'context': context
        }
        
        logger.error(json.dumps(log_data), exc_info=True)
    
    def get_logs(self, log_type: str, start_date: datetime, 
                end_date: datetime, limit: int = 100) -> list:
        """
        Obtiene logs filtrados por tipo y fecha
        
        Args:
            log_type: Tipo de log (security/audit/error)
            start_date: Fecha inicial
            end_date: Fecha final
            limit: Límite de registros
            
        Returns:
            list: Lista de registros de log
        """
        log_file = os.path.join(current_app.root_path, 'logs', f'{log_type}.log')
        if not os.path.exists(log_file):
            return []
        
        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.split(' - ')[-1])
                    log_date = datetime.fromisoformat(log_entry['timestamp'])
                    
                    if start_date <= log_date <= end_date:
                        logs.append(log_entry)
                        if len(logs) >= limit:
                            break
                except:
                    continue
        
        return logs 