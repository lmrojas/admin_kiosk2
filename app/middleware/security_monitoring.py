"""
Middleware de monitoreo de seguridad en tiempo real.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from functools import wraps
from flask import request, current_app, g
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import logging
import redis
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Clase para eventos de seguridad."""
    timestamp: datetime
    event_type: str
    severity: str
    source_ip: str
    user_id: Optional[int]
    details: Dict
    
class SecurityMonitor:
    """Monitor de seguridad en tiempo real."""
    
    def __init__(self):
        """Inicializar monitor."""
        self._redis_client = None
        self.alert_thresholds = {
            'failed_login': {'count': 5, 'window': 300},  # 5 en 5 min
            'invalid_token': {'count': 10, 'window': 300},  # 10 en 5 min
            'rate_limit': {'count': 20, 'window': 60},  # 20 en 1 min
            'invalid_input': {'count': 15, 'window': 300},  # 15 en 5 min
            'suspicious_ip': {'count': 30, 'window': 3600}  # 30 en 1 hora
        }
        self.suspicious_patterns = [
            r'(?i)(union|select|insert|update|delete|drop).*sql',
            r'(?i)script.*alert\(.*\)',
            r'(?i)(eval|exec|system)\(.*\)',
            r'(?i)/etc/passwd',
            r'(?i)((\.\.)/+)+',
            r'(?i)\.(php|asp|aspx|jsp)$'
        ]
        
    @property
    def redis_client(self):
        """Obtener cliente Redis."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=current_app.config.get('REDIS_HOST', 'localhost'),
                port=current_app.config.get('REDIS_PORT', 6379),
                db=current_app.config.get('REDIS_SECURITY_DB', 1),
                password=current_app.config.get('REDIS_PASSWORD', '')
            )
        return self._redis_client
        
    def track_event(self, event: SecurityEvent) -> None:
        """
        Registra un evento de seguridad.
        
        Args:
            event: Evento a registrar
        """
        try:
            # Guardar evento en Redis
            event_key = f"security_event:{event.timestamp.isoformat()}"
            event_data = {
                'type': event.event_type,
                'severity': event.severity,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'details': json.dumps(event.details)
            }
            self.redis_client.hmset(event_key, event_data)
            self.redis_client.expire(event_key, 86400)  # 24 horas
            
            # Incrementar contadores
            self._increment_counters(event)
            
            # Verificar alertas
            self._check_alerts(event)
            
        except Exception as e:
            logger.error(f"Error registrando evento de seguridad: {str(e)}")
            
    def _increment_counters(self, event: SecurityEvent) -> None:
        """
        Incrementa contadores para el evento.
        
        Args:
            event: Evento a contabilizar
        """
        # Contador por tipo de evento
        type_key = f"count:{event.event_type}:{event.source_ip}"
        self.redis_client.incr(type_key)
        self.redis_client.expire(type_key, 3600)
        
        # Contador por IP
        ip_key = f"ip_events:{event.source_ip}"
        self.redis_client.incr(ip_key)
        self.redis_client.expire(ip_key, 3600)
        
        if event.user_id:
            # Contador por usuario
            user_key = f"user_events:{event.user_id}"
            self.redis_client.incr(user_key)
            self.redis_client.expire(user_key, 3600)
            
    def _check_alerts(self, event: SecurityEvent) -> None:
        """
        Verifica si se deben generar alertas.
        
        Args:
            event: Evento a verificar
        """
        if event.event_type in self.alert_thresholds:
            threshold = self.alert_thresholds[event.event_type]
            count_key = f"count:{event.event_type}:{event.source_ip}"
            count = int(self.redis_client.get(count_key) or 0)
            
            if count >= threshold['count']:
                self._generate_alert(event, count)
                
    def _generate_alert(self, event: SecurityEvent, count: int) -> None:
        """
        Genera una alerta de seguridad.
        
        Args:
            event: Evento que generó la alerta
            count: Contador actual
        """
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event.event_type,
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'count': count,
            'details': event.details
        }
        
        # Guardar alerta
        alert_key = f"security_alert:{alert['timestamp']}"
        self.redis_client.hmset(alert_key, alert)
        self.redis_client.expire(alert_key, 86400 * 7)  # 7 días
        
        # Log de alerta
        logger.warning(f"Alerta de seguridad: {json.dumps(alert)}")
        
    def check_request_patterns(self) -> Optional[str]:
        """
        Verifica patrones sospechosos en la request actual.
        
        Returns:
            str: Patrón detectado o None
        """
        # Verificar URL
        url = request.url
        for pattern in self.suspicious_patterns:
            if re.search(pattern, url):
                return pattern
                
        # Verificar parámetros
        for value in request.values.values():
            for pattern in self.suspicious_patterns:
                if re.search(pattern, str(value)):
                    return pattern
                    
        # Verificar headers
        for header in request.headers.values():
            for pattern in self.suspicious_patterns:
                if re.search(pattern, str(header)):
                    return pattern
                    
        return None
        
    def monitor_request(self):
        """Decorador para monitorear requests."""
        def decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                # Verificar patrones sospechosos
                suspicious_pattern = self.check_request_patterns()
                if suspicious_pattern:
                    self.track_event(SecurityEvent(
                        timestamp=datetime.utcnow(),
                        event_type='suspicious_pattern',
                        severity='high',
                        source_ip=request.remote_addr,
                        user_id=getattr(g.get('user', None), 'id', None),
                        details={
                            'pattern': suspicious_pattern,
                            'url': request.url,
                            'method': request.method
                        }
                    ))
                    
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    # Registrar errores de seguridad
                    if isinstance(e, (
                        ValueError, TypeError, AttributeError,
                        KeyError, IndexError
                    )):
                        self.track_event(SecurityEvent(
                            timestamp=datetime.utcnow(),
                            event_type='security_error',
                            severity='medium',
                            source_ip=request.remote_addr,
                            user_id=getattr(g.get('user', None), 'id', None),
                            details={
                                'error': str(e),
                                'type': e.__class__.__name__,
                                'url': request.url,
                                'method': request.method
                            }
                        ))
                    raise
                    
            return wrapped
        return decorator 