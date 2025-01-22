# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import logging
import redis
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app
from typing import Optional, Callable, Dict, Any
from app.models.security import SecurityAudit, SecurityEvent
from app import db

class SecurityService:
    """Servicio unificado para gestionar todos los aspectos de seguridad"""
    
    def __init__(self):
        """Inicializa el servicio de seguridad"""
        self.logger = logging.getLogger(__name__)
        self._redis_client = None
    
    @property
    def redis_client(self):
        """Obtiene el cliente Redis, inicializándolo si es necesario"""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=current_app.config.get('REDIS_HOST', 'localhost'),
                port=current_app.config.get('REDIS_PORT', 6379),
                db=current_app.config.get('REDIS_DB', 0),
                password=current_app.config.get('REDIS_PASSWORD', '')
            )
        return self._redis_client
    
    def generate_jwt(self, user_id: int, additional_claims: Dict[str, Any] = None) -> str:
        """
        Genera un token JWT para un usuario
        
        Args:
            user_id: ID del usuario
            additional_claims: Claims adicionales para el token
            
        Returns:
            str: Token JWT generado
        """
        try:
            payload = {
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(days=1),
                'iat': datetime.utcnow()
            }
            
            if additional_claims:
                payload.update(additional_claims)
                
            return jwt.encode(
                payload,
                current_app.config['SECRET_KEY'],
                algorithm='HS256'
            )
            
        except Exception as e:
            self.logger.error(f"Error generando JWT: {str(e)}")
            raise

    def verify_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verifica un token JWT
        
        Args:
            token: Token JWT a verificar
            
        Returns:
            Dict o None: Payload del token si es válido, None si no
        """
        try:
            payload = jwt.decode(
                token,
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token JWT expirado")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Token JWT inválido: {str(e)}")
            return None

    def rate_limit(self, key: str, limit: int, period: int) -> bool:
        """
        Verifica el rate limiting para una clave
        
        Args:
            key: Clave para el rate limiting
            limit: Límite de peticiones
            period: Período en segundos
            
        Returns:
            bool: True si está dentro del límite, False si no
        """
        try:
            current = self.redis_client.get(key)
            if current is None:
                self.redis_client.setex(key, period, 1)
                return True
            
            count = int(current)
            if count >= limit:
                return False
                
            self.redis_client.incr(key)
            return True
            
        except Exception as e:
            self.logger.error(f"Error en rate limiting: {str(e)}")
            return True  # En caso de error, permitir el request

    def audit_log(self, user_id: Optional[int], action: str, resource: str, details: Dict[str, Any]) -> None:
        """
        Registra una acción en el log de auditoría
        
        Args:
            user_id: ID del usuario que realiza la acción
            action: Tipo de acción realizada
            resource: Recurso afectado
            details: Detalles adicionales de la acción
        """
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'details': details
            }
            
            self.logger.info(f"Audit log: {log_entry}")
            
        except Exception as e:
            self.logger.error(f"Error en audit log: {str(e)}")
    
    def rate_limit_decorator(self, limit: int = 60, period: int = 60) -> Callable:
        """
        Decorador para aplicar rate limiting a rutas.
        
        Args:
            limit (int): Número máximo de peticiones permitidas
            period (int): Período en segundos
            
        Returns:
            Callable: Decorador configurado
        """
        def decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                key = f"rate_limit:{request.remote_addr}:{request.endpoint}"
                
                if not self.rate_limit(key, limit, period):
                    return jsonify({
                        'error': 'Too many requests',
                        'retry_after': period
                    }), 429
                
                return f(*args, **kwargs)
            return wrapped
        return decorator

    def log_access(self, ip: str, method: str, path: str, user_id: Optional[int] = None) -> None:
        """
        Registra un intento de acceso al sistema.
        
        Args:
            ip: Dirección IP del acceso
            method: Método HTTP
            path: Ruta accedida
            user_id: ID del usuario (opcional)
        """
        try:
            audit = SecurityAudit(
                ip_address=ip,
                method=method,
                path=path,
                user_id=user_id,
                timestamp=datetime.now(),
                event_type='ACCESS'
            )
            db.session.add(audit)
            db.session.commit()
        except Exception as e:
            self.logger.error(f"Error al registrar acceso: {str(e)}", 
                        extra={'ip': ip})
            db.session.rollback()

    def log_security_event(self, 
                          event_type: str, 
                          description: str, 
                          severity: str,
                          metadata: Optional[Dict] = None,
                          user_id: Optional[int] = None) -> None:
        """
        Registra un evento de seguridad.
        
        Args:
            event_type: Tipo de evento
            description: Descripción del evento
            severity: Severidad del evento
            metadata: Metadatos adicionales (opcional)
            user_id: ID del usuario (opcional)
        """
        try:
            event = SecurityEvent(
                event_type=event_type,
                description=description,
                severity=severity,
                event_metadata=metadata or {},
                user_id=user_id,
                timestamp=datetime.now()
            )
            db.session.add(event)
            db.session.commit()
            
            if severity in ['HIGH', 'CRITICAL']:
                self.logger.critical(
                    f"Evento de seguridad crítico: {description}",
                    extra={
                        'event_type': event_type,
                        'severity': severity,
                        'user_id': user_id
                    }
                )
        except Exception as e:
            self.logger.error(f"Error al registrar evento de seguridad: {str(e)}")
            db.session.rollback()
            raise 