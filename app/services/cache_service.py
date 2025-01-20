"""
Servicio de caché.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from functools import wraps
from flask import current_app
import redis
import json
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import logging
import hashlib
import pickle

logger = logging.getLogger(__name__)

class CacheService:
    """Servicio para gestionar el caché del sistema."""
    
    def __init__(self):
        """Inicializar servicio de caché."""
        self._redis_client = None
        self.default_ttl = 3600  # 1 hora
        self.key_prefix = 'kiosk_cache:'
        
    @property
    def redis_client(self) -> redis.Redis:
        """Obtener cliente Redis."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=current_app.config.get('REDIS_HOST', 'localhost'),
                port=current_app.config.get('REDIS_PORT', 6379),
                db=current_app.config.get('REDIS_CACHE_DB', 0),
                password=current_app.config.get('REDIS_PASSWORD', ''),
                decode_responses=True
            )
        return self._redis_client
        
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del caché.
        
        Args:
            key: Clave a buscar
            
        Returns:
            Any: Valor almacenado o None si no existe
        """
        try:
            full_key = f"{self.key_prefix}{key}"
            value = self.redis_client.get(full_key)
            
            if value is None:
                return None
                
            return pickle.loads(value.encode())
            
        except Exception as e:
            logger.error(f"Error obteniendo caché para {key}: {str(e)}")
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Almacenar valor en caché.
        
        Args:
            key: Clave para almacenar
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos
            
        Returns:
            bool: True si se almacenó correctamente
        """
        try:
            full_key = f"{self.key_prefix}{key}"
            ttl = ttl or self.default_ttl
            
            # Serializar valor
            serialized = pickle.dumps(value)
            
            return self.redis_client.setex(
                full_key,
                ttl,
                serialized.decode()
            )
            
        except Exception as e:
            logger.error(f"Error almacenando en caché {key}: {str(e)}")
            return False
            
    def delete(self, key: str) -> bool:
        """
        Eliminar valor del caché.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        try:
            full_key = f"{self.key_prefix}{key}"
            return bool(self.redis_client.delete(full_key))
            
        except Exception as e:
            logger.error(f"Error eliminando caché {key}: {str(e)}")
            return False
            
    def clear_pattern(self, pattern: str) -> int:
        """
        Eliminar todas las claves que coincidan con el patrón.
        
        Args:
            pattern: Patrón de claves a eliminar
            
        Returns:
            int: Número de claves eliminadas
        """
        try:
            full_pattern = f"{self.key_prefix}{pattern}"
            keys = self.redis_client.keys(full_pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Error limpiando caché con patrón {pattern}: {str(e)}")
            return 0
            
    def cache_decorator(self, prefix: str, ttl: Optional[int] = None):
        """
        Decorador para cachear resultados de funciones.
        
        Args:
            prefix: Prefijo para la clave de caché
            ttl: Tiempo de vida en segundos
            
        Returns:
            Callable: Decorador configurado
        """
        def decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                # Generar clave única
                key_parts = [prefix, f.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                
                cache_key = "_".join(key_parts)
                
                # Intentar obtener del caché
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                    
                # Ejecutar función y almacenar resultado
                result = f(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            return wrapped
        return decorator
        
    def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Obtener múltiples valores del caché.
        
        Args:
            keys: Lista de claves a obtener
            
        Returns:
            Dict: Diccionario con los valores encontrados
        """
        try:
            full_keys = [f"{self.key_prefix}{key}" for key in keys]
            values = self.redis_client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = pickle.loads(value.encode())
                    
            return result
            
        except Exception as e:
            logger.error(f"Error en bulk_get: {str(e)}")
            return {}
            
    def bulk_set(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Almacenar múltiples valores en caché.
        
        Args:
            data: Diccionario con datos a almacenar
            ttl: Tiempo de vida en segundos
            
        Returns:
            bool: True si todos los valores se almacenaron correctamente
        """
        try:
            pipeline = self.redis_client.pipeline()
            ttl = ttl or self.default_ttl
            
            for key, value in data.items():
                full_key = f"{self.key_prefix}{key}"
                serialized = pickle.dumps(value)
                pipeline.setex(full_key, ttl, serialized.decode())
                
            pipeline.execute()
            return True
            
        except Exception as e:
            logger.error(f"Error en bulk_set: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del caché.
        
        Returns:
            Dict: Estadísticas del sistema de caché
        """
        try:
            info = self.redis_client.info()
            keys = len(self.redis_client.keys(f"{self.key_prefix}*"))
            
            return {
                'total_keys': keys,
                'used_memory': info.get('used_memory_human'),
                'hits': info.get('keyspace_hits'),
                'misses': info.get('keyspace_misses'),
                'evicted_keys': info.get('evicted_keys')
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {} 