"""
Tests de rendimiento para operaciones de caché.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
import json
import redis
from datetime import datetime, timedelta
from app.services.cache_service import CacheService
from app.services.kiosk_service import KioskService
from app.models import Kiosk

@pytest.fixture(scope="module")
def redis_client():
    """Fixture para crear un cliente Redis."""
    client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True
    )
    
    # Limpiar base de datos de prueba
    client.flushdb()
    
    yield client
    
    # Limpiar después de las pruebas
    client.flushdb()
    client.close()

@pytest.fixture(scope="module")
def cache_service(redis_client):
    """Fixture para crear un servicio de caché."""
    return CacheService(redis_client)

def test_simple_cache_performance(benchmark, cache_service):
    """Test de rendimiento para operaciones simples de caché."""
    def cache_operation():
        # Escribir
        cache_service.set("test_key", "test_value", 300)
        # Leer
        return cache_service.get("test_key")
    
    result = benchmark(cache_operation)
    assert result == "test_value"

def test_bulk_cache_performance(benchmark, cache_service):
    """Test de rendimiento para operaciones masivas de caché."""
    def bulk_operation():
        pipeline = cache_service.redis.pipeline()
        
        # Escribir múltiples valores
        for i in range(1000):
            pipeline.set(f"bulk_key_{i}", f"value_{i}", 300)
        
        # Ejecutar pipeline
        pipeline.execute()
        
        # Leer múltiples valores
        return cache_service.redis.mget([f"bulk_key_{i}" for i in range(1000)])
    
    results = benchmark(bulk_operation)
    assert len(results) == 1000
    assert all(v is not None for v in results)

def test_json_cache_performance(benchmark, cache_service):
    """Test de rendimiento para caché de objetos JSON."""
    test_data = {
        "id": 1,
        "name": "Test Object",
        "attributes": {
            "color": "blue",
            "size": "large",
            "tags": ["tag1", "tag2", "tag3"]
        },
        "metrics": {
            "value1": 100,
            "value2": 200,
            "nested": {
                "value3": 300,
                "value4": 400
            }
        }
    }
    
    def json_operation():
        # Serializar y guardar
        cache_service.set_json("json_key", test_data, 300)
        # Leer y deserializar
        return cache_service.get_json("json_key")
    
    result = benchmark(json_operation)
    assert result["name"] == "Test Object"
    assert len(result["attributes"]["tags"]) == 3

def test_cache_pattern_performance(benchmark, cache_service):
    """Test de rendimiento para operaciones con patrones de caché."""
    def pattern_operation():
        # Escribir valores con patrón
        for i in range(100):
            cache_service.set(f"pattern:test:{i}", f"value_{i}", 300)
        
        # Buscar claves por patrón
        keys = cache_service.redis.keys("pattern:test:*")
        
        # Leer valores encontrados
        return cache_service.redis.mget(keys)
    
    results = benchmark(pattern_operation)
    assert len(results) == 100

def test_cache_expiry_performance(benchmark, cache_service):
    """Test de rendimiento para operaciones con expiración."""
    def expiry_operation():
        # Escribir valores con diferentes tiempos de expiración
        for i in range(100):
            cache_service.set(
                f"expiry_key_{i}",
                f"value_{i}",
                (i + 1) * 60  # Diferentes tiempos de expiración
            )
        
        # Verificar tiempos de expiración
        ttls = [
            cache_service.redis.ttl(f"expiry_key_{i}")
            for i in range(100)
        ]
        
        return ttls
    
    results = benchmark(expiry_operation)
    assert len(results) == 100
    assert all(ttl > 0 for ttl in results)

def test_cache_invalidation_performance(benchmark, cache_service):
    """Test de rendimiento para invalidación de caché."""
    def invalidation_operation():
        # Escribir valores
        for i in range(100):
            cache_service.set(f"invalid:key:{i}", f"value_{i}", 300)
        
        # Invalidar por patrón
        pattern = "invalid:key:*"
        keys = cache_service.redis.keys(pattern)
        
        if keys:
            return cache_service.redis.delete(*keys)
        return 0
    
    deleted_count = benchmark(invalidation_operation)
    assert deleted_count == 100
    
def test_concurrent_cache_performance(benchmark, cache_service):
    """Test de rendimiento para operaciones concurrentes."""
    def concurrent_operation():
        pipeline = cache_service.redis.pipeline()
        
        # Simular operaciones concurrentes
        for i in range(100):
            # Incrementar contador
            pipeline.incr(f"counter:{i}")
            # Verificar existencia
            pipeline.exists(f"counter:{i}")
            # Obtener valor
            pipeline.get(f"counter:{i}")
        
        return pipeline.execute()
    
    results = benchmark(concurrent_operation)
    assert len(results) == 300  # 3 operaciones por iteración 