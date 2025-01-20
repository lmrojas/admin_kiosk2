"""
Tests unitarios para el servicio de caché.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from app.services.cache_service import CacheService
from datetime import datetime, timedelta
import time

class TestCacheService:
    """Suite de tests para CacheService."""
    
    @pytest.fixture
    def cache_service(self, app):
        """Fixture que proporciona el servicio de caché."""
        return CacheService()
        
    def test_set_get_basic(self, cache_service):
        """Test de operaciones básicas set/get."""
        # Test con string
        assert cache_service.set('test_key', 'test_value')
        assert cache_service.get('test_key') == 'test_value'
        
        # Test con int
        assert cache_service.set('test_int', 42)
        assert cache_service.get('test_int') == 42
        
        # Test con dict
        test_dict = {'name': 'test', 'value': 123}
        assert cache_service.set('test_dict', test_dict)
        assert cache_service.get('test_dict') == test_dict
        
    def test_ttl_expiration(self, cache_service):
        """Test de expiración de valores."""
        # Guardar con TTL corto
        assert cache_service.set('test_ttl', 'will_expire', ttl=1)
        
        # Verificar valor inmediatamente
        assert cache_service.get('test_ttl') == 'will_expire'
        
        # Esperar expiración
        time.sleep(1.1)
        assert cache_service.get('test_ttl') is None
        
    def test_delete(self, cache_service):
        """Test de eliminación de valores."""
        # Guardar y verificar
        assert cache_service.set('test_delete', 'to_be_deleted')
        assert cache_service.get('test_delete') == 'to_be_deleted'
        
        # Eliminar y verificar
        assert cache_service.delete('test_delete')
        assert cache_service.get('test_delete') is None
        
    def test_clear_pattern(self, cache_service):
        """Test de limpieza por patrón."""
        # Guardar varios valores
        cache_service.set('test:1', 'value1')
        cache_service.set('test:2', 'value2')
        cache_service.set('other:1', 'value3')
        
        # Limpiar por patrón
        deleted = cache_service.clear_pattern('test:*')
        assert deleted == 2
        
        # Verificar eliminación
        assert cache_service.get('test:1') is None
        assert cache_service.get('test:2') is None
        assert cache_service.get('other:1') == 'value3'
        
    def test_cache_decorator(self, cache_service):
        """Test del decorador de caché."""
        call_count = 0
        
        @cache_service.cache_decorator('test')
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
            
        # Primera llamada
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Segunda llamada (debería usar caché)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # No incrementa
        
        # Llamada con diferente argumento
        result3 = expensive_function(7)
        assert result3 == 14
        assert call_count == 2
        
    def test_bulk_operations(self, cache_service):
        """Test de operaciones en lote."""
        # Datos de prueba
        test_data = {
            'key1': 'value1',
            'key2': 42,
            'key3': {'nested': 'value'}
        }
        
        # Bulk set
        assert cache_service.bulk_set(test_data)
        
        # Bulk get
        results = cache_service.bulk_get(['key1', 'key2', 'key3', 'nonexistent'])
        assert len(results) == 3
        assert results['key1'] == 'value1'
        assert results['key2'] == 42
        assert results['key3'] == {'nested': 'value'}
        assert 'nonexistent' not in results
        
    def test_complex_objects(self, cache_service):
        """Test con objetos complejos."""
        # Datetime
        now = datetime.utcnow()
        assert cache_service.set('test_datetime', now)
        cached_dt = cache_service.get('test_datetime')
        assert isinstance(cached_dt, datetime)
        assert cached_dt == now
        
        # Lista de diccionarios
        complex_data = [
            {'id': 1, 'data': [1, 2, 3]},
            {'id': 2, 'data': ['a', 'b', 'c']}
        ]
        assert cache_service.set('test_complex', complex_data)
        assert cache_service.get('test_complex') == complex_data
        
    def test_error_handling(self, cache_service):
        """Test de manejo de errores."""
        # Intentar guardar objeto no serializable
        class NonSerializable:
            pass
            
        assert not cache_service.set('test_error', NonSerializable())
        assert cache_service.get('test_error') is None
        
        # Intentar obtener clave no existente
        assert cache_service.get('nonexistent_key') is None
        
        # Intentar eliminar clave no existente
        assert not cache_service.delete('nonexistent_key')
        
    def test_stats(self, cache_service):
        """Test de estadísticas del caché."""
        # Generar algunos datos
        for i in range(5):
            cache_service.set(f'stats_test_{i}', f'value_{i}')
            
        # Obtener estadísticas
        stats = cache_service.get_stats()
        
        assert 'total_keys' in stats
        assert 'used_memory' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'evicted_keys' in stats
        assert stats['total_keys'] >= 5 