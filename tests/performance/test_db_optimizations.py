"""
Tests de rendimiento específicos para optimizaciones de base de datos.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Kiosk, SensorData, User
from datetime import datetime, timedelta
import random

@pytest.fixture(scope="module")
def db_session():
    """Fixture para crear una sesión de base de datos para pruebas."""
    engine = create_engine('postgresql://test_user:test_password@localhost/test_db')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Crear datos de prueba masivos
    create_test_data(session)
    
    yield session
    
    # Limpiar datos de prueba
    cleanup_test_data(session)
    session.close()

def create_test_data(session):
    """Crear datos de prueba masivos."""
    # Crear kiosks
    kiosks = []
    for i in range(100):
        kiosk = Kiosk(
            name=f"Perf Test Kiosk {i}",
            location=f"Location {i}",
            status="active",
            health_score=random.uniform(80.0, 100.0)
        )
        kiosks.append(kiosk)
    
    session.bulk_save_objects(kiosks)
    session.commit()
    
    # Crear datos de sensores
    now = datetime.utcnow()
    sensor_data = []
    for kiosk in kiosks:
        for i in range(1000):  # 1000 registros por kiosk
            sensor_data.append(
                SensorData(
                    kiosk_id=kiosk.id,
                    cpu_usage=random.uniform(0, 100),
                    memory_usage=random.uniform(0, 100),
                    network_latency=random.uniform(1, 1000),
                    timestamp=now - timedelta(minutes=i)
                )
            )
    
    # Insertar en lotes para mejor rendimiento
    for i in range(0, len(sensor_data), 1000):
        session.bulk_save_objects(sensor_data[i:i+1000])
        session.commit()

def cleanup_test_data(session):
    """Limpiar datos de prueba."""
    session.query(SensorData).delete()
    session.query(Kiosk).filter(Kiosk.name.like("Perf Test%")).delete()
    session.commit()

def test_indexed_vs_nonindexed_query(benchmark, db_session):
    """Comparar rendimiento de consultas con y sin índices."""
    def query_without_index():
        return db_session.query(SensorData)\
            .filter(SensorData.cpu_usage > 90)\
            .all()
    
    def query_with_index():
        # Asumiendo que ya existe un índice en cpu_usage
        return db_session.query(SensorData)\
            .filter(SensorData.cpu_usage > 90)\
            .all()
    
    # Medir ambas consultas
    result_without = benchmark(query_without_index)
    # Crear índice temporalmente
    db_session.execute(text('CREATE INDEX IF NOT EXISTS idx_sensor_cpu ON sensor_data (cpu_usage)'))
    result_with = benchmark(query_with_index)
    # Eliminar índice
    db_session.execute(text('DROP INDEX IF EXISTS idx_sensor_cpu'))
    
    assert len(result_without) == len(result_with)

def test_join_optimization(benchmark, db_session):
    """Probar optimización de consultas JOIN."""
    def complex_join_query():
        return db_session.query(Kiosk, SensorData)\
            .join(SensorData)\
            .filter(Kiosk.status == 'active')\
            .filter(SensorData.cpu_usage > 80)\
            .all()
    
    results = benchmark(complex_join_query)
    assert len(results) > 0

def test_bulk_insert_performance(benchmark, db_session):
    """Probar rendimiento de inserción masiva."""
    def single_inserts():
        for i in range(100):
            sensor_data = SensorData(
                kiosk_id=1,
                cpu_usage=random.uniform(0, 100),
                memory_usage=random.uniform(0, 100),
                timestamp=datetime.utcnow()
            )
            db_session.add(sensor_data)
        db_session.commit()
    
    def bulk_insert():
        sensor_data = [
            SensorData(
                kiosk_id=1,
                cpu_usage=random.uniform(0, 100),
                memory_usage=random.uniform(0, 100),
                timestamp=datetime.utcnow()
            )
            for _ in range(100)
        ]
        db_session.bulk_save_objects(sensor_data)
        db_session.commit()
    
    # Comparar rendimiento
    benchmark(single_inserts)
    benchmark(bulk_insert)

def test_query_optimization_with_limit(benchmark, db_session):
    """Probar optimización de consultas con LIMIT."""
    def query_with_limit():
        return db_session.query(SensorData)\
            .order_by(SensorData.timestamp.desc())\
            .limit(100)\
            .all()
    
    def query_without_limit():
        return db_session.query(SensorData)\
            .order_by(SensorData.timestamp.desc())\
            .all()[:100]
    
    # Comparar rendimiento
    result_with = benchmark(query_with_limit)
    result_without = benchmark(query_without_limit)
    
    assert len(result_with) == len(result_without) == 100

def test_aggregation_performance(benchmark, db_session):
    """Probar rendimiento de agregaciones."""
    def complex_aggregation():
        return db_session.query(
            Kiosk.id,
            text('AVG(sensor_data.cpu_usage) as avg_cpu'),
            text('MAX(sensor_data.memory_usage) as max_memory'),
            text('COUNT(*) as total_readings')
        )\
        .join(SensorData)\
        .group_by(Kiosk.id)\
        .all()
    
    results = benchmark(complex_aggregation)
    assert len(results) > 0

def test_subquery_vs_join(benchmark, db_session):
    """Comparar rendimiento entre subqueries y joins."""
    def using_subquery():
        subquery = db_session.query(
            SensorData.kiosk_id,
            text('AVG(cpu_usage) as avg_cpu')
        )\
        .group_by(SensorData.kiosk_id)\
        .subquery()
        
        return db_session.query(Kiosk, subquery.c.avg_cpu)\
            .join(subquery, Kiosk.id == subquery.c.kiosk_id)\
            .all()
    
    def using_join():
        return db_session.query(
            Kiosk,
            text('AVG(sensor_data.cpu_usage) as avg_cpu')
        )\
        .join(SensorData)\
        .group_by(Kiosk.id)\
        .all()
    
    # Comparar rendimiento
    result_subquery = benchmark(using_subquery)
    result_join = benchmark(using_join)
    
    assert len(result_subquery) == len(result_join) 