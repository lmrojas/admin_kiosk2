"""
Tests de rendimiento para operaciones de base de datos.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import User, Kiosk
from app.services.kiosk_service import KioskService
from app.services.auth_service import AuthService

@pytest.fixture(scope="module")
def db_session():
    """Fixture para crear una sesión de base de datos para pruebas."""
    engine = create_engine('postgresql://postgres:postgres@localhost/admin_kiosk2_test')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Crear datos de prueba
    for i in range(100):
        user = User(
            username=f"test_user_{i}",
            email=f"test{i}@example.com",
            role_name="OPERATOR"
        )
        user.set_password("test_password")
        session.add(user)
        
        kiosk = Kiosk(
            name=f"Kiosk {i}",
            location=f"Location {i}",
            status="active",
            health_score=95.0
        )
        session.add(kiosk)
    
    session.commit()
    
    yield session
    
    # Limpiar datos de prueba
    session.query(User).delete()
    session.query(Kiosk).delete()
    session.commit()
    session.close()

def test_user_query_performance(benchmark, db_session):
    """Test de rendimiento para consultas de usuarios."""
    def query_users():
        return db_session.query(User).filter(
            User.role_name == "OPERATOR"
        ).all()
    
    users = benchmark(query_users)
    assert len(users) == 100

def test_kiosk_query_performance(benchmark, db_session):
    """Test de rendimiento para consultas de kiosks."""
    def query_kiosks():
        return db_session.query(Kiosk).filter(
            Kiosk.status == "active"
        ).all()
    
    kiosks = benchmark(query_kiosks)
    assert len(kiosks) == 100

def test_kiosk_service_performance(benchmark, db_session):
    """Test de rendimiento para operaciones del servicio de kiosks."""
    kiosk_service = KioskService(db_session)
    
    def get_kiosk_metrics():
        return [
            kiosk_service.get_kiosk_metrics(kiosk.id)
            for kiosk in db_session.query(Kiosk).limit(10)
        ]
    
    metrics = benchmark(get_kiosk_metrics)
    assert len(metrics) == 10

def test_auth_service_performance(benchmark, db_session):
    """Test de rendimiento para operaciones de autenticación."""
    auth_service = AuthService(db_session)
    
    def authenticate_users():
        results = []
        for i in range(10):
            try:
                results.append(
                    auth_service.authenticate(
                        f"test_user_{i}",
                        "test_password"
                    )
                )
            except Exception:
                continue
        return results
    
    tokens = benchmark(authenticate_users)
    assert len(tokens) == 10

def test_bulk_insert_performance(benchmark, db_session):
    """Test de rendimiento para inserciones masivas."""
    def bulk_insert():
        users = [
            User(
                username=f"bulk_user_{i}",
                email=f"bulk{i}@example.com",
                role_name="VIEWER"
            )
            for i in range(1000)
        ]
        db_session.bulk_save_objects(users)
        db_session.commit()
    
    benchmark(bulk_insert)
    count = db_session.query(User).filter(
        User.username.like("bulk_user_%")
    ).count()
    assert count == 1000

def test_complex_query_performance(benchmark, db_session):
    """Test de rendimiento para consultas complejas."""
    def complex_query():
        return db_session.query(Kiosk).\
            filter(Kiosk.status == "active").\
            filter(Kiosk.health_score > 90).\
            order_by(Kiosk.health_score.desc()).\
            limit(50).\
            all()
    
    results = benchmark(complex_query)
    assert len(results) > 0 