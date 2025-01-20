# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import pytest
from app import create_app, db
from app.models.user import User

@pytest.fixture
def app():
    """Fixture para crear una aplicación de prueba"""
    app = create_app('config.testing')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

def test_user_creation(app):
    """Probar creación de usuario"""
    user = User(username='testuser', email='test@example.com')
    user.set_password('testpassword')
    
    db.session.add(user)
    db.session.commit()
    
    assert user.username == 'testuser'
    assert user.email == 'test@example.com'
    assert user.check_password('testpassword') is True

def test_user_password_hashing(app):
    """Probar hash de contraseña"""
    user = User(username='testuser', email='test@example.com')
    user.set_password('testpassword')
    
    assert user.password_hash is not None
    assert user.password_hash != 'testpassword'
    assert user.check_password('testpassword') is True
    assert user.check_password('wrongpassword') is False

def test_user_unique_constraints(app):
    """Probar restricciones de unicidad"""
    user1 = User(username='testuser', email='test1@example.com')
    user1.set_password('password1')
    
    user2 = User(username='testuser2', email='test1@example.com')
    user2.set_password('password2')
    
    db.session.add(user1)
    db.session.commit()
    
    with pytest.raises(Exception):
        db.session.add(user2)
        db.session.commit() 