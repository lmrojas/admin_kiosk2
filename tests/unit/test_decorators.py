# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
from flask import Flask, jsonify
from flask_login import login_user
from app.utils.decorators import jwt_required, permission_required, role_required, audit_action
from app.models.user import User, Role, Permission, UserRole, UserPermission
from app import db

@pytest.fixture
def app():
    """Fixture que proporciona una aplicación Flask de prueba"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/admin_kiosk2_test'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    with app.app_context():
        db.init_app(app)
        db.create_all()
        
        # Crear permisos de prueba
        for permission in UserPermission:
            db.session.add(Permission(
                name=permission.value,
                description=permission.value
            ))
        
        # Crear roles de prueba
        admin_role = Role(name=UserRole.ADMIN.value, description='Admin')
        admin_role.permissions = Permission.query.all()
        db.session.add(admin_role)
        
        viewer_role = Role(name=UserRole.VIEWER.value, description='Viewer')
        viewer_role.permissions = [
            Permission.query.filter_by(name=UserPermission.VIEW_DASHBOARD.value).first()
        ]
        db.session.add(viewer_role)
        
        db.session.commit()
        
        yield app
        
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    """Fixture que proporciona un cliente de prueba"""
    return app.test_client()

@pytest.fixture
def admin_user(app):
    """Fixture que proporciona un usuario administrador"""
    user = User(
        username='admin',
        email='admin@test.com'
    )
    user.set_password('password')
    user.roles = [Role.query.filter_by(name=UserRole.ADMIN.value).first()]
    db.session.add(user)
    db.session.commit()
    return user

@pytest.fixture
def viewer_user(app):
    """Fixture que proporciona un usuario viewer"""
    user = User(
        username='viewer',
        email='viewer@test.com'
    )
    user.set_password('password')
    user.roles = [Role.query.filter_by(name=UserRole.VIEWER.value).first()]
    db.session.add(user)
    db.session.commit()
    return user

def test_jwt_required_no_token(app, client):
    """Prueba el decorador jwt_required sin token"""
    @app.route('/test')
    @jwt_required()
    def test_route():
        return jsonify({'message': 'success'})
    
    response = client.get('/test')
    assert response.status_code == 401
    assert b'Token no proporcionado' in response.data

def test_jwt_required_invalid_token(app, client):
    """Prueba el decorador jwt_required con token inválido"""
    @app.route('/test')
    @jwt_required()
    def test_route():
        return jsonify({'message': 'success'})
    
    response = client.get('/test', headers={'Authorization': 'Bearer invalid-token'})
    assert response.status_code == 401
    assert b'Token inv' in response.data

def test_permission_required_no_auth(app, client):
    """Prueba el decorador permission_required sin autenticación"""
    @app.route('/test')
    @permission_required(UserPermission.VIEW_DASHBOARD.value)
    def test_route():
        return jsonify({'message': 'success'})
    
    response = client.get('/test')
    assert response.status_code == 401

def test_permission_required_with_permission(app, client, admin_user):
    """Prueba el decorador permission_required con permiso válido"""
    @app.route('/test')
    @permission_required(UserPermission.VIEW_DASHBOARD.value)
    def test_route():
        return jsonify({'message': 'success'})
    
    with client:
        login_user(admin_user)
        response = client.get('/test')
        assert response.status_code == 200

def test_permission_required_without_permission(app, client, viewer_user):
    """Prueba el decorador permission_required sin permiso necesario"""
    @app.route('/test')
    @permission_required(UserPermission.SYSTEM_CONFIG.value)
    def test_route():
        return jsonify({'message': 'success'})
    
    with client:
        login_user(viewer_user)
        response = client.get('/test')
        assert response.status_code == 403

def test_role_required_with_role(app, client, admin_user):
    """Prueba el decorador role_required con rol válido"""
    @app.route('/test')
    @role_required(UserRole.ADMIN.value)
    def test_route():
        return jsonify({'message': 'success'})
    
    with client:
        login_user(admin_user)
        response = client.get('/test')
        assert response.status_code == 200

def test_role_required_without_role(app, client, viewer_user):
    """Prueba el decorador role_required sin rol necesario"""
    @app.route('/test')
    @role_required(UserRole.ADMIN.value)
    def test_route():
        return jsonify({'message': 'success'})
    
    with client:
        login_user(viewer_user)
        response = client.get('/test')
        assert response.status_code == 403

def test_audit_action_success(app, client, admin_user, caplog):
    """Prueba el decorador audit_action en caso exitoso"""
    @app.route('/test')
    @audit_action('test_action')
    def test_route():
        return jsonify({'message': 'success'})
    
    with client:
        login_user(admin_user)
        response = client.get('/test')
        assert response.status_code == 200
        assert 'test_action' in caplog.text
        assert 'success' in caplog.text

def test_audit_action_error(app, client, admin_user, caplog):
    """Prueba el decorador audit_action en caso de error"""
    @app.route('/test')
    @audit_action('test_action')
    def test_route():
        raise ValueError('Test error')
    
    with client:
        login_user(admin_user)
        with pytest.raises(ValueError):
            client.get('/test')
        assert 'test_action' in caplog.text
        assert 'error' in caplog.text 