# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import pytest
from scripts.init_data import initialize_data
from app.models.user import User, Role, Permission, UserRole, UserPermission
from app import db

@pytest.fixture
def app_with_db(app):
    """Fixture que proporciona una aplicación con base de datos limpia"""
    with app.app_context():
        # Ejecutar migraciones
        from flask_migrate import upgrade
        upgrade()
        
        # Crear tablas
        db.create_all()
        
        yield app
        
        # Limpiar base de datos
        db.session.remove()
        db.drop_all()

def test_initialize_data_creates_permissions(app_with_db):
    """Prueba que se creen todos los permisos correctamente"""
    initialize_data()
    
    # Verificar que se crearon todos los permisos
    permisos = Permission.query.all()
    assert len(permisos) == len(UserPermission)
    
    # Verificar permisos específicos
    assert Permission.query.filter_by(name=UserPermission.KIOSK_CREATE.value).first() is not None
    assert Permission.query.filter_by(name=UserPermission.USER_MANAGE.value).first() is not None

def test_initialize_data_creates_roles(app_with_db):
    """Prueba que se creen todos los roles correctamente"""
    initialize_data()
    
    # Verificar que se crearon todos los roles
    roles = Role.query.all()
    assert len(roles) == len(UserRole)
    
    # Verificar rol admin y sus permisos
    admin_role = Role.query.filter_by(name=UserRole.ADMIN.value).first()
    assert admin_role is not None
    assert len(admin_role.permissions) == len(UserPermission)
    
    # Verificar rol operator y sus permisos
    operator_role = Role.query.filter_by(name=UserRole.OPERATOR.value).first()
    assert operator_role is not None
    assert len(operator_role.permissions) == 4
    assert Permission.query.filter_by(name=UserPermission.KIOSK_UPDATE.value).first() in operator_role.permissions
    
    # Verificar rol viewer y sus permisos
    viewer_role = Role.query.filter_by(name=UserRole.VIEWER.value).first()
    assert viewer_role is not None
    assert len(viewer_role.permissions) == 2
    assert Permission.query.filter_by(name=UserPermission.VIEW_DASHBOARD.value).first() in viewer_role.permissions

def test_initialize_data_creates_admin_user(app_with_db):
    """Prueba que se cree el usuario administrador correctamente"""
    initialize_data()
    
    # Verificar que se creó el usuario admin
    admin = User.query.filter_by(username='admin').first()
    assert admin is not None
    assert admin.email == 'admin@admin.com'
    assert admin.is_active is True
    assert admin.role_name == UserRole.ADMIN.value
    
    # Verificar que el admin tiene todos los permisos a través de su rol
    admin_role = Role.query.filter_by(name=UserRole.ADMIN.value).first()
    assert len(admin_role.permissions) == len(UserPermission)

def test_initialize_data_idempotent(app_with_db):
    """Prueba que ejecutar initialize_data() múltiples veces no duplique datos"""
    # Primera ejecución
    initialize_data()
    initial_permissions = len(Permission.query.all())
    initial_roles = len(Role.query.all())
    initial_users = len(User.query.all())
    
    # Segunda ejecución
    initialize_data()
    assert len(Permission.query.all()) == initial_permissions
    assert len(Role.query.all()) == initial_roles
    assert len(User.query.all()) == initial_users 