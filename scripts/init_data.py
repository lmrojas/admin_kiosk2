"""
Este script inicializa los datos básicos necesarios para el funcionamiento del sistema.

Funcionalidad:
- Inicializa los permisos del sistema con sus descripciones
- Configura los roles y sus permisos asociados
- Crea el usuario administrador inicial (si no existe)
  - Username: admin
  - Password: admin123
  - Email: admin@example.com
  - Rol: ADMIN

Uso:
python scripts/init_data.py

Notas:
- Debe ejecutarse después de init_roles.py
- Crea solo el usuario admin si no existe
- No modifica usuarios existentes
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import sys
from pathlib import Path

# Agregar directorio raíz al path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from app import db, create_app
from app.models.user import User, UserRole, UserPermission
from werkzeug.security import generate_password_hash
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def init_permissions():
    """Inicializa los permisos del sistema"""
    try:
        # Diccionario de permisos y sus descripciones
        permissions = {
            UserPermission.VIEW_KIOSK.value: "Ver kiosks",
            UserPermission.CREATE_KIOSK.value: "Crear kiosks",
            UserPermission.UPDATE_KIOSK.value: "Actualizar kiosks",
            UserPermission.DELETE_KIOSK.value: "Eliminar kiosks",
            UserPermission.VIEW_METRICS.value: "Ver métricas",
            UserPermission.VIEW_ALERTS.value: "Ver alertas",
            UserPermission.MANAGE_USERS.value: "Gestionar usuarios"
        }
        
        logger.info("Permisos inicializados correctamente")
        return permissions
        
    except Exception as e:
        logger.error(f"Error inicializando permisos: {str(e)}")
        raise

def init_roles():
    """Inicializa los roles del sistema"""
    try:
        # Definir permisos para cada rol
        role_permissions = {
            UserRole.ADMIN.value: [p.value for p in UserPermission],
            UserRole.OPERATOR.value: [
                UserPermission.UPDATE_KIOSK.value,
                UserPermission.VIEW_KIOSK.value,
                UserPermission.VIEW_METRICS.value,
                UserPermission.VIEW_ALERTS.value
            ],
            UserRole.VIEWER.value: [
                UserPermission.VIEW_KIOSK.value,
                UserPermission.VIEW_METRICS.value,
                UserPermission.VIEW_ALERTS.value
            ]
        }
        
        logger.info("Roles inicializados correctamente")
        return role_permissions
        
    except Exception as e:
        logger.error(f"Error inicializando roles: {str(e)}")
        raise

def init_admin_user():
    """Crea el usuario administrador inicial"""
    try:
        # Verificar si ya existe un admin
        admin = User.query.filter_by(username='admin').first()
        if admin:
            logger.info("Usuario admin ya existe")
            return admin
            
        # Crear usuario admin
        admin = User(
            username='admin',
            email='admin@example.com',
            role_name=UserRole.ADMIN.value,
            is_active=True
        )
        admin.set_password('admin123')
        
        db.session.add(admin)
        db.session.commit()
        
        logger.info("Usuario admin creado correctamente")
        return admin
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creando usuario admin: {str(e)}")
        raise

def init_all():
    """Inicializa todos los datos del sistema"""
    app = create_app()
    with app.app_context():
        try:
            logger.info("Iniciando inicialización de datos...")
            
            # Inicializar permisos y roles
            permissions = init_permissions()
            role_permissions = init_roles()
            
            # Crear usuario admin
            admin = init_admin_user()
            
            logger.info("Inicialización de datos completada exitosamente")
            
        except Exception as e:
            logger.error(f"Error en la inicialización de datos: {str(e)}")
            raise

if __name__ == '__main__':
    init_all() 