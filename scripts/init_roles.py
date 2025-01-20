# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import create_app, db
from app.models.user import UserRole, UserPermission, Role, Permission
import logging

logger = logging.getLogger(__name__)

def init_permissions():
    """Inicializa los permisos del sistema"""
    logger.info("Inicializando permisos...")
    
    for permission in UserPermission:
        if not Permission.query.get(permission.value):
            perm = Permission(
                name=permission.value,
                description=permission.value.replace('_', ' ').title()
            )
            db.session.add(perm)
            logger.info(f"Creado permiso: {permission.value}")
    
    try:
        db.session.commit()
        logger.info("Permisos inicializados correctamente")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error inicializando permisos: {str(e)}")
        raise

def init_roles():
    """Inicializa los roles del sistema con sus permisos"""
    logger.info("Inicializando roles...")
    
    # Definir roles y sus permisos
    roles_config = {
        UserRole.ADMIN.value: {
            'description': 'Administrador del sistema con acceso total',
            'permissions': [p.value for p in UserPermission]
        },
        UserRole.OPERATOR.value: {
            'description': 'Operador con acceso a gestión de kiosks',
            'permissions': [
                UserPermission.VIEW_DASHBOARD.value,
                UserPermission.VIEW_KIOSKS.value,
                UserPermission.KIOSK_UPDATE.value,
                UserPermission.VIEW_REPORTS.value
            ]
        },
        UserRole.VIEWER.value: {
            'description': 'Usuario con acceso solo de lectura',
            'permissions': [
                UserPermission.VIEW_DASHBOARD.value,
                UserPermission.VIEW_KIOSKS.value,
                UserPermission.VIEW_REPORTS.value
            ]
        }
    }
    
    # Crear o actualizar roles
    for role_name, config in roles_config.items():
        role = Role.query.get(role_name)
        if not role:
            role = Role(name=role_name, description=config['description'])
            db.session.add(role)
            logger.info(f"Creado rol: {role_name}")
        else:
            role.description = config['description']
            logger.info(f"Actualizado rol: {role_name}")
        
        # Asignar permisos
        role.permissions = []
        for perm_name in config['permissions']:
            perm = Permission.query.get(perm_name)
            if perm:
                role.permissions.append(perm)
    
    try:
        db.session.commit()
        logger.info("Roles inicializados correctamente")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error inicializando roles: {str(e)}")
        raise

def init_all():
    """Inicializa todos los roles y permisos"""
    app = create_app()
    with app.app_context():
        init_permissions()
        init_roles()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    init_all() 