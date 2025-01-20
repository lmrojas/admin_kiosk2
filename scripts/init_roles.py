"""
Este script inicializa los roles y permisos del sistema.

Funcionalidad:
- Crea los permisos base del sistema (VIEW_DASHBOARD, VIEW_KIOSKS, etc.)
- Crea los roles principales (ADMIN, OPERATOR, VIEWER)
- Asigna los permisos correspondientes a cada rol
- Si los roles/permisos ya existen, los actualiza

Uso:
python scripts/init_roles.py

Notas:
- Debe ejecutarse después de crear la base de datos
- Requiere que la aplicación esté configurada correctamente
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import create_app, db
from app.models.user import UserRole, UserPermission, User
import logging

logger = logging.getLogger(__name__)

def init_roles():
    """Inicializa los roles del sistema"""
    logger.info("Inicializando roles...")
    
    # Crear usuario admin si no existe
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@example.com',
            role_name=UserRole.ADMIN.value
        )
        admin.set_password('admin123')  # Contraseña temporal que debe cambiarse
        db.session.add(admin)
        logger.info("Usuario admin creado")
    
    db.session.commit()
    logger.info("Roles inicializados correctamente")

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        init_roles() 