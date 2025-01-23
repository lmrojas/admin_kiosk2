"""
Módulo de autenticación y autorización.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify, abort
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User, UserRole, UserPermission
from app.services.auth_service import AuthService
from app.services.two_factor_service import TwoFactorService
import logging
from functools import wraps

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
auth_service = AuthService()
two_factor_service = TwoFactorService()
logger = logging.getLogger(__name__)

def admin_required(f):
    """Decorador que verifica que el usuario sea administrador."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Por favor inicia sesión para acceder a esta página.', 'warning')
            return redirect(url_for('auth.login'))
        
        if not current_user.has_role(UserRole.ADMIN.value):
            flash('Se requiere rol de administrador para acceder a esta página.', 'danger')
            return abort(403)
        
        return f(*args, **kwargs)
    return decorated_function

def permission_required(permission):
    """Decorador que verifica que el usuario tenga el permiso requerido."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash('Por favor inicia sesión para acceder a esta página.', 'warning')
                return redirect(url_for('auth.login'))
            
            if not current_user.has_permission(permission):
                flash('No tienes los permisos necesarios para acceder a esta página.', 'danger')
                return abort(403)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route('/')
def index():
    """Ruta raíz del blueprint de autenticación."""
    return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Vista de login con soporte 2FA."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        code_2fa = request.form.get('code_2fa')
        remember = True
        
        try:
            result = auth_service.process_login(username, password, code_2fa)
            if result.get('success'):
                user = result['user']
                login_user(user, remember=remember)
                session.permanent = True
                return redirect(url_for('main.dashboard'))
            elif result.get('needs_2fa'):
                session['pending_user_id'] = result['user_id']
                session.permanent = True
                return redirect(url_for('auth.verify_2fa'))
            else:
                flash(result.get('error', 'Error de autenticación'), 'error')
                return redirect(url_for('auth.login'))
            
        except Exception as e:
            logger.error(f'Error en login: {str(e)}')
            flash('Error en el proceso de login', 'error')
            
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Vista de registro de usuarios."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            result = auth_service.process_registration(username, email, password)
            if result.get('success'):
                flash('Usuario registrado exitosamente', 'success')
                return redirect(url_for('auth.login'))
            else:
                flash(result.get('error', 'Error al registrar usuario'), 'error')
        except Exception as e:
            logger.error(f'Error en registro: {str(e)}')
            flash('Error en el proceso de registro', 'error')
    
    return render_template('auth/register.html')

@auth_bp.route('/verify/<verification_type>', methods=['GET', 'POST'])
def verify():
    """
    Vista unificada de verificación.
    Maneja 2FA, códigos de respaldo y códigos temporales.
    """
    if 'pending_user_id' not in session:
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        code = request.form.get('code')
        result = auth_service.verify_auth(
            session['pending_user_id'], 
            code,
            verification_type
        )
        
        if result.get('success'):
            user = result['user']
            login_user(user, remember=True)
            session.permanent = True
            session.pop('pending_user_id', None)
            return redirect(url_for('main.dashboard'))
        else:
            flash(result.get('error', 'Código inválido'), 'error')
    
    # Para solicitudes GET o verificación fallida
    if verification_type == '2fa':
        return render_template('auth/verify_2fa.html')
    elif verification_type == 'backup':
        return render_template('auth/verify_backup.html')
    elif verification_type == 'temp':
        return render_template('auth/verify_temp.html')
    else:
        abort(404)

@auth_bp.route('/logout')
@login_required
def logout():
    """Vista de cierre de sesión."""
    username = current_user.username
    # Limpiar la sesión completamente
    session.clear()
    logout_user()
    logger.info(f'Usuario {username} ha cerrado sesión')
    flash('Has cerrado sesión exitosamente', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/setup-2fa', methods=['GET', 'POST'])
@login_required
def setup_2fa():
    """Vista de configuración inicial 2FA."""
    if current_user.two_factor_enabled:
        flash('2FA ya está habilitado', 'info')
        return redirect(url_for('auth.profile'))
    
    if request.method == 'POST':
        code = request.form.get('code')
        
        if two_factor_service.enable_2fa(current_user, code):
            flash('2FA habilitado exitosamente', 'success')
            logger.info(f'Usuario {current_user.username} habilitó 2FA')
            return redirect(url_for('auth.backup_codes'))
        
        flash('Código inválido', 'error')
        return redirect(url_for('auth.setup_2fa'))
    
    # Generar QR code
    qr_code = two_factor_service.generate_qr_code(current_user)
    return render_template('auth/setup_2fa.html', qr_code=qr_code)

@auth_bp.route('/disable-2fa', methods=['POST'])
@login_required
def disable_2fa():
    """Endpoint para deshabilitar 2FA."""
    code = request.form.get('code')
    
    if two_factor_service.disable_2fa(current_user, code):
        flash('2FA deshabilitado exitosamente', 'success')
        logger.info(f'Usuario {current_user.username} deshabilitó 2FA')
    else:
        flash('Código inválido', 'error')
    
    return redirect(url_for('auth.profile'))

@auth_bp.route('/backup-codes')
@login_required
def backup_codes():
    """Vista de códigos de respaldo."""
    if not current_user.two_factor_enabled:
        flash('2FA no está habilitado', 'error')
        return redirect(url_for('auth.profile'))
    
    codes = two_factor_service.generate_backup_codes()
    return render_template('auth/backup_codes.html', backup_codes=codes)

@auth_bp.route('/request-temp-code', methods=['POST'])
def request_temp_code():
    """Endpoint para solicitar código temporal."""
    if 'pending_user_id' not in session:
        return jsonify({'error': 'No hay sesión pendiente'}), 400
    
    user = User.query.get(session['pending_user_id'])
    if not user:
        return jsonify({'error': 'Usuario no encontrado'}), 404
    
    if two_factor_service.send_temp_code(user):
        logger.info(f'Código temporal enviado a {user.username}')
        return jsonify({'message': 'Código enviado'})
    
    return jsonify({'error': 'Error enviando código'}), 500 