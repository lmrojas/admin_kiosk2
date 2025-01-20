"""
Módulo de autenticación y autorización.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User, UserRole, UserPermission
from app.services.auth_service import AuthService
from app.services.two_factor_service import TwoFactorService
from app.utils.decorators import admin_required, permission_required
import logging

auth_bp = Blueprint('auth', __name__)
auth_service = AuthService()
two_factor_service = TwoFactorService()
logger = logging.getLogger(__name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Vista de login con soporte 2FA."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        code_2fa = request.form.get('code_2fa')
        
        try:
            user = auth_service.authenticate(username, password)
            if not user:
                flash('Credenciales inválidas', 'error')
                return redirect(url_for('auth.login'))
            
            # Verificar 2FA si está habilitado
            if user.two_factor_enabled:
                if not code_2fa:
                    # Guardar usuario en sesión y redirigir a verificación 2FA
                    session['pending_user_id'] = user.id
                    return redirect(url_for('auth.verify_2fa'))
                
                if not two_factor_service.verify_code(user, code_2fa):
                    flash('Código 2FA inválido', 'error')
                    return redirect(url_for('auth.verify_2fa'))
            
            login_user(user)
            logger.info(f'Usuario {user.username} ha iniciado sesión')
            return redirect(url_for('main.dashboard'))
            
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
            user = auth_service.register_user(username, email, password)
            if user:
                flash('Usuario registrado exitosamente', 'success')
                logger.info(f'Nuevo usuario registrado: {username}')
                return redirect(url_for('auth.login'))
            else:
                flash('Error al registrar usuario', 'error')
        except ValueError as e:
            flash(str(e), 'error')
            logger.warning(f'Error de validación en registro: {str(e)}')
        except Exception as e:
            logger.error(f'Error en registro: {str(e)}')
            flash('Error en el proceso de registro', 'error')
    
    return render_template('auth/register.html')

@auth_bp.route('/verify-2fa', methods=['GET', 'POST'])
def verify_2fa():
    """Vista de verificación 2FA."""
    if 'pending_user_id' not in session:
        return redirect(url_for('auth.login'))
    
    user = User.query.get(session['pending_user_id'])
    if not user:
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        code = request.form.get('code')
        
        # Verificar código
        if two_factor_service.verify_code(user, code):
            login_user(user)
            session.pop('pending_user_id', None)
            logger.info(f'Usuario {user.username} verificó 2FA exitosamente')
            return redirect(url_for('dashboard.index'))
        
        flash('Código inválido', 'error')
    
    return render_template('auth/2fa.html', username=user.username)

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

@auth_bp.route('/verify-backup-code', methods=['POST'])
def verify_backup_code():
    """Endpoint para verificar código de respaldo."""
    if 'pending_user_id' not in session:
        return jsonify({'error': 'No hay sesión pendiente'}), 400
    
    user = User.query.get(session['pending_user_id'])
    if not user:
        return jsonify({'error': 'Usuario no encontrado'}), 404
    
    code = request.form.get('code')
    if two_factor_service.verify_backup_code(user, code):
        login_user(user)
        session.pop('pending_user_id', None)
        logger.info(f'Usuario {user.username} usó código de respaldo')
        return jsonify({'success': True})
    
    return jsonify({'error': 'Código inválido'}), 400

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

@auth_bp.route('/verify-temp-code', methods=['POST'])
def verify_temp_code():
    """Endpoint para verificar código temporal."""
    if 'pending_user_id' not in session:
        return jsonify({'error': 'No hay sesión pendiente'}), 400
    
    user = User.query.get(session['pending_user_id'])
    if not user:
        return jsonify({'error': 'Usuario no encontrado'}), 404
    
    code = request.form.get('code')
    if two_factor_service.verify_temp_code(user, code):
        login_user(user)
        session.pop('pending_user_id', None)
        logger.info(f'Usuario {user.username} verificó con código temporal')
        return jsonify({'success': True})
    
    return jsonify({'error': 'Código inválido'}), 400 