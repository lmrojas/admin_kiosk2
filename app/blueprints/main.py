# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from app.models.kiosk import Kiosk
from app.models.user import User
from app import db

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Página de inicio del sistema - Muestra el mapa de kiosks"""
    return redirect(url_for('kiosk.show_map'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Panel de control del usuario"""
    # Obtener estadísticas de kiosks
    total_kiosks = Kiosk.query.filter_by(owner_id=current_user.id).count()
    active_kiosks = Kiosk.query.filter_by(owner_id=current_user.id, status='active').count()
    
    # Obtener estadísticas de usuarios
    total_users = User.query.count()
    
    # Obtener kiosks recientes del usuario
    recent_kiosks = Kiosk.query.filter_by(owner_id=current_user.id).order_by(Kiosk.created_at.desc()).limit(5).all()
    
    return render_template(
        'main/dashboard.html', 
        total_kiosks=total_kiosks, 
        active_kiosks=active_kiosks,
        total_users=total_users,
        recent_kiosks=recent_kiosks
    )

@main_bp.route('/dashboard/logs')
@login_required
def logs():
    """Vista de logs del sistema"""
    return render_template('main/logs.html')

@main_bp.route('/dashboard/security')
@login_required
def security():
    """Vista de seguridad del sistema"""
    return render_template('main/security.html')

@main_bp.route('/dashboard/backups')
@login_required
def backups():
    """Vista de backups del sistema"""
    return render_template('main/backups.html')

@main_bp.route('/dashboard/ai')
@login_required
def ai_dashboard():
    """Vista del dashboard de IA integrada en el dashboard principal."""
    try:
        from app.services.kiosk_ai_service import KioskAIService
        ai_service = KioskAIService()
        
        metrics = ai_service.get_model_metrics()
        if not metrics:
            metrics = {
                'model_version': 'N/A',
                'accuracy_rate': 0.0,
                'avg_prediction_time': 0,
                'anomaly_threshold': 0.7,
                'is_training': False,
                'training_progress': 0,
                'training_loss': 0.0,
                'current_epoch': 0,
                'total_epochs': 50
            }
            
        return render_template('main/dashboard.html',
                            active_tab='ai',
                            ai_metrics=metrics)
                            
    except Exception as e:
        current_app.logger.error(f"Error al cargar el dashboard de IA: {str(e)}")
        flash('Error al cargar los datos de IA', 'error')
        return redirect(url_for('main.dashboard'))

@main_bp.route('/profile')
@login_required
def profile():
    """Perfil de usuario"""
    return render_template('main/profile.html', user=current_user)

@main_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Editar perfil de usuario"""
    if request.method == 'POST':
        # Obtener datos del formulario
        email = request.form.get('email')
        
        # Validar y actualizar email
        if email and email != current_user.email:
            # Verificar si el email ya existe
            existing_user = User.query.filter_by(email=email).first()
            if existing_user and existing_user.id != current_user.id:
                flash('El correo electrónico ya está en uso.', 'danger')
                return render_template('main/edit_profile.html')
            
            current_user.email = email
            db.session.commit()
            flash('Perfil actualizado exitosamente.', 'success')
            return redirect(url_for('main.profile'))
    
    return render_template('main/edit_profile.html', user=current_user) 