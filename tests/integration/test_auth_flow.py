"""
Tests de integración para el flujo de autenticación.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from flask import session, url_for
from app.models.user import User
from app.services.auth_service import AuthService
from app.services.two_factor_service import TwoFactorService
import json

class TestAuthFlow:
    """Suite de tests de integración para el flujo completo de autenticación."""
    
    @pytest.fixture
    def auth_service(self, app):
        """Fixture para el servicio de autenticación."""
        return AuthService()
        
    @pytest.fixture
    def two_factor_service(self, app):
        """Fixture para el servicio de 2FA."""
        return TwoFactorService()
        
    def test_registro_login_2fa_flow(self, client, auth_service, two_factor_service):
        """Test del flujo completo: registro -> login -> configuración 2FA -> login con 2FA."""
        # 1. Registro de usuario
        registro_data = {
            'username': 'test_integration',
            'email': 'test_integration@example.com',
            'password': 'Test123!@#',
            'confirm_password': 'Test123!@#'
        }
        
        response = client.post('/auth/register', data=registro_data)
        assert response.status_code == 302  # Redirección después del registro
        
        # 2. Login inicial
        login_data = {
            'username': registro_data['email'],
            'password': registro_data['password']
        }
        
        response = client.post('/auth/login', data=login_data)
        assert response.status_code == 302  # Redirección al dashboard
        
        # 3. Configuración de 2FA
        response = client.get('/auth/setup-2fa')
        assert response.status_code == 200
        assert b'Configure Two-Factor Authentication' in response.data
        
        # Obtener y verificar código 2FA
        user = User.query.filter_by(email=registro_data['email']).first()
        code = two_factor_service.generate_code(user)
        
        response = client.post('/auth/setup-2fa', data={'code': code})
        assert response.status_code == 302  # Redirección a backup codes
        
        # 4. Logout
        response = client.get('/auth/logout')
        assert response.status_code == 302
        
        # 5. Login con 2FA
        response = client.post('/auth/login', data=login_data)
        assert response.status_code == 302  # Redirección a verificación 2FA
        
        # Verificar código 2FA
        code = two_factor_service.generate_code(user)
        response = client.post('/auth/verify-2fa', data={'code': code})
        assert response.status_code == 302  # Redirección al dashboard
        
    def test_bloqueo_cuenta_flow(self, client, auth_service):
        """Test del flujo de bloqueo de cuenta por intentos fallidos."""
        # 1. Crear usuario
        user_data = {
            'username': 'test_lockout',
            'email': 'test_lockout@example.com',
            'password': 'Test123!@#'
        }
        
        user = User(**user_data)
        auth_service.register_user(user)
        
        # 2. Intentos fallidos de login
        for _ in range(5):
            response = client.post('/auth/login', data={
                'username': user_data['email'],
                'password': 'wrong_password'
            })
            assert response.status_code in [401, 302]
            
        # 3. Verificar bloqueo
        response = client.post('/auth/login', data={
            'username': user_data['email'],
            'password': user_data['password']
        })
        assert response.status_code == 403  # Cuenta bloqueada
        
    def test_recuperacion_password_flow(self, client, auth_service):
        """Test del flujo de recuperación de contraseña."""
        # 1. Crear usuario
        user_data = {
            'username': 'test_recovery',
            'email': 'test_recovery@example.com',
            'password': 'Test123!@#'
        }
        
        user = User(**user_data)
        auth_service.register_user(user)
        
        # 2. Solicitar recuperación
        response = client.post('/auth/forgot-password', data={
            'email': user_data['email']
        })
        assert response.status_code == 200
        
        # 3. Obtener token de reset
        token = auth_service.generate_reset_token(user)
        
        # 4. Reset password
        new_password = 'NewTest123!@#'
        response = client.post(f'/auth/reset-password/{token}', data={
            'password': new_password,
            'confirm_password': new_password
        })
        assert response.status_code == 302
        
        # 5. Login con nueva contraseña
        response = client.post('/auth/login', data={
            'username': user_data['email'],
            'password': new_password
        })
        assert response.status_code == 302  # Login exitoso
        
    def test_session_timeout_flow(self, client, auth_service):
        """Test del flujo de timeout de sesión."""
        # 1. Login
        user_data = {
            'username': 'test_session',
            'email': 'test_session@example.com',
            'password': 'Test123!@#'
        }
        
        user = User(**user_data)
        auth_service.register_user(user)
        
        response = client.post('/auth/login', data={
            'username': user_data['email'],
            'password': user_data['password']
        })
        assert response.status_code == 302
        
        # 2. Simular timeout
        with client.session_transaction() as sess:
            sess['last_activity'] = '2000-01-01T00:00:00'  # Fecha antigua
            
        # 3. Intentar acceder a ruta protegida
        response = client.get('/dashboard')
        assert response.status_code == 401  # Sesión expirada 