"""
Servicio de autenticación de dos factores.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pyotp
import qrcode
import io
import base64
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import secrets
from flask import current_app
from app.models.user import User
from app.services.notification_service import NotificationService
import logging

logger = logging.getLogger(__name__)

class TwoFactorService:
    """Servicio para gestionar autenticación de dos factores."""
    
    def __init__(self):
        """Inicializar servicio."""
        self.notification_service = NotificationService()
        self.temp_code_expiry = timedelta(minutes=10)
        self.backup_codes_count = 8
        self.backup_code_length = 10
    
    def generate_secret(self) -> str:
        """Genera una nueva clave secreta para 2FA."""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user: User) -> str:
        """
        Genera código QR para configuración de 2FA.
        
        Args:
            user: Usuario para el que se genera el QR
            
        Returns:
            str: Imagen QR en base64
        """
        try:
            # Generar nueva clave secreta
            secret = self.generate_secret()
            user.two_factor_secret = secret
            
            # Generar URI para QR
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                user.email,
                issuer_name=current_app.config['TWO_FACTOR_ISSUER']
            )
            
            # Generar QR
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            # Convertir a imagen
            img = qr.make_image(fill_color="black", back_color="white")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generando QR para {user.email}: {str(e)}")
            raise
    
    def enable_2fa(self, user: User, code: str) -> bool:
        """
        Habilita 2FA para un usuario.
        
        Args:
            user: Usuario a habilitar
            code: Código de verificación inicial
            
        Returns:
            bool: True si se habilitó correctamente
        """
        try:
            if self.verify_code(user, code):
                user.two_factor_enabled = True
                user.backup_codes = self.generate_backup_codes()
                logger.info(f"2FA habilitado para {user.email}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error habilitando 2FA para {user.email}: {str(e)}")
            return False
    
    def disable_2fa(self, user: User, code: str) -> bool:
        """
        Deshabilita 2FA para un usuario.
        
        Args:
            user: Usuario a deshabilitar
            code: Código de verificación
            
        Returns:
            bool: True si se deshabilitó correctamente
        """
        try:
            if self.verify_code(user, code):
                user.two_factor_enabled = False
                user.two_factor_secret = None
                user.backup_codes = None
                user.temp_2fa_code = None
                logger.info(f"2FA deshabilitado para {user.email}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deshabilitando 2FA para {user.email}: {str(e)}")
            return False
    
    def verify_code(self, user: User, code: str) -> bool:
        """
        Verifica un código 2FA.
        
        Args:
            user: Usuario a verificar
            code: Código a verificar
            
        Returns:
            bool: True si el código es válido
        """
        try:
            if not user.two_factor_secret:
                return False
            
            totp = pyotp.TOTP(user.two_factor_secret)
            return totp.verify(code)
            
        except Exception as e:
            logger.error(f"Error verificando código para {user.email}: {str(e)}")
            return False
    
    def generate_backup_codes(self, count: Optional[int] = None) -> List[str]:
        """
        Genera códigos de respaldo.
        
        Args:
            count: Número de códigos a generar
            
        Returns:
            List[str]: Lista de códigos generados
        """
        try:
            count = count or self.backup_codes_count
            return [
                secrets.token_urlsafe(self.backup_code_length)[:self.backup_code_length]
                for _ in range(count)
            ]
            
        except Exception as e:
            logger.error(f"Error generando códigos de respaldo: {str(e)}")
            return []
    
    def verify_backup_code(self, user: User, code: str) -> bool:
        """
        Verifica un código de respaldo.
        
        Args:
            user: Usuario a verificar
            code: Código de respaldo
            
        Returns:
            bool: True si el código es válido
        """
        try:
            if not user.backup_codes:
                return False
            
            if code in user.backup_codes:
                # Eliminar código usado
                user.backup_codes.remove(code)
                logger.info(f"Código de respaldo usado por {user.email}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error verificando código de respaldo para {user.email}: {str(e)}")
            return False
    
    def send_temp_code(self, user: User) -> bool:
        """
        Envía un código temporal por email.
        
        Args:
            user: Usuario destinatario
            
        Returns:
            bool: True si se envió correctamente
        """
        try:
            # Generar código temporal
            code = ''.join(secrets.choice('0123456789') for _ in range(6))
            expires = datetime.utcnow() + self.temp_code_expiry
            
            # Guardar código
            user.temp_2fa_code = {
                'code': code,
                'expires': expires.isoformat()
            }
            
            # Enviar por email
            self.notification_service.send_email(
                to=user.email,
                subject='Código Temporal 2FA',
                body=f'Tu código temporal es: {code}\nExpira en {self.temp_code_expiry.total_seconds()/60} minutos.'
            )
            
            logger.info(f"Código temporal enviado a {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando código temporal a {user.email}: {str(e)}")
            return False
    
    def verify_temp_code(self, user: User, code: str) -> bool:
        """
        Verifica un código temporal.
        
        Args:
            user: Usuario a verificar
            code: Código temporal
            
        Returns:
            bool: True si el código es válido
        """
        try:
            if not user.temp_2fa_code:
                return False
            
            temp_code = user.temp_2fa_code.get('code')
            expires = datetime.fromisoformat(user.temp_2fa_code.get('expires'))
            
            if datetime.utcnow() > expires:
                user.temp_2fa_code = None
                return False
            
            if code == temp_code:
                user.temp_2fa_code = None
                logger.info(f"Código temporal verificado para {user.email}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error verificando código temporal para {user.email}: {str(e)}")
            return False 