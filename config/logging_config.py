# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class LoggingConfig:
    """Configuración centralizada de logging para la aplicación de administración de kiosks"""
    
    @staticmethod
    def configure_logging(app):
        """
        Configurar logging para la aplicación
        
        Args:
            app (Flask): Instancia de la aplicación Flask
        """
        # Crear directorio de logs si no existe
        logs_dir = os.path.join(app.root_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Nombre de archivo de log con fecha actual
        log_filename = os.path.join(
            logs_dir, 
            f'admin_kiosk_{datetime.now().strftime("%Y-%m-%d")}.log'
        )
        
        # Configuración de formato de log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configurar handler de archivo con rotación
        file_handler = RotatingFileHandler(
            log_filename, 
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Configurar handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        
        # Configurar logger de la aplicación
        app_logger = logging.getLogger('admin_kiosk')
        app_logger.setLevel(logging.INFO)
        app_logger.addHandler(file_handler)
        app_logger.addHandler(console_handler)
        
        # Configurar loggers de librerías externas
        external_loggers = [
            'werkzeug',  # Flask's built-in logger
            'sqlalchemy',
            'flask_login'
        ]
        
        for logger_name in external_loggers:
            ext_logger = logging.getLogger(logger_name)
            ext_logger.setLevel(logging.WARNING)
            ext_logger.addHandler(file_handler)
        
        # Registrar inicio de la aplicación
        app_logger.info("Iniciando aplicación Admin Kiosk")
    
    @staticmethod
    def log_kiosk_event(event_type, kiosk_uuid, details=None):
        """
        Registrar eventos específicos de kiosks
        
        Args:
            event_type (str): Tipo de evento (creación, actualización, eliminación)
            kiosk_uuid (str): UUID del kiosk
            details (dict, opcional): Detalles adicionales del evento
        """
        logger = logging.getLogger('admin_kiosk.kiosk')
        log_message = f"Evento de Kiosk: {event_type} - UUID: {kiosk_uuid}"
        
        if details:
            log_message += f" - Detalles: {details}"
        
        logger.info(log_message)
    
    @staticmethod
    def log_auth_event(event_type, username, success=True):
        """
        Registrar eventos de autenticación
        
        Args:
            event_type (str): Tipo de evento (login, logout, registro)
            username (str): Nombre de usuario
            success (bool): Indica si el evento fue exitoso
        """
        logger = logging.getLogger('admin_kiosk.auth')
        log_level = logging.INFO if success else logging.WARNING
        
        log_message = f"Evento de Autenticación: {event_type} - Usuario: {username}"
        
        if not success:
            log_message += " - Estado: Fallido"
        
        logger.log(log_level, log_message)
    
    @staticmethod
    def log_system_error(error_type, error_message, traceback_info=None):
        """
        Registrar errores del sistema
        
        Args:
            error_type (str): Tipo de error
            error_message (str): Mensaje de error
            traceback_info (str, opcional): Información de seguimiento
        """
        logger = logging.getLogger('admin_kiosk.errors')
        log_message = f"Error del Sistema: {error_type} - {error_message}"
        
        if traceback_info:
            log_message += f"\nTraceback: {traceback_info}"
        
        logger.error(log_message) 