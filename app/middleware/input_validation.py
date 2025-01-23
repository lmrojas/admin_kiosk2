"""
Middleware de validación y sanitización de entrada.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from functools import wraps
from flask import request, abort, current_app
import bleach
import re
import json
from typing import Dict, Any, List, Optional
import logging
from app.config.security_config import INPUT_SANITIZATION
from app.services.config_service import config_service
from bleach import clean

logger = logging.getLogger(__name__)

class InputValidationMiddleware:
    """Middleware para validación y sanitización de entrada."""
    
    def __init__(self):
        """Inicializar reglas de validación."""
        self.validation_rules = {
            'text': r'^[\w\s\-.,!?@#$%&*()]+$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'username': r'^[a-zA-Z0-9_-]{3,16}$',
            'password': r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{12,}$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'time': r'^\d{2}:\d{2}(:\d{2})?$',
            'number': r'^\d+$',
            'float': r'^\d*\.?\d+$',
            'boolean': r'^(true|false|0|1)$',
            'url': r'^https?:\/\/.+',
            'file_path': r'^[\w\-. \/\\]+$'
        }
        
    def sanitize_input(self, value: str, input_type: str = 'text') -> str:
        """
        Sanitiza un valor de entrada.
        
        Args:
            value: Valor a sanitizar
            input_type: Tipo de entrada (text, html, etc)
            
        Returns:
            str: Valor sanitizado
        """
        if not isinstance(value, str):
            return value
            
        # Sanitización básica
        value = value.strip()
        
        if input_type == 'html':
            # Sanitizar HTML permitiendo solo tags y atributos seguros
            return bleach.clean(
                value,
                tags=INPUT_SANITIZATION['ALLOWED_HTML_TAGS'],
                attributes=INPUT_SANITIZATION['ALLOWED_ATTRIBUTES'],
                strip=INPUT_SANITIZATION['STRIP_COMMENTS']
            )
        else:
            # Sanitización para texto plano
            return bleach.clean(value, tags=[], strip=True)
            
    def validate_input(self, value: str, input_type: str) -> bool:
        """
        Valida un valor de entrada contra las reglas definidas.
        
        Args:
            value: Valor a validar
            input_type: Tipo de entrada a validar
            
        Returns:
            bool: True si es válido, False si no
        """
        if not isinstance(value, str):
            return False
            
        if input_type not in self.validation_rules:
            return False
            
        pattern = self.validation_rules[input_type]
        return bool(re.match(pattern, value))
        
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza un diccionario de datos.
        
        Args:
            data: Diccionario a sanitizar
            
        Returns:
            Dict: Diccionario sanitizado
        """
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_dict(item) if isinstance(item, dict)
                    else self.sanitize_input(str(item))
                    for item in value
                ]
            else:
                sanitized[key] = self.sanitize_input(str(value))
        return sanitized
        
    def validate_request_data(self) -> None:
        """
        Valida y sanitiza los datos de la request actual.
        Aborta con 400 si hay datos inválidos.
        """
        try:
            # Validar form data
            if request.form:
                for key, value in request.form.items():
                    if not self.validate_input(value, self._get_input_type(key)):
                        logger.warning(f"Dato inválido en form: {key}={value}")
                        abort(400, f"Dato inválido: {key}")
                        
            # Validar query params
            if request.args:
                for key, value in request.args.items():
                    if not self.validate_input(value, self._get_input_type(key)):
                        logger.warning(f"Dato inválido en query: {key}={value}")
                        abort(400, f"Dato inválido: {key}")
                        
            # Validar JSON
            if request.is_json:
                data = request.get_json()
                if isinstance(data, dict):
                    self._validate_json_data(data)
                    
        except Exception as e:
            logger.error(f"Error validando request: {str(e)}")
            abort(400, "Error validando datos de entrada")
            
    def _validate_json_data(self, data: Dict[str, Any], path: str = '') -> None:
        """
        Valida recursivamente datos JSON.
        
        Args:
            data: Datos a validar
            path: Path actual en la estructura JSON
        """
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                self._validate_json_data(value, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_json_data(item, f"{current_path}[{i}]")
                    else:
                        if not self.validate_input(str(item), self._get_input_type(key)):
                            logger.warning(f"Dato JSON inválido: {current_path}[{i}]={item}")
                            abort(400, f"Dato inválido: {current_path}[{i}]")
            else:
                if not self.validate_input(str(value), self._get_input_type(key)):
                    logger.warning(f"Dato JSON inválido: {current_path}={value}")
                    abort(400, f"Dato inválido: {current_path}")
                    
    def _get_input_type(self, field_name: str) -> str:
        """
        Determina el tipo de validación basado en el nombre del campo.
        
        Args:
            field_name: Nombre del campo
            
        Returns:
            str: Tipo de validación a aplicar
        """
        field_name = field_name.lower()
        
        if 'email' in field_name:
            return 'email'
        elif 'password' in field_name:
            return 'password'
        elif 'username' in field_name:
            return 'username'
        elif 'date' in field_name:
            return 'date'
        elif 'time' in field_name:
            return 'time'
        elif any(word in field_name for word in ['number', 'count', 'id']):
            return 'number'
        elif 'url' in field_name:
            return 'url'
        elif 'path' in field_name:
            return 'file_path'
        elif field_name in ['true', 'false', 'enabled', 'active']:
            return 'boolean'
        
        return 'text'
        
    def validate_input_decorator(self):
        """Decorador para validar input en rutas."""
        def decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                self.validate_request_data()
                return f(*args, **kwargs)
            return wrapped
        return decorator

def sanitize_input():
    """Sanitiza la entrada del usuario usando la configuración del servicio."""
    if request.method == 'POST':
        sanitization_config = config_service.get_security_config()['INPUT_SANITIZATION']
        
        for key, value in request.form.items():
            if isinstance(value, str):
                request.form[key] = clean(
                    value,
                    tags=sanitization_config['ALLOWED_TAGS'],
                    attributes=sanitization_config['ALLOWED_ATTRIBUTES'],
                    strip=sanitization_config['STRIP']
                )
                
                if len(request.form[key]) > sanitization_config['MAX_LENGTH']:
                    raise ValueError(f'Input {key} exceeds maximum length') 