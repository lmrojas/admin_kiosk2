"""
Tests de penetración para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
import requests
from typing import Dict, List
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class TestPenetration:
    """Suite de tests de penetración."""
    
    BASE_URL = "https://staging.admin-kiosk.com"  # URL de staging para pruebas
    
    @pytest.fixture
    def session(self):
        """Fixture para mantener una sesión de requests."""
        return requests.Session()
    
    def test_sql_injection_vulnerabilities(self, session):
        """Test para detectar vulnerabilidades de SQL Injection."""
        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users; --",
            "admin' --",
            "' OR 1=1; --"
        ]
        
        endpoints = [
            "/api/auth/login",
            "/api/kiosks/search",
            "/api/users/search"
        ]
        
        for endpoint in endpoints:
            url = urljoin(self.BASE_URL, endpoint)
            for payload in payloads:
                data = {"username": payload, "password": payload}
                response = session.post(url, json=data)
                
                # Verificar que no hay fugas de información
                assert response.status_code in [400, 401, 403]
                assert "error" in response.json()
                assert "sql" not in response.text.lower()
                assert "database" not in response.text.lower()

    def test_xss_vulnerabilities(self, session):
        """Test para detectar vulnerabilidades XSS."""
        payloads = [
            "<script>alert('xss')</script>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert('xss')",
            "<svg onload='alert(1)'>",
            "'-alert(1)-'"
        ]
        
        endpoints = [
            "/api/kiosks",
            "/api/users/profile",
            "/api/messages"
        ]
        
        for endpoint in endpoints:
            url = urljoin(self.BASE_URL, endpoint)
            for payload in payloads:
                data = {"name": payload, "description": payload}
                response = session.post(url, json=data)
                
                # Verificar que el contenido malicioso es sanitizado
                assert payload not in response.text
                assert "<script>" not in response.text
                assert "javascript:" not in response.text

    def test_csrf_protection(self, session):
        """Test para verificar protección CSRF."""
        endpoints = [
            "/api/auth/password/change",
            "/api/kiosks/create",
            "/api/users/update"
        ]
        
        # Intentar requests sin token CSRF
        for endpoint in endpoints:
            url = urljoin(self.BASE_URL, endpoint)
            response = session.post(url, json={})
            assert response.status_code in [403, 401]
            assert "csrf" in response.text.lower()
        
        # Intentar requests con token CSRF inválido
        headers = {"X-CSRFToken": "invalid-token"}
        for endpoint in endpoints:
            url = urljoin(self.BASE_URL, endpoint)
            response = session.post(url, json={}, headers=headers)
            assert response.status_code in [403, 401]

    def test_authentication_security(self, session):
        """Test para verificar seguridad en autenticación."""
        login_url = urljoin(self.BASE_URL, "/api/auth/login")
        
        # Probar límite de intentos
        for _ in range(6):
            response = session.post(login_url, json={
                "username": "test",
                "password": "wrong"
            })
        assert response.status_code == 429
        
        # Verificar complejidad de contraseña
        weak_passwords = ["123456", "password", "admin", "qwerty"]
        register_url = urljoin(self.BASE_URL, "/api/auth/register")
        for password in weak_passwords:
            response = session.post(register_url, json={
                "username": "test_user",
                "password": password
            })
            assert response.status_code == 400
            assert "password" in response.json()["error"].lower()

    def test_sensitive_data_exposure(self, session):
        """Test para detectar exposición de datos sensibles."""
        endpoints = [
            "/api/users",
            "/api/kiosks",
            "/api/logs",
            "/api/config"
        ]
        
        sensitive_patterns = [
            r"\b[\w\.-]+@[\w\.-]+\.\w+\b",  # Email
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # Tarjetas de crédito
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"password|secret|key|token|credential",  # Palabras clave sensibles
            r"(\b25[0-5]|\b2[0-4][0-9]|\b[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}"  # IPs
        ]
        
        for endpoint in endpoints:
            url = urljoin(self.BASE_URL, endpoint)
            response = session.get(url)
            
            if response.status_code == 200:
                content = response.text
                for pattern in sensitive_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    assert not list(matches), f"Datos sensibles encontrados: {pattern}"

    def test_security_headers(self, session):
        """Test para verificar headers de seguridad."""
        required_headers = {
            'Strict-Transport-Security': r'max-age=\d+',
            'X-Frame-Options': r'DENY',
            'X-Content-Type-Options': r'nosniff',
            'X-XSS-Protection': r'1; mode=block',
            'Content-Security-Policy': r".+",
            'Referrer-Policy': r'strict-origin-when-cross-origin'
        }
        
        response = session.get(self.BASE_URL)
        
        for header, pattern in required_headers.items():
            assert header in response.headers
            assert re.match(pattern, response.headers[header])

    def test_file_upload_vulnerabilities(self, session):
        """Test para detectar vulnerabilidades en carga de archivos."""
        upload_url = urljoin(self.BASE_URL, "/api/files/upload")
        
        malicious_files = [
            ("malicious.php", b"<?php echo 'hacked'; ?>"),
            ("malicious.js", b"<script>alert('xss')</script>"),
            ("malicious.jpg.php", b"GIF89a<?php system($_GET['cmd']); ?>"),
            ("../../../etc/passwd", b"attempt to traverse directories"),
        ]
        
        for filename, content in malicious_files:
            files = {'file': (filename, content, 'application/octet-stream')}
            response = session.post(upload_url, files=files)
            
            assert response.status_code in [400, 403]
            assert "error" in response.json()

    def test_api_security(self, session):
        """Test para verificar seguridad en la API."""
        # Verificar versionado de API
        response = session.get(urljoin(self.BASE_URL, "/api"))
        assert "version" in response.json()
        
        # Verificar rate limiting
        for _ in range(101):
            response = session.get(urljoin(self.BASE_URL, "/api/kiosks"))
        assert response.status_code == 429
        
        # Verificar métodos HTTP permitidos
        endpoints = ["/api/kiosks", "/api/users", "/api/auth/login"]
        for endpoint in endpoints:
            url = urljoin(self.BASE_URL, endpoint)
            response = session.options(url)
            allowed_methods = response.headers.get('Allow', '').split(', ')
            assert all(method in ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'] 
                      for method in allowed_methods)

    def test_session_security(self, session):
        """Test para verificar seguridad en el manejo de sesiones."""
        # Login para obtener una sesión
        login_url = urljoin(self.BASE_URL, "/api/auth/login")
        response = session.post(login_url, json={
            "username": "test_user",
            "password": "Test123!@#"
        })
        
        if response.status_code == 200:
            cookies = session.cookies
            
            # Verificar flags de cookies
            for cookie in cookies:
                assert cookie.secure  # Requiere HTTPS
                assert cookie.has_nonstandard_attr('HttpOnly')  # No accesible por JS
                assert cookie.has_nonstandard_attr('SameSite')  # Protección CSRF
            
            # Verificar expiración de sesión
            response = session.get(urljoin(self.BASE_URL, "/api/kiosks"))
            assert response.status_code == 200
            
            # Esperar más del tiempo de expiración
            import time
            time.sleep(3700)  # 1 hora + 100 segundos
            
            response = session.get(urljoin(self.BASE_URL, "/api/kiosks"))
            assert response.status_code == 401  # Sesión expirada 