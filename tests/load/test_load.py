#!/usr/bin/env python
# Este código solo puede ser modificado según cura.md y project_custom_structure.txt

from locust import HttpUser, task, events, between
from app import create_app
from app.services.monitoring_service import MonitoringService
from app.services.logging_service import LoggingService

# Crear la aplicación y el contexto
app = create_app()
app.app_context().push()

# Configurar servicios
monitoring_service = MonitoringService()
logging_service = LoggingService()
logger = logging_service.get_logger(__name__)

class LoadTestUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Inicializar usuario de prueba"""
        try:
            # Login
            response = self.client.post("/api/auth/login", json={
                "username": "test_user",
                "password": "test_password"
            })
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
                logger.info("Usuario de prueba autenticado exitosamente")
            else:
                logger.error(f"Error en login: {response.text}")
        except Exception as e:
            logger.error(f"Error iniciando usuario de prueba: {str(e)}")
            
    @task(3)
    def view_dashboard(self):
        """Ver dashboard"""
        try:
            with self.client.get("/api/dashboard/status", headers=self.headers, catch_response=True) as response:
                if response.status_code == 200:
                    monitoring_service.register_metric("dashboard_view_success", 1)
                else:
                    monitoring_service.register_metric("dashboard_view_error", 1)
                    response.failure(f"Error al ver dashboard: {response.text}")
        except Exception as e:
            logger.error(f"Error en view_dashboard: {str(e)}")
            
    @task(2)
    def check_kiosk_status(self):
        """Verificar estado de kiosks"""
        try:
            with self.client.get("/api/kiosks/status", headers=self.headers, catch_response=True) as response:
                if response.status_code == 200:
                    monitoring_service.register_metric("kiosk_status_check_success", 1)
                else:
                    monitoring_service.register_metric("kiosk_status_check_error", 1)
                    response.failure(f"Error al verificar kiosks: {response.text}")
        except Exception as e:
            logger.error(f"Error en check_kiosk_status: {str(e)}")
            
    @task(1)
    def update_kiosk(self):
        """Actualizar información de kiosk"""
        try:
            kiosk_data = {
                "name": "Test Kiosk",
                "location": "Test Location",
                "status": "active"
            }
            with self.client.put("/api/kiosks/1", json=kiosk_data, headers=self.headers, catch_response=True) as response:
                if response.status_code == 200:
                    monitoring_service.register_metric("kiosk_update_success", 1)
                else:
                    monitoring_service.register_metric("kiosk_update_error", 1)
                    response.failure(f"Error al actualizar kiosk: {response.text}")
        except Exception as e:
            logger.error(f"Error en update_kiosk: {str(e)}")

# Event handlers
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Registrar inicio de prueba"""
    try:
        monitoring_service.register_event("load_test_started", {
            "num_users": environment.runner.target_user_count,
            "spawn_rate": environment.runner.spawn_rate
        })
        logger.info("Prueba de carga iniciada")
    except Exception as e:
        logger.error(f"Error al registrar inicio de prueba: {str(e)}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Registrar fin de prueba"""
    try:
        stats = environment.runner.stats
        monitoring_service.register_event("load_test_completed", {
            "total_requests": stats.total.num_requests,
            "total_failures": stats.total.num_failures,
            "response_time_avg": stats.total.avg_response_time
        })
        logger.info("Prueba de carga completada")
    except Exception as e:
        logger.error(f"Error al registrar fin de prueba: {str(e)}") 