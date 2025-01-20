"""
Tests de integración para el flujo del sistema de IA.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from app.services.ai_service import AIService
from app.services.data_service import DataService
from app.models.sensor_data import SensorData
import numpy as np
from datetime import datetime, timedelta
import json

class TestAIFlow:
    """Suite de tests de integración para el flujo completo del sistema de IA."""
    
    @pytest.fixture
    def ai_service(self, app):
        """Fixture para el servicio de IA."""
        return AIService()
        
    @pytest.fixture
    def data_service(self, app):
        """Fixture para el servicio de datos."""
        return DataService()
        
    def test_entrenamiento_prediccion_flow(self, client, ai_service, data_service):
        """Test del flujo completo de entrenamiento y predicción."""
        # 1. Generar datos de entrenamiento
        train_data = []
        for i in range(100):
            timestamp = datetime.utcnow() - timedelta(hours=i)
            data = {
                'kiosk_id': 1,
                'cpu_usage': np.random.uniform(20, 80),
                'ram_usage': np.random.uniform(500, 2000),
                'temperature': np.random.uniform(35, 45),
                'timestamp': timestamp.isoformat()
            }
            train_data.append(data)
            
        # Guardar datos de entrenamiento
        data_service.bulk_save_sensor_data(train_data)
        
        # 2. Entrenar modelo
        response = client.post('/api/ai/train', json={
            'model_type': 'anomaly_detection',
            'parameters': {
                'contamination': 0.1,
                'random_state': 42
            }
        })
        assert response.status_code == 200
        model_id = response.json['model_id']
        
        # 3. Verificar métricas de entrenamiento
        response = client.get(f'/api/ai/models/{model_id}/metrics')
        assert response.status_code == 200
        metrics = response.json
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # 4. Realizar predicción
        test_data = {
            'cpu_usage': 90.0,  # Valor anómalo
            'ram_usage': 1500,
            'temperature': 50.0  # Temperatura alta
        }
        
        response = client.post(
            f'/api/ai/models/{model_id}/predict',
            json=test_data
        )
        assert response.status_code == 200
        prediction = response.json
        assert prediction['is_anomaly'] == True
        
    def test_validacion_cruzada_flow(self, client, ai_service):
        """Test del flujo de validación cruzada."""
        # 1. Configurar validación
        cv_config = {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 42
        }
        
        response = client.post('/api/ai/cross-validate', json={
            'model_type': 'anomaly_detection',
            'cv_config': cv_config
        })
        assert response.status_code == 200
        cv_id = response.json['cv_id']
        
        # 2. Verificar resultados
        response = client.get(f'/api/ai/cross-validate/{cv_id}/results')
        assert response.status_code == 200
        results = response.json
        
        assert len(results['fold_scores']) == cv_config['n_splits']
        assert 'mean_score' in results
        assert 'std_score' in results
        
    def test_explicabilidad_predicciones_flow(self, client, ai_service):
        """Test del flujo de explicabilidad de predicciones."""
        # 1. Entrenar modelo
        response = client.post('/api/ai/train', json={
            'model_type': 'anomaly_detection',
            'parameters': {'random_state': 42}
        })
        assert response.status_code == 200
        model_id = response.json['model_id']
        
        # 2. Realizar predicción
        test_data = {
            'cpu_usage': 95.0,
            'ram_usage': 2500,
            'temperature': 55.0
        }
        
        response = client.post(
            f'/api/ai/models/{model_id}/predict',
            json=test_data
        )
        assert response.status_code == 200
        prediction_id = response.json['prediction_id']
        
        # 3. Obtener explicación
        response = client.get(
            f'/api/ai/predictions/{prediction_id}/explain'
        )
        assert response.status_code == 200
        explanation = response.json
        
        assert 'feature_importance' in explanation
        assert 'shap_values' in explanation
        assert len(explanation['feature_importance']) > 0
        
    def test_monitoreo_drift_flow(self, client, ai_service, data_service):
        """Test del flujo de monitoreo de drift."""
        # 1. Configurar monitoreo
        drift_config = {
            'window_size': 24,  # horas
            'threshold': 0.05,
            'features': ['cpu_usage', 'ram_usage', 'temperature']
        }
        
        response = client.post('/api/ai/drift/configure', json=drift_config)
        assert response.status_code == 200
        
        # 2. Generar datos con drift
        drift_data = []
        for i in range(48):  # 2 días
            timestamp = datetime.utcnow() - timedelta(hours=i)
            # Introducir drift gradual en temperatura
            temp_drift = 0 if i < 24 else 10
            data = {
                'kiosk_id': 1,
                'cpu_usage': np.random.uniform(20, 80),
                'ram_usage': np.random.uniform(500, 2000),
                'temperature': np.random.uniform(35, 45) + temp_drift,
                'timestamp': timestamp.isoformat()
            }
            drift_data.append(data)
            
        # Guardar datos
        data_service.bulk_save_sensor_data(drift_data)
        
        # 3. Detectar drift
        response = client.post('/api/ai/drift/detect')
        assert response.status_code == 200
        drift_results = response.json
        
        assert 'has_drift' in drift_results
        assert 'drift_metrics' in drift_results
        assert drift_results['has_drift'] == True  # Debería detectar drift
        
        # 4. Obtener reporte de drift
        response = client.get('/api/ai/drift/report')
        assert response.status_code == 200
        report = response.json
        
        assert 'feature_drifts' in report
        assert 'temperature' in report['feature_drifts']
        assert report['feature_drifts']['temperature']['has_drift'] == True 