A continuación se presenta la **versión final y actualizada** de la Guía Definitiva de Desarrollo para un **Sistema de Kiosks Inteligentes** con Flask + IA, **aclarando expresamente** que:

1. La **lógica de testing** o simulación de kiosks (alta, comportamientos, estados, etc.) se realiza **en un microservicio/app aparte**, **por fuera** del sistema principal.  
2. El **sistema en sí** (admin_kiosk) únicamente **recibe** datos (vía WebSockets/API) y responde a los kiosks reales o simulados (ej.: reiniciar, bloquear…).  
3. **No hay simulación interna** en Flask: todos los "datos simulados" llegan como si fueran reales.

> **Nota**: Este documento **reemplaza** versiones anteriores de la guía, enfatizando la existencia de una **Kiosk App separada** para simular o testear comportamientos (en paralelo) mientras el sistema Flask + IA se mantiene ligero, seguro y enfocado en la lógica real.

---

# Guía Definitiva de Desarrollo: Sistema de Kiosks Inteligentes con Flask + IA

## 0. Preámbulo: Objetivo y Filosofía

### 0.1 Objetivo Principal
Crear un **sistema** que:
- **Escale** con facilidad (altamente escalable).  
- **Garantice la seguridad** (datos y autenticación).  
- **Aproveche IA** para análisis de anomalías, alertas, etc.  
- **Minimice la intervención humana**, incluso para reentrenar el modelo IA.  
- **Separe** completamente la simulación/testing del sistema productivo.

### 0.2 Principios Fundamentales
- **Arquitectura**: MVT (Model-View-Template) + Services.  
- **Separación estricta** de responsabilidades (módulos, servicios).  
- **Buenas prácticas** (CI/CD, testing, MLOps, etc.).  
- **Cambios** solo bajo las normas de `cura.md`, actualizando `project_custom_structure.txt`.
- **No simulación interna**: Los kiosks son entidades externas que se conectan al sistema.

> **Importante**: El sistema NO simula kiosks internamente. Todo kiosk debe registrarse a través del sistema administrativo y conectarse vía WebSocket.

---

## 1. Estructura del Proyecto

```
admin_kiosk/
│
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kiosk.py        # Modelo de kiosk y datos de sensores
│   │   └── user.py         # Modelo de usuario y autenticación
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── kiosk_service.py  # Manejo de kiosks reales
│   │   └── kiosk_ai_service.py  # Análisis de anomalías
│   ├── blueprints/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── kiosk.py
│   └── templates/
│       ├── base.html
│       └── kiosk/
│           ├── index.html      # Lista de kiosks
│           └── dashboard.html  # Estado del kiosk
│
├── scripts/
│   ├── init_roles.py      # Inicializa roles y permisos
│   ├── init_data.py       # Crea usuario admin inicial
│   └── init_kiosks.py     # Verifica estructura de tablas
│
├── requirements.txt
├── run.py
└── README.md
```

### 1.1 Comentario Obligatorio
En **todos** los archivos principales:
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md
```

---

## 2. Configuración del Entorno

### 2.1 Requisitos Previos
- **Python 3.9+**  
- `pip`  
- `virtualenv / venv`

### 2.2 Crear Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: .\venv\Scripts\activate
```

### 2.3 Instalación de Dependencias
```bash
pip install flask flask-sqlalchemy flask-login flask-migrate torch
pip freeze > requirements.txt
```

### 2.4 Versiones Sugeridas
```
Flask==2.1.0
Flask-SQLAlchemy==2.5.1
Flask-Login==0.5.0
Flask-Migrate==3.1.0
torch==1.9.0
```

---

## 3. Inicialización de la App Flask

### 3.1 `app/__init__.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_object='config.default.Config'):
    app = Flask(__name__)
    app.config.from_object(config_object)

    db.init_app(app)
    login_manager.init_app(app)

    # Registro de blueprints
    from .blueprints.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    # Otros blueprints, p.ej. kiosk

    return app
```

---

## 4. Modelos de Datos

### 4.1 `app/models/user.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
```
> Añade más modelos (Kiosk, Logs, etc.) según sea necesario.

---

## 5. Servicios de Negocio (Capa Services)

### 5.1 `app/services/auth_service.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

from app.models.user import User
from app import db

class AuthService:
    @staticmethod
    def register_user(username, password):
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user

    @staticmethod
    def authenticate(username, password):
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            return user
        return None
```

### 5.2 `app/services/kiosk_ai_service.py` (IA)
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import torch

class KioskAIService:
    """
    Servicio para análisis de anomalías en kiosks.
    Procesa datos recibidos de kiosks reales.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not self._initialized:
            self.model = None
            self._initialized = True
            self._load_model()
            
    def _load_model(self):
        """Carga el modelo de detección de anomalías."""
        try:
            model_path = current_app.config['AI_MODEL_PATH']
            self.model = torch.load(model_path)
            self.model.eval()
            logger.info("Modelo AI cargado correctamente")
        except Exception as e:
            logger.error(f"Error cargando modelo AI: {str(e)}")
            
    def predict_anomaly(self, kiosk_data):
        """
        Predice anomalías en datos recibidos de un kiosk.
        Args:
            kiosk_data (dict): Datos de CPU, memoria, latencia, etc.
        Returns:
            float: Probabilidad de anomalía
        """
        if not self.model:
            return 0.0
            
        try:
            features = self._prepare_features(kiosk_data)
            with torch.no_grad():
                prediction = self.model(features)
                return float(torch.sigmoid(prediction).item())
        except Exception as e:
            logger.error(f"Error prediciendo anomalía: {str(e)}")
            return 0.0
```

### 5.3 `app/services/kiosk_service.py`
- Manejar aquí la lógica que relaciona Kiosk con DB, logs, alertas…

---

## 6. Blueprints y Rutas

### 6.1 `app/blueprints/auth.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_user, logout_user, login_required
from app.services.auth_service import AuthService

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = AuthService.authenticate(username, password)
        if user:
            login_user(user)
            return redirect(url_for('main.index'))  # Ajusta tu blueprint principal
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
```

---

## 7. Configuraciones

### 7.1 `config/default.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'desarrollo-secreto')

    # Base de datos (PostgreSQL por defecto)
    DB_USER = os.environ.get('DB_USER', 'postgres')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', 'admin_kiosk2')

    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
```

---

## 8. Punto de Entrada

### 8.1 `run.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

from app import create_app, db

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
```

---

## 9. Migraciones de Base de Datos

```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

---

## 10. Testing (SIN Simulación Interna)

**Relevante**:  
- No existe un "motor de simulación" para kiosks dentro de Flask.  
- Tests de "kiosk behaviors" se hacen externamente, enviando datos vía WebSockets o API.  
- Aquí, solo tests unitarios/integración del core.

```python
# tests/test_auth_service.py
import pytest
from app import create_app, db
from app.services.auth_service import AuthService

@pytest.fixture
def app():
    app = create_app('config.default.Config')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

def test_user_registration(app):
    user = AuthService.register_user('testuser', 'password123')
    assert user.username == 'testuser'
```

---

## 11. Entrenamiento de IA (Reales + Sintéticos)

### 11.1 `scripts/train_ai_model.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from generate_synthetic_data import generate_synthetic_kiosk_data

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)  # 3 features -> 1 output

    def forward(self, x):
        return self.fc(x)

def train_model(real_data_path=None, output_model='models/kiosk_anomaly_model.pth'):
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Carga de datos REALES (si existen)
    data_real = pd.DataFrame()
    if real_data_path:
        data_real = pd.read_csv(real_data_path)

    # Generar DATOS SINTÉTICOS
    data_synthetic = generate_synthetic_kiosk_data(num_samples=500)

    # Combinar ambos
    combined_data = pd.concat([data_real, data_synthetic], ignore_index=True)

    # Convertir a tensores
    X = torch.tensor(combined_data[['cpu_usage','memory_usage','network_latency']].values, dtype=torch.float32)
    Y = torch.tensor(combined_data['label'].values, dtype=torch.float32).unsqueeze(1)

    # Entrenamiento
    for epoch in range(15):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    # Guardar modelo
    torch.save(model, output_model)
    print(f"Modelo guardado en {output_model}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_data_path', type=str, default=None)
    parser.add_argument('--output_model', type=str, default='models/kiosk_anomaly_model.pth')
    args = parser.parse_args()

    train_model(real_data_path=args.real_data_path, output_model=args.output_model)
```

### 11.2 `scripts/generate_synthetic_data.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import pandas as pd
import random

def generate_synthetic_kiosk_data(num_samples=500):
    """
    Crea un DataFrame con: cpu_usage, memory_usage, network_latency, label
    label: 0 => normal, 1 => anomalía
    """
    data = {
        'cpu_usage': [],
        'memory_usage': [],
        'network_latency': [],
        'label': []
    }
    for _ in range(num_samples):
        cpu = random.uniform(0, 100)
        mem = random.uniform(0, 100)
        net = random.uniform(10, 300)
        anomaly = 1 if (cpu > 90 or mem > 90 or net > 250) else 0

        data['cpu_usage'].append(cpu)
        data['memory_usage'].append(mem)
        data['network_latency'].append(net)
        data['label'].append(anomaly)

    return pd.DataFrame(data)
```

---

## 12. Pipeline de Reentrenamiento Automático

### 12.1 `scripts/auto_retrain_pipeline.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import os
import subprocess

def auto_retrain(real_data_path=None, output_model='models/kiosk_anomaly_model.pth'):
    """
    1) Llama a train_ai_model.py
    2) (Opcional) valida performance
    3) Actualiza modelo si pasa umbrales
    """
    cmd = [
        'python', 'scripts/train_ai_model.py',
        f'--output_model={output_model}'
    ]
    if real_data_path:
        cmd.append(f'--real_data_path={real_data_path}')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Entrenamiento fallido.")
        print(result.stdout)
        print(result.stderr)
        return

    print(f"[INFO] Entrenamiento finalizado. Modelo en {output_model}")

if __name__ == "__main__":
    auto_retrain(real_data_path='data/real_kiosks.csv')
```

---

## 13. Manejo de Credenciales y Entornos

- Usar **variables de entorno** para `SECRET_KEY`, `DATABASE_URL`, etc.  
- No subir contraseñas en texto plano a repos.  
- Revisar `@cura.md` ante cambios sensibles.

---

## 14. Despliegue

- **Desarrollo**: `python run.py` (Flask).  
- **Producción**: Gunicorn/WSGI + Nginx, Docker, K8s, etc.

---

## 15. Git y Control de Versiones

- `git init && git remote add origin ...`  
- Ramas `feature/...`, commits con mensajes claros.  
- Revisar `@cura.md` antes de merges.

---

## 16. Scripts de Estructura y Documentación

- `scripts/export_structure.py`: actualiza `project_custom_structure.txt` con la estructura final.

---

## 17. Comunicación en Tiempo Real

- Flask-SocketIO (u otro) para recibir métricas en vivo.  
- **Los kiosks** (reales o simulados) **conectan** a `admin_kiosk` y **emiten** sus datos.  
- **admin_kiosk** solo **escucha** y responde (por ejemplo, enviando comandos como "reiniciar kiosk").

---

## 18. Próximos Pasos

- Roles/permisos.  
- Optimizar IA.  
- Docker/Kubernetes.  
- Logs centralizados (Grafana/ELK).  
- **Geolocalización** + dashboards.

---

## 19. Manejo de Cambios en BD y Backups

1. Planear migraciones, backups.  
2. `flask db migrate && flask db upgrade`.  
3. Test con `pytest`.  
4. Actualizar `project_custom_structure.txt`.

---

## 20. **Aclaración sobre el Testing de Kiosks**

- **Simulaciones** (kiosk offline, alta temperatura, etc.) se hacen en un **microservicio/app aparte**.  
- **admin_kiosk** solo recibe los datos por WebSocket/API.  
- Sin motor interno de simulación => la lógica de test está "fuera" de la aplicación Flask.

---

## 21. Resumen y Conclusión

- **Objetivo**: Un Flask MVT + Services con IA, que mezcla datos reales y sintéticos para entrenar su modelo, **sin** mezclar la simulación de kiosks en la app.  
- **Scripts**:  
  - `train_ai_model.py` (entrena IA).  
  - `generate_synthetic_data.py` (crea datos sintéticos).  
  - `auto_retrain_pipeline.py` (pipeline automático).  
  - *Kiosk-simulator apps* (externas) que envían datos al sistema.  
- **Resultado**: Una plataforma que recibe "datos reales" (aunque simulados externamente), registra estados y alertas, e integra un proceso MLOps (reentrenamiento) independiente.  

