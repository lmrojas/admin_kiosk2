A continuación, se presenta una versión **todavía más completa** de la Guía Definitiva de Desarrollo para un **Sistema de Kiosks Inteligentes** con Flask + IA, **incluyendo** un **ciclo de entrenamiento automático** que fusione tanto datos **reales** como datos **sintéticos** (simulados), con la posibilidad de reentrenar el modelo de IA de manera **periódica** o **en base a eventos**. De esta forma, el sistema **aprende** y se **actualiza** sin intervención manual constante.

---

# Guía Definitiva de Desarrollo: Sistema de Kiosks Inteligentes con Flask + IA (Entrenamiento Automático)

## 0. Preámbulo: Objetivo y Filosofía

### 0.1 Objetivo Principal
Crear un sistema de kiosks que sea:
- 🚀 **Altamente escalable**  
- 🔒 **Seguro**  
- 🧠 **Inteligente** (soportado por IA)  
- 💰 **Eficiente en costos**  
- 👥 **Mínima intervención humana** (incluyendo la opción de **reentrenar** la IA automáticamente)

### 0.2 Principios Fundamentales
- **Patrón de Arquitectura**: MVT (Model-View-Template) + Services  
- **Separación estricta** de responsabilidades  
- **Código modular** y mantenible  
- **Desarrollo guiado** por buenas prácticas (testing, CI/CD, IA explicable, etc.)  
- **Regla de oro**: Todo cambio en el proyecto **debe** acatar las normas de `@cura.md` (o `cursor_ai_rules.md`) y consultar `project_custom_structure.txt` antes de modificarse la estructura o introducir nuevos archivos.

> **Nuevo Énfasis**: **Automatizar** el ciclo de vida del modelo de IA, combinando **datos reales** y **sintéticos** para reentrenar periódicamente y mantener un sistema de kiosks **en constante aprendizaje**.

---

## 1. Estructura del Proyecto

```
admin_kiosk/
│
├── app/                    
│   ├── __init__.py         
│   ├── models/             
│   │   ├── __init__.py
│   │   └── user.py         
│   ├── services/           
│   │   ├── __init__.py
│   │   ├── auth_service.py      
│   │   ├── kiosk_ai_service.py  
│   │   ├── kiosk_service.py     
│   │   └── ...
│   ├── blueprints/         
│   │   ├── __init__.py
│   │   ├── auth.py         
│   │   └── kiosk.py        
│   ├── templates/          
│   │   ├── base.html
│   │   └── login.html
│   └── utils/              
│       └── __init__.py
│
├── config/                 
│   └── default.py
│
├── scripts/                
│   ├── export_structure.py       # Genera/actualiza project_custom_structure.txt
│   ├── train_ai_model.py         # Entrena el modelo de IA (manual o programado)
│   ├── generate_synthetic_data.py# Genera datos sintéticos de kiosks
│   └── auto_retrain_pipeline.py  # Ejecución automática de reentrenamiento
│
├── tests/                  
│   ├── unit/
│   └── integration/
│
├── migrations/             
│
├── requirements.txt        
├── run.py                  
├── README.md               
└── project_custom_structure.txt  
```

### 1.1 Regla Fundamental: Comentario Obligatorio

En **CADA** archivo del proyecto, la **primera línea** será:
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
```

---

## 2. Configuración del Entorno

### 2.1 Requisitos Previos
- **Python 3.9+**  
- `pip`  
- `virtualenv` (o `venv`)  

### 2.2 Crear Entorno Virtual

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Instalación de Dependencias
```bash
pip install flask flask-sqlalchemy flask-login flask-migrate torch
pip freeze > requirements.txt
```
> Mantén tu `venv` activo para evitar conflictos con el sistema global.

### 2.4 Versiones Específicas Recomendadas
```
Flask==2.1.0
Flask-SQLAlchemy==2.5.1
Flask-Login==0.5.0
Flask-Migrate==3.1.0
torch==1.9.0
```
### 2.5 Actualización de Dependencias
```bash
pip freeze > requirements.txt
```

---

## 3. Inicialización de la Aplicación Flask

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

    from .blueprints.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    # Opcional: registro de otros blueprints (kiosk, dashboard, etc.)

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
> Añade más modelos (Kiosk, SensorData, etc.) según tus necesidades.

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

### 5.2 `app/services/kiosk_ai_service.py` (Integración IA)
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import torch

class KioskAIService:
    """Servicio de IA para detección de anomalías en kiosks."""

    def __init__(self, model_path=None):
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        if path:
            try:
                model = torch.load(path, map_location=torch.device('cpu'))
                model.eval()
                return model
            except Exception as e:
                print(f"[ERROR] No se pudo cargar el modelo IA: {e}")
                return None
        return None

    def predict_anomaly(self, kiosk_data):
        """
        kiosk_data: dict con métricas (cpu_usage, memory_usage, network_latency).
        Retorna un valor float con la probabilidad de anomalía.
        """
        if not self.model:
            return 0.0

        features = torch.tensor([
            kiosk_data.get('cpu_usage', 0.0),
            kiosk_data.get('memory_usage', 0.0),
            kiosk_data.get('network_latency', 0.0)
        ], dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            output = self.model(features)
            return float(torch.sigmoid(output).item())
```

### 5.3 `app/services/kiosk_service.py`
Podría manejar:
- Estados de kiosks (online/offline)
- Métricas periódicas
- Interacción con DB (registro histórico)

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
            return redirect(url_for('main.index'))  # Ajusta la ruta a tu blueprint principal
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
    
    # Configuración base de datos PostgreSQL
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

## 10. Testing

### 10.1 `tests/test_auth_service.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

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

## 11. Entrenamiento de IA: Datos Reales + Sintéticos

### 11.1 Script de Entrenamiento Manual: `scripts/train_ai_model.py`
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
        data_real = pd.read_csv(real_data_path)  # CSV con cpu_usage, memory_usage, network_latency, label

    # Generar DATOS SINTÉTICOS
    data_synthetic = generate_synthetic_kiosk_data(num_samples=500)  
    # Retorna un DataFrame con: cpu_usage, memory_usage, network_latency, label (0 o 1)

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
    # Ejemplo de ejecución:
    # python scripts/train_ai_model.py --real_data_path data/real_kiosks.csv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--real_data_path', type=str, default=None)
    parser.add_argument('--output_model', type=str, default='models/kiosk_anomaly_model.pth')
    args = parser.parse_args()

    train_model(real_data_path=args.real_data_path, output_model=args.output_model)
```

> Este script **combina** datos reales (si existen) con **datos sintéticos** generados al vuelo, entrenando un modelo y guardándolo. Puede ejecutarse **manualmente** o **ser llamado** por un pipeline automático.

### 11.2 Generación de Datos Sintéticos: `scripts/generate_synthetic_data.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import pandas as pd
import random

def generate_synthetic_kiosk_data(num_samples=500):
    """
    Crea un DataFrame con columnas: cpu_usage, memory_usage, network_latency, label
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
        # Heurística simple para etiquetar anomalías
        anomaly = 1 if (cpu > 90.0 or mem > 90.0 or net > 250) else 0

        data['cpu_usage'].append(cpu)
        data['memory_usage'].append(mem)
        data['network_latency'].append(net)
        data['label'].append(anomaly)

    return pd.DataFrame(data)
```

---

## 12. **Entrenamiento Automático** (Opcional, Recomendado)

Para automatizar, podemos tener un **script** o **job** que corra periódicamente (vía cron, Celery, Airflow, etc.) y que:

1. Descargue / Recopile los datos **reales** del sistema (si existen).  
2. **Llame** a `generate_synthetic_kiosk_data()` para crear datos de refuerzo.  
3. **Ejecute** `train_model()` con ambos datasets combinados.  
4. **Reemplace** el modelo en `models/kiosk_anomaly_model.pth` si la **nueva versión** supera ciertos umbrales de calidad.

### 12.1 Ejemplo: `scripts/auto_retrain_pipeline.py`
```python
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE 
# SIGUIENDO LO ESTABLECIDO EN @cura.md

import os
import subprocess
import datetime

def auto_retrain(real_data_path=None, output_model='models/kiosk_anomaly_model.pth'):
    """
    Script que ejecuta el pipeline completo:
    1) Llama a train_ai_model.py con real_data_path + datos sintéticos
    2) (Opcional) Valida la performance (comparar nuevo modelo vs. viejo)
    3) Actualiza modelo en producción si está bien
    """
    # 1) Ejecutar script de entrenamiento
    cmd = [
        'python', 'scripts/train_ai_model.py',
        f'--output_model={output_model}'
    ]
    if real_data_path:
        cmd.append(f'--real_data_path={real_data_path}')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Entrenamiento fallido. Logs:")
        print(result.stdout)
        print(result.stderr)
        return

    # 2) (Opcional) Se podría hacer una validación extra aquí
    print(f"[INFO] Entrenamiento finalizado. Modelo actualizado: {output_model}")

if __name__ == "__main__":
    # Podrías usar un job de cron/celery/airflow para llamar este script
    # Ejemplo: 0 3 * * * /usr/bin/python scripts/auto_retrain_pipeline.py
    auto_retrain(real_data_path='data/real_kiosks.csv')
```

> Con esto, tu sistema de kiosks puede tener un **pipeline** de reentrenamiento automático, combinando datos **reales** y **sintéticos**.

---

## 13. Manejo de Credenciales y Entornos

1. **Variables de Entorno**: `SECRET_KEY`, `DATABASE_URL`, etc.  
2. **Evitar** subir contraseñas/llaves.  
3. **Backups** antes de grandes cambios en la DB.  
4. **@cura.md**: Revisa reglas antes de modificar algo sensible.

---

## 14. Acceso al Servidor

- Para **desarrollo local**: `python run.py`.  
- Para **producción**: WSGI + Nginx, Docker, Kubernetes, etc.

---

## 15. Configuración del Repositorio en Git

1. `git init && git remote add origin ...`  
2. Crea ramas `feature/xxx`.  
3. Commits claros (`feat:`, `fix:`).  
4. Push tras pasar `pytest` y leer `@cura.md`.

---

## 16. Scripts de Estructura y Documentación

- `scripts/export_structure.py`: actualiza `project_custom_structure.txt`.
- **Obligatorio** tras crear/borrar archivos.

---

## 17. Comunicación en Tiempo Real (Opcional)

- Flask-SocketIO para recibir métricas en vivo.  
- Notificar anomalías a un dashboard.

---

## 18. Próximos Pasos y Microservicios

1. **Autorización** (roles, permisos).  
2. **Optimizar** el modelo IA (hiperparámetros, data augmentation).  
3. **Despliegue** en contenedores (Docker) y orquestación (Kubernetes).  
4. **Logs centralizados** (Grafana, Kibana).  
5. **Geolocalización** si manejas ubicaciones de kiosks.

---

## 19. Consideraciones Finales

- **MVT + Services**: No mezclar lógica de negocio en modelos/vistas.  
- **`@cura.md`**: Dicta las reglas de edición, testing, backups, versionado.  
- **`project_custom_structure.txt`**: actualiza tras cada cambio importante.  
- **IA**: Entrenamiento **automático** gracias a scripts (real + sintético).  
- **Seguridad**: variables de entorno, backups, logs, Docker, CI/CD.

> **Nota**: La innovación es un **viaje continuo** de aprendizaje y mejora. Ahora, con un **pipeline automático** de reentrenamiento IA (que fusiona datos reales y simulados), tu **Sistema de Kiosks Inteligentes** mantendrá su **precisión** y **robustez** de forma casi desatendida.  

---

## 20. Apéndice: Manejo de Cambios en BD y Backups

1. **Planifica** grandes cambios (nuevas columnas, alter table).  
2. **Backup** (pg_dump en PostgreSQL u otro).  
3. **Migraciones** (`flask db migrate`, `flask db upgrade`).  
4. **Validar** con `pytest`.  
5. **Actualizar** `project_custom_structure.txt`.

---

## 21. Resumen y Conclusión

- **Objetivo**: Un proyecto Flask MVT + Services, con IA (PyTorch), con un **proceso de entrenamiento automático** que usa **datos reales** + **datos sintéticos**.  
- **Scripts clave**:  
  - `train_ai_model.py`: entrena el modelo manual o automatizado.  
  - `generate_synthetic_data.py`: produce datos sintéticos.  
  - `auto_retrain_pipeline.py`: pipeline automático (cron/Airflow/Celery).  
- **Futuro**: CI/CD de IA (MLOps), contenedores, orquestación, dashboards en tiempo real.  

Con esta **Guía Definitiva**, tendrás un **Sistema de Kiosks Inteligentes** que aprende y **se entrena por sí solo** conforme llegan más datos. Disfruta de la **automatización**, la **seguridad** y la **eficiencia** que brinda este enfoque. ¡A seguir innovando!