# Admin Kiosk

Este repositorio contiene el proyecto **Admin Kiosk**, un sistema de gestión y monitoreo integral para kiosks interactivos. Se basa en Flask siguiendo el **patrón MVT + Services** y cumple con las pautas definidas en [`cura.md`](./cura.md).

## Características Principales
- Monitoreo de sensores y recursos (CPU, RAM, Disco, etc.).
- Comunicación en tiempo real vía WebSocket.
- Gestión de ubicaciones geográficas.
- Estructura ordenada (MVT + Services) para un fácil mantenimiento y escalabilidad.

## Requisitos
- Python 3.9+
- Entorno virtual de Python (`venv`)
- [Flask](https://pypi.org/project/Flask/), [SQLAlchemy](https://pypi.org/project/SQLAlchemy/), [Flask-Login](https://pypi.org/project/Flask-Login/), etc. (ver `requirements.txt`)

## Instalación y Configuración
1. **Clona** este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/admin_kiosk.git
   ```
2. **Crea y activa** tu entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ó
   venv\Scripts\activate     # Windows
   ```
3. **Instala** dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configura** la base de datos (por ejemplo, `sqlite:///admin_kiosk.db` en `config/default.py` o la variable de entorno `DATABASE_URL`).
5. **Ejecuta** la aplicación:
   ```bash
   flask run
   ```

## Estructura del Proyecto
- `app/`: Núcleo de la aplicación (modelos, servicios, blueprints, templates).
- `config/`: Configuraciones de la aplicación.
- `scripts/`: Scripts de utilidad (exportar estructura, generar datos sintéticos, etc.).
- `tests/`: Pruebas unitarias e integradas.
- `docs/`: Documentación adicional.

## Cómo Contribuir
- Crea una rama a partir de `main` (por ejemplo, `feature/tumarama`).
- Asegúrate de **leer** [`cura.md`](./cura.md) para respetar las reglas de MVT + Services.
- Realiza tus cambios, añade tests y verifica que todo pase en verde con `pytest`.
- Envía tus PR (pull request) para revisión.

## Licencia
Este proyecto está bajo la licencia **MIT** (o la que corresponda). Mira el archivo `LICENSE` para más detalles.
