"""
Módulo base para modelos.
Sigue el patrón MVT + S.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    """Inicializa la base de datos."""
    db.init_app(app) 