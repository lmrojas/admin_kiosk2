"""initial migration

Revision ID: 0db44f6c341e
Revises: 
Create Date: 2025-01-20 14:10:37.651150

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0db44f6c341e'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('drift_metrics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('model_version', sa.String(length=50), nullable=False),
    sa.Column('analysis_window', sa.Integer(), nullable=True),
    sa.Column('distribution_shift', sa.Float(), nullable=True),
    sa.Column('feature_drift', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('performance_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('alerts', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('severity', sa.String(length=20), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('kiosks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('uuid', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('location', sa.String(length=200), nullable=True),
    sa.Column('latitude', sa.Float(), nullable=True),
    sa.Column('longitude', sa.Float(), nullable=True),
    sa.Column('altitude', sa.Float(), nullable=True),
    sa.Column('location_updated_at', sa.DateTime(), nullable=True),
    sa.Column('location_accuracy', sa.Float(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=True),
    sa.Column('last_online', sa.DateTime(), nullable=True),
    sa.Column('cpu_model', sa.String(length=100), nullable=True),
    sa.Column('ram_total', sa.Float(), nullable=True),
    sa.Column('storage_total', sa.Float(), nullable=True),
    sa.Column('ip_address', sa.String(length=45), nullable=True),
    sa.Column('mac_address', sa.String(length=17), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('owner_id', sa.Integer(), nullable=True),
    sa.Column('health_score', sa.Float(), nullable=True),
    sa.Column('anomaly_probability', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name='fk_kiosk_owner', ondelete='SET NULL', use_alter=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('uuid')
    )
    op.create_table('model_metrics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('version', sa.String(length=50), nullable=False, comment='Identificador de la versión del modelo'),
    sa.Column('timestamp', sa.DateTime(), nullable=True, comment='Fecha y hora del registro de métricas'),
    sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True, comment='Métricas detalladas del modelo'),
    sa.Column('roc_auc', sa.Float(), nullable=True, comment='Área bajo la curva ROC'),
    sa.Column('pr_auc', sa.Float(), nullable=True, comment='Área bajo la curva Precisión-Recall'),
    sa.Column('confusion_matrix', postgresql.JSON(astext_type=sa.Text()), nullable=True, comment='Matriz de confusión del modelo'),
    sa.Column('class_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True, comment='Métricas detalladas por clase'),
    sa.Column('calibration_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True, comment='Métricas de calibración'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('prediction_logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('model_version', sa.String(length=50), nullable=False),
    sa.Column('features', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('predicted_value', sa.Float(), nullable=True),
    sa.Column('actual_value', sa.Float(), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('prediction_time', sa.Float(), nullable=True),
    sa.Column('extra_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=50), nullable=False),
    sa.Column('email', sa.String(length=120), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('role_name', sa.String(length=20), nullable=False),
    sa.Column('two_factor_enabled', sa.Boolean(), nullable=True),
    sa.Column('two_factor_secret', sa.String(length=32), nullable=True),
    sa.Column('backup_codes', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('temp_2fa_code', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('last_login', sa.DateTime(), nullable=True),
    sa.Column('failed_login_attempts', sa.Integer(), nullable=True),
    sa.Column('account_locked', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_table('sensor_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('kiosk_id', sa.Integer(), nullable=False),
    sa.Column('cpu_usage', sa.Float(), nullable=False),
    sa.Column('memory_usage', sa.Float(), nullable=False),
    sa.Column('network_latency', sa.Float(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['kiosk_id'], ['kiosks.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('sensor_data')
    op.drop_table('user')
    op.drop_table('prediction_logs')
    op.drop_table('model_metrics')
    op.drop_table('kiosks')
    op.drop_table('drift_metrics')
    # ### end Alembic commands ###