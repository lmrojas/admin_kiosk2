# Documentación de Mantenimiento - Admin Kiosk
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

## 1. Mantenimiento del Sistema

### 1.1 Requisitos de Sistema
- Python 3.9+
- Node.js 16+
- PostgreSQL 13+
- Redis 6+
- Docker 20+

### 1.2 Entornos
- Desarrollo: `dev.admin-kiosk.com`
- Staging: `staging.admin-kiosk.com`
- Producción: `admin-kiosk.com`

## 2. Procedimientos de Despliegue

### 2.1 Despliegue en Producción
```bash
# 1. Preparación
git checkout main
git pull origin main

# 2. Tests
pytest
npm run test

# 3. Build
docker build -t admin-kiosk:latest .

# 4. Despliegue
./scripts/deploy.sh production
```

### 2.2 Rollback
```bash
# Revertir a versión anterior
./scripts/rollback.sh production <version>
```

## 3. Monitoreo y Logs

### 3.1 Logs del Sistema
- Aplicación: `/var/log/admin-kiosk/app.log`
- Nginx: `/var/log/nginx/admin-kiosk.log`
- PostgreSQL: `/var/log/postgresql/postgresql.log`

### 3.2 Monitoreo
- Prometheus: `prometheus.admin-kiosk.com`
- Grafana: `grafana.admin-kiosk.com`
- AlertManager: `alerts.admin-kiosk.com`

## 4. Backup y Recuperación

### 4.1 Backup Automático
```bash
# Backup diario
0 2 * * * /scripts/backup.sh daily

# Backup semanal
0 3 * * 0 /scripts/backup.sh weekly
```

### 4.2 Restauración
```bash
# Restaurar desde backup
./scripts/restore.sh <backup_file>
```

## 5. Mantenimiento de Base de Datos

### 5.1 Migraciones
```bash
# Crear migración
python manage.py makemigrations

# Aplicar migraciones
python manage.py migrate
```

### 5.2 Optimización
```bash
# Vacuum
VACUUM ANALYZE;

# Reindex
REINDEX DATABASE admin_kiosk;
```

## 6. Seguridad

### 6.1 Actualizaciones de Seguridad
```bash
# Actualizar dependencias
pip install --upgrade -r requirements.txt
npm audit fix
```

### 6.2 Certificados SSL
```bash
# Renovar certificados
certbot renew
```

## 7. Troubleshooting

### 7.1 Problemas Comunes

#### Error de Conexión a BD
```bash
# Verificar servicio
systemctl status postgresql

# Verificar logs
tail -f /var/log/postgresql/postgresql.log
```

#### Problemas de Caché
```bash
# Limpiar caché
redis-cli FLUSHALL

# Verificar estado
redis-cli INFO
```

### 7.2 Monitoreo de Rendimiento
```bash
# CPU y Memoria
top -b -n 1

# Disco
df -h

# IO
iostat -x 1
```

## 8. Mantenimiento de IA

### 8.1 Reentrenamiento de Modelos
```bash
# Entrenamiento programado
python manage.py train_models

# Validación
python manage.py validate_models
```

### 8.2 Monitoreo de Drift
```bash
# Verificar drift
python manage.py check_drift

# Actualizar baseline
python manage.py update_baseline
```

## 9. Gestión de Configuración

### 9.1 Variables de Entorno
```bash
# Producción
source /etc/admin-kiosk/prod.env

# Staging
source /etc/admin-kiosk/staging.env
```

### 9.2 Configuración de Servicios
- Nginx: `/etc/nginx/sites-available/admin-kiosk`
- Supervisor: `/etc/supervisor/conf.d/admin-kiosk.conf`
- PostgreSQL: `/etc/postgresql/13/main/postgresql.conf`

## 10. Procedimientos de Emergencia

### 10.1 Failover
```bash
# Activar sitio de contingencia
./scripts/failover.sh activate

# Revertir a sitio principal
./scripts/failover.sh revert
```

### 10.2 Recuperación de Desastres
1. Activar plan DR
2. Restaurar último backup
3. Verificar integridad
4. Actualizar DNS

## 11. Contactos

### 11.1 Equipo de Soporte
- Nivel 1: support@admin-kiosk.com
- Nivel 2: tech@admin-kiosk.com
- Emergencias: +1-800-KIOSK-911

### 11.2 Proveedores
- AWS: account-team@aws.com
- CDN: support@cloudflare.com
- SSL: support@letsencrypt.org 