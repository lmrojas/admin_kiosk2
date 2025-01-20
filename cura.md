Versión **actualizada** de `cura.md` que **incluye** las reglas de manejo de **git**, **push**, **backups** y la administración de **cambios en la base de datos**. Puedes **reemplazar** o **sobrescribir** tu `cura.md` existente con este contenido para que todos los desarrolladores (humanos e IA) lo sigan.

---

# CURA.md

## 0. Introducción

El presente archivo (`cura.md`) es la **fuente de verdad** para todos los lineamientos y reglas a seguir en nuestro sistema. **Ninguna modificación** o **adición** de código puede hacerse **sin revisarlo primero**. Este documento:

1. **Establece** el patrón de diseño obligatorio: **MVT + Services**.  
2. **Dicta** la forma de trabajo (uso de `venv`, no duplicaciones, scripts de testing, backups, etc.).  
3. **Indica** los procedimientos para **crear**, **modificar** o **borrar** archivos, métodos, clases o servicios.  
4. **Exige** la documentación y actualización continua (`project_custom_structure.txt`) para no perder la vista global del proyecto.  
5. **Describe** la manera de manejar errores, refactors, pruebas y despliegues.
6. **Regula** cómo versionar el código con **Git** y mantener **backups** (push en host remoto, snapshots de BD, etc.).

> **ADVERTENCIA**: **No** se permite **bajo ninguna circunstancia** saltarse estas normas. Cualquier violación a `cura.md` puede generar inconsistencias graves en el proyecto.

---

## 1. Trabajar Siempre en `venv`

1. **Uso Obligatorio** de entorno virtual:
   - Está **prohibido** instalar o modificar paquetes (o librerías) fuera del `venv`.
   - El archivo `requirements.txt` se **actualiza** solo tras haber validado las instalaciones en el `venv`.

2. **Control de Dependencias**:
   - Documentar cada **nueva** dependencia (en `requirements.txt` o en un subdocumento de instalación).
   - Verificar que no existan **paquetes duplicados** o en conflicto.

3. **Objetivo**:
   - Evitar conflictos con librerías globales del sistema.
   - Mantener un registro **exacto** y **repetible** de los paquetes utilizados.

---

## 2. Patrón MVT + Services (Model-View-Template + Capa de Servicios)

1. **Separación Obligatoria de Lógica**:
   - **Models**: Contienen la definición de **tablas y esquemas** de la base de datos (SQLAlchemy). **No** deben contener lógica de negocio.
   - **Views / Blueprints**: Son rutas Flask que **solo** manejan la capa de presentación y control. No deben llevar lógica de negocio (más allá de lo estrictamente necesario para enrutamiento).
   - **Templates**: Son archivos `.html`/Jinja2 que **solo** contienen la parte visual (layout, HTML, CSS, etc.). Nada de lógica pesada.
   - **Services**: Módulos o clases dedicadas a la **lógica de negocio** (p. ej., encriptación de contraseñas, cálculos, validaciones, interacciones con APIs externas).  
     > Toda lógica compleja **debe** ir en “Services” en lugar de duplicarla en varios archivos.

2. **En Resumen**:
   - **No** meter lógica de negocio en los **Modelos** ni en las **Vistas**.
   - **No** duplicar funciones existentes en los **Services**.

---

## 3. Lectura Obligatoria de `project_custom_structure.txt` Antes de Cualquier Cambio

1. **Fechas y Versionado**:
   - Antes de **crear**, **modificar** o **borrar** un archivo, clase o método, verifica la **fecha de última actualización** de `project_custom_structure.txt`.
   - Si detectas que está desactualizado, **solicita** una actualización al responsable o confirma que la estructura se mantiene.

2. **Contexto Central**:
   - El archivo `project_custom_structure.txt` es la **“fuente de verdad”** sobre la organización actual del proyecto.
   - Allí se documenta **dónde** están cada modelo, vista, servicio, script o template.

3. **Evitar Duplicaciones**:
   - Revisar que **no** exista ya un archivo o método parecido.
   - Mantener la estructura original o adaptarla con criterios claros y aprobados (evitando colisiones).

---

## 4. Procedimiento para Crear/Modificar/Borrar Archivos o Funciones

1. **Crear**:  
   1. **Paso 1**: Leer `cura.md` y `project_custom_structure.txt` para encontrar la ubicación correcta (modelo, servicio, vista, etc.).  
   2. **Paso 2**: Crear el archivo/clase/método con el **comentario obligatorio** al inicio (ver sección 10).  
   3. **Paso 3**: Registrar (o actualizar) **importaciones** y **rutas** (si aplica) en el blueprint correspondiente.  
   4. **Paso 4**: Inmediatamente **ejecutar** los scripts de testing (ej. `pytest`) para asegurar compatibilidad.

2. **Modificar**:  
   1. **Paso 1**: Chequear la última versión en `project_custom_structure.txt`.  
   2. **Paso 2**: Actualizar **todas** las referencias (imports, llamadas en vistas o servicios).  
   3. **Paso 3**: Corregir/añadir **tests** que involucren la funcionalidad cambiada.  
   4. **Paso 4**: Documentar brevemente en el commit de qué se trató la modificación.

3. **Borrar**:  
   1. **Paso 1**: Verificar que no existan dependencias actuales.  
   2. **Paso 2**: Eliminar o marcar como *deprecated* la clase/método en el repo.  
   3. **Paso 3**: Actualizar `project_custom_structure.txt`.  
   4. **Paso 4**: Pasar los scripts de testing para detectar fallos por referencias rotas.

---

## 5. Scripts de Testing y Cobertura

1. **Ubicación y Estructura**:
   - Cada test debe residir en la carpeta que le corresponda (`tests/unit/`, `tests/integration/`, `tests/e2e/`, etc.).
   - Nombrar archivos y métodos de test de forma clara (`test_auth_service.py`, `test_kiosk_model.py`, etc.).

2. **Ejecución Solo en venv**:
   - Está prohibido correr `pytest`, `coverage` o similares fuera del `venv`.
   - Documentar en un `README.md` dentro de `tests/` los pasos para correr las pruebas.

3. **Cobertura y Revisión**:
   - Procurar una **alta cobertura** (lo más cercana posible a 100%).
   - Revisar que **no** haya duplicaciones de tests.

---

## 6. Actualización Continua (API, WebSocket, Documentación)

1. **API y Sistema Mayor**:
   - Cada vez que cambie la estructura de un endpoint (REST/gRPC), **documentar** ese cambio y **actualizar** la referencia en `project_custom_structure.txt`.
   - Llevar un **histórico** de endpoints, con su fecha de versión.

2. **Comunicación WebSocket**:
   - Antes de modificar la **estructura** de datos enviada por WebSocket, asegurarse de que el **sistema mayor** (o microservicios) conozca el nuevo formato.
   - Actualizar la documentación y/o el diagrama de flujo de la comunicación.

3. **Documentación**:
   - Todo cambio significativo debe reflejarse en la documentación interna o en `docs/`.
   - Mantener `project_custom_structure.txt` **siempre sincronizado**.

---

## 7. Manejo de Errores, Excepciones y Logs

1. **Errores en Desarrollo**:
   - Agregar logs claros y detallados (nivel DEBUG o INFO) para permitir un diagnóstico rápido.
   - Evitar exponer información sensible en pantallas de error.

2. **Errores en Producción**:
   - Mantener logs (nivel ERROR o WARNING) y notificar al equipo responsable.
   - Seguir el **principio de mínimo privilegio** para no revelar stack traces completos al usuario final.

3. **Manejo de Excepciones**:
   - Manejar excepciones específicas (p. ej., `ValueError`, `IntegrityError`) en lugar de hacer `catch all`.
   - Centralizar la lógica de retomar errores en un **Service** o en un **middleware** (si corresponde).

---

## 8. Refactor y Correcciones Mayores

1. **Plan de Refactor**:
   - Antes de un refactor extenso, crear un **plan** (puede ser un gist, una doc breve) que describa qué se va a hacer y por qué.
   - Asegurarse de **no duplicar** archivos de modelos (ej. `models.py` y `models/`) o servicios.

2. **Cambios de Estructura**:
   - Consultar `project_custom_structure.txt` y actualizarlo con un commit separado, indicando **fecha** de actualización.
   - Confirmar que la IA no proponga soluciones que contradigan `cura.md`.

3. **Verificación**:
   - Correr todos los tests (unitarios, de integración, e2e).
   - Revisar logs de errors, warnings y coverage para asegurar que el refactor no rompió nada.

---

## 9. Roles de la IA y de Desarrolladores Humanos

1. **IA (Claude-3.5-haiku o Similar)**:
   - Antes de **crear**, **modificar** o **borrar** código, la IA debe:
     1. Leer `cura.md`.
     2. Revisar `project_custom_structure.txt`.
     3. Verificar que no haya duplicaciones, imports rotos ni conflictos con la estructura MVT + Services.  
   - Incluir siempre el **comentario obligatorio** al inicio de cada archivo que cree o modifique.

2. **Desarrolladores Humanos**:
   - Verificar que la IA cumpla las reglas.
   - Revisar commits y logs de la IA (o de cualquier otra fuente).
   - Ejecutar scripts de testing localmente y confirmar la **consistencia**.

3. **Responsabilidad**:
   - Tanto la IA como los humanos son responsables de mantener la integridad del proyecto.
   - Si se detecta una violación de `cura.md`, se debe **retroceder** el cambio o arreglarlo inmediatamente.

---

## 10. Comentario Obligatorio en Cada Archivo

En **cada archivo** (código Python, plantillas HTML, JS, etc.) al **inicio**:
```
# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'
```
> Esto asegura que **cualquiera** que abra el archivo sepa de las **reglas** y **procedimientos** a seguir.

---

## 11. Repositorio Git y Control de Versiones

1. **Repositorio Remoto**:
   - Repositorio oficial: https://github.com/lmrojas/admin_kiosk2
   - **Obligatorio** mantener sincronizado el repositorio local con el remoto
   - Usar HTTPS para clonar y hacer push

2. **Flujo de Ramas (Branches)**:
   - Se recomienda usar un esquema como `main` (o `master`) para producción y ramas de características (`feature/xxx`) para nuevas funcionalidades.
   - Antes de mezclar (`merge`) a `main`, debe haber:
     1. Revisión de código (pull request).
     2. Todos los tests en **verde**.

3. **Commits Claros**:
   - Cada commit debe llevar un mensaje que describa **qué** se cambió y **por qué**.
   - Evitar commits “en blanco” o con mensajes triviales.

4. **Push y Backup**:
   - **Realizar** push a un **repositorio remoto** (GitHub, GitLab, Bitbucket, o el que se use) con frecuencia, para no perder cambios.
   - Mantener un **backup** (por ejemplo, un mirror de Git) en caso de que el repositorio principal falle.
   - Antes de `git push`, asegurarse de haber corrido los tests y verificado que no haya duplicaciones o rompimientos.

5. **Resolución de Conflictos**:
   - Siempre revisar `project_custom_structure.txt` antes de resolver un conflicto, para no crear duplicados.
   - En caso de duda, **preguntar** o verificar en la rama principal cómo se organizó el código.

---

## 12. Migraciones y Cambios en la Base de Datos

1. **Herramienta de Migración**:
   - Usar preferentemente [Flask-Migrate](https://flask-migrate.readthedocs.io/) o Alembic para generar y aplicar migraciones.
   - Cada vez que se modifiquen los modelos, crear una migración (p. ej., `flask db migrate -m "Descripción del cambio"`).

2. **Pruebas de Migración**:
   - Antes de aplicar la migración en un entorno de staging/producción, **probar** en local (con `venv`) para confirmar que no rompa la estructura.
   - Si se detectan errores, corregirlos **antes** de subir la migración definitiva.

3. **Respaldo (Backup) de BD**:
   - Antes de cualquier cambio mayor en la base de datos (columna nueva, alter table, drop table, etc.), generar un **backup** (p. ej. `pg_dump` si se usa PostgreSQL).
   - Conservar un histórico de backups (idealmente, versionados por fecha).
   - Si el cambio falla, restaurar la base anterior y documentar el incidente.

4. **Sincronía con `project_custom_structure.txt`**:
   - Actualizar la sección de modelos en `project_custom_structure.txt` para reflejar las nuevas tablas o columnas.
   - Incluir la fecha y razón del cambio (p. ej. “Se agregó campo `last_login` en el modelo `User`”).

---

## 13. Análisis, Retroalimentación y Mejoras Continuas

1. **Revisión Constante**:
   - Periódicamente, revisar este `cura.md` para actualizarlo si el proyecto evoluciona a nuevas tecnologías o metodologías.
   - **Fecha de Última Actualización**: _(mantener aquí un timestamp cada vez que se cambie algo importante)._  

2. **Procesos de Mejora**:
   - Si un procedimiento resulta ineficiente o genera conflictos, discutirlo en el equipo y proponer mejoras (manteniendo la esencia de MVT + Services y evitando duplicaciones de código).
   - La innovación y la evolución del proyecto son bienvenidas, pero **sin** romper estas reglas de base.

3. **Objetivo Final**:
   - Garantizar que el desarrollo sea **coherente**, **estable**, **seguro** y **escalable**.
   - Fomentar una cultura de **orden**, **buena documentación** y **comunicación transparente**.

---

## 14. Cierre

`cura.md` es un **documento vivo** que establece el marco de trabajo. **Todos** los participantes en el proyecto (humanos e IA) deben **acatarlo** y **mantenerlo** al día. De este modo, aseguramos:

- Un proyecto sin duplicaciones de código.
- Mantenimiento de la estructura MVT + Services.
- Uso correcto de scripts de testing y exportación de estructura.
- Procesos de Git y backups **claros**, sin riesgos de pérdida de datos.
- Iteraciones continuas de forma ordenada y segura.

**Cumplirlo es obligatorio**.  
En caso de duda, se prioriza siempre lo indicado aquí y en `project_custom_structure.txt`.  

> **Recuerda**: *La calidad y la organización en el desarrollo no son metas finales, sino un camino de mejora continua.*  

¡Listo! Con estas reglas, estaremos **100% alineados** y evitaremos conflictos en el proyecto, tanto a nivel de **código** como de **control de versiones** y **base de datos**.