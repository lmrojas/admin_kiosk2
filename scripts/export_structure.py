# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set

def should_ignore(path: str, ignore_patterns: Set[str] = None) -> bool:
    """
    Determina si un archivo o directorio debe ser ignorado.
    
    Args:
        path (str): Ruta del archivo o directorio
        ignore_patterns (set): Patrones a ignorar
    
    Returns:
        bool: True si debe ignorarse, False en caso contrario
    """
    if ignore_patterns is None:
        ignore_patterns = {
            '__pycache__', 
            '.git', 
            '.pytest_cache', 
            '.coverage',
            'venv',
            'env',
            '.env',
            '.vscode',
            '.idea',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '.DS_Store'
        }
    
    # Ignorar archivos y directorios ocultos
    if os.path.basename(path).startswith('.'):
        return True
    
    # Ignorar patrones específicos
    for pattern in ignore_patterns:
        if pattern in path:
            return True
    
    return False

def get_file_info(file_path: str) -> Dict[str, str]:
    """
    Obtiene información sobre un archivo.
    
    Args:
        file_path (str): Ruta del archivo
    
    Returns:
        dict: Información del archivo (tamaño, última modificación)
    """
    stats = os.stat(file_path)
    size = stats.st_size
    modified = datetime.fromtimestamp(stats.st_mtime)
    
    # Convertir tamaño a formato legible
    if size < 1024:
        size_str = f"{size}B"
    elif size < 1024 * 1024:
        size_str = f"{size/1024:.1f}KB"
    else:
        size_str = f"{size/(1024*1024):.1f}MB"
    
    # Contar líneas si es archivo de texto
    num_lines = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
    except:
        pass
    
    return {
        'size': size_str,
        'modified': modified.strftime("%b %d, %I:%M %p"),
        'lines': num_lines
    }

def scan_directory(root_dir: str, current_dir: str = '', level: int = 0) -> List[str]:
    """
    Escanea un directorio recursivamente y genera su estructura.
    
    Args:
        root_dir (str): Directorio raíz del proyecto
        current_dir (str): Directorio actual siendo escaneado
        level (int): Nivel de profundidad actual
    
    Returns:
        list: Lista de líneas describiendo la estructura
    """
    structure = []
    full_path = os.path.join(root_dir, current_dir)
    
    try:
        # Obtener lista de archivos y directorios
        items = os.listdir(full_path)
        items.sort()
        
        for item in items:
            item_path = os.path.join(full_path, item)
            relative_path = os.path.join(current_dir, item)
            
            # Ignorar items según patrones
            if should_ignore(item_path):
                continue
            
            # Formatear línea según tipo
            indent = "  " * level
            if os.path.isdir(item_path):
                structure.append(f"{indent}[dir]  {item}/ (? items) - {datetime.fromtimestamp(os.path.getmtime(item_path)).strftime('%b %d, %I:%M %p')}")
                structure.extend(scan_directory(root_dir, relative_path, level + 1))
            else:
                info = get_file_info(item_path)
                structure.append(f"{indent}[file] {item} ({info['size']}, {info['lines']} lines) - {info['modified']}")
    
    except Exception as e:
        logging.error(f"Error escaneando directorio {full_path}: {str(e)}")
    
    return structure

def generate_structure(root_dir: str) -> str:
    """
    Genera la estructura completa del proyecto.
    
    Args:
        root_dir (str): Directorio raíz del proyecto
    
    Returns:
        str: Estructura del proyecto formateada
    """
    header = f"""# Estructura del Proyecto
# Última actualización: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# ADVERTENCIA: Este archivo es generado automáticamente.
# NO MODIFICAR MANUALMENTE.
#
# Para actualizar, ejecutar:
# python scripts/export_structure.py
#
# Símbolos:
# [dir]  - Directorio
# [file] - Archivo
#
# Reglas:
# 1. Cualquier modificación a la estructura del proyecto debe:
#    a) Seguir las guías en @cura.md
#    b) Actualizar este archivo ejecutando el script
# 2. Los archivos ignorados no aparecen en esta estructura
# 3. Las rutas son relativas al directorio raíz del proyecto
#
"""
    
    structure = scan_directory(root_dir)
    return header + "\n".join(structure)

def export_structure(root_dir: str, output_file: str = 'project_custom_structure.txt'):
    """
    Exporta la estructura del proyecto a un archivo.
    
    Args:
        root_dir (str): Directorio raíz del proyecto
        output_file (str): Archivo de salida
    """
    try:
        structure = generate_structure(root_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(structure)
        
        logging.info(f"Estructura exportada exitosamente a {output_file}")
        
    except Exception as e:
        logging.error(f"Error exportando estructura: {str(e)}")
        raise

def main():
    """Función principal para exportar la estructura"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Obtener directorio raíz (directorio padre del directorio scripts)
        root_dir = str(Path(__file__).resolve().parent.parent)
        
        # Exportar estructura
        export_structure(root_dir)
        
    except Exception as e:
        logging.error(f"Error en script: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 