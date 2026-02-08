@python -x "%~f0" %* & exit /b

import os
import sys

# --- Inicio del Script de Python ---

def main():
    """
    Función principal para generar el archivo markdown.
    """
    try:
        # Asegura que el script se ejecute en su propio directorio (para doble clic)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(script_dir)

        # --- CONFIGURACIÓN: Carpetas a excluir ---
        # Modifica esta lista con las carpetas que quieres excluir
        EXCLUDED_DIRS = [
            'bin',
            'obj', 
            'Debug',
            'Release',
            '.vs',
            '.git',
            'packages',
            'node_modules',
            '__pycache__'
        ]
        
        # También puedes excluir por patrones si lo prefieres
        EXCLUDED_PATTERNS = [
            '.vscode',
            '.idea'
        ]

        output_filename = 'Project_structure.md'
        print(f"Generando '{output_filename}' con el contenido de los archivos .py...")
        print(f"Carpetas excluidas: {EXCLUDED_DIRS}")

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for root, dirs, files in os.walk('.'):
                # Eliminar carpetas excluidas de la lista 'dirs' para que os.walk no las procese
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not any(pattern in d for pattern in EXCLUDED_PATTERNS)]
                
                # Verificar si el directorio actual está en la ruta excluida
                current_dir_components = root.split(os.sep)
                if any(excluded_dir in current_dir_components for excluded_dir in EXCLUDED_DIRS):
                    continue
                
                if any(any(pattern in component for component in current_dir_components) for pattern in EXCLUDED_PATTERNS):
                    continue

                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file).replace('\\', '/')
                        
                        outfile.write(f"## `{full_path}`\n\n")
                        
                        try:
                            with open(full_path, 'r', encoding='utf-8') as infile:
                                content = infile.read()
                            
                            outfile.write("```py\n")
                            outfile.write(content)
                            outfile.write("\n```\n\n")
                            
                        except Exception as e:
                            outfile.write(f"```\nError al leer el archivo: {e}\n```\n\n")

        print(f"'{output_filename}' ha sido actualizado exitosamente.")
        print("Proceso completado.")

    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()