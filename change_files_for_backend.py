import os

def add_import_statements_and_replace(root_dir):
    items = [
        "meshgrid",
    ]
    script_path = os.path.abspath(__file__)  # Get the path of this script
    for subdir, dirs, files in os.walk(root_dir):
        # Skip hidden directories and directories starting with _
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        for file in files:
            # Only process Python files
            if file.endswith('.py') and not file.startswith('.'):
                file_path = os.path.join(subdir, file)
                if file_path == script_path:
                    # Skip processing this script file
                    continue
                with open(file_path, 'r') as f:
                    file_content = f.read()
                updated_content = file_content
                # Check if any item is in the file content
                import_statements = []
                for item in items:
                    if f'np.{item}' in file_content:
                        import_statements.append(f'from pyrecest.backend import {item}')
                        updated_content = updated_content.replace(f'np.{item}', item)
                # Prepend import statements if not already present
                for import_statement in import_statements:
                    if import_statement not in updated_content:
                        updated_content = f'{import_statement}\n{updated_content}'
                # Write updated content back to file if any changes were made
                if updated_content != file_content:
                    with open(file_path, 'w') as f:
                        f.write(updated_content)
                        print(f'Updated {file_path}')

# Specify the directory to start the search from
root_directory = './pyrecest'
add_import_statements_and_replace(root_directory)
