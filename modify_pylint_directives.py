import os
import re

def modify_pylint_directives(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Step 1: Replace existing pylint disable comment
    content = re.sub(
        r'# pylint: disable=redefined-builtin',
        '# pylint: disable=redefined-builtin,no-name-in-module,no-member',
        content
    )

    # Step 2: Add new pylint disable comment if not already added
    def add_pylint_comment(match):
        if '# pylint: disable=no-name-in-module,no-member' not in content:
            return f'# pylint: disable=no-name-in-module,no-member\n{match.group()}'
        return match.group()

    pattern_import = re.compile(r'(from pyrecest\.backend import|import pyrecest\.backend)')
    content = re.sub(pattern_import, add_pylint_comment, content)

    # Ensure there's a newline at the end of the file
    content = content.rstrip('\n') + '\n'

    with open(filename, 'w') as file:
        file.write(content)

def process_files(root_dir):
    script_path = os.path.abspath(__file__)  # Get the path of this script
    for subdir, dirs, files in os.walk(root_dir):
        # Skip hidden directories and directories starting with _
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        for file in files:
            if file.endswith('.py') and not file == os.path.basename(script_path):
                file_path = os.path.join(subdir, file)
                modify_pylint_directives(file_path)

# Specify the directory to start the search from
root_directory = './pyrecest'
process_files(root_directory)
