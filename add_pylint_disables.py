import os
import re

def add_pylint_disable(filename):
    with open(filename, 'r') as file:
        content = file.read()

    def replacer(match):
        import_statement = match.group(1)
        if '# pylint: disable=redefined-builtin' not in import_statement:
            return f'# pylint: disable=redefined-builtin\n{import_statement}'
        return import_statement

    pattern_single = re.compile(r'(from pyrecest\.backend import.*\b(?:all|any|sum)\b.*)')
    pattern_multi = re.compile(r'(from pyrecest\.backend import \([\s\S]*?\b(?:all|any|sum)\b[\s\S]*?\))', re.MULTILINE)

    content = re.sub(pattern_single, replacer, content)
    content = re.sub(pattern_multi, replacer, content)

    # Ensure there's a newline at the end of the file
    content = content.rstrip('\n') + '\n'

    with open(filename, 'w') as file:
        file.write(content)


def add_import_statements_and_replace(root_dir):
    script_path = os.path.abspath(__file__)  # Get the path of this script
    for subdir, dirs, files in os.walk(root_dir):
        # Skip hidden directories and directories starting with _
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        for file in files:
            if file.endswith('.py') and not file == os.path.basename(script_path):
                file_path = os.path.join(subdir, file)
                add_pylint_disable(file_path)


# Specify the directory to start the search from
root_directory = './pyrecest'
add_import_statements_and_replace(root_directory)
