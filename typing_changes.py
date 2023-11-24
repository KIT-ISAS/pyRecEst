import re
import os
from typing import Union

def modify_code(directory_path):
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                #print('Checking ' + file_path)
                temp_file_path = os.path.join(dirpath, 'temp.py')
                with open(file_path, 'r') as file:
                    code = file.read()

                    # Check if the specific type hint exists in the code
                    found = re.search(r'int\s*\|\s*int32\s*\|\s*int64', code)
                    if found:
                        print(f'Found in {file_path}')
                        # Replace type hint
                        code = re.sub(
                            r'int\s*\|\s*int32\s*\|\s*int64',
                            'Union[int, int32, int64]',
                            code
                        )

                        # Ensure Union is imported
                        if 'from typing import Union' not in code:
                            # Find the first import statement or the beginning of the file
                            # Find the first standalone import statement or the beginning of the file
                            first_import = re.search(r'^\s*(import [\w]+|from [\w\s\d.]+ import)', code, re.MULTILINE)
                            if first_import:
                                insert_pos = first_import.start()
                            else:
                                insert_pos = 0
                            code = code[:insert_pos] + 'from typing import Union\n' + code[insert_pos:]


                    # Remove '@beartype' lines
                    lines = code.splitlines()
                    modified_lines = [line for line in lines if line.strip() != '@beartype']

                # Write to temp file
                with open(temp_file_path, 'w') as temp_file:
                    temp_file.write('\n'.join(modified_lines))

                # Replace the original file with the modified temporary file
                os.replace(temp_file_path, file_path)

# Usage
modify_code('./pyrecest')
