#!/bin/bash

# Create the tmp folder if it doesn't exist
mkdir -p tmp

# Find all python files, and process them one by one
find . -type f -name "*.py" | while read -r file; do
    # Get the file path without the leading ./
    file_path="${file#./}"

    # Get the file name without the path
    file_name="$(basename "$file_path")"

    # Ensure unique names by appending a unique number if a file with the same name exists
    counter=1
    while [ -e "tmp/${file_name%.py}_$counter.py" ]; do
        counter=$((counter+1))
    done

    # Copy the file to the tmp folder with a unique name
    cp "$file_path" "tmp/${file_name%.py}_$counter.py"

    # Truncate the copied file to 200 lines
    head -n 200 "tmp/${file_name%.py}_$counter.py" > "tmp/${file_name%.py}_$counter.tmp"
    mv -f "tmp/${file_name%.py}_$counter.tmp" "tmp/${file_name%.py}_$counter.py"
done
