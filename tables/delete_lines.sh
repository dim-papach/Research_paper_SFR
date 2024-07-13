#!/bin/bash

# Usage: ./delete_lines.sh file1 file2 ...

# Strings to be deleted
strings_to_delete=("MKT J045920.2-252959" "MKT J125225.4-124304." "6dF J2218489-461303")

# Function to delete lines containing specific strings
delete_lines() {
  local file=$1
  local tmp_file=$(mktemp)
  
  # Loop through each string and delete lines containing it
  cp "$file" "$tmp_file"
  for str in "${strings_to_delete[@]}"; do
    sed -i "/$str/d" "$tmp_file"
  done
  
  # Replace original file with the modified one
  mv "$tmp_file" "$file"
}

# Iterate through all files provided as arguments
for file in "$@"; do
  if [ -f "$file" ]; then
    delete_lines "$file"
    echo "Processed $file"
  else
    echo "File $file not found"
  fi
done