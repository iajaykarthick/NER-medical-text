#!/bin/bash

# Check if the input directory exists
if [ ! -d "$1" ]; then
  echo "Error: Directory not found!"
  exit 1
fi

# Loop through each .bio file in the input directory
for file in "$1"/*.bio; do
  # Check if the file exists
  if [ ! -f "$file" ]; then
    echo "Error: File not found!"
    exit 1
  fi

  # Loop through each line in the file
  prev_tag=""
  while read -r line; do

    # Skip empty lines
    if [[ "$line" =~ ^[[:space:]]*$ ]]; then
      continue
    fi

    # Split the line into columns
    columns=($line)

    # Check if the line contains exactly two columns
    if [ "${#columns[@]}" -ne 2 ]; then
      echo "Error: Invalid number of columns on line: $line in file $file"
      continue
    fi

    # Check if the second column contains a valid BIO tag
    if [[ ! "${columns[1]}" =~ ^(B|I|O)(-[A-Za-z_]+)?$ ]]; then
      echo "Error: Invalid BIO tag on line: $line in file $file"
      continue
    fi

    # Check if the tag sequence is valid
    if [ "${columns[1]:0:1}" == "I" ]; then
      if [ "${prev_tag}" == "" ]; then
        echo "Error: Invalid tag sequence on line: $line in file $file"
        continue
      fi
      if [ "${columns[1]:2}" != "${prev_tag}" ]; then
        echo "Error: Invalid tag sequence on line: $line in file $file"
        continue
      fi
    else
      prev_tag="${columns[1]:2}"
    fi

  done < "$file"

  echo "$file contains valid BIO format."
done

