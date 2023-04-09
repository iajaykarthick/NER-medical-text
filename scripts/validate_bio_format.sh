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
    continue
  fi

  # Loop through each line in the file
  prev_tag=""
  while IFS=$'\t' read -r word tag; do
    # Skip empty lines
    if [[ "$word" =~ ^[[:space:]]*$ ]]; then
      continue
    fi

    # Check if the second column contains a valid BIO tag
    if [[ ! "$tag" =~ ^(B|I|O)-[A-Za-z_]+$ ]]; then
      echo "Error: Invalid BIO tag on line: $word\t$tag in file $file"
    fi

    # Check if the tag sequence is valid
    if [ "${tag:0:1}" == "I" ]; then
      if [ "$prev_tag" == "" ]; then
        echo "Error: Invalid tag sequence on line: $word\t$tag in file $file"
      fi
      if [ "${tag:2}" != "$prev_tag" ]; then
        echo "Error: Invalid tag sequence on line: $word\t$tag in file $file"
      fi
    else
      prev_tag="${tag:2}"
    fi

  done < "$file"

  echo "$file contains valid BIO format."
done

