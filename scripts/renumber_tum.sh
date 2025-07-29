#!/bin/bash

# This script renumbers the timestamps in a TUM file to be zero-based integer indices.
# It preserves the header and creates a new file with the suffix "_renumbered".

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_tum_file>"
    exit 1
fi

input_file="$1"

# Create a new filename for the output
if [[ "$input_file" == *.tum ]]; then
    output_file="${input_file%.tum}_renumbered.tum"
else
    output_file="${input_file}_renumbered"
fi


if [ ! -f "$input_file" ]; then
    echo "Error: File not found at $input_file"
    exit 1
fi

echo "Renumbering timestamps in $input_file"
echo "Output will be saved to $output_file"

# Use awk to process the file:
# - For lines starting with '#', print them as-is.
# - For all other lines, replace the first column with a counter `c` that starts at 0.
awk 'BEGIN { c=0; } /^#/ { print; next; } { $1=c++; print; }' "$input_file" > "$output_file"

echo "Done. Renumbered file saved to $output_file" 