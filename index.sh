#!/bin/bash

# Directory containing the resources
resource_dir=$1

# Output HTML file
output_file="dist/index.html"

# Start of the HTML file
echo "<html><head><title>Resource Index</title></head><body><h1>Links for nntool</h1><ul>" > "$output_file"

# Loop through each file in the directory and add it to the HTML file
for file in "$resource_dir"/*; do
    filename=$(basename "$file")
    echo "<li><a href='$filename'>$filename</a></li>" >> "$output_file"
done

# End of the HTML file
echo "</ul></body></html>" >> "$output_file"

echo "Index page created as $output_file"
