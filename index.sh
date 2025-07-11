#!/bin/bash

# Directory containing the resources
resource_dir=$1

# Output HTML file
output_file="$2/index.html"

# Start of the HTML file
echo "<html><head><title>Resource Index</title></head><body><h1>Links for nntool</h1>" > "$output_file"

# Loop through each file in the directory and add it to the HTML file
for file in "$resource_dir"/*; do
    filename=$(basename "$file")

    # Skip index.html file to avoid self-reference
    if [ "$filename" != "index.html" ]; then
        echo "<a href='$filename'>$filename</a><br>" >> "$output_file"
    fi
done

# End of the HTML file
echo "</body></html>" >> "$output_file"

echo "Index page created as $output_file"
