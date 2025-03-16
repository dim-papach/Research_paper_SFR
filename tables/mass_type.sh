#!/bin/bash

# Define file paths
FILE_PATH="outer_join.ecsv"
FILE_PATH_INNER="inner_join.ecsv"
TEMP_FILE="temp.ecsv"
CMD_FILE="commands.txt"  # Temporary command file for STILTS

# Define mass threshold
MASS_THRESHOLD=9

# Write the STILTS commands to a temporary file
cat > "$CMD_FILE" << 'EOL'
addcol mass_type '(!isBlank(Tdw1) || !isBlank(Tdw2)) ? "Dwarf" :
                  ( (logM_HEC != null) ?
                    ( (logM_HEC < 9) ? "Dwarf" : "Massive" )
                    : "Undefined" )'
addcol mass_classification_method '(!isBlank(Tdw1) || !isBlank(Tdw2)) ? "Tdw" :
                                  ( (logM_HEC != null) ? "Mass" : "" )'
EOL

# Process the table using the commands in CMD_FILE
stilts tpipe \
    in="$FILE_PATH" \
    cmd=@"$CMD_FILE" \
    out="$TEMP_FILE" ofmt=ecsv

# Remove the temporary command file
rm "$CMD_FILE"

# Replace the original file with the processed file
mv "$TEMP_FILE" "$FILE_PATH"

# Copy the processed file to inner_join.ecsv
cp "$FILE_PATH" "$FILE_PATH_INNER"

