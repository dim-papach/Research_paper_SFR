#!/bin/bash

# Define file paths
FILE_PATH="outer_join.ecsv"
FILE_PATH_INNER="inner_join.ecsv"
TEMP_FILE="temp.ecsv"
touch $TEMP_FILE
CMD_FILE="commands.txt"  # Temporary command file for STILTS
touch $CMD_FILE
# Define mass threshold
MASS_THRESHOLD=9

# Write the STILTS commands to a temporary file
cat > "$CMD_FILE" << EOL
addcol mass_type \
    "if((strlen(Tdw1) > 0) or (strlen(Tdw2) > 0), 'Dwarf', \
    if(logM_HEC < $MASS_THRESHOLD, 'Dwarf', \
    if(logM_HEC >= $MASS_THRESHOLD, 'Massive', 'Undefined')))"


addcol mass_classification_method \
    "if((Tdw1 != '') or (Tdw2 != ''), 'Tdw', \
    if(logM_HEC < $MASS_THRESHOLD, 'Mass', \
    if(logM_HEC >= $MASS_THRESHOLD, 'Mass', '')))"
EOL

# Process the table using the commands in CMD_FILE
topcat -stilts tpipe \
    in="$FILE_PATH" \
    cmd=@"$CMD_FILE" \
    out="$TEMP_FILE" ofmt=ecsv

# Remove the temporary command file
rm "$CMD_FILE"

# Replace the original file with the processed file
mv "$TEMP_FILE" "$FILE_PATH"

# Copy the processed file to inner_join.ecsv
cp "$FILE_PATH" "$FILE_PATH_INNER"
