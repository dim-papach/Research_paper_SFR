#!/usr/bin/env python3


import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table, join, QTable
import glob
import os

# Change directory to where the notes are located
file_path = "./tables/outer_join.ecsv"
file_path_inner = "./tables/inner_join.ecsv"
# Get data
data = QTable(ascii.read(file_path))

# Define the mass and luminosity thresholds
mass_threshold = 9  # Log(Mass) threshold for dwarf vs massive


# Classify galaxies based on logM_HEC, logKLum, Tdw1, and Tdw2 with method flags
def classify_galaxy(row):
    # Initialize the classification and method flag
    classification = "Undefined"
    method = ""

    # Check Tdw1 and Tdw2 columns for morphological classification
    if not row["Tdw1"] is None or not row["Tdw2"] is None:
        classification = "Dwarf"
        method = "Tdw"  # Classification based on morphology (Tdw1 or Tdw2)
    # Check mass classification
    elif row["logM_HEC"] < mass_threshold:
        classification = "Dwarf"
        method = "Mass"  # Classification based on mass
    elif row["logM_HEC"] >= mass_threshold:
        classification = "Massive"
        method = "Mass"

    return classification, method


# Apply classification to each row and add new columns to QTable
classifications, methods = [], []

for row in data:
    classification, method = classify_galaxy(row)
    classifications.append(classification)
    methods.append(method)

# Add the new classification and method columns to the QTable
data["mass_type"] = classifications
data["mass_type"].description = "Mass classification, Dwarf, Massive, Undefined"
data["mass_classification_method"] = methods

ascii.write(
    data, "temp_join_class.csv", format="csv", overwrite=True, fast_writer=False
)
