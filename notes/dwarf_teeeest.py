#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, QTable, MaskedColumn
import astropy.units as u
from astropy.visualization import quantity_support
import glob
import os


# Change directory to where the notes are located
os.chdir("/home/dp/Documents/Research_paper_SFR/tables")
# open outer_join.ecsv
data = QTable.read("outer_join.ecsv")

from astropy.table import QTable, MaskedColumn
import numpy as np


def recreate_masked_table(data, missing_value=None):
    """
    Convert a regular QTable into a masked QTable by detecting missing values.

    Parameters:
    - data (QTable): Input QTable without masks.
    - missing_value: Value to treat as "missing" for non-numeric columns (e.g., None, '').

    Returns:
    - QTable: A new masked QTable.
    """
    masked_data = QTable()

    for col_name in data.colnames:
        col = data[col_name]
        if col.dtype.kind in "if":  # Numeric types (float or int)
            mask = np.isnan(col)  # Detect NaN for numeric columns
        else:  # Non-numeric columns (e.g., strings)
            mask = [val == missing_value for val in col]  # Detect missing_value
        # Create a MaskedColumn with the mask
        masked_data[col_name] = MaskedColumn(col, mask=mask)

    return masked_data


data = recreate_masked_table(data, missing_value=None)

from astropy.table import QTable


def classify_galaxy_astropy(row):
    # Initialize the classification and method flag
    classification = "undefined"
    method = ""

    # Handle masking and string checks for Tdw1 and Tdw2
    Tdw1_valid = not data["Tdw1"].mask[row.index] and row["Tdw1"] != ""
    Tdw2_valid = not data["Tdw2"].mask[row.index] and row["Tdw2"] != ""

    # Handle masking for logM_HEC
    logM_HEC_valid = not data["logM_HEC"].mask[row.index]

    # Morphological classification
    if Tdw1_valid or Tdw2_valid:
        classification = "dwarf"
        method = "T"  # Classification based on morphology
    # Mass-based classification
    elif logM_HEC_valid:
        if row["logM_HEC"] < mass_threshold:
            classification = "dwarf"
            method = "M"
        else:
            classification = "massive"
            method = "M"

    return classification, method


# Apply the function to all rows
classifications = [classify_galaxy_astropy(row) for row in data]

# Add the new columns to the QTable
mass_type, classification_method = zip(*classifications)
data["mass_type"] = mass_type
data["classification_method"] = classification_method
# print data["mass_type"] == massive

print(data.info())
ascii.write(data, "mass_types_outer_test.ecsv", format="ecsv", overwrite=True)
