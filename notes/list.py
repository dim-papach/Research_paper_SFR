#!/bin/python3
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

# Set non-interactive backend for matplotlib
plt.switch_backend("Agg")


def clean_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        if line[19:21].strip().endswith("."):
            line = line[:19] + line[19:21].replace(".", "") + line[21:]
        cleaned_lines.append(line)

    cleaned_content = "".join(cleaned_lines)
    return cleaned_content


# Change directory to where the notes are located
os.chdir("/home/dp/Documents/Research_paper_SFR/notes")

# Find all matching files in the tables directory
file_pattern = "../tables/lvg_table*.dat"
file_paths = sorted(glob.glob(file_pattern))

# Create a list to store Tables
dq = []
for file_path in file_paths:
    cleaned_content = clean_file(file_path)
    print(f"Processing {file_path}")
    dq.append(Table(ascii.read(cleaned_content, format="mrt"), masked=True))

# Table names
nm = ["cat", "param", "magn", "vel", "kin", "dist", "sfr"]

# Associate table names with the tables
for name, table in zip(nm, dq):
    globals()[name] = table


# Function to create scatter plot and calculate R^2
def scatter_and_r2(table1, table2, col1_name, col2_name, title):
    # Join tables based on 'Name'
    temp = join(
        table1["Name", col1_name],
        table2["Name", col2_name],
        join_type="inner",
        keys="Name",
    )

    x = temp[temp.colnames[1]]
    y = temp[temp.colnames[2]]
    r2 = r2_score(x, y)

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, alpha=0.5)
    plt.xlabel(col1_name)
    plt.ylabel(col2_name)
    plt.title(f"{title}\n$R^2$ = {r2:.2f}")
    plt.grid(True)
    plt.savefig(title)
    plt.close()


### Modify magnitudes table
magn.remove_column("r_mag")
magn["Filter"][magn["Filter"] == "FU"] = "FUV"

filters = magn.group_by("Filter")
keys = filters.groups.keys["Filter"]
indices = filters.groups.indices

# Create color tables based on filters
dcolor = []
pain, n = 0, 0

for ind in indices[1:]:
    magn_table = filters[pain:ind]
    magn_table.rename_column("l_mag", f"l_mag_{keys[n]}")
    magn_table.rename_column("mag", f"mag_{keys[n]}")
    magn_table.rename_column("e_mag", f"e_mag_{keys[n]}")
    magn_table.remove_column("Filter")
    pain = ind
    n += 1
    dcolor.append(magn_table)

# Merge color tables
colors = Table(dcolor[0])
for dcolor_item in dcolor[1:]:
    colors = Table(join(colors, dcolor_item, keys="Name", join_type="outer"))

# Rename columns in param table
param.rename_column("a26", "A26")
param.rename_column("AB", "AB_int")

### Comparisons
# Compare and remove columns in cat and kin tables
scatter_and_r2(cat, kin, "W50", "W50", "cat vs kin W50")
if len(cat["W50"]) > len(kin["W50"]):
    kin.remove_column("W50")
else:
    cat.remove_column("W50")

# Compare param and sfr tables, and adjust BMag columns
scatter_and_r2(param, sfr, "BMag", "BMag", "param vs sfr BMag")
if len(param["BMag"]) > len(sfr["BMag"]):
    param.remove_column("BMag")
else:
    sfr.remove_column("BMag")

# Compare and remove RA columns in param and sfr tables
scatter_and_r2(param, sfr, "RAh", "RAh", "param vs sfr RAh")

columns_to_remove = ["RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs"]
for col in columns_to_remove:
    sfr.remove_column(col)

# Compare cat and sfr tables, and adjust TType columns
scatter_and_r2(cat, sfr, "TType", "T", "cat vs sfr TType")
ttype = Table(
    join(cat["Name", "TType"], sfr["Name", "T"], join_type="outer", keys="Name")
)
sfr.remove_column("T")

# Remove unnecessary columns in vel and dist tables
vel.remove_column("r_cz")
dist.remove_column("r_DM")
dist.remove_column("n_DM")

# Replace "magn" with "colors" in the list of table names
nm[nm.index("magn")] = "colors"

# Combine all tables into one
dtables = [Table(globals()[name]) for name in nm]
dt = dtables[0]
for data in dtables[1:]:
    dt = Table(join(dt, data, join_type="outer"))

# Filter and adjust rows
mask = ~((dt["DE-"] == "-") | (dt["DE-"] == "+"))
dt = dt[~mask]

# Convert log values to linear scale
dt["KLum"] = 10 ** dt["logKLum"]
dt["KLum"].unit = u.Lsun
dt["KLum"].description = "Linear K_S_ band luminosity"

dt["M26"] = 10 ** dt["logM26"]
dt["M26"].unit = u.Msun
dt["M26"].description = "Linear mass within Holmberg radius"

dt["MHI"] = 10 ** dt["logMHI"]
dt["MHI"].unit = u.Msun
dt["MHI"].description = "Linear hydrogen mass"

dt.rename_column("SFRHa", "logSFRHa")
dt.rename_column("SFRFUV", "logSFRFUV")

dt["SFRHa"] = 10 ** dt["logSFRHa"]
dt["SFRFUV"] = 10 ** dt["logSFRFUV"]
dt["SFRFUV"].unit = u.Msun / u.yr
dt["SFRHa"].unit = u.Msun / u.yr
dt["SFRHa"].description = "Linear Star Formation Rate (Hα)"
dt["SFRFUV"].description = "Linear Star Formation Rate (FUV)"

log_columns_to_linear = ["KLum", "M26", "MHI, SFRHa, SFRFUV"]

# create a column with the average value of SFRHa and SFRFUV
# Extract the data and masks from the columns
sfr_ha_data = dt["SFRHa"].data
sfr_ha_mask = dt["SFRHa"].mask

sfr_fuv_data = dt["SFRFUV"].data
sfr_fuv_mask = dt["SFRFUV"].mask

# Create masked arrays using the data and masks
sfr_ha_masked = np.ma.array(sfr_ha_data, mask=sfr_ha_mask)
sfr_fuv_masked = np.ma.array(sfr_fuv_data, mask=sfr_fuv_mask)

# Stack the masked arrays along a new axis
stacked_sfr = np.ma.stack([sfr_ha_masked, sfr_fuv_masked], axis=-1)

# Compute the mean along the new axis, ignoring masked values
sfr_avg_masked = np.ma.mean(stacked_sfr, axis=-1)

# Create a MaskedColumn for 'SFR_avg' using the computed masked average
dt["SFR_UNGC"] = MaskedColumn(
    sfr_avg_masked.data, mask=sfr_avg_masked.mask, unit=u.Msun / u.yr
)
dt["SFR_UNGC"].description = "Average Star Formation Rate, from the Ha and FUV"

# log
dt["logSFR_UNGC"] = np.log10(dt["SFR_UNGC"].data)
dt["logSFR_UNGC"].description = "Log Average Star Formation Rate, from the Ha and FUV"

# new cols of linear  av_SFR to u.Msun/u.Gyr
dt["SFR_UNGC_Gyr"] = dt["SFR_UNGC"].to(u.Msun / u.Gyr)
dt["SFR_UNGC_Gyr"].description = (
    "Average Star Formation Rate, from the Ha and FUV, in Gyr"
)
# To log
dt["logSFR_UNGC_Gyr"] = np.log10(dt["SFR_UNGC_Gyr"].data)
dt["logSFR_UNGC_Gyr"].description = (
    "Log Average Star Formation Rate, from the Ha and FUV, in Gyr"
)

# Check the description of the new column

print(dt["SFR_UNGC"].description)

# Add Stellar Mass from Mass/Light ratio M*/KLum = 0.6
dt["StellarMass"] = 0.6 * dt["KLum"]
dt["StellarMass"].unit = u.Msun
dt["StellarMass"].description = "Stellar Mass (M*/K luminosity)"

# Add column of logStellarMass
dt["logStellarMass"] = np.log10(dt["StellarMass"])
dt["logStellarMass"].description = "Log Stellar Mass (M*/K luminosity)"

# Create SkyCoord objects for coordinates
ra_str = [
    f"{hour}:{minute}:{second:.1f}"
    for hour, minute, second in zip(dt["RAh"], dt["RAm"], dt["RAs"])
]
dec_str = [
    f"{sign}{degree}:{minute}:{second:.1f}"
    for sign, degree, minute, second in zip(dt["DE-"], dt["DEd"], dt["DEm"], dt["DEs"])
]

dt["Coordinates"] = SkyCoord(
    ra_str, dec_str, obstime="J2000", unit=(u.hourangle, u.deg)
)  # make it 2 columns RA and DEC
dt["Ra"] = dt["Coordinates"].ra
dt["Ra"].description = "Right Ascension"
dt["Dec"] = dt["Coordinates"].dec
dt["Dec"].description = "Declination"
dt.remove_column("Coordinates")

# Remove original coordinate columns
for col in ["RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs"]:
    dt.remove_column(col)

# Reorder columns
column_order = ["Name", "Ra", "Dec"] + [
    col for col in dt.colnames if col not in ["Name", "Ra", "Dec"]
]
dt = dt[column_order]

# create a list of pairs of colums to be plotted
pairs = [
    ("FUVmag", "mag_FUV"),
    ("Bmag", "mag_B"),
    ("Hamag", "mag_Ha"),
    ("Kmag", "mag_Ks"),
    ("Bmag", "BMag"),
    ("21mag", "mag_HI"),
    ("AB", "AB_int"),
]

# Create plots for each pair of columns
for pair in pairs:
    col1, col2 = pair
    plt.figure(figsize=(8, 6))
    plt.scatter(dt[col1], dt[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"{col1} vs {col2}")
    plt.grid(True)
    plt.savefig(f"{col1} vs {col2}")
    plt.close()

pairs.remove(("Bmag", "BMag"))
pairs.remove(("AB", "AB_int"))


def compare_and_remove(dt, col1, col2):
    """
    Compare two columns in an Astropy Table and remove the one with more masked (NaN) values.

    This function compares the number of masked (NaN) values in two specified columns of an Astropy Table.
    It removes the column with the greater number of masked values. Additionally, it removes any associated
    flag columns (columns starting with 'l_', 'e_', or 'f_') corresponding to the removed column.

    Parameters
    ----------
    dt : astropy.table.Table
        The table containing the columns to be compared and potentially removed.
    col1 : str
        The name of the first column to be compared.
    col2 : str
        The name of the second column to be compared.

    Returns
    -------
    removed : str
        The name of the column that was removed from the table.
    """
    mask1 = dt[col1].mask.sum()
    mask2 = dt[col2].mask.sum()
    print(f"{col1}: {mask1} NaN", f"{col2}: {mask2} NaN")
    if mask1 >= mask2:
        dt.remove_column(col1)
        print(f"Removed {col1}")
        removed = col1
    else:
        dt.remove_column(col2)
        print(f"Removed {col2}")
        removed = col2

    if f"l_{removed}" in dt.colnames:
        dt.remove_column(f"l_{removed}")
    if f"e_{removed}" in dt.colnames:
        dt.remove_column(f"e_{removed}")
    if f"f_{removed}" in dt.colnames:
        dt.remove_column(f"f_{removed}")
    return removed


all_removed = []
for pair in pairs:
    col1, col2 = pair
    compiled_removed = compare_and_remove(dt, col1, col2)
    # List of all removed columns
    all_removed.append(compiled_removed)


# fix the units
dt["logKLum"].unit = ""
dt["logM26"].unit = ""
dt["logMHI"].unit = ""
dt["Thetaj"].unit = ""


fixed_units = ["logKLum", "logM26", "logMHI", "Thetaj", "SFRFUV", "SFRHa"]


# Function to check flag consistency
def check_flags(table, log_file="flag_errors_table.txt"):
    """
    Check for consistency between masked values and flag columns in an Astropy Table.

    This function iterates over the columns of the provided table and checks if the
    masked values in the main columns are consistent with the corresponding flag columns
    (columns starting with 'l_' or 'f_'). If inconsistencies are found, it records the
    positions of these inconsistencies.

    Parameters
    ----------
    table : astropy.table.Table
        The table to be checked for flag consistency. The table should contain columns
        with names starting with 'l_' or 'f_' which are used as flag indicators for the
        main columns.

    Returns
    -------
    flag_errors : dict
        A dictionary where the keys are the names of the flag columns with inconsistencies,
        and the values are arrays of positions (indices) where the inconsistencies occur.
        If no inconsistencies are found, the dictionary will be empty.
    """
    flag_errors = []

    # Iterate over columns with 'l_' or 'f_' prefix to identify limit flags or other flag indicators
    for colname in table.colnames:
        if colname.startswith("l_") or colname.startswith("f_"):
            main_col = colname[
                2:
            ]  # Remove 'l_' or 'f_' prefix to get the main column name
            if main_col in table.colnames:
                mask = table[main_col].mask
                flags = table[colname]

                # Check where the flag indicates a limit but the main column isn't masked
                flag_positions = (
                    flags != ""
                )  # Adjust for specific flag indicators if needed
                inconsistent_mask = (
                    ~mask
                ) & flag_positions  # Only add masks, don't remove existing ones

                # If inconsistencies exist, log them by adding masks
                if np.any(inconsistent_mask):
                    # Apply mask where the flag indicates a limit but the column is not masked
                    for i in np.where(inconsistent_mask)[0]:
                        if (
                            flags[i] != "" and not mask[i]
                        ):  # Flag indicates a limit, but the main column is not masked
                            table[main_col].mask[i] = True  # Apply mask
                            # Log the error
                            flag_errors.append(
                                [colname, i, f"Mask applied to {main_col} at index {i}"]
                            )

    # Print names of affected columns
    affected_columns = set([error[0] for error in flag_errors])
    if affected_columns:
        print("Inconsistencies found in the following columns:")
        for col in affected_columns:
            print(col)
    else:
        print("All flags are consistent with masked values.")

    # Log inconsistencies in a table format
    if flag_errors:
        error_table = Table(rows=flag_errors, names=("Column", "Position", "Action"))
        error_table.write(log_file, format="ascii", overwrite=True)
        print(f"Flag inconsistencies logged as a table to {log_file}.")

    return flag_errors


# Check the flags
flag_errors = check_flags(table)

# Make sure that all distances are less or equal to 11 Mpc, remove the galaxy otherwise
dt = dt[dt["Dis"] <= 11]

# Save the final table to an ECSV file

file_path = "../tables/final_table.ecsv"
print(dt.info())
print("WHAT WE HAVE DONE:")
print("\nFixed the coordinates to sky coordinates.")
print(
    "\nChanged the way table 3 was displayed, so we can compare it to the other tables."
)
print(f"\nWe converted the logarithmic values to linear scale:{log_columns_to_linear} ")
print(f"\nWe have removed:{all_removed}, after comparison with other columns.")
print(f"\nThe columns {fixed_units} are now in the correct units.")
print("\nThe masks are not the same as the flag columns in the input data.")
print("\nMerged all tables into one.")
print(
    "\nThe total number of columns in the final table is:",
    len(dt.colnames),
    "with number of rows:",
    len(dt),
)
print(f"\nThe final table has been saved to {file_path}.")
# Save the final table to a FITS file
ascii.write(dt, file_path, format="ecsv", overwrite=True)
