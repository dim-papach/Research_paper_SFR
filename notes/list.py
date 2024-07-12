import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, QTable
from astropy.utils.diff import report_diff_values
import astropy.units as u
import glob
import os

 

# Change directory to where the notes are located
os.chdir("/home/dp/Documents/Research_paper_SFR/notes")

# Find all matching files in the tables directory
file_pattern = "../tables/lvg_table*.dat"
file_paths = sorted(glob.glob(file_pattern))

# Create a list to store Tables
dq = [Table(ascii.read(file_path, format="mrt"), masked=True) for file_path in file_paths]

# Table names
nm = ["cat", "param", "magn", "vel", "kin", "dist", "sfr"]

# Associate table names with the tables
for name, table in zip(nm, dq):
    globals()[name] = table

# Clean RAh column in all tables that contain it
def clean_ra_column(table):
    if 'RAh' in table.colnames:
        table['RAh'] = [int(str(x).replace('.', '')) if '.' in str(x) else x for x in table['RAh']]

for table in dq:
    clean_ra_column(table)

# Modify magnitudes table
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

# Compare and remove columns in cat and kin tables
identical = report_diff_values(cat["Name", "W50"], kin["Name", "W50"])
cat.remove_column("W50")
kin.remove_column("r_W50")

# Compare param and sfr tables, and adjust BMag columns
with open('diff_report.txt', 'w') as file:
    report_diff_values(param["Name", "BMag"], sfr["Name", "BMag"], rtol=0.7, atol=0.7, fileobj=file)

bmag = join(param["Name", "BMag"], sfr["Name", "BMag"], join_type='outer', keys="Name")
missing_indices = np.isnan(bmag['BMag_2'])
bmag['BMag_2'][missing_indices] = bmag['BMag_1'][missing_indices]
bmag.rename_column("BMag_2", "BMag")
bmag.remove_column("BMag_1")

# Update sfr table
sfr.remove_column("BMag")
param.remove_column("BMag")
sfr = QTable(join(sfr, bmag, join_type="outer"))

# Compare and remove RA columns in param and sfr tables
with open('diff_report_RAh.txt', 'w') as file:
    report_diff_values(param["Name", "RAh"], sfr["Name", "RAh"], fileobj=file)

columns_to_remove = ["RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs"]
for col in columns_to_remove:
    sfr.remove_column(col)

# Compare cat and sfr tables, and adjust TType columns
report_diff_values(cat["Name", "TType"], sfr["Name", "T"])
ttype = Table(join(cat["Name", "TType"], sfr["Name", "T"], join_type='outer', keys="Name"))
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
mask = ~((dt['DE-'] == '-') | (dt['DE-'] == '+'))
dt = dt[~mask]

# Convert log values to linear scale
dt["KLum"] = (10**dt["logKLum"])
dt["KLum"].unit = u.Lsun
dt["KLum"].description = "Linear K_S_ band luminosity"

dt["M26"] = (10**dt["logM26"])
dt["M26"].unit = u.Msun
dt["M26"].description = "Linear mass within Holmberg radius"

dt["MHI"] = (10**dt["logMHI"])
dt["MHI"].unit = u.Msun
dt["MHI"].description = "Linear hydrogen mass"

# Create SkyCoord objects for coordinates
ra_str = [f"{hour}:{minute}:{second:.1f}" for hour, minute, second in zip(dt['RAh'], dt['RAm'], dt['RAs'])]
dec_str = [f"{sign}{degree}:{minute}:{second:.1f}" for sign, degree, minute, second in zip(dt['DE-'], dt['DEd'], dt['DEm'], dt['DEs'])]

dt['Coordinates'] = SkyCoord(ra_str, dec_str, obstime="J2000", unit=(u.hourangle, u.deg))

# Remove original coordinate columns
for col in ['RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs']:
    dt.remove_column(col)

# Reorder columns
column_order = ["Name", "Coordinates"] + [col for col in dt.colnames if col not in ["Name", "Coordinates"]]
dt = dt[column_order]

# Save the final table to a FITS file
dt.write("../tables/final_table.fits", overwrite=True)
