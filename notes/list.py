import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, QTable
import astropy.units as u
from astropy.visualization import quantity_support
import glob
import os

# Set non-interactive backend for matplotlib
plt.switch_backend('Agg')

def clean_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    cleaned_lines = []
    for line in lines:
        if line[19:21].strip().endswith('.'):
            line = line[:19] + line[19:21].replace('.', '') + line[21:]
        cleaned_lines.append(line)
    
    cleaned_content = ''.join(cleaned_lines)
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
    dq.append(Table(ascii.read(cleaned_content, format="mrt"), masked=True))

# Table names
nm = ["cat", "param", "magn", "vel", "kin", "dist", "sfr"]

# Associate table names with the tables
for name, table in zip(nm, dq):
    globals()[name] = table


# Function to create scatter plot and calculate R^2
def scatter_and_r2(table1, table2, col1_name, col2_name, title):
    # Join tables based on 'Name'
    temp = join(table1["Name", col1_name], table2["Name", col2_name], join_type='inner', keys="Name")
    
    x = temp[temp.colnames[1]]
    y = temp[temp.colnames[2]]
    r2 = r2_score(x, y)

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, alpha=0.5)
    plt.xlabel(col1_name)
    plt.ylabel(col2_name)
    plt.title(f'{title}\n$R^2$ = {r2:.2f}')
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

# create a list of pairs of colums to be plotted
pairs = [('FUVmag','mag_FUV'),
         ('Bmag','mag_B'),
         ('Hamag', 'mag_Ha'),
         ('Kmag','mag_Ks'),
         ('Bmag', 'BMag'),
         ("21mag", "mag_HI"),
         ('AB','AB_int')
         ]

# Create plots for each pair of columns
for pair in pairs:
    col1, col2 = pair
    plt.figure(figsize=(8, 6))
    plt.scatter(dt[col1], dt[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'{col1} vs {col2}')
    plt.grid(True)
    plt.savefig(f'{col1} vs {col2}')
    plt.close()

pairs.remove(('Bmag', 'BMag'))
pairs.remove(('AB', 'AB_int'))


for pair in pairs:
    col1, col2 = pair
    mask1 = dt[col1].mask.sum()
    mask2 = dt[col2].mask.sum()
    print(f'{col1}: {mask1} NaN', f'{col2}: {mask2} NaN')
    if mask1>=mask2:
        dt.remove_column(col1)        
        print(f'Removed {col1}')
        removed = col1
    else:
        dt.remove_column(col2)
        print(f'Removed {col2}')
        removed = col2
    
    if f"l_{removed}" in dt.colnames:
        dt.remove_column(f"l_{removed}")
    if f"e_{removed}" in dt.colnames:
        dt.remove_column(f"e_{removed}")
    if f"f_{removed}" in dt.colnames:
        dt.remove_column(f"f_{removed}")
        
   
# fix the units
dt["logKLum"].unit = ""
dt["logM26"].unit = ""
dt["logMHI"].unit = ""
dt["Thetaj"].unit = ""

## SFR units
dt["SFRFUV"].unit = u.Msun / u.yr
dt["SFRHa"].unit = u.Msun / u.yr

print(dt.info())

# Save the final table to a FITS file
dt.write("../tables/final_table.fits", overwrite=True)