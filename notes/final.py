cd ~/Documents/Research_paper_SFR/
wget --no-check-certificate -r -nH --cut-dirs=2 --accept="*.dat" --reject="*8.dat" -e robots=off -U mozilla https://www.sao.ru/lv/lvgdb/tables/

import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord, ICRS, Galactic, FK4, FK5, Angle
from astropy.table import Table, QTable, join, hstack, vstack, unique, Column, MaskedColumn, setdiff
from astropy.utils.masked import Masked
from astropy.coordinates import SkyCoord, Galactic, Angle
from astropy.time import Time
import astropy.units as u
from astropy.utils.diff import report_diff_values
import astropy.constants as const
import glob as glob
import pandas as pd
import sys
import os

os.chdir("/home/dp/Documents/Research_paper_SFR/notes")
file_pattern = "../tables/lvg_table*.dat"

# Use glob to find all matching files
file_paths = sorted(glob.glob(file_pattern))

dq = []  # Create an empty list to store Tables

for file_path in file_paths:
    table = Table(ascii.read(file_path, format="mrt"),masked=True)
    dq.append(table)

nm = []
nm = ["cat", "param", "magn", "vel", "kin", "dist", "sfr"]

# Automatically associate the names with the Tables
for i in range(len(nm)):
    globals()[nm[i]] = dq[i]

magn.remove_column("r_mag")

magn["Filter"][magn["Filter"] == "FU"] = "FUV"

filters = magn.group_by("Filter")

keys = filters.groups.keys["Filter"]
keys[keys == "FU"] = "FUV"
indices = filters.groups.indices


magn_table = Table()
dcolor = []
pain = 0
n = 0
print(indices,"\n", keys,"\n")
for ind in indices[1:]:
    magn_table = filters[pain:ind]
    magn_table.rename_column("l_mag", f"l_mag_{keys[n]}")
    magn_table.rename_column("mag", f"mag_{keys[n]}")
    magn_table.rename_column("e_mag", f"e_mag_{keys[n]}")
    magn_table.remove_column("Filter")
    print("\n********************",keys[n],"********************\n",magn_table.info, )
    pain = ind
    n+=1
    dcolor.append(magn_table)
colors = Table(dcolor[0])

for dcolor_item in dcolor[1:]:
    colors = Table(join(colors, dcolor_item, keys="Name", join_type="outer"))

colors.info

param.rename_column("a26", "A26")
param.rename_column("AB","AB_int")

identical = report_diff_values(cat["Name","W50"], kin["Name", "W50"])
print(identical)

cat.remove_column("W50")
kin.remove_column("r_W50")

# Open a file for writing
with open('diff_report.txt', 'w') as file:
    # Redirect the output to the file
   identical = report_diff_values(param["Name","BMag"], sfr["Name", "BMag"], rtol=0.7, atol=0.7, fileobj=file)
identical

bmag = join(param["Name","BMag"], sfr["Name","BMag"], join_type='outer', keys = "Name")

# Find indices where 'BMag_2' is NaN
missing_indices = np.isnan(bmag['BMag_2'])

# Replace NaN values in 'BMag_2' with corresponding values from 'BMag_1'
bmag['BMag_2'][missing_indices] = bmag['BMag_1'][missing_indices]

bmag.rename_column("BMag_2", "BMag")
bmag.remove_column("BMag_1")

print(bmag["BMag"])

sfr.remove_column("BMag")
param.remove_column("BMag")
sfr = QTable(join(sfr,bmag, join_type = "outer"))

sfr.info()

with open('diff_report_RAh.txt', 'w') as file:
    # Redirect the output to the file
   identical = report_diff_values(param["Name","RAh"], sfr["Name", "RAh"], fileobj=file)
print(identical)

columns_to_remove = ["RAh", "RAm", "RAs", "DE-", "DEd","DEm", "DEs"]
# Remove the identified columns
for col in columns_to_remove:
    sfr.remove_column(col)

identical = report_diff_values(cat["Name","TType"], sfr["Name", "T"])

ttype = Table(join(cat["Name","TType"], sfr["Name","T"], join_type='outer', keys = "Name"))
print(ttype.info)

sfr.remove_column("T")

vel.remove_column("r_cz")
dist.remove_column("r_DM")
dist.remove_column("n_DM") # Method used to determine DM

# Find the index of "magn" in the list
index_to_replace = nm.index("magn")

# Replace "magn" with "color"
nm[index_to_replace] = "colors"

dtables = []


for i in range(len(nm)):
    lists = Table(globals()[nm[i]])
    dtables.append(lists)

dt = dtables[0]
for data in dtables[1:]:
    dt = Table(join(dt, data, join_type="outer"))
print(dt.info)

print(dt[dt["Name"] == "6dF J2218489-46130"])

mask = ~((dt['DE-'] == '-') | (dt['DE-'] == '+'))

# Get the rows to delete
rows_to_delete = dt[mask]

# Filter the table to keep only the rows where 'DE-' is either '+' or '-'
dt = dt[~mask]

# Print the rows to delete
print("Rows to delete:", len(rows_to_delete))
print(rows_to_delete)
print("Remaining Galaxies:", len(dt))

dt["KLum"] = (10**dt["logKLum"])
dt["KLum"].unit = u.Lsun
dt["KLum"].description = "Linear K_S_ band luminosity"

dt["M26"] = (10**dt["logM26"])
dt["M26"].unit = u.Msun
dt["M26"].description = "Linear mass within Holmberg radius"

dt["MHI"] = (10**dt["logMHI"])
dt["MHI"].unit = u.Msun
dt["MHI"].description = "Linear hydrogen mass"
dt[["KLum", "M26", "MHI"]].info

SFR_unit = u.Msun/u.year
dt["SFRHa"].unit = SFR_unit
dt["SFRFUV"].unit = SFR_unit

dt.info

dt["SFR_0"] = np.ma.mean([dt["SFRHa"].data,dt["SFRFUV"].data], axis = 0)
dt["SFR_0"].unit = SFR_unit

dt["StellarMass"] = 0.6 * dt["KLum"]
dt["StellarMass"].unit = dt["KLum"].unit * u.Msun/u.solLum
dt["StellarMass"].description = "K-band luminosity using a mass-to-light ratio of 0.6"
dt["StellarMass"].info

dt["av_SFR_theor"] = 1.3 * dt["StellarMass"] /12.5
dt["av_SFR_theor"].unit = dt["StellarMass"].unit/u.Gyr
dt["av_SFR_theor"].description = "The theoretical average SFR with zeta=1.3 and t_sf = 12.5 Gyr"
dt["av_SFR_theor"].info

dt.info()

data_table = dt.copy() # At first I was afraid, I was petrified, that this would break everything so I did it after I saved the file. Now I put it here and we will see (:

ra_hour_column = data_table['RAh']
ra_minute_column = data_table['RAm']
ra_second_column = data_table['RAs']
dec_sign_column = data_table['DE-']
dec_degree_column = data_table['DEd']
dec_minute_column = data_table['DEm']
dec_second_column = data_table['DEs']

# Create SkyCoord objects with strings
ra_str = [f"{hour}:{minute}:{second:.1f}" for hour, minute, second in zip(ra_hour_column, ra_minute_column, ra_second_column)]
dec_str = [f"{sign}{degree}:{minute}:{second:.1f}" for sign,degree, minute, second in zip(dec_sign_column, dec_degree_column, dec_minute_column, dec_second_column)]

# Create SkyCoord objects in the Galactic coordinate system
galactic_coords = data_table['Coordinates'] = SkyCoord(ra_str, dec_str, obstime = "J2000", unit=(u.hourangle, u.deg))
# Print the Galactic coordinates
print(galactic_coords)
data_table.remove_column('RAh')
data_table.remove_column('RAm')
data_table.remove_column('RAs')
data_table.remove_column('DE-')
data_table.remove_column('DEd')
data_table.remove_column('DEm')
data_table.remove_column('DEs')

column_order = ["Name","Coordinates"] + [col for col in data_table.colnames if col not in ["Name","Coordinates"]]

# Reorder columns
data_table = data_table[column_order]
data_table.info()

ascii.write(data_table, "../tables/final_table.ecsv", format= "ecsv" , overwrite=True)
ascii.write(data_table, "../tables/final_table.csv", format= "csv" , overwrite=True)

paint = QTable(ascii.read("../tables/final_table.ecsv", format="ecsv"),masked=True)

paint.show_in_browser(jsviewer = True)

paint.info()

# Example usage:
dt = CustomQTable()
dt['Column1'] = [1, 2, 3] * u.m
dt['Column2'] = [4, 5, 6] * u.s

# Changing the unit for Column1 to kilometers
dt.change_latex_name('Column1', "padsfain")
dt.update_metadata("Column2",  latex_name = "fads")

dt["Column1"] = dt["Column1"].to(u.km) * u.yr
# Printing LaTeX metadata for all columns

print(dt.print_latex("Column2"))

# custom_table.py
dc = CustomQTable(ascii.read("../tables/final_table.ecsv", format="ecsv"),masked=True)

dc.update_metadata("Coordinates", unit = "deg, deg")

for col_name in dc.columns:
    if col_name != "Coordinates":
        dc.update_metadata(col_name)


dc.change_latex_name("a26", r"$a_{26}$")

dc.change_latex_name("FUVmag", r"$FUV_{mag}$")
dc.change_latex_name("Bmag", r"$B_{mag}$")
dc.change_latex_name("Hamag", r"$H\alpha_{mag}$")
dc.change_latex_name("Kmag", r"$K_{mag}$")
dc.change_latex_name("21mag", r"${HI_{mag}}^{21\. cm}$")
dc.change_latex_name("TType", r"Galaxy Type")
dc.change_latex_name("Dis", r"Distance")
dc.change_latex_name("A26", r"$A_{26}$")
dc.change_latex_name("inc", r"inclination")
dc.change_latex_name("AB_int", r"$AB_{int}$")
dc.change_latex_name("logKLum", r"$\log{K_{Lum}}$")
dc.change_latex_name("logM26", r"$\log{M_{26}}$")
dc.change_latex_name("logMHI", r"$\log{M_{HI}}$")
dc.change_latex_name("Theta1", r"$\Theta_1$")
dc.change_latex_name("Theta5", r"$\Theta_5$")
dc.change_latex_name("Thetaj", r"$\Theta_j$")

dc.change_latex_name("SFRFUV", r"$SFR_{FUV}$")
dc.change_latex_name("PFUV", r"$P_{FUV}$")
dc.change_latex_name("FFUV", r"$F_{FUV}$")
dc.change_latex_name("SFRHa", r"$SFR_{H\alpha}$")
dc.change_latex_name("PHa", r"$P_{H\alpha}$")
dc.change_latex_name("FHa", r"$F_{H\alpha}$")
dc.meta["latex"]["21mag"]

ascii.write(dc, "../tables/final_table.ecsv", format= "ecsv" , overwrite=True)
