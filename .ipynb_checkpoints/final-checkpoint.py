# #+title:Investigations of the galaxies of the LCV
# #+subtitle: Finding the normalization constant of $SFR_{del}$ and the relations between the various masses of the Galaxies
# #+author: Dimitrios Papachistopoulos
# #+PROPERTY: header-args :lang python :eval python :exports results :tangle final.py :comments both :results output :session main

# :latex_prop:
# #+OPTIONS: toc:nil
# #+LaTeX_CLASS_OPTIONS: [a4paper,twocolumn]
# #+LaTeX_HEADER: \usepackage{breakcites}
# #+LaTeX_HEADER: \usepackage{paralist}
# #+LaTeX_HEADER: \usepackage{amsmath}
# #+LaTeX_HEADER: \usepackage{biblatex}
# #+LaTeX_HEADER: \usepackage{hyperref}
# #+LaTeX_HEADER: \usepackage{graphicx}
# #+LaTeX_HEADER: \usepackage{caption}
# #+LaTeX_HEADER: \usepackage{booktabs}
# #+LaTeX_HEADER: \usepackage[T1]{fontenc}
# #+LaTeX_HEADER: \usepackage{tgbonum}
# #+LaTeX_HEADER: \let\itemize\compactitem
# #+LaTeX_HEADER: \let\description\compactdesc
# #+LaTeX_HEADER: \let\enumerate\compactenum
# #+OPTIONS: tex:imagemagick
# #+bibliography:My Library.bib
# :end:


# [[file:paper.org::+begin_src python :results none][No heading:1]]
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table, QTable, join, hstack, vstack, unique, Column, MaskedColumn, setdiff
from astropy.utils.masked import Masked
from astropy.coordinates import SkyCoord, Galactic, Angle
from astropy.time import Time
import astropy.units as u
from astropy.utils.diff import report_diff_values
from astropy.visualization import quantity_support
import glob as glob
import sys
import matplotlib.pyplot as plt
from plotnine import (
    ggplot,qplot, ggsave, aes,
    after_stat,
    stage,
    geom_point, geom_smooth, geom_qq, geom_bar, geom_text, geom_label,
    position_dodge2,
    facet_wrap,
    labs,
    scale_x_continuous, scale_x_log10, scale_y_log10,
    xlim, ylim,
    stat_qq
)
from scipy.optimize import fsolve
from sympy import symbols, exp, log, lambdify


dt = QTable.read("./tables/final_table.ecsv")
dt.info()
# No heading:1 ends here

# How are the galaxies chosen

# According to [cite:@kraan-kortewegCatalogueGalaxiesIo1979] the Local Cosmological Volume is defined as the galaxies inside the radius of 10 Mpc and having radial velocities with respect to centroid of the Local Group $V_{lg} \le 500 \, km \cdot s^{-1}$. However, this assumed a Hubble constant of $H_0 = 50\, km \cdot s^{-1}$.

# 1. *Initial Selection Criteria*: Galaxies within a 10 Mpc radius were initially selected based on a radial velocity limit (VLG) of 500 km/s, considering a Hubble parameter (H0) of 50 km/s/Mpc.

# 2. *Updated Criteria*: To accommodate the revised H0 value of 73 km/s/Mpc, the VLG limit needs to be raised to 730 km/s.

# 3. *Local Velocity Field*: The presence of the Virgo cluster and the Local Void introduces additional velocity components, complicating distance estimation based solely on radial velocities.

# 4. *Peculiar Motions*: Collective motions within large-scale structures can introduce peculiar velocities, complicating distance estimation.

# 5. *Distance Measurement Methods*: Direct distance measurements using methods like the tip of the red giant branch (TRGB) provide accurate distances but are resource-intensive, requiring extensive observation time with instruments like the Hubble Space Telescope (HST).

# 6. *Inclusion Criteria*: Galaxies are included based on either radial velocities or distance estimates, considering the limitations and uncertainties in both methods.

# 7. *Extension to 11 Mpc*: Galaxies with distance estimates beyond 10 Mpc may still be included due to uncertainties in distance measurements and the potential influence of coherent motions and large-scale structures.

# 8. *Sample Composition*: The LV sample comprises src_python[:results value org]{len(dt)} {{{results(src_org{1440})}}} galaxies, with considerations for galaxies near the boundaries of the selection criteria and the potential influence of measurement errors.


# [[file:paper.org::*How are the galaxies chosen][How are the galaxies chosen:1]]
hubble = (ggplot(dt[["Dis", "VLG"]].to_pandas())
          + aes( x = "Dis", y =  "VLG")
          + labs( x = f"Distance [{dt['Dis'].unit}]" , y = "Radial velocities $V_{LG} $"+f"[{dt['VLG'].unit:latex}]" )
          + xlim(0,11) + ylim(0,1200)
          + geom_smooth(color = "red")
          + geom_point()
          )
fname = "figure/hubble.png"
hubble.save(fname)
fname

# How are the galaxies chosen:1 ends here

# Mapping the galaxies
# Because matplotlib needs the coordinates in radians and between $-\pi$ and $\pi$
# and, not 0 and $2\pi$, we have to convert coordinates.


# [[file:paper.org::*Mapping the galaxies][Mapping the galaxies:1]]
filename = "figure/mapping"

# Assuming dt is your data table containing coordinates, mass, and distance
c = dt["Coordinates"]
mass = dt["M26"].data  # Assuming mass is provided in some unit
distance = dt["Dis"].data  # Assuming distance is provided in some unit
distance_units = dt["Dis"].units

# Extract Galactic Coordinates
galactic_coords = c.galactic

# Extract Equatorial Coordinates
equatorial_coords = c.transform_to('icrs')

# Define the size and color based on mass and distance
marker_size = np.sqrt(mass) * 0.0002  # Adjust scaling factor as needed
marker_color = distance   # Use distance directly for marker color

# Plot Galactic Coordinates
plt.figure(figsize=(8, 8))
plt.subplot(211, projection="aitoff")
plt.grid(True)
plt.scatter(galactic_coords.l.wrap_at(180 * u.deg).radian, galactic_coords.b.radian, s=marker_size, c=marker_color, cmap='viridis')
plt.colorbar(label='Distance [{}]'.format(dt['Dis'].unit))  # Add colorbar for distance
plt.title("Galactic Coordinates")

# Plot Equatorial Coordinates
plt.subplot(212, projection="mollweide")
plt.grid(True)
plt.scatter(equatorial_coords.ra.wrap_at(180 * u.deg).radian, equatorial_coords.dec.radian, s=marker_size, c=marker_color, cmap='viridis')
plt.colorbar(label='Distance [{}]'.format(dt['Dis'].unit))  # Add colorbar for distance
plt.title("Equatorial Coordinates")


plt.suptitle("Galaxies of the LCV with Mass and Distance Representation")

plt.tight_layout()  # Adjust spacing between subplots
plt.savefig(filename)
plt.close()

filename+".png"
# Mapping the galaxies:1 ends here

# Morphology
# #+name: morphology

# [[file:paper.org::morphology][morphology]]
x="TType"
file="Types"
label="Morphology type code"
dttype = pd.DataFrame({"x": dt["{}".format(x)]}).dropna()
morphology = (
    ggplot(dttype, aes("factor(x)"))
    + geom_bar(color="black", fill="#1f77b4", show_legend=False)
    + geom_text(
        aes(label=after_stat("count")),
        stat="count",
        nudge_y=15,
        va="bottom",
        size = 9
    )
    + geom_text(
        aes(label=after_stat("prop*100"), group=1),
        stat="count",
        va="bottom",
        format_string="({:.1f}%)",
        size = 6
    )
    + labs(x = "{}".format(label))
)
fname = "figure/{}.png".format(file)
morphology.save(fname)

"[[./"+fname+"]]"
# morphology ends here

# Understanding the limit flags

# Some of those values contain limit flags, which we will mask for our present analysis. However, those values will be shown in the plots, and afterwards will be compared with the theoretical values.


# [[file:paper.org::*Understanding the limit flags][Understanding the limit flags:1]]
for column in dt.columns:
    if column.startswith("l_") or column.startswith("f_"):
        if column.startswith("l_"):
            corresponding_column_name = column[2:]  # Remove the 'l_' prefix
        else:
            corresponding_column_name = column[2:]  # Remove the 'f_' prefix

        try:
            all_masks_in_corresponding = all(mask in dt[corresponding_column_name].mask for mask in dt[column].mask)
            if all_masks_in_corresponding:
                print(f"All masks in {column} are also masks in {corresponding_column_name}")
            else:
                print(f"Not all masks in {column} are masks in {corresponding_column_name}")
        except AttributeError:
            print(f"We have no mask for {column}")
# Understanding the limit flags:1 ends here

# Total stellar masses and the total gas mass of the galaxies
# The K-band values are converted to the total Stellar Masses of each galaxy according to the mass-to-light ratio of 0.6 ($M_\odot/Lum$)(\cite{lelliSPARCMASSMODELS2016}), and the $MHI$ can be converted to the total mass of the gas of the galaxy using the equation $M_g=1.33\, MHI$


# [[file:paper.org::*Total stellar masses and the total gas mass of the galaxies][Total stellar masses and the total gas mass of the galaxies:1]]
dt["M_g"] = 1.33 * dt["MHI"]
dt["M_g"].unit = dt["MHI"].unit
# Total stellar masses and the total gas mass of the galaxies:1 ends here

# The total and the average SFR

# The total SFR of each galaxy can be calcuated by the mean values of SFR_{Ha} and SFR_{FUV}


# [[file:paper.org::*The total and the average SFR][The total and the average SFR:1]]
sfrs = (ggplot(dt[["SFRFUV", "SFRHa"]].to_pandas())
         + aes(x = "SFRFUV", y = "SFRHa")
         + labs(x = "$SFR_{FUV}$"+f" [{dt['SFRFUV'].unit:latex}]",
                y = "$SFR_{Hα}$"+f" [{dt['SFRHa'].unit:latex}]")
         + geom_point()
         + geom_smooth(method = "lm")
)


fname = "figure/sfrs.png"
sfrs.save(fname)

fname
# The total and the average SFR:1 ends here

# Constant t_{sf}
# The observed ages of galactic discs are $tsf≈ 12$ Gyr[cite:@knoxSurveyCoolWhite1999], so assuming an approximation of $tsf=12.5$ Gyr, the $\overline{SFRdel}$ can be calcuated, from the equation (\ref{eq:av_SFR M*}).


# [[file:paper.org::*Constant t_{sf}][Constant t_{sf}:1]]
###Constant tsf
dts=dt.copy()
tsf=12.5*10**9
zeta=1.3

dts['av_SFR']=dts['StellarMass']*1.3/(12.5*10**9)
dts['log_av_SFR']=np.log10(dts['av_SFR'].data)
adsf
dts['ratio']=dts['av_SFR']/dts['SFR_0']
# Constant t_{sf}:1 ends here



# #+RESULTS:
# : [0;33mWARNING[0m: column logKLum has a unit but is kept as a MaskedColumn as an attempt to convert it to Quantity failed with:
# : UnitTypeError("MaskedQuantity instances require normal units, not <class 'astropy.units.function.logarithmic.DexUnit'> instances.") [astropy.table.table]
# : [0;33mWARNING[0m: column logM26 has a unit but is kept as a MaskedColumn as an attempt to convert it to Quantity failed with:
# : UnitTypeError("MaskedQuantity instances require normal units, not <class 'astropy.units.function.logarithmic.DexUnit'> instances.") [astropy.table.table]
# : [0;33mWARNING[0m: column logMHI has a unit but is kept as a MaskedColumn as an attempt to convert it to Quantity failed with:
# : UnitTypeError("MaskedQuantity instances require normal units, not <class 'astropy.units.function.logarithmic.DexUnit'> instances.") [astropy.table.table]
# : /tmp/babel-d6GMUU/python-cPnIyx:8: RuntimeWarning: divide by zero encountered in log10
# : /home/dp/.local/lib/python3.10/site-packages/astropy/utils/masked/core.py:879: RuntimeWarning: divide by zero encountered in divide
# : /home/dp/.local/lib/python3.10/site-packages/astropy/utils/masked/core.py:879: RuntimeWarning: invalid value encountered in divide

# After that the equation of ratio

# \begin{equation} \label{eq:ratio}                                        \frac{\overline{SFRdel}}{SFR0,del}=\frac{e^x-x-1}{x^2}
# \end{equation}

# can be solved numerically for x and using the equations (\Ref{eq:SFR}) and (\Ref{eq:av_SFR-x}) the $Adel$ and of each galaxy are found.


# [[file:paper.org::*Constant t_{sf}][Constant t_{sf}:2]]
# Define symbols
x_sym = symbols('x')

# Define the function
def sfrx(x, ratio):
    x = max(0, x)
    return ratio - (exp(x) - x - 1) / x**2

# Convert the function to a callable function using lambdify
x_ratio = symbols('x_ratio')
sfrx_callable = lambdify([x_sym, x_ratio], sfrx(x_sym, x_ratio), modules='numpy')

# Assuming you have your data in an Astropy Table (similar to pandas DataFrame)

# Define a function to be solved
def solve_z(ratio):
    # Define a modified function with x shifted by 1 to ensure initial guess is positive
    def modified_sfrx(x):
        return sfrx_callable(x + 1, ratio)

    # Solve the modified function starting from initial guess of 2
    return fsolve(modified_sfrx, 2.0)[0] - 1  # Shift back by 1 to get original solution

# Use Astropy's Table functionality to iterate over rows efficiently
z_values = np.zeros(len(dts))
for i, ratio in enumerate(dts['ratio'].data):
    z_values[i] = solve_z(ratio)

# Convert the z values to dimensionless quantities
z_values *= u.dimensionless_unscaled

# Assign the result back to the table
dts['x_tsf'] = z_values

# Optionally, compute the log
dts["log_x_tsf"] = np.log10(dts["x_tsf"])

print(dts["x_tsf"])
# Constant t_{sf}:2 ends here

# Filling the Catalogue


# [[file:paper.org::*Filling the Catalogue][Filling the Catalogue:1]]
from astropy.table import QTable
import astropy.units as u

class CustomQTable(QTable):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if isinstance(value, u.Quantity):
            self.update_metadata(key, value.unit)

    def add_column(self, column, index=None, name=None, copy=True):
        super().add_column(column, index=index, name=name, copy=copy)
        if isinstance(column, u.Quantity):
            self.update_metadata(name, column.unit)

    def update_metadata(self, column_name, unit):
        if 'latex' not in self.meta:
            self.meta['latex'] = {}
        self.meta['latex'][column_name] = {
            'latex_name': f'${column_name}$',
            'latex_unit': str(unit)
        }

# Example usage:
dt = CustomQTable()
dt['Column1'] = [1, 2, 3] * u.m
dt['Column2'] = [4, 5, 6] * u.s
print(dt.meta['latex'])

# Changing the unit of Column2
dt['Column2'] = dt['Column2'] * u.cm
print(dt.meta['latex'])

print(dt["Column2"].meta["latex"])
# Filling the Catalogue:1 ends here
