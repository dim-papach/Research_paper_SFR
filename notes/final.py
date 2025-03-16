# #+title:Investigations of the galaxies in the LCV
# #+subtitle: Finding the normalization constant of $SFR_{del}$ and the relations between the parameters of the Galaxies
# #+author: Dimitrios Papachistopoulos
# #+PROPERTY: header-args :lang python :eval python :exports results :tangle final.py :comments both :results output :session main_paper

# :latex_prop:
# #+OPTIONS: toc:nil
# #+LaTeX_CLASS_OPTIONS: [a4paper]
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
import numpy.ma as ma
import pandas as pd
from astropy.io import ascii
from astropy.table import Table, QTable, join, hstack, vstack, unique, Column, MaskedColumn, setdiff
from astropy.utils.masked import Masked
from astropy.coordinates import SkyCoord, Galactic, Angle
from astropy.time import Time
import astropy.units as u
from astropy.utils.diff import report_diff_values
from astropy.visualization import quantity_support, hist
quantity_support()
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
from scipy import optimize, stats
from sympy import symbols, exp, log, lambdify
import seaborn as sns
import plotly.express as px
# No heading:1 ends here

# [[file:paper.org::+begin_src python :results none][No heading:2]]
dt = QTable.read("./tables/outer_join.ecsv")
# No heading:2 ends here

# [[file:paper.org::+begin_src python :results none][No heading:3]]
sns.set_theme(style="darkgrid",context = "notebook", palette="deep")
# No heading:3 ends here

# How are the galaxies chosen

# According to [cite:@kraan-kortewegCatalogueGalaxiesIo1979] the Local Cosmological Volume is defined as the galaxies inside the radius of 10 Mpc and having radial velocities with respect to centroid of the Local Group $V_{lg} \le 500 \, km \cdot s^{-1}$. However, this assumed a Hubble constant of $H_0 = 50\, km \cdot s^{-1}$.

# 1. *Initial Selection Criteria*: Galaxies within a 10 Mpc radius were initially selected based on a radial velocity limit (VLG) of 500 km/s, considering a Hubble parameter (H0) of 50 km/s/Mpc.

# 2. *Updated Criteria*: To accommodate the revised H0 value of 73 km/s/Mpc, the VLG limit needs to be raised to 730 km/s.

# 3. *Local Velocity Field*: The presence of the Virgo cluster and the Local Void introduces additional velocity components, complicating distance estimation based solely on radial velocities.

# 4. *Peculiar Motions*: Collective motions within large-scale structures can introduce peculiar velocities, complicating distance estimation.

# 5. *Distance Measurement Methods*: Direct distance measurements using methods like the tip of the red giant branch (TRGB) provide accurate distances but are resource-intensive, requiring extensive observation time with instruments like the Hubble Space Telescope (HST).

# 6. *Inclusion Criteria*: Galaxies are included based on either radial velocities or distance estimates, considering the limitations and uncertainties in both methods.

# 7. *Extension to 11 Mpc*: Galaxies with distance estimates beyond 10 Mpc may still be included due to uncertainties in distance measurements and the potential influence of coherent motions and large-scale structures.

# 8. *Sample Composition*: The LV sample comprises src_python[:results value org]{len(dt)} {{{results(src_org{3934})}}} galaxies, with considerations for galaxies near the boundaries of the selection criteria and the potential influence of measurement errors.


# [[file:paper.org::*How are the galaxies chosen][How are the galaxies chosen:1]]
hubble = (ggplot(dt.to_pandas())
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

# The limit flags are placed in values, were the uncertainty of the value is high, usualy because of how accurate the measurement is (way to high or low)


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

# Standarized constants

# We should use some standart consistent values for our analysis.

# 1. According to [cite:@speagleHighlyConsistentFramework2014] and[cite:@kroupaConstraintsStarFormation2020] the $t_{sf} = 12\, Gyr$ represents a strong and consistent constraint of galaxy evolution, across many studies. While other researchers adopt a t_{sf}= 13.6 Gyr[cite:@haslbauerCosmologicalStarFormation2023], we use the 12 Gyr assumption following the framework of SP14
# 2. $\zeta =$ accommodates mass-loss through stellar evolution. According to the IGIMF theory the galaxies of the the LCV are expected to have 1< $\zeta$ <1.3, so by adopting $\zeta =1.3$ we are working conservatively
# 3. Main Sequence z = 5


# [[file:paper.org::*Standarized constants][Standarized constants:1]]
t_sf = 13.6 * u.Gyr
zeta = 1.3
z = 5
# Standarized constants:1 ends here

# Total stellar masses, the total gas mass and total barionic of the galaxies

# The $MHI$ can be converted to the total mass of the gas of the galaxy using the equation $M_g=1.33\, MHI$



# [[file:paper.org::*Total stellar masses, the total gas mass and total barionic of the galaxies][Total stellar masses, the total gas mass and total barionic of the galaxies:1]]
dt["M_g"] = 1.33 * dt["MHI"]
dt["M_g"].info()
# Total stellar masses, the total gas mass and total barionic of the galaxies:1 ends here



# #+RESULTS:
# : name = M_g
# : dtype = float64
# : unit = solMass
# : class = Quantity
# : n_bad = 3110
# : length = 3934

# The K-band values are converted to the total Stellar Masses of each galaxy according to the mass-to-light ratio of 0.6 ($M_\odot/Lum$)[cite:@lelliSPARCMASSMODELS2016]


# [[file:paper.org::*Total stellar masses, the total gas mass and total barionic of the galaxies][Total stellar masses, the total gas mass and total barionic of the galaxies:2]]
dt["StellarMass"] = 0.6 * dt["KLum"]* u.Msun/u.solLum
dt["StellarMass"].description = "K-band luminosity using a mass-to-light ratio of 0.6"
dt["StellarMass"].info()
# Total stellar masses, the total gas mass and total barionic of the galaxies:2 ends here



# #+RESULTS:
# : name = StellarMass
# : dtype = float64
# : unit = solMass
# : class = Quantity
# : n_bad = 2626
# : length = 3934

# The total barionic mass can be calcuated as the sum of the total gas mass of the galaxy with the Stellar mass


# [[file:paper.org::*Total stellar masses, the total gas mass and total barionic of the galaxies][Total stellar masses, the total gas mass and total barionic of the galaxies:3]]
dt["BarMass"] = dt["M_g"] + dt["StellarMass"]
dt["BarMass"].info()
# Total stellar masses, the total gas mass and total barionic of the galaxies:3 ends here

# Ratio of M_g and StellarMass


# [[file:paper.org::*Ratio of M_g and StellarMass][Ratio of M_g and StellarMass:1]]
dt["mass_ratio"] = dt["M_g"] / dt["StellarMass"]
dt["mass_ratio"].info(["attributes", "stats"])
# Ratio of M_g and StellarMass:1 ends here



# #+RESULTS:
# : name = mass_ratio
# : dtype = float64
# : class = Quantity
# : mean = 2.07787
# : std = 3.47845
# : min = 7.51105e-05
# : max = 43.2216
# : n_bad = 3121
# : length = 3934

# Histogram of dt["mass_ratio"]


# [[file:paper.org::*Ratio of M_g and StellarMass][Ratio of M_g and StellarMass:2]]
#seaborn plot of mass_ratio
sns.histplot(np.log10(dt["mass_ratio"].value))
plt.xlabel(r"Mass ratio $\frac{M_g}{M_*}$")
plt.ylabel("Number of Galaxies")
plt.title("Mass ratio distribution")
#save
plt.tight_layout()
plt.savefig("figure/mass_ratio.png")
plt.close()
#print in org
"./figure/mass_ratio.png"
# Ratio of M_g and StellarMass:2 ends here

# Color index

# Here we calculate the color indexes <FUV-B>


# [[file:paper.org::*Color index][Color index:1]]
dt["color"] = dt["FUVmag"]-dt["Bmag"]
# Color index:1 ends here



# #+RESULTS:

# The lower the value, the bluer the stars, thus the younger the star populations


# [[file:paper.org::*Color index][Color index:2]]
#hist
hist(dt["color"], bins = "freedman")
plt.xlabel("Color index")
plt.ylabel("Number of stars")
plt.title("Color index <FUV - B> distribution")
#save
plt.savefig("figure/color_index.png")
plt.close()
#print in org
"./figure/color_index.png"
# Color index:2 ends here

# SFR_0

# Now we have to calculate the total SFR from the equation:

# $$
#     SFR_o=\frac{SFR_{FUV}+SFR_{Ha}}{2}
# $$

# if we have both the SFR. If we only have one of them then:

# $$
#     SFR_{0}=SFR_{i},\ \text{if } SFR_{j}=0,\ i\neq j,\ i,j=SFR_{FUV},\, SFR_{Ha}
# $$


# [[file:paper.org::*SFR_0][SFR_0:1]]
sns.lmplot(data = dt.to_pandas(), x="SFRHa",y="SFRFUV")
#labels in latex with units
plt.xlabel(r"$SFR_{Ha}$"+r" [{}]".format(dt["SFRHa"].unit.to_string("latex")))
plt.ylabel(r"$SFR_{FUV}$"+r" [{}]".format(dt["SFRHa"].unit.to_string("latex")))

#save
plt.tight_layout()
plt.savefig("figure/SFR_FUV-Ha.png")
plt.close()


sns.jointplot(data = dt.to_pandas(), x="logSFRHa",y="logSFRFUV", kind = "reg", ratio=2,marginal_ticks=True)
#labels in latex with units
plt.xlabel(r"$\log_{10}(SFR_{Ha}$"+r"/ [{}])".format(dt["SFRHa"].unit.to_string("latex")))
plt.ylabel(r"$\log_{10}(SFR_{FUV}$"+r"/ [{}])".format(dt["SFRHa"].unit.to_string("latex")) )

plt.tight_layout()
#save
plt.savefig("figure/log_SFR_FUV_Ha.png")
plt.close()
#print in org
print("[[./figure/SFR_FUV-Ha.png]]")
print("[[./figure/log_SFR_FUV_Ha.png]]")
# SFR_0:1 ends here



# #+RESULTS:
# :results:
# [[./figure/SFR_FUV-Ha.png]]
# [[./figure/log_SFR_FUV_Ha.png]]
# :end:


# create the average SFR_0 from SFRHa SFRFUV with np.ma.average (already done)


# [[file:paper.org::*SFR_0][SFR_0:2]]
# Calculate the mean of SFRHa and SFRFUV, ignoring masked values

SFR_0 = dt["SFR_UNGC"]
# SFR_0:2 ends here

# Applying the cut SFR_0 >= 1e-3 solMass/yr

# keep only the SFR_0 data were >1e-3


# [[file:paper.org::*Applying the cut SFR_0 >= 1e-3 solMass/yr][Applying the cut SFR_0 >= 1e-3 solMass/yr:1]]
dc = dt.copy()
dc = dc[dc["SFR_UNGC"].value >= 1e-3]

print(dc["SFR_UNGC"].info())
# Applying the cut SFR_0 >= 1e-3 solMass/yr:1 ends here



# #+RESULTS:
# : name = SFR_UNGC
# : dtype = float64
# : unit = solMass / yr
# : description = Average Star Formation Rate, from the Ha and FUV
# : class = Quantity
# : n_bad = 0
# : length = 518
# : None


# [[file:paper.org::*Applying the cut SFR_0 >= 1e-3 solMass/yr][Applying the cut SFR_0 >= 1e-3 solMass/yr:2]]
dc[["SFR_UNGC","SFRFUV", 'SFRHa']].info("stats")
# Applying the cut SFR_0 >= 1e-3 solMass/yr:2 ends here



# #+RESULTS:
# : <QTable length=518>
# :   name            mean                  std                    min                    max          n_bad
# : -------- --------------------- --------------------- ------------------------ -------------------- -----
# : SFR_UNGC 0.152153 solMass / yr 0.457533 solMass / yr  0.00102329 solMass / yr 4.38718 solMass / yr     0
# :   SFRFUV 0.163954 solMass / yr 0.530241 solMass / yr 6.60693e-05 solMass / yr  5.7544 solMass / yr    82
# :    SFRHa 0.176991 solMass / yr 0.496778 solMass / yr 2.04174e-05 solMass / yr  4.2658 solMass / yr    94



# Histogram of SFR_0


# [[file:paper.org::*Applying the cut SFR_0 >= 1e-3 solMass/yr][Applying the cut SFR_0 >= 1e-3 solMass/yr:3]]
hist(dc["logSFR_UNGC"].value, bins = "freedman")

plt.title("Histogram of $SFR_0$")
plt.xlabel(r"$\log(SFR_0/[{}])$".format(dc["SFR_0"].unit.to_string("unicode")))
plt.ylabel("Number of Galaxies")
#save fig, plt close and print fig file name
fname = "figure/log_SFR_0_hist.png"
plt.savefig(fname)
plt.close()
fname
# Applying the cut SFR_0 >= 1e-3 solMass/yr:3 ends here

# Morphology
# #+name: morphology_cut

# [[file:paper.org::morphology_cut][morphology_cut]]
def morphology_hist_cut(x_data, label):

    dttype = pd.DataFrame({"x_data": dc["{}".format(x_data)]}).dropna()
    morphology = (
        ggplot(dttype, aes("factor(x_data)"))
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
    fname = "figure/{}_cut.png".format(x_data)
    morphology.save(fname)
    return "[[./"+fname+"]]"


morphology_hist_cut("TType", "Morphology type code")
# morphology_cut ends here

# Morphology of dwarf galaxies

# [[file:paper.org::*Morphology of dwarf galaxies][Morphology of dwarf galaxies:1]]
morphology_hist_cut("Tdw1","Morphology of dwarf galaxies")
# Morphology of dwarf galaxies:1 ends here

# Dwarf galaxy surface brightness morphology



# [[file:paper.org::*Dwarf galaxy surface brightness morphology][Dwarf galaxy surface brightness morphology:1]]
morphology_hist_cut("Tdw2", "Dwarf galaxy surface brightness morphology")
# Dwarf galaxy surface brightness morphology:1 ends here

# Theoretical Average SFR

# To calculate the average Star Formation Rate $\overline{SFR}$ we can use the equation

# $$
#     \overline{SFR}=\frac{\zeta M_*}{t_{sf}}
# $$

# where ζ is the mass-loss through stellar evolution and we assume that $\zeta\approx 1.3$ (see explanation in the paper`), M* is the stellar mass of each galaxy and we assume that is   $t_{sf}=13.6\ Gyr$


# [[file:paper.org::*Theoretical Average SFR][Theoretical Average SFR:1]]
dc["av_SFR_theor"] = 1.3 * dc["StellarMass"] / t_sf.to(u.yr)
dc["av_SFR_theor"].info()
# Theoretical Average SFR:1 ends here



# #+RESULTS:
# : name = av_SFR_theor
# : dtype = float64
# : unit = solMass / yr
# : class = Quantity
# : n_bad = 1
# : length = 518


# [[file:paper.org::*Theoretical Average SFR][Theoretical Average SFR:2]]
plt.scatter(y = dc["av_SFR_theor"],x = dc["SFR_UNGC"], c = dc["color"].value)
#colobar
plt.colorbar()

plt.xscale("log")
plt.yscale("log")
plt.show()
# Theoretical Average SFR:2 ends here

# Ratio av_SFR/SFR_0

# Now we have to calculate the ratio $\frac{\overline{SFR}}{SFR_0}$


# [[file:paper.org::*Ratio av_SFR/SFR_0][Ratio av_SFR/SFR_0:1]]
dc["SFR_ratio"] = dc["av_SFR_theor"] / dc["SFR_UNGC"]

#log10 of ratio
dc["logSFR_ratio"] = np.log10(dc["SFR_ratio"])

dc[["SFR_ratio", "logSFR_ratio"]].info(["attributes","stats"])
# Ratio av_SFR/SFR_0:1 ends here

# Histogram of SFR ratio

# [[file:paper.org::*Histogram of SFR ratio][Histogram of SFR ratio:1]]
dc["logSFR_ratio"].info("stats")
# Histogram of SFR ratio:1 ends here



# #+RESULTS:
# : name = logSFR_ratio
# : mean = -0.0103318
# : std = 0.500618
# : min = -1.54195
# : max = 2.96856
# : n_bad = 1
# : length = 518


# [[file:paper.org::*Histogram of SFR ratio][Histogram of SFR ratio:2]]
#hist(dc["SFR_ratio"], bins = "freedman")
sns.displot(dc.to_pandas(), x="logSFR_ratio")

#plt.xscale("log")
plt.axvline(-0.01, color = "black", linestyle = "--", alpha = 0.8)
#text next to mean line "Mean = mean.value"
plt.text(0.1, plt.ylim()[1]-4, r"Mean of ratio = 4.87", color = "blue")


plt.title(r"Histogram of $\frac{\overline{SFR}}{SFR_0}$")
plt.xlabel(r"$\log_{10}\left(\frac{\overline{SFR}}{SFR}\right)$")
plt.ylabel("Number of Galaxies")

plt.tight_layout()
#save fig, plt close and print fig file name
fname = "figure/SFR_ratio_hist.png"
plt.savefig(fname)
plt.close()
fname
# Histogram of SFR ratio:2 ends here

# Scatter color and ratio


# [[file:paper.org::*Scatter color and ratio][Scatter color and ratio:1]]
plt.scatter(y = dc["SFR_ratio"],x =dc["color"], c = np.log10(dc["mass_ratio"].value), cmap = "viridis" )
#title and labels with units in latex
plt.yscale("log")
plt.title(r"$\frac{\overline{SFR}}{SFR_0}$ - <FUV-B>")
plt.ylabel(r"$\frac{\overline{SFR}}{SFR_0}$")
plt.xlabel("<FUV-B>")
plt.colorbar(label = r"$M_g/M_*$")
#save in dir figure
plt.tight_layout()
plt.savefig("figure/ratio_vs_color.png")
plt.close()
#print file
"figure/ratio_vs_color.png"
# Scatter color and ratio:1 ends here

# The gas depletion timescale \tau_g

# "The gas depletion timescale τg measures the time taken by a galaxy to exhaust its gas content Mg given the current SFR (Pflamm-Altenburg & Kroupa 2009).

# $$
# \tau_g = \frac{M_g}{\dot{M_*}}
# $$

# where Mg is the neutral gas mass at the desired time and $\dot{M_*}$ is the SFR then."[cite:@nageshSimulationsStarformingMainsequence2023]


# [[file:paper.org::*The gas depletion timescale \tau_g][The gas depletion timescale \tau_g:1]]
dc["tau_g"] = dc["M_g"]/dc["SFR_UNGC"].to(u.solMass/u.Gyr)
dc["tau_g"].info("stats")
# The gas depletion timescale \tau_g:1 ends here



# #+RESULTS:
# : name = tau_g
# : mean = 15.4527 Gyr
# : std = 28.4194 Gyr
# : min = 0.401654 Gyr
# : max = 461.16 Gyr
# : n_bad = 38
# : length = 518


# [[file:paper.org::*The gas depletion timescale \tau_g][The gas depletion timescale \tau_g:2]]
plt.close("all")
sns.displot(np.log10(dc["tau_g"].value))
#mean line
plt.axvline(np.log10(np.mean(dc["tau_g"].value)), color = "black", linestyle = "--", alpha = 0.8)
#text next to mean line "Mean = mean.value"
plt.text(np.log10(np.mean(dc["tau_g"].value))*1.1, plt.ylim()[1]-4, r"Mean = {:.2f}".format(np.mean(dc["tau_g"])), color = "blue")

plt.xlabel(r'$\tau_g$ '+"[{}]".format(dc["tau_g"].unit.to_string("latex")))
plt.ylabel("Number of galaxies")
plt.title(r'Histogram of $\tau_g$')
plt.tight_layout()
#save
plt.savefig("figure/tau_g-hist.png")
plt.close()

"figure/tau_g-hist.png"
# The gas depletion timescale \tau_g:2 ends here

# Constant t_{sf}
# The observed ages of galactic discs are $tsf≈ 12$ Gyr[cite:@knoxSurveyCoolWhite1999a], so assuming an approximation of $tsf=12$ Gyr, the $\overline{SFR_{del}}$ can be calcuated, from the equation (\ref{eq:av_SFR M*}).


# After that the equation of ratio

# \begin{equation} \label{eq:ratio}                                        \frac{\overline{SFRdel}}{SFR0,del}=\frac{e^x-x-1}{x^2}
# \end{equation}

# can be solved numerically for x and using the equations (\Ref{eq:SFR}) and (\Ref{eq:av_SFR-x}) the $Adel$ and of each galaxy are found.


# [[file:paper.org::*Constant t_{sf}][Constant t_{sf}:1]]
dc["SFR_UNGC", "SFR_ratio", "StellarMass"].info()
# Constant t_{sf}:1 ends here



# #+RESULTS:
# : <QTable length=518>
# :     name     dtype      unit                       description                     class   n_bad
# : ----------- ------- ------------ ------------------------------------------------ -------- -----
# :    SFR_UNGC float64 solMass / yr Average Star Formation Rate, from the Ha and FUV Quantity     0
# :   SFR_ratio float64                                                               Quantity     1
# : StellarMass float64      solMass                                                  Quantity     1


# [[file:paper.org::*Constant t_{sf}][Constant t_{sf}:2]]
ratio_array = np.array(dc["SFR_ratio"])
sfr_array = np.array(dc["SFR_UNGC"])
mass_array = np.array(dc["StellarMass"])
av_array = np.array(dc["av_SFR_theor"])

tsf = t_sf/u.yr
x2 = np.empty(len(dc))
# Constant t_{sf}:2 ends here

# Newton


# [[file:paper.org::*Newton][Newton:1]]
for i in range(len(dc)-1):
    ratio = ratio_array[i]
    mass = mass_array[i]
    sfr = sfr_array[i]
    def f(x):
        return (-sfr + zeta*mass*x**2/(np.exp(x)-1-x)/tsf )  # only one real root

    def f_prime(x):
        return -zeta*mass*(x*(np.exp(x)*(x-2)+x+2)/(np.exp(x)-x-1)**2)/tsf

    sol = optimize.root_scalar(f, bracket=[0, 4], x0 = 3.4, fprime = f_prime, method="newton")
    x2[i] = sol.root

dc["x_n"] = x2
dc["A_n"] =MaskedColumn(dc["SFR_UNGC"]*t_sf.to(u.yr)*np.exp(dc["x_n"])/(dc["x_n"]**2))
# A_n = nan where inf

dc["x_n"] = np.where(np.isinf(dc["x_n"]), np.nan, dc["x_n"])
dc["x_n"] = np.where(np.isnan(dc["x_n"]), 1.66518, dc["x_n"])
dc["A_n"] = np.where(np.isinf(dc["A_n"]), np.nan, dc["A_n"])
# Newton:1 ends here



# #+RESULTS:
# : /home/dp/.local/lib/python3.10/site-packages/astropy/units/quantity.py:671: RuntimeWarning: divide by zero encountered in divide


# [[file:paper.org::*Newton][Newton:2]]
print(dc["x_n", "A_n"].info("stats"))
# Newton:2 ends here

# fsolve


# [[file:paper.org::*fsolve][fsolve:1]]
from scipy.optimize import fsolve
# Example loop
x = np.ma.empty(len(dc))
A = np.ma.empty(len(dc))
for i in range(len(dc)-1):

    ratio = ratio_array[i]
    mass = mass_array[i]
    sfr = sfr_array[i]
    av = av_array[i]

    def sfrx(z):
        x = z[0]
        A = z[1]

        f = np.zeros(2)
        f[0] = ratio - (np.exp(x) - x - 1) / x**2
        #f[0] = av*tsf - A * [1-(1-x)*np.exp(-x)]
        f[1] = sfr - A*10**(-9) * x * tsf * np.exp(-x) / x
        return f

    # Solve the equation
    z = fsolve(sfrx, [3,1e+8])
    x[i] = z[0]
    A[i] = z[1]*10**9

    ## mask If sfr ratio or mass is nan
    if np.isnan(ratio) or np.isnan(mass):
        x[i] = np.nan
        A[i] = np.nan

dc["x_f"] = MaskedColumn(x, name = "x")
dc["A_f"] = MaskedColumn(A, name = "A", unit = u.solMass)
# fsolve:1 ends here



# #+RESULTS:
# : /tmp/babel-etXArC/python-M39CsK:23: RuntimeWarning: The iteration is not making good progress, as measured by the
# :   improvement from the last ten iterations.


# [[file:paper.org::*fsolve][fsolve:2]]
print(dc["x_f","A_f"].info(["attributes" ,"stats"]))
# fsolve:2 ends here

# For X


# [[file:paper.org::*For X][For X:1]]
hist(dc["x_f"], bins = "freedman")                   #
hist(dc["x_n"], bins = "freedman", alpha = 0.5)      #
plt.xlabel('x')                                      #
plt.ylabel("Number of galaxies")                     #
plt.title('Hist of x solved with fsolve and Newton') #
#show the labels                                     #
plt.legend(["fsolve", "Newton"])                     #
                                                     #
plt.savefig("figure/x-hist.png")                     #
plt.close()                                          #
                                                     #
"figure/x-hist.png"                                  #
# For X:1 ends here



# #+RESULTS:
# :results:
# [[file:figure/x-hist.png]]
# :end:


# [[file:paper.org::*For X][For X:2]]
print(dc["x_f","x_n"].info(["attributes" ,"stats"]))
# For X:2 ends here



# #+RESULTS:
# :results:
# <QTable length=518>
# name  dtype     class       mean    std     min      max
# ---- ------- ------------ ------- ------- -------- -------
#  x_f float64 MaskedColumn 1.16944 3.12235 -33.7994 11.7659
#  x_n float64       Column 1.20898 3.17842 -33.7994 11.7659
# None
# :end:



# [[file:paper.org::*For X][For X:3]]
plt.scatter(dc["x_f"], dc["x_n"])
plt.xlabel('$x_{fsolve}$')
plt.ylabel(r'$x_{Newton}$')
plt.title('scatter of $x_{Newton}$ and $x_{fsolve}$')
#savefig and print the file
plt.savefig("figure/x-scatter.png")
plt.close()

"figure/x-scatter.png"
# For X:3 ends here

# They have the same max and min

# Let's find the Galaxies (Name) with the min and max x_f and x_n


# [[file:paper.org::*They have the same max and min][They have the same max and min:1]]
print("MAX of x_n:", dc["Name"][np.argmax(dc["x_n"])])
print("MIN of x_n:", dc["Name"][np.argmin(dc["x_n"])])
print("MAX of x_f:", dc["Name"][np.argmax(dc["x_f"])])
print("MIN of x_f:", dc["Name"][np.argmin(dc["x_f"])])
# They have the same max and min:1 ends here



# #+RESULTS:
# : MAX of x_n: Maffei1
# : MIN of x_n: AGC124056
# : MAX of x_f: Maffei1
# : MIN of x_f: AGC124056

# What are those galaxies?


# [[file:paper.org::*They have the same max and min][They have the same max and min:2]]
print("MAX of SFR Ratio:", dc["Name"][np.argmax(dc["SFR_ratio"])])
print("MIN of SFR Ratio:", dc["Name"][np.argmin(dc["SFR_ratio"])])
# They have the same max and min:2 ends here

# Compare for A



# [[file:paper.org::*Compare for A][Compare for A:1]]
print(dc["A_f","A_n"].info(["attributes" ,"stats"]))
# Compare for A:1 ends here



# #+RESULTS:
# : <QTable length=518>
# : name  dtype    unit      class              mean                std                 min                  max
# : ---- ------- ------- -------------- ------------------- ------------------- -------------------- -------------------
# :  A_f float64 solMass MaskedQuantity 5.34269e+08 solMass 2.68206e+09 solMass -1.21983e+07 solMass 3.35987e+10 solMass
# :  A_n float64 solMass MaskedQuantity  1.0472e+12 solMass  2.1989e+13 solMass  2.32981e-10 solMass 4.99763e+14 solMass
# : None



# [[file:paper.org::*Compare for A][Compare for A:2]]
sns.displot(np.log10(dc["A_n"].value))
plt.xlabel('$\log(A_{del})$'+f'[{dc["A_n"].unit:latex}]')

plt.xlim(np.min(dc["A_f"]))
plt.ylabel("Number of galaxies")
plt.title('Histogram of $A_{del}^N$')
plt.tight_layout()
#save
plt.savefig("figure/A_n-hist.png")
plt.close()

"figure/A_n-hist.png"
# Compare for A:2 ends here



# #+RESULTS:
# [[file:figure/A_n-hist.png]]


# [[file:paper.org::*Compare for A][Compare for A:3]]
sns.displot(np.log10(dc["A_f"].value))
plt.xlabel('$\log(A_{del})$'+f'[{dc["A_n"].unit:latex}]')

plt.xlim(np.min(dc["A_f"]))
plt.ylabel("Number of galaxies")
plt.title('Histogram of $A_{del}^f$')
plt.tight_layout()
#save
plt.savefig("figure/A_f-hist.png")
plt.close()

"figure/A_f-hist.png"
# Compare for A:3 ends here

# Calculating and \tau of the galaxies


# [[file:paper.org::*Calculating and \tau of the galaxies][Calculating and \tau of the galaxies:1]]
dc['tau'] = t_sf/dc["x_f"]


print(dc[["tau"]].info("stats"))
# Calculating and \tau of the galaxies:1 ends here



# #+RESULTS:
# : /home/dp/.local/lib/python3.10/site-packages/astropy/units/quantity.py:671: RuntimeWarning: overflow encountered in divide
# : <QTable length=518>
# : name     mean         std         min          max     n_bad
# : ---- ------------ ----------- ------------ ----------- -----
# :  tau -9.26834 Gyr 308.395 Gyr -4750.87 Gyr 2412.26 Gyr     1
# : None


# [[file:paper.org::*Calculating and \tau of the galaxies][Calculating and \tau of the galaxies:2]]
plt.close("all")
sns.displot((dc["tau"].value))
#mean line
plt.axvline(-9.27, color = "black", linestyle = "--", alpha = 0.8)
#mean line
#plt.axvline((np.mean(dc["tau"].value)), color = "black", linestyle = "--", alpha = 0.8)
#text next to mean line "Mean = mean.value"
#plt.text((np.mean(dc["tau"].value))*1.1, plt.ylim()[1]-4, r"Mean = {:.2f}".format(np.mean(dc["tau"])), color = "blue")
plt.text(-9.2, plt.ylim()[1]-4, r"Mean = {:.2f}".format(-9.27), color = "blue")

#text next to mean line "Mean = mean.value"
#plt.text(np.log10(6.3)*1.1, 65, r"Mean = 6.29 Gyr", color = "blue")

plt.xlabel(r'$\tau$ '+"[{}]".format(dc["tau"].unit.to_string("latex")))
plt.ylabel("Number of galaxies")
plt.title(r'Histogram of $\tau$ [Gyr]')
plt.tight_layout()
plt.xlim(-40,60)
#save
plt.savefig("figure/tau-hist.png")
plt.close()

"figure/tau-hist.png"
# Calculating and \tau of the galaxies:2 ends here

# TODO Add zoom and theoretical lines


# [[file:paper.org::*Add zoom and theoretical lines][Add zoom and theoretical lines:1]]
tmp = dc.to_pandas()
fig = px.scatter(tmp, "tau", "A_f", hover_data = ["Name", "TType"], log_y = False)
fig.show()
# Add zoom and theoretical lines:1 ends here



# #+RESULTS:
# : Gtk-[1;32mMessage[0m: [34m00:08:33.411[0m: Failed to load module "canberra-gtk-module"
# : Gtk-[1;32mMessage[0m: [34m00:08:33.414[0m: Failed to load module "canberra-gtk-module"
# : Gtk-[1;32mMessage[0m: [34m00:08:34.657[0m: Failed to load module "canberra-gtk-module"
# : Gtk-[1;32mMessage[0m: [34m00:08:34.658[0m: Failed to load module "canberra-gtk-module"
# : ATTENTION: default value of option mesa_glthread overridden by environment.
# : [Parent 2, Main Thread] WARNING: Failed to enumerate devices of org.freedesktop.UPower: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown: org.freedesktop.DBus.Error.ServiceUnknown
# : : 'glib warning', file /home/runner/work/desktop/desktop/engine/toolkit/xre/nsSigHandlers.cpp:187
# :
# : ** (zen-alpha:2): [1;33mWARNING[0m **: [34m00:08:34.797[0m: Failed to enumerate devices of org.freedesktop.UPower: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown: org.freedesktop.DBus.Error.ServiceUnknown


# [[file:paper.org::*Add zoom and theoretical lines][Add zoom and theoretical lines:2]]
colval = dc["StellarMass"]
colormap = np.log10(colval.data)
plt.scatter(dc["x_n"], dc["A_n"], c=colormap, cmap="viridis")
plt.title('Scatter plot of A vs x')
plt.xlabel("x")
plt.ylabel(r'$A_{del}$ ' + f'[{dc["A_f"].unit:latex}]')
plt.yscale("log")

plt.colorbar(label=r"$\log(M_*/{})$".format(colval.unit.to_string(format="unicode")))

# Define the function to plot
def func(x, a):
    return 1 / (1 - (1 + x) * np.exp(-x)) * a

# Generate x values
x_values = np.linspace(-20, 11, 1000)
# Define the value of a
a_value = dc["av_SFR_theor"].mean() * t_sf.to(u.yr)

# Calculate y values using the function
y_values = func(x_values, a_value.value)

# Plot the function
plt.plot(x_values, y_values, label=r"$\overline{SFR} \cdot t_{sf} \cdot \frac{1}{1-\left(1+x\right)e^{-x}}$", color="green")
plt.legend()

plt.savefig("figure/x-A.png")
plt.show()  # Display the plot
plt.close()

"figure/x-A.png"
# Add zoom and theoretical lines:2 ends here



# #+RESULTS:
# [[file:figure/x-A.png]]



# [[file:paper.org::*Add zoom and theoretical lines][Add zoom and theoretical lines:3]]
filename = "figure/tau-A"

plt.scatter(dc["tau"], dc["A_n"], c=np.log10(dc["StellarMass"]/dc["StellarMass"].unit), cmap = "viridis")
plt.title(r'Scatter plot of A vs $\tau$')
plt.xlabel(r'$\tau$ '+ f'[{dc["tau"].unit:latex}]')
plt.ylabel(r'$A_{del}$ '+ f'[{dc["A_f"].unit:latex}]')
plt.yscale("log")

plt.colorbar(label = "$\log(M_*/$" +"[{}])".format(dc["StellarMass"].unit.to_string("latex")))

plt.savefig(filename+".png")
plt.close()

filename + ".png"
# Add zoom and theoretical lines:3 ends here



# #+RESULTS:
# :results:
# [[file:figure/tau-A.png]]
# :end:



# [[file:paper.org::*Add zoom and theoretical lines][Add zoom and theoretical lines:4]]
filename = "figure/tau-A_zoom.png"

plt.scatter(dc["tau"], dc["A_f"], c=np.log10(dc["StellarMass"]/dc["StellarMass"].unit), cmap = "viridis")

plt.colorbar(label = "$\log(M_*/$" +"[{}])".format(dc["StellarMass"].unit.to_string("latex")))

plt.ylim(10**3, 10**11.7)
plt.xlim(-27,27)

def func(x, a):
    tt = 13.6
    return zeta * a*np.exp(tt/x)/(np.exp(tt/x)-1-tt/x)

x_values = np.linspace(plt.xlim()[0], plt.xlim()[1], len(dc["StellarMass"])+1)
#a_value = dc["StellarMass"].mean().value
a_value = 3.57e9

for i in range(0,6):
    y_values = func(x_values, (a_value) * 0.1**i )
    plt.plot(x_values, y_values, label = r"$M_* = $"+ "{:.1e}".format((a_value+5) * 0.2**i))


plt.title(r'Scatter plot of A vs $\tau$, zoom in')
plt.xlabel(r'$\tau$ '+ f'[{dc["tau"].unit:latex}]')
plt.ylabel(r'$A_{del}$ '+ f'[{dc["A_f"].unit:latex}]')
plt.yscale("log")

plt.legend(loc = "upper left")

plt.savefig(filename)
plt.close()

filename
# Add zoom and theoretical lines:4 ends here

# Scatter of SFR_0 vs x_n

# [[file:paper.org::*Scatter of SFR_0 vs x_n][Scatter of SFR_0 vs x_n:1]]
dc["av_SFR_theor"].mean()
# Scatter of SFR_0 vs x_n:1 ends here



# #+RESULTS:
# : <Quantity nan solMass / yr>


# [[file:paper.org::*Scatter of SFR_0 vs x_n][Scatter of SFR_0 vs x_n:2]]
plt.scatter( x=dc["x_n"], y=dc["SFR_UNGC"], c = np.log10(dc["av_SFR_theor"].value), cmap = "viridis")
plt.colorbar(label = r"$\overline{SFR}$ "+ "[{}]".format(dc["av_SFR_theor"].unit.to_string("latex")))


def func(x, a):
    return  a * x**2/(np.exp(x) - x - 1)

x_values = np.linspace(0.01, 10, 1000)
#a_value = dc["av_SFR_theor"].mean().value
a_value = 0.342195

for i in range(0,6):
    y_values = func(x_values, (a_value+5) * 0.2**i )
    plt.plot(x_values, y_values, label = r"$\overline{SFR} = $"+ r"{:.2f}".format((a_value+5) * 0.2**i))

plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.5*10**-3,10)
plt.xlabel(r"x")
plt.ylabel(r"$SFR_{UNGC}$"+" [{}]".format(dc["SFR_UNGC"].unit.to_string("latex")))
plt.tight_layout()
plt.savefig("figure/SFR-x.png")
plt.close()
"figure/SFR-x.png"
# Scatter of SFR_0 vs x_n:2 ends here



# #+RESULTS:
# [[file:figure/SFR-x.png]]



# [[file:paper.org::*Scatter of SFR_0 vs x_n][Scatter of SFR_0 vs x_n:3]]
plt.scatter( x=dc["x_n"], y=dc["A_n"]/dc["StellarMass"]/zeta, label = "Data")


x_values = np.linspace(plt.xlim()[0], plt.xlim()[1], 1000)

def func(x):
    return np.exp(x)/(np.exp(x)-x-1)

plt.plot(x_values, func(x_values), label = r"$\frac{\exp{x}}{\exp{x}-1-x}$")

plt.legend()
plt.xlabel(r"x")
plt.ylabel(r"$A/zM_*$")
plt.yscale("log")
plt.tight_layout()
fname = "figure/AzM_*-x.png"
plt.savefig(fname)
plt.close()
fname
# Scatter of SFR_0 vs x_n:3 ends here

# PROJ The relations of the Masses
# Since the aim of the paper is to find the SFR lets first understand and calculate the masses of the galaxies and see if we can find any relation with the SFR.


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:1]]
dff = dc.to_pandas()
# Assuming df is your DataFrame
# Step 1: Identify non-numeric columns
non_numeric_columns = dff.select_dtypes(exclude=['float', 'int']).columns

# Step 2: Drop non-numeric columns or handle them appropriately
df_numeric = dff.drop(columns=non_numeric_columns)
# Step 3: Replace NaN values with zeros or other appropriate values

df_numeric = df_numeric.loc[:, ~df_numeric.columns.str.startswith('e_')]

df_numeric = df_numeric.loc[:, ~df_numeric.columns.str.startswith('Name')]
df_numeric = df_numeric.loc[:, ~df_numeric.columns.str.startswith('Coord')]
df_numeric = df_numeric.loc[:, ~df_numeric.columns.str.startswith('log')]
# Step 4: Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

plt.close("all")
# The relations of the Masses:1 ends here



# #+RESULTS:


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:2]]
# Plot heatmap using seaborn
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# Set x-axis tick labels
plt.xticks(ticks=np.arange(0.5, len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90)

# Set y-axis tick labels
plt.yticks(ticks=np.arange(0.5, len(correlation_matrix.index)), labels=correlation_matrix.index, rotation=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("figure/heatmap")
plt.close("all")
# The relations of the Masses:2 ends here



# #+RESULTS:


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:3]]
clustermap = sns.clustermap(correlation_matrix, cmap='coolwarm', fmt=".2f", linewidths=0.5,
                             row_cluster=False, col_cluster=False)

ax = clustermap.ax_heatmap

# Set x-axis tick labels
ax.set_xticks(np.arange(0.5, len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=90)

# Set y-axis tick labels
ax.set_yticks(np.arange(0.5, len(correlation_matrix.index)))
ax.set_yticklabels(correlation_matrix.index, rotation=0)

plt.tight_layout()

plt.savefig("figure/clustermap")
plt.close("all")
# The relations of the Masses:3 ends here



# #+RESULTS:


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:4]]
fig = px.imshow(correlation_matrix, text_auto = True, color_continuous_scale='RdBu_r')
fig.show()
fig.write_html("figure/correlation.html")
# The relations of the Masses:4 ends here



# #+RESULTS:
# : Gtk-[1;32mMessage[0m: [34m00:08:46.739[0m: Failed to load module "canberra-gtk-module"
# : Gtk-[1;32mMessage[0m: [34m00:08:46.741[0m: Failed to load module "canberra-gtk-module"

# Pairplot with StellarMass, MHI, SFR_UNGC and av_SFR, M26


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:5]]
#PairGrid with StellarMass, MHI, SFR_UNGC and av_SFR_theor, M26
#log scale axes

sns.pairplot(correlation_matrix, vars=["StellarMass", "MHI", "SFR_UNGC", "av_SFR_theor", "M26"], kind="reg", diag_kind="kde")
plt.tight_layout()
plt.savefig("figure/pairplot")
plt.close("all")
print(correlation_matrix[["StellarMass", "MHI", "SFR_UNGC", "av_SFR_theor", "M26", "M_g", "tau", "A_f"]].corr())
# The relations of the Masses:5 ends here



# #+RESULTS:
# :               StellarMass       MHI  SFR_UNGC  av_SFR_theor       M26       M_g       tau       A_f
# : StellarMass      1.000000  0.895269  0.911743      1.000000  0.981763  0.895269 -0.069947  0.932965
# : MHI              0.895269  1.000000  0.975992      0.895269  0.911023  1.000000 -0.095762  0.764614
# : SFR_UNGC         0.911743  0.975992  1.000000      0.911743  0.925263  0.975992 -0.126696  0.769262
# : av_SFR_theor     1.000000  0.895269  0.911743      1.000000  0.981763  0.895269 -0.069947  0.932965
# : M26              0.981763  0.911023  0.925263      0.981763  1.000000  0.911023 -0.076677  0.900513
# : M_g              0.895269  1.000000  0.975992      0.895269  0.911023  1.000000 -0.095762  0.764614
# : tau             -0.069947 -0.095762 -0.126696     -0.069947 -0.076677 -0.095762  1.000000 -0.043090
# : A_f              0.932965  0.764614  0.769262      0.932965  0.900513  0.764614 -0.043090  1.000000


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:6]]
#define a new dataframe
df_log = pd.DataFrame()

temp = pd.DataFrame()
for i in ["StellarMass", "MHI", "SFR_UNGC", "av_SFR_theor", "M26", "tau", "A_f", "M_g", "tau_g", "BarMass"]:
    temp [i] =dff[i]
    temp[i].loc[temp[i] == 0] = np.nan
    df_log[i] = np.log(temp[i].dropna())

for i in ["TType", "Tdw1", "Tdw2"]:
    df_log[i] = dff[i]
# The relations of the Masses:6 ends here

# [[file:paper.org::*The relations of the Masses][The relations of the Masses:7]]
#PairGrid with StellarMass, MHI, SFR_UNGC and av_SFR_theor, M26
#log scale axes

sns.pairplot(df_log, vars=["StellarMass", "MHI", "SFR_UNGC", "M26", "BarMass", "A_f"], kind="reg", diag_kind="kde")
plt.tight_layout()
plt.savefig("figure/pairplot.png")
plt.close("all")

#print file
"figure/pairplot.png"
# The relations of the Masses:7 ends here



# #+RESULTS:


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:8]]
print(df_log[["MHI", "SFR_UNGC", "av_SFR_theor", "M26", "M_g", "tau", "A_f", "tau_g"]].corr())
# The relations of the Masses:8 ends here



# #+RESULTS:
# :                    MHI  SFR_UNGC  av_SFR_theor       M26       M_g       tau       A_f     tau_g
# : MHI           1.000000  0.873227      0.807654  0.864241  1.000000 -0.017130  0.028412 -0.067180
# : SFR_UNGC      0.873227  1.000000      0.867858  0.860180  0.873227 -0.020644  0.012013 -0.544876
# : av_SFR_theor  0.807654  0.867858      1.000000  0.895931  0.807654 -0.334168  0.056187 -0.430786
# : M26           0.864241  0.860180      0.895931  1.000000  0.864241 -0.188427  0.063607 -0.280310
# : M_g           1.000000  0.873227      0.807654  0.864241  1.000000 -0.017130  0.028412 -0.067180
# : tau          -0.017130 -0.020644     -0.334168 -0.188427 -0.017130  1.000000  0.065413  0.031146
# : A_f           0.028412  0.012013      0.056187  0.063607  0.028412  0.065413  1.000000  0.023517
# : tau_g        -0.067180 -0.544876     -0.430786 -0.280310 -0.067180  0.031146  0.023517  1.000000


# [[file:paper.org::*The relations of the Masses][The relations of the Masses:9]]
#heatmap of masses and luminosities
sns.heatmap(df_log[["MHI", "SFR_UNGC", "av_SFR_theor", "M26", "M_g", "BarMass","tau", "A_f", "tau_g"]].corr(), annot = True)
plt.tight_layout()
fname = "figure/heatmap_mass_sfr_a_t.png"
plt.savefig(fname)
plt.close()

fname
# The relations of the Masses:9 ends here

# Luminosities

# Let's find some relations between the magnitudes
# We will use the mags of table 3


# [[file:paper.org::*Luminosities][Luminosities:1]]
#find all the columns of df with mag_

mag_cols = [col for col in dff.columns if "mag_" in col]

#drop the but the ones with l_mag
mag_cols = [col for col in mag_cols if "l_mag" not in col]
mag_cols = [col for col in mag_cols if "e_mag" not in col]

#create a new dataframe
df_mags = dff[mag_cols]

#heatmap
filename = "figure/heatmap_mags.png"
sns.heatmap(df_mags.corr(), annot = True, fmt = ".3g")
plt.tight_layout()
plt.savefig(filename)
plt.close("all")

#print the file name
print(filename)
# Luminosities:1 ends here

# TODO Luminosity and Masses


# [[file:paper.org::*Luminosity and Masses][Luminosity and Masses:1]]

# Luminosity and Masses:1 ends here
