

































#| echo: false
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import astropy
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, QTable
import astropy.units as u
from astropy.visualization import quantity_support, hist
import scipy
from scipy.optimize import curve_fit
#quantity_support()
from astropy.stats import sigma_clip, SigmaClip
from astropy.modeling import models, fitting
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import glob
import os
from tabulate import tabulate
from IPython.display import Markdown, display

plt.style.use('ggplot')
pd.set_option('display.float_format', lambda x: '%.f' % x)

dt = QTable(ascii.read("../tables/inner_join.ecsv"), masked=True)

inner = dt
outer = QTable(ascii.read("../tables/outer_join.ecsv"), masked=True)
hec_not_lvg = QTable(ascii.read("../tables/HEC_not_LVG_join.ecsv"), masked=True)
lvg_not_hec = QTable(ascii.read("../tables/LVG_not_HEC_join.ecsv"), masked=True)


lvg = QTable(ascii.read("../tables/final_table.ecsv"), masked=True)
hec = QTable(ascii.read("../tables/HECATE_LCV.ecsv"), masked=True)






























































































#| output: asis
# Creating a dataframe with the lengths of each table
data = {
    "Table": ["Inner join", "Outer join", "LVG", "HECATE", "Unique galaxies in LVG", "Unique Galaxies in Hecate"],
    "Number of galaxies": [len(dt), len(outer), len(lvg), len(hec), len(lvg_not_hec), len(hec_not_lvg)]
}

df_lengths = pd.DataFrame(data)

# Pretty print the dataframe in markdown format
df_lengths_md = df_lengths.to_markdown(index=False)
print(df_lengths_md)




























def table_completeness(dataset_choice, column_name="Dis", description = "Distance", keep_nan = False):
    """
    Creates a table of counts and percentages for each bin based on the chosen dataset (LVG or HEC).
    
    Parameters:
    - dataset_choice: A string indicating which dataset to use ("lvg" or "hec")
    - column_name: The column name in the datasets to analyze (default is "Dis")
    - description: what to show in the title and the axis (default "Distance")
    """
    
    if dataset_choice == "lvg":
        main_df, unique_df, label = lvg, lvg_not_hec, "LVG"
    elif dataset_choice == "hec" and hec is not None and hec_not_lvg is not None:
        main_df, unique_df, label = hec, hec_not_lvg, "HEC"
    else:
        raise ValueError("Invalid dataset choice. Please choose 'lvg' or 'hec' with valid datasets.")

    # Remove NaN values before plotting
    if keep_nan == False:
        main_df = main_df[~np.isnan(main_df[column_name].data)]
        unique_df = unique_df[~np.isnan(unique_df[column_name].data)]
    else:
        main_df = main_df[column_name].data
        unique_df = unique_df[column_name].data
    counts, bin_edges = np.histogram(main_df[column_name].data, bins='auto')

    main_counts, _ = np.histogram(main_df[column_name].data, bins=bin_edges)
    unique_counts, _ = np.histogram(unique_df[column_name].data, bins=bin_edges)
    inner_counts, _ = np.histogram(inner[column_name].data, bins=bin_edges)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    table_data = []

    # Calculate percentages in a vectorized way
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero safely
        unique_pct = np.divide(unique_counts, main_counts, out=np.zeros_like(unique_counts, dtype=float), where=main_counts!=0) * 100
        inner_pct = np.divide(inner_counts, main_counts, out=np.zeros_like(inner_counts, dtype=float), where=main_counts!=0) * 100

    # Construct the DataFrame in a vectorized manner
    table_df = pd.DataFrame({
        "Bin Start": np.round(bin_edges[:-1]),
        "Bin End": np.round(bin_edges[1:]),
        "Main Counts": main_counts,
        "Unique Counts": unique_counts,
        "Unique %": unique_pct,
        "Inner Counts": inner_counts,
        "Inner %": inner_pct
    })
    return table_df



#| fig-cap: "Histograms showing the Distance Completeness of the Catalogs"
#| layout-ncol: 2
#| fig-subcap: 
#| - "HECATE"
#| - "LVG"
#| label: fig-dis-comp

#HECATE

main = hec["D"].data
unique = hec_not_lvg["D"].data
counts, bin_edges = np.histogram(main)

main_counts, _ = np.histogram(main, bins=bin_edges)
unique_counts, _ = np.histogram(unique, bins=bin_edges)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.histplot(main, bins = bin_edges, kde = True, label = "HECATE", ax = ax1)
sns.histplot(unique, bins = bin_edges, kde = True, label = "Unique for HECATE", ax = ax1)
ax1.set_title("Completenes of Distance for HECATE")
ax1.tick_params(labelbottom=True)
unique_pct = (unique_counts/main_counts)*100 
###
plt.subplot(2,1,2)
sns.lineplot(x = bin_edges[1:], y = unique_pct, marker = "o", ax = ax2)
#changing ylables ticks
# Set y-axis ticks for ax2 (completeness percentage)
ax2.set_yticks(np.arange(min(unique_pct), max(unique_pct) + 1, 5.0))

# Format the y-tick labels to display as percentages
y_value = ['{:,.0f}%'.format(x) for x in ax2.get_yticks()]
ax2.set_yticklabels(y_value)
ax2.set(xlabel = "Distance [{}]".format(lvg["Dis"].unit), ylabel="Completeness %")
plt.show()

#LVG
main = lvg["Dis"].data
unique = lvg_not_hec["Dis"].data
counts, bin_edges = np.histogram(main)

main_counts, _ = np.histogram(main, bins=bin_edges)
unique_counts, _ = np.histogram(unique, bins=bin_edges)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.histplot(main, bins = bin_edges, kde = True, label = "LVG", ax = ax1)
sns.histplot(unique, bins = bin_edges, kde = True, label = "Unique for LVG", ax = ax1)
ax1.legend()
ax1.set_title("Completenes of Distance for LVG")
ax1.tick_params(labelbottom=True)
unique_pct = (unique_counts/main_counts)*100 
###
sns.lineplot(x = bin_edges[1:], y = unique_pct, marker = "o", ax = ax2)

#changing ylables ticks
# Set y-axis ticks for ax2 (completeness percentage)
ax2.set_yticks(np.arange(min(unique_pct), max(unique_pct) + 1, 5.0))

# Format the y-tick labels to display as percentages
y_value = ['{:,.0f}%'.format(x) for x in ax2.get_yticks()]
ax2.set_yticklabels(y_value)
ax2.set(xlabel = "Distance [{}]".format(lvg["Dis"].unit), ylabel="Completeness %")
plt.show()

plt.close()



#| fig-cap: "Histograms showing the Type Completeness of the Catalogs"
#| layout-ncol: 2
#| fig-subcap: 
#| - "HECATE"
#| - "LVG"
#| label: fig-type-comp
main = hec["T"].data
main = main[~np.isnan(main)]
unique = hec_not_lvg["T"].data
unique = unique[~np.isnan(unique)]
counts, bin_edges = np.histogram(main, bins = 12)

main_counts, _ = np.histogram(main, bins=bin_edges)
unique_counts, _ = np.histogram(unique, bins=bin_edges)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.histplot(main, bins = bin_edges, kde = True, label = "HECATE",ax = ax1)
sns.histplot(unique, bins = bin_edges, kde = True, label = "Unique for HECATE",ax = ax1)
ax1.legend()
ax1.set_title("Completenes of Types for HECATE")
ax1.tick_params(labelbottom=True)
unique_pct = (unique_counts/main_counts)*100 
###

sns.lineplot(x = bin_edges[1:], y = unique_pct, marker = "o",ax = ax2)

#changing ylables ticks
# Set y-axis ticks for ax2 (completeness percentage)
ax2.set_yticks(np.arange(min(unique_pct), max(unique_pct) + 1, 5.0))

# Format the y-tick labels to display as percentages
y_value = ['{:,.0f}%'.format(x) for x in ax2.get_yticks()]
ax2.set_yticklabels(y_value)

ax2.set(xlabel = "Types of galaxies", ylabel="Completeness %")
plt.show()
######
#LVG
main = lvg["TType"].data
main = main[~np.isnan(main)]
unique = lvg_not_hec["TType"].data
unique = unique[~np.isnan(unique)]
counts, bin_edges = np.histogram(main)

main_counts, _ = np.histogram(main, bins=bin_edges)
unique_counts, _ = np.histogram(unique, bins=bin_edges)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.histplot(main, bins = bin_edges, kde = True, label = "LVG", ax = ax1)
sns.histplot(unique, bins = bin_edges, kde = True, label = "Unique for LVG", ax = ax1)
ax1.legend()
ax1.set_title("Completenes of Types for LVG")
ax1.tick_params(labelbottom=True)
unique_pct = (unique_counts/main_counts)*100 

sns.lineplot(x = bin_edges[1:] ,y = unique_pct, marker = "o", ax = ax2)

#changing ylables ticks
# Set y-axis ticks for ax2 (completeness percentage)
ax2.set_yticks(np.arange(min(unique_pct), max(unique_pct) + 1, 10.0))

# Format the y-tick labels to display as percentages
y_value = ['{:,.0f}%'.format(x) for x in ax2.get_yticks()]
ax2.set_yticklabels(y_value)

ax2.set(xlabel = "Types of galaxies", ylabel="Completeness %")
plt.show()






















def compare_data(x, y, sigma = False):
    """
    Performs a linear comparison between two datasets.
    
    This function fits a linear model to the data, calculates the slope and intercept of the fitted line,
    and computes the R-squared value and Pearson correlation coefficient to assess the fit quality.
    
    Parameters:
    - x (array-like): The independent variable data.
    - y (array-like): The dependent variable data.
    - unc (array-like with units, optional): The uncertainties associated with the a variable data. Default is None.

    Returns:
    tuple: A tuple containing the following elements:
        - slope (float): The slope of the fitted linear model.
        - intercept (float): The intercept of the fitted linear model.
        - r2 (float): The R-squared value, indicating the proportion of variance explained by the linear model.
        - corr (float): The Pearson correlation coefficient, measuring the linear correlation between x and y.
    """
    try:
        x_data = np.ma.array(x.value, mask=np.isnan(x.value))
        y_data = np.ma.array(y.value, mask=np.isnan(y.value))
        # initialize a linear fitter
        fit = fitting.LinearLSQFitter()

        # initialize a linear model
        line_init = models.Linear1D()

        # fit the data with the fitter
        # check if sigma
        if sigma is False:
            fitted_line = fit(line_init, x_data, y_data)
            outliers = None
        else:
            or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter =3, sigma=3)
            fitted_line, outliers = or_fit(line_init, x_data, y_data)
        slope = fitted_line.slope.value
        intercept = fitted_line.intercept.value

        # Predict values using the fitted model
        y_pred = fitted_line(x_data)

        # Remove NaN values
        mask = ~np.isnan(y_data)
        if outliers is not None:
            mask = ~np.isnan(y_data) & ~outliers
        y_data_clean = y_data[mask]
        y_pred_clean = y_pred[mask]

        # Calculate R-squared
        r2 = r2_score(y_data_clean, y_pred_clean)

        # Calculate Pearson correlation coefficient
        corr = np.sqrt(np.abs(r2))

        return slope, intercept, r2, corr, outliers
    except Exception:
        return 0,0,0,0,None








# Define a function that creates a pairplot with correlation coefficients
def pairplot_with_correlation(df, cmap="coolwarm", font_size=12, font_style='italic', font_weight='bold', box_color='white'):
    """
    Creates a Seaborn pairplot with correlation coefficients displayed on the upper triangle.
    
    Parameters:
    df : pandas.DataFrame
        The dataframe containing the data for the pairplot.
    cmap : str, optional
        The colormap to use for coloring the correlation coefficients (default is 'coolwarm_r').
    font_size : int, optional
        The font size for the correlation coefficient annotations (default is 12).
    font_style : str, optional
        The font style for the correlation coefficient annotations (default is 'italic').
    font_weight : str, optional
        The font weight for the correlation coefficient annotations (default is 'bold').
    box_color : str, optional
        The background color of the box around the correlation coefficients (default is 'white').
    
    Returns:
    A Seaborn pairplot with correlation coefficients displayed in a heatmap style.
    """
    plt.close()
    # Create Pairplot
    pairplot = sns.pairplot(df, plot_kws={"color": "green"}, diag_kws = {"color" : "green"})

    # Calculate Correlations
    corr_matrix = df.corr()

    # Define a colormap
    colormap = plt.get_cmap(cmap)

    # Overlay Correlation Coefficients with customized annotations
    for i, j in zip(*np.triu_indices_from(corr_matrix, 1)):
        # Get the correlation value
        corr_value = corr_matrix.iloc[i, j]

        # Get a color from the colormap based on the correlation value
        color = colormap((corr_value + 1) / 2)  # Normalize between 0 and 1

        # Annotate the plot with the correlation value, using a box, bold, and italic
        pairplot.axes[i, j].annotate(f"{corr_value:.2f}", 
                                     xy=(0.5, 0.5), 
                                     xycoords='axes fraction',
                                     ha='center', 
                                     va='center', 
                                     fontsize=font_size,  # Font size
                                     fontstyle=font_style,  # Font style
                                     fontweight=font_weight,  # Font weight
                                     bbox=dict(facecolor=box_color, edgecolor=color, boxstyle='round,pad=0.5'),  # Box around the text
                                     color=color)  # Use heatmap color for text

    # Show the plot
    plt.show()












def relative_diff(hec, lvg):
    """
    Calculate the relative difference between two sets of values.

    Parameters:
    hec (numpy.ndarray): An array of values representing the HEC dataset.
    lvg (numpy.ndarray): An array of values representing the LVG dataset.

    Returns:
    numpy.ndarray: An array of relative differences between the HEC and LVG datasets.
    The relative difference is calculated as ((HEC - LVG) * 100) / |HEC|.
    Any infinite values in the result are replaced with NaN.
    """
    diff = (hec - lvg) * 100 / np.abs(hec)
    diff[np.isinf(diff)] = np.nan
    return diff











#| label: fig-coord-compare 
#| layout-align: center
#| fig-cap: "Comparison of the Distances"

plt.close()
fig, ax = plt.subplots()
sns.regplot(x = dt["Dis"].data, y = dt["D"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
ax.errorbar(dt["Dis"].data, dt["D"].data, yerr = dt["E_D"].data, fmt='none', capsize=2, zorder=1, color = "blue", alpha = 0.3)
results = scipy.stats.linregress(x = dt["Dis"], y=dt["D"])
plt.xlabel("$D_{LVG}$"+" [{}]".format(dt["Dis"].unit))
plt.ylabel("$D_{HEC}$"+" [{}]".format(dt["D"].unit))
plt.legend(["Data",
            "$D_{HEC}=$" +f"{results.slope:.2f}"+"$\cdot D_{LVG}+$"+
                f"$({results.intercept:+.2f})$"+
            f"\n $R^2 = {results.rvalue**2*100:.0f}\%$"])
plt.tight_layout()
plt.show()















#| label: fig-vel-compare
#| layout-align: center 
#| fig-cap: "Comparison of the Radial Velocities"
plt.close()
fig, ax = plt.subplots()
sns.regplot(x = dt["RVel"].data, y = dt["V"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))

ax.errorbar(dt["RVel"].data, dt["V"].data, yerr = dt["E_V"].data, fmt='none', capsize=2, zorder=1, color = "blue")
results = scipy.stats.linregress(x = dt["RVel"], y=dt["V"])
plt.xlabel("$V_{LVG}$"+" [{}]".format(dt["RVel"].unit))
plt.ylabel("$V_{HEC}$"+" [{}]".format(dt["V"].unit))
plt.legend(["Data",
            "$V_{HEC}=$" +f"{results.slope:.2f}"+"$\cdot V_{LVG}+$"+
            f"$({results.intercept:+.2f})$"+
            f"\n $R^2 = {results.rvalue**2*100:.0f}\%$"])
plt.show()






vel_data = dt[["RVel", "V", "VLG", "cz", "V_VIR"]].to_pandas()



#| label: fig-vel-pairplot
#| fig-cap: "The correlation Matrix of the Velocities. The lower left triangle is composed of the scatter plots of the various velocities, the diagonal shows their distrubution and the upper right triangle shows the correlations of the Velocities" 
pairplot_with_correlation(vel_data)










dt["INCL"].mask = np.isnan(dt["INCL"])
















#| label: fig-types-compare
#| layout-ncol: 2
#| layout-align: center
#| fig-cap: "Comparison of the Types of galaxies"
#| fig-subcap: 
#| - "All the galaxies, without the correction" 
#| - "Comparison with the correction $|{T_{HECATE}-T_{LVG}}|<σ_{Τ_{LVG}}$"

plt.close()
fig, ax = plt.subplots()
sns.regplot(x = dt["TType"].data, y = dt["T"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))

ax.errorbar(dt["TType"].data, dt["T"].data, yerr = dt["E_T"].data, fmt='none', capsize=2, zorder=1, color = "blue")
temp = dt[["TType", "T"]].to_pandas().dropna()
temp["TType"] = temp["TType"].astype("float64")
temp["T"] = temp["T"].astype("float64")

results = scipy.stats.linregress(x = temp["TType"],
                                y = temp["T"])
plt.xlabel("$Type_{LVG}$")
plt.ylabel("$Type_{HEC}$")
plt.legend(["Data",
            "$T_{HEC}=$" +f"{results.slope:.2f}"+"$\cdot T_{LVG}+$"+
            f"$({results.intercept:+.2f})$"+
            f"\n $R^2 = {results.rvalue**2*100:.0f}\%$"])
plt.show()

types = dt[["T", "TType","E_T"]]
types["diff"] = dt["T"]-dt["TType"]
types = types[np.abs(types["diff"])<dt["TType"].std()]


plt.close()
fig, ax = plt.subplots()
sns.regplot(x = types["TType"].data, y = types["T"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))

ax.errorbar(types["TType"].data, types["T"].data, yerr = types["E_T"].data, fmt='none', capsize=2, zorder=1, color = "blue")
results = scipy.stats.linregress(x= types["TType"].data,
                                y = types["T"].data)
plt.xlabel("$Type_{LVG}$")
plt.ylabel("$Type_{HEC}$")
plt.legend(["Data",
            "$T_{HEC}=$" +f"{results.slope:.2f}"+"$\cdot T_{LVG}+$"+
            f"$({results.intercept:+.2f})$"+
            f"\n $R^2 = {results.rvalue**2*100:.0f}\%$"])
plt.show()

















#| label: fig-incl-compare
#| layout-ncol: 2
#| layout-align: center
#| fig-cap: "Comparison of the Inclination of the galaxies"
#| fig-subcap: 
#| - "Distribution of the Inclination of the galaxies" 
#| - "Distribution of the Percentage Change" 
plt.close()
temp = dt[["inc", "INCL"]].to_pandas()

sns.histplot(temp["INCL"], kde = True, label = "HECATE")
sns.histplot(temp["inc"], kde = True, label = "LVG")
plt.legend()
plt.xlabel("Inclination of galaxies [deg]")
plt.show()

temp["Percentage Change [%]"] = (-temp["inc"] + temp["INCL"])/temp["INCL"]
temp.loc[np.isinf(temp["Percentage Change [%]"]), "Percentage Change [%]"] = np.nan
sns.histplot(temp["Percentage Change [%]"], kde = True)
plt.show()



# Display specific statistics, excluding the 25th and 75th percentiles
summary_stats = temp.describe().drop(['25%', '75%'])
summary_stats







#| label: fig-axis-compare
#| layout-ncol: 2
#| layout-align: center
#| fig-cap: "Comparison of the Major Axises of the galaxies"
#| fig-subcap: 
#| - "Linear scale" 
#| - "$log_{10}$ scale"
plt.close()
fig, ax = plt.subplots()
sns.regplot(x = dt["a26_1"].data, y = dt["R1"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
temp = dt[["a26_1","R1"]].to_pandas().dropna()
results = scipy.stats.linregress(x = temp["a26_1"], y = temp["R1"])

plt.xlabel("(Major angular diameter$)_{LVG}$")
plt.ylabel("(Semi major axis$)_{HEC}$")
plt.legend(["Data",
            "$R1_{HEC}=$" +f"{results.slope:.2f}"+"$\cdot a_{26,LVG}$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()
dt["log_a26"] = np.log10(dt["a26_1"].data)
dt["log_R1"] = np.log10(dt["R1"].data)
plt.close()
fig, ax = plt.subplots()
p = sns.regplot(x = dt["log_a26"].data, y = dt["log_R1"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))

temp = dt[["log_a26","log_R1"]].to_pandas().dropna()
results = scipy.stats.linregress(x = temp["log_a26"], y = temp["log_R1"])
plt.xlabel("(Major angular diameter$)_{LVG}$"+" [{}]".format(dt["a26_1"].unit))
plt.ylabel("(Semi major axis$)_{HEC}$"+" [{}]".format(dt["R1"].unit))
plt.legend(["Data",
            "$log_{10}(R1_{HEC})=$" +f"{results.slope:.2f}"+"$\cdot log_{10}(a_{26,LVG})$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.0f}\%$"])
plt.show()




















#| echo: false
#| output: asis 
#| warning: false
logKLum_corr = round(compare_data(dt["logKLum"], dt["logL_K"], sigma = True)[3], 3)








#| label: fig-klum-compare
#| layout-ncol: 2 
#| fig-cap: "Comparison of the $L_K$ of the galaxies."
#| fig-subcap: 
#| - "Linear Regression with free paramaters, $y=ax+b$ "
#| - "Linear Regrassion $y=x$ "
 
plt.close()
fig, ax = plt.subplots()
sns.regplot(x = dt["logKLum"], y = dt["logL_K"],
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
temp = dt[["logKLum","logL_K"]].to_pandas().dropna()
results = scipy.stats.linregress(x = temp["logKLum"], y = temp["logL_K"])
plt.xlabel("log($L_{K, LVG} $"+"/{})".format(dt["KLum"].unit.to_string("latex")))
plt.ylabel("log($L_{K, HECATE} $"+"/{})".format(dt["KLum"].unit.to_string("latex")))
plt.legend(["Data",
            "$log(L_{K,HEC})=$" +f"{results.slope:.2f}"+"$\cdot log(L_{K,LVG})$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()

data = dt[["logKLum", "logL_K"]]
data = dt[~np.isnan(data["logL_K"])]
logL_K_LVG = data['logKLum']  # Replace with actual column names
logL_K_HECATE = data['logL_K']  # Replace with actual column names


# Plot the data and both fits
plt.figure(figsize=(8, 6))
plt.scatter(logL_K_LVG, logL_K_HECATE, color='blue', label='Data', alpha =0.5, edgecolors="white")

# Plot free fit (with intercept)
plt.plot(logL_K_LVG, results.slope * logL_K_LVG + results.intercept, color='green', label='Free Fit: $log(L_{K,HEC})=$'+ f'{results.slope:.2f}'+ "$\cdot log(L_{K,LVG})$"+ f'{results.intercept:.2f}'+f"\n $R^2 = {results.rvalue**2*100:.2f}\%$")

slope_forced = 1
# Plot forced fit (through the origin)
results_f = scipy.stats.linregress(x = logL_K_HECATE, y = slope_forced*logL_K_LVG)
plt.plot(logL_K_LVG, slope_forced * logL_K_LVG, color='red', label='Forced Fit: $log(L_{K,HEC})=$' +f' = {slope_forced:.2f}'"$\cdot log(L_{K,LVG})$"+f"\n $R^2 = {results_f.rvalue**2*100:.2f}\%$")


plt.xlabel("log($L_{K, LVG} $"+"/{})".format(dt["KLum"].unit.to_string("latex")))
plt.ylabel("log($L_{K, HECATE} $"+"/{})".format(dt["KLum"].unit.to_string("latex")))
plt.legend()
plt.grid(True)
plt.show()



















#| output: false 
#| warning: false
mag_B_corr = round(compare_data(dt["mag_B"], dt["BT"], sigma = True)[3], 3)
Kmag_corr = round(compare_data(dt["Kmag"], dt["K"], sigma = True)[3], 3)








#| label: fig-mag-compare
#| layout-ncol: 2
#| layout-align: center
#| fig-cap: "Comparison of the Magnitudes of the galaxies"
#| fig-subcap: 
#| - "$M_B$" 
#| - "$M_K$"
#| 
plt.close()
fig, ax = plt.subplots()
sns.regplot(x = dt["mag_B"].data, y = dt["BT"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
ax.errorbar(dt["mag_B"].data, dt["BT"].data, xerr = dt["e_mag_B"].data, yerr = dt["E_BT"].data, fmt='none', capsize=2, zorder=1, color = "blue")
temp = dt[["mag_B","BT"]].to_pandas().dropna()
results = scipy.stats.linregress(x = temp["mag_B"], y = temp["BT"])
plt.xlabel("$M_{B,LVG}$"+" [{}]".format(dt["mag_B"].unit.to_string("latex")))
plt.ylabel("$M_{B,HECATE} $"+" [{}]".format(dt["BT"].unit.to_string("latex")))
plt.legend(["Data",
            "$M_{B,HEC}=$" +f"{results.slope:.2f}"+"$\cdot M_{B,LVG}$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()

#M_K
plt.close()
fig, ax = plt.subplots()
sns.regplot(x = dt["Kmag"].data, y = dt["K"].data,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
ax.errorbar(dt["Kmag"].data, dt["K"].data, yerr = dt["E_K"].data, fmt='none', capsize=2, zorder=1, color = "blue")
temp = dt[["Kmag","K"]].to_pandas().dropna()
results = scipy.stats.linregress(x = temp["Kmag"], y = temp["K"])
plt.xlabel("$M_{K,LVG}$"+" [{}]".format(dt["Kmag"].unit.to_string("latex")))
plt.ylabel("$M_{K,HECATE} $"+" [{}]".format(dt["K"].unit.to_string("latex")))
plt.legend(["Data",
            "$M_{K,HEC}=$" +f"{results.slope:.2f}"+"$\cdot M_{K,LVG}$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()






#| label: fig-kmag-compare
#| layout-ncol: 2 
#| fig-cap: "Comparison of the $L_K$ of the galaxies."
#| fig-subcap: 
#| - "Linear Regression with free paramaters, $y=ax+b$ "
#| - "Linear Regrassion $y=x$ "
 
plt.close()
fig, ax = plt.subplots()
temp = dt[["Kmag","K"]].to_pandas().dropna()
K_LVG = temp['Kmag']  # Replace with actual column names
K_HECATE = temp['K']  # Replace with actual column names


sns.regplot(x = K_LVG, y = K_HECATE,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
results = scipy.stats.linregress(x = temp["Kmag"], y = temp["K"])
r2_free = results.rvalue
plt.xlabel("$M_{K,LVG}$"+" [{}]".format(dt["Kmag"].unit.to_string("latex")))
plt.ylabel("$M_{K,HECATE} $"+" [{}]".format(dt["K"].unit.to_string("latex")))
plt.legend(["Data",
            "$M_{K,HEC}=$" +f"{results.slope:.2f}"+"$\cdot M_{K,LVG}$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()

# Plot the data and both fits
plt.figure(figsize=(8, 6))
plt.scatter(K_LVG, K_HECATE, color='blue', label='Data', alpha =0.5, edgecolors="white")

# Plot free fit (with intercept)
plt.plot(K_LVG, results.slope * K_LVG + results.intercept, color='green', label='Free Fit: $M_{K,HEC}=$'+ f'{results.slope:.2f}'+ "$\cdot M_{K,LVG}$"+ f'{results.intercept:.2f}'+f"\n $R^2 = {results.rvalue**2*100:.2f}\%$")

slope_forced = 1
# Plot forced fit (through the origin)
results_f = pearsonr(x = K_HECATE, y = slope_forced*K_LVG)
plt.plot(K_LVG, slope_forced * K_LVG, color='red', label='Forced Fit: $M_{K,HEC}=$' +f' = {slope_forced:.2f}'"$\cdot M_{K,LVG}$"+f"\n $R^2 = {results_f.statistic**2*100:.2f}\%$")


plt.xlabel("$M_{K,LVG}$"+" [{}]".format(dt["Kmag"].unit.to_string("latex")))
plt.ylabel("$M_{K,HECATE} $"+" [{}]".format(dt["K"].unit.to_string("latex")))
plt.legend()
plt.grid(True)
plt.show()










#| echo: false
# Extract the relevant SFR columns
sfr_columns = ["logSFR_TIR", "logSFR_FIR", "logSFR_60u", "logSFR_12u", "logSFR_22u", "logSFR_HEC", "logSFR_GSW", "logSFRFUV", "logSFRHa"]
sfr_data = dt[sfr_columns].to_pandas()

# Count the number of non-NaN cells for each column
non_nan_counts = sfr_data.notna().sum()















#| label: fig-sfr-pairplot
#| layout-align: center
#| fig-cap: "Comparison of the $SFR_{i}$ of the galaxies"
#| echo: false
sfr_data.drop("logSFR_GSW", axis=1, inplace=True)

# Example usage of the function
pairplot_with_correlation(sfr_data)










#| label: fig-sfr-lvg
#| layout-ncol: 2
#| layout-align: center
#| fig-cap: "Comparison of the $SFR_{FUV}-SFR_{Ha}$ of the galaxies"
#| fig-subcap: 
#| - "Linear scale" 
#| - "Decimal logarithmic scale"

##linear
plt.close()
fig, ax = plt.subplots()
temp = lvg[["SFRHa","SFRFUV"]].to_pandas().dropna()
x = temp["SFRHa"]
y = temp["SFRFUV"]
sns.regplot(x = x, y = y,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
results = scipy.stats.linregress(x = x, y = y)
plt.axvline(x=1e-3, color='green', linestyle='--', linewidth=2)
plt.xlabel("$SFR_{Ha}$"+" [{}]".format(dt["SFRHa"].unit.to_string("latex")))
plt.ylabel("$SFR_{FUV} $"+" [{}]".format(dt["SFRFUV"].unit.to_string("latex")))
plt.legend(["Data",
            "$SFR_{FUV}=$" +f"{results.slope:.2f}"+"$\cdot SFR_{Ha}$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()

##log

plt.close()

x = np.log10(temp["SFRHa"])
y = np.log10(temp["SFRFUV"])
fig, ax = plt.subplots()
sns.regplot(x = x, y = y,
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
results = scipy.stats.linregress(x = x, y = y)
plt.axvline(x=-3, color='green', linestyle='--', linewidth=2)

plt.xlabel("$log(SFR_{Ha})$"+" [{}]".format(dt["logSFR_HEC"].unit.to_string("latex")))
plt.ylabel("$log(SFR_{FUV})$"+" [{}]".format(dt["logSFR_HEC"].unit.to_string("latex")))
plt.legend(["Data",
            "$log(SFR_{FUV})=$" +f"{results.slope:.2f}"+"$\cdot log(SFR_{Ha})$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()



sfr_fuv = dt['SFRFUV']
sfr_ha = dt['SFRHa']

dt["SFR_t"] =np.log10(np.nanmean([sfr_fuv, sfr_ha], axis=0))
dt["SFR_t"].unit = dt["SFRFUV"].unit



#| label: fig-sfr-compare
#| layout-ncol: 2
#| layout-align: center
#| fig-cap: "Comparison of the SFR's of the galaxies"
#| fig-subcap: 
#| - "Linear scale" 
#| - "Decimal logarithmic scale"

##linear
plt.close()
fig, ax = plt.subplots()
temp = dt[["SFR_t","logSFR_HEC"]].to_pandas().dropna()
temp = temp[temp["SFR_t"]>-3]
sns.regplot(x = 10**temp["SFR_t"], y = 10**temp["logSFR_HEC"],
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
results = scipy.stats.linregress(x = 10**temp["SFR_t"], y = 10**temp["logSFR_HEC"])
plt.xlabel("$SFR_{LVG}$"+" [{}]".format(dt["SFR_t"].unit.to_string("latex")))
plt.ylabel("$SFR_{HECATE} $"+" [{}]".format(dt["SFR_t"].unit.to_string("latex")))
plt.legend(["Data",
            "$SFR_{HEC}=$" +f"{results.slope:.2f}"+"$\cdot SFR_{LVG}$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()

##log

plt.close()
fig, ax = plt.subplots()
temp = dt[["SFR_t","logSFR_HEC"]].to_pandas().dropna()
temp = temp[temp["SFR_t"]>-3]
sns.regplot(x = temp["SFR_t"], y = temp["logSFR_HEC"],
                 scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'),
                 line_kws=dict(alpha=0.7, color='red', linewidth=3))
results = scipy.stats.linregress(x = temp["SFR_t"], y = temp["logSFR_HEC"])
plt.xlabel("$log(SFR_{LVG})$"+" [{}]".format(dt["logSFR_HEC"].unit.to_string("latex")))
plt.ylabel("$log(SFR_{HECATE})$"+" [{}]".format(dt["logSFR_HEC"].unit.to_string("latex")))
plt.legend(["Data",
            "$log(SFR_{HEC})=$" +f"{results.slope:.2f}"+"$\cdot log(SFR_{LVG})$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()



















#| echo: false
# Extract the relevant mass columns
mass_columns = ["logM26", "logMHI", "logM_HEC", "logM_GSW", "logStellarMass"]
mass_data = dt[mass_columns].to_pandas()

# Count the number of non-NaN cells for each column
non_nan_counts = mass_data.notna().sum()


















#| label: fig-Mass-compare
#| fig-cap: "Comparison of the Stellar Masses of the galaxies"

plt.close()
fig, ax = plt.subplots()
temp = dt[["logStellarMass","logM_HEC"]].to_pandas().dropna()
x = temp["logStellarMass"]
y = temp["logM_HEC"]
sns.regplot(x = x, y = y, 
                scatter_kws=dict(alpha=0.5, color='blue', edgecolors='white'), 
                line_kws=dict(alpha=0.7, color='red', linewidth=3))
results = scipy.stats.linregress(x = x, y = y)
plt.xlabel("$log(M_{*,LVG})$"+" [{}]".format(dt["logM_HEC"].unit.to_string("latex")))
plt.ylabel("$log(M_{*,HECATE} )$"+" [{}]".format(dt["logM_HEC"].unit.to_string("latex")))
plt.legend(["Data",
            "$log(M_{*,HEC})=$" +f"{results.slope:.2f}"+"$\cdot log(M_{*,LVG})$"+
            f"${results.intercept:+.2f}$"+
            f"\n $R^2 = {results.rvalue**2*100:.2f}\%$"])
plt.show()














mass_data.drop("logM_GSW", axis=1, inplace=True)
# Compute the correlation matrix

pairplot_with_correlation(mass_data)
