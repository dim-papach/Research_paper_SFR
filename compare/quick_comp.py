#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import sigmaclip
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, QTable
import astropy.units as u

# plt.style.use("seaborn-v0_8-whitegrid")
# plt.style.use("ggplot")

# Get data
data = pd.read_csv("../tables/inner_join.csv")

# Define the mass and luminosity thresholds
mass_threshold = 9  # Log(Mass) threshold for dwarf vs massive


# Classify galaxies based on logM_HEC, logKLum, Tdw1, and Tdw2 with method flags
def classify_galaxy(row):
    # Initialize the classification and method flag
    classification = "undefined"
    method = ""

    # Check Tdw1 and Tdw2 columns for morphological classification
    if not pd.isna(row["Tdw1"]) or not pd.isna(row["Tdw2"]):
        classification = "dwarf"
        method = "T"  # Classification based on morphology (Tdw1 or Tdw2)
    # Check mass classification
    elif row["logM_HEC"] < mass_threshold:
        classification = "dwarf"
        method = "M"  # Classification based on mass
    elif row["logM_HEC"] >= mass_threshold:
        classification = "massive"
        method = "M"
    return classification, method


# Apply the classification function and split into two new columns
data[["mass_type", "classification_method"]] = data.apply(
    lambda row: pd.Series(classify_galaxy(row)), axis=1
)

cols = pd.read_csv("Reordered_Final_Comparison_Table.csv")

data["SFR_mean"] = data[["SFRFUV", "SFRHa"]].mean(axis=1)
# log SFR_mean
data["logSFR_mean"] = np.log10(data["SFR_mean"])


def perform_linear_regression_analysis_fixed(
    x,
    y,
    desc=None,
    x_error=None,
    y_error=None,
    log_transform_x=False,
    log_transform_y=False,
):
    # Check if the y_error column is None or NaN
    if x_error is None or x_error == "None":
        print(f"Warning: X-error column for {x} vs {y} is not specified or is None.")
    if y_error is None or y_error == "None":
        print(f"Warning: Y_Error column for {x} vs {y} is not specified or is None.")
    # Extract and align data, dropping NaNs
    error_cols = [
        col for col in [x, y, x_error, y_error, "mass_type"] if col in data.columns
    ]
    aligned_data = data[error_cols].dropna()

    # Proceed only if there's sufficient data
    if len(aligned_data) < 2:
        print(f"Insufficient data for the pair ({x}, {y})")
        return

    x_data = aligned_data[x]
    y_data = aligned_data[y]
    groups = data.groupby("mass_type")

    # Apply log transformation if specified
    if log_transform_x:
        x_data = np.log10(x_data)
        x_data = x_data.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()  # Remove inf and NaN values after log transformation

    if log_transform_y:
        y_data = np.log10(y_data)
        y_data = y_data.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()  # Remove inf and NaN values after log transformation

    # Weight calculation based on the y_error column if available
    if y_error and y_error in aligned_data.columns:
        y_err_data = aligned_data[y_error]
        weights = 1 / y_err_data**2
        model = sm.WLS(y_data, sm.add_constant(x_data), weights=weights)
    else:
        model = sm.OLS(y_data, sm.add_constant(x_data))

    results = model.fit()

    # Fit parameters
    slope = results.params[1]
    intercept = results.params[0]
    slope_err = results.bse[1]
    intercept_err = results.bse[0]
    r_squared = results.rsquared

    # Calculate residuals
    residuals = y_data - (intercept + slope * x_data)

    # Define markers for each mass type
    marker_dict = {"Dwarf": "o", "Massive": "s", "Undefined": "D"}

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Scatter plot with fit
    for name, group in groups:
        marker_shape = marker_dict[name] if marker_dict and name in marker_dict else "o"

        x_data_g = group[x] if log_transform_x is False else np.log10(group[x])
        y_data_g = group[y] if log_transform_y is False else np.log10(group[y])
        x_err_data = group[x_error] if x_error in group.columns else None
        y_err_data = group[y_error] if y_error in group.columns else None

        axs[0].errorbar(
            x_data_g,
            y_data_g,
            xerr=x_err_data,
            yerr=y_err_data,
            fmt=marker_shape,
            ecolor="grey",
            capsize=3,
            label=name,
        )
    # Plot the fitted line
    x_range = axs[0].get_xlim()
    x_fit = np.linspace(x_range[0], x_range[1], 250)
    y_fit = intercept + slope * x_fit
    axs[0].plot(
        x_fit,
        y_fit,
        color="red",
        label=f"Fit: y = ({slope:.3f} ± {slope_err:.3f})x + ({intercept:.3f} ± {intercept_err:.3f})",
    )
    axs[0].set_xlabel(
        f"UNGC({'log10(' if log_transform_x else ''}{x}{')' if log_transform_x else ''})"
    )
    axs[0].set_ylabel(
        f"HECATE({'log10(' if log_transform_y else ''}{y}{')' if log_transform_y else ''})"
    )
    axs[0].set_title(f"Linear Fit of {desc}\n$R^2$ = {r_squared:.2f}")
    axs[0].legend()
    axs[0].grid(True)

    # Residuals plot
    for name, group in groups:
        marker_shape = marker_dict.get(name, "o") if marker_dict else "o"
        x_data_g = group[x] if log_transform_x is False else np.log10(group[x])
        residuals_g = group[y] - (intercept + slope * group[x])
        axs[1].scatter(x_data_g, residuals_g, marker=marker_shape, label=name)

    axs[1].axhline(0, color="red", linestyle="--", label="Zero Residual Line")
    axs[1].set_xlabel(
        f"UNGC({'log10(' if log_transform_x else ''}{x}{')' if log_transform_x else ''})"
    )
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residuals of the Linear Fit")
    axs[1].legend()
    axs[1].grid(True)

    # Residuals histogram plot
    axs[2].hist(residuals, bins=30, label="Residuals", edgecolor="black")
    axs[2].axvline(0, color="red", linestyle="--", label="Zero Residual Line")
    axs[2].axvline(
        np.mean(residuals),
        linestyle="--",
        color="green",
        label=f"Mean Residual Line = {np.mean(residuals):.2f}",
    )
    axs[2].set_xlabel("Residuals")
    axs[2].set_ylabel("Count")
    axs[2].set_title("Histogram of Residuals")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("./quickplots/" + desc + ".png", dpi=300)
    plt.close()


# Example usage:
# Run the analysis for a specific pair of columns, specifying whether to apply log transformations
for _, row in cols.iterrows():
    print(row)
    perform_linear_regression_analysis_fixed(
        x=row["x"],
        y=row["y"],
        x_error=row["x_error"],
        y_error=row["y_error"],
        log_transform_x=row["log_x"],
        log_transform_y=row["log_y"],
        desc=row["description"],
    )


def sigma_clip(df, x, y, y_clip, sigma=3):
    if x and y in df.columns:  # Ensure the column exists in the DataFrame
        # Replace values outside the sigma range with NaN
        df[y_clip] = df[y].where(abs(df[x] - df[y]) <= sigma, np.nan)
    else:
        print(f"Column not found in DataFrame")


# Apply sigma clipping on the 'data' column
sigma_clip(data, x="TType", y="T", y_clip="T_clip", sigma=7)

perform_linear_regression_analysis_fixed(
    x="TType", y="T_clip", y_error="E_T", x_error=None, desc="Type_clip"
)
