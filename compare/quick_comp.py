#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8-whitegrid")
# plt.style.use("ggplot")

# Load the data
data = pd.read_csv("../tables/inner_join.csv")
cols = pd.read_csv("Reordered_Final_Comparison_Table.csv")

data["SFR_total"] = data[["SFRFUV", "SFRHa"]].mean(axis=1)

# Define the mass and luminosity thresholds
mass_threshold = 9  # Log(Mass) threshold for dwarf vs massive


# Classify galaxies based on logM_HEC, logKLum, Tdw1, and Tdw2 with method flags
def classify_galaxy(row):
    # Initialize the classification and method flag
    classification = "undefined"
    method = ""

    # Check Tdw1 and Tdw2 columns for morphological classification
    if not pd.isna(row["Tdw1"]) or not pd.isna(row["Tdw2"]):
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


# Apply the classification function and split into two new columns
data[["mass_type", "classification_method"]] = data.apply(
    lambda row: pd.Series(classify_galaxy(row)), axis=1
)


def perform_linear_regression_analysis_fixed(
    x, y, desc=None, error=None, log_transform_x=False, log_transform_y=False
):
    # Check if the error column is None or NaN
    if error is None or error == "None":
        print(f"Warning: Error column for {x} vs {y} is not specified or is None.")

    # Extract and align data, dropping NaNs
    if error and error in data.columns:
        aligned_data = data[[x, y, error, "mass_type"]].dropna()
    else:
        aligned_data = data[[x, y, "mass_type"]].dropna()

    # Proceed only if there's sufficient data
    if len(aligned_data) < 2:
        print(f"Insufficient data for the pair ({x}, {y})")
        return

    x_data = aligned_data[x]
    y_data = aligned_data[y]
    groups = aligned_data.groupby("mass_type")

    # Apply log transformation if specified
    if log_transform_x:
        x_data = np.log10(x_data)
    if log_transform_y:
        y_data = np.log10(y_data)

    # Weight calculation based on the error column if available
    if error and error in aligned_data.columns:
        y_err_data = aligned_data[error]
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
    marker_dict = {"dwarf": "o", "massive": "s", "undefined": "D"}

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Scatter plot with fit
    if error and error in data.columns:
        for name, group in groups:
            marker_shape = marker_dict.get(
                name, "o"
            )  # Default to 'o' if name not in dict
            axs[0].errorbar(
                group[x],
                group[y],
                yerr=group[error],
                fmt=marker_shape,
                ecolor="grey",
                capsize=3,
                label=name,
            )
    else:
        axs[0].scatter(x_data, y_data, color="blue", label="Data")

    axs[0].plot(
        x_data,
        intercept + slope * x_data,
        color="red",
        zorder=5,
        label=f"Fit: y = ({slope:.3f} ± {slope_err:.3f})x +  ({intercept:.3f} ± {intercept_err:.3f}) ",
    )
    axs[0].set_xlabel("UNGC" + f'{"log10" if log_transform_x else ""}({x})')
    axs[0].set_ylabel("HECATE" + f'{"log10" if log_transform_y else ""}({y})')
    axs[0].set_title(f"Linear Fit of {desc}\n$R^2$ = {r_squared:.2f}")
    axs[0].legend()
    axs[0].grid(True)

    # Residuals plot
    axs[1].scatter(x_data, residuals, color="blue", label="Residuals")
    axs[1].axhline(0, color="red", linestyle="--", label="Zero Residual Line")
    axs[1].set_xlabel(f'{"log10" if log_transform_x else ""}({x})')
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residuals of the Linear Fit")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# Example usage:
# Run the analysis for a specific pair of columns, specifying whether to apply log transformations
for _, row in cols.iterrows():
    perform_linear_regression_analysis_fixed(
        x=row["x"], y=row["y"], error=row["error"], desc=row["description"]
    )
