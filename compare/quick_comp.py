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


def perform_linear_regression_analysis_fixed(
    x, y, desc=None, error=None, log_transform_x=False, log_transform_y=False
):
    # Check if the error column is None or NaN
    if error is None or error == "None":
        print(f"Warning: Error column for {x} vs {y} is not specified or is None.")

    # Extract and align data, dropping NaNs
    if error and error in data.columns:
        aligned_data = data[[x, y, error]].dropna()
    else:
        aligned_data = data[[x, y]].dropna()

    # Proceed only if there's sufficient data
    if len(aligned_data) < 2:
        print(f"Insufficient data for the pair ({x}, {y})")
        return

    x_data = aligned_data[x]
    y_data = aligned_data[y]

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

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Scatter plot with fit
    if error and error in data.columns:
        axs[0].errorbar(
            x_data,
            y_data,
            yerr=y_err_data,
            fmt="ob",
            ecolor="grey",
            capsize=3,
            label="Data",
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
