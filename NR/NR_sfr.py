import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable
from joblib import Parallel, delayed

#set working directory
# import os.path
# os.chdir("/home/dp/Documents/Research_paper_SFR/NR/")  # Replace with your directory

# Read the data

dt = QTable.read("./tables/filled.ecsv")
# Define constants
zeta = 1.3  # Mass-loss through stellar evolution
t_sf = 13.6 * u.Gyr  # Star formation timescale
tsf = t_sf.to(u.yr).value  # Convert t_sf to years
#tsf = t_sf.value  # Convert t_sf to years

# Initialize arrays (same as original)
x_n = np.empty(len(dt))
A_n = np.empty(len(dt))

# Extract SFR and M arrays
SFR_arr = np.array(dt["SFR_total"])
M_arr = np.array(dt["M_total"])

K = SFR_arr*tsf/(zeta*M_arr)
k_arr=1/K
# Function to solve for x using the Newton-Raphson method
def solve_x_n(i):
    SFR, M, k = SFR_arr[i], M_arr[i], k_arr[i]
    if np.isnan(SFR) or np.isnan(M) or np.isnan(k):
        return np.nan  # Handle NaNs early
    try:
    # Define the function and its derivative for the Newton-Raphson method
        def f(x):
            return (k*x**2-np.exp(x)+x+1)

        def f_prime(x):
            return (2*k*x -np.exp(x)+1)
        sol = optimize.root_scalar(f, x0=3, fprime=f_prime, method="newton")
        return sol.root if sol.converged else np.nan
    except ValueError:
        return np.nan

# Run the solver in parallel
num_cores = 4  # Use all CPU cores
x_n[:] = Parallel(n_jobs=num_cores)(delayed(solve_x_n)(i) for i in range(len(dt)))

# Compute A_n for all valid x_n (same as original, but vectorized)
valid_x = ~np.isnan(x_n)
A_n[valid_x] = SFR_arr[valid_x] * tsf * np.exp(x_n[valid_x]) / (x_n[valid_x]**2)
A_n[~valid_x] = np.nan  # Keep NaNs consistent

# Add results to the DataFrame (same as original)
dt["x_n"] = x_n
dt['tau'] = np.empty(len(dt))
dt["tau"][valid_x] = (t_sf/dt["x_n"][valid_x])
dt["tau"][~valid_x] = np.nan
dt["A_n"] = A_n * u.solMass  # Add units to A_n

# Print stats (same as original)
print(dt["x_n", "A_n", "tau"].info("stats"))

#remove rows with NaN values in the tau and A_n columns
dt=dt[~np.isnan(dt["tau"])]
dt=dt[~np.isnan(dt["A_n"])]

# Save the updated table with NR results
output_filename = "NR/filled_with_NR.csv"
dt.write(output_filename, format="ascii.csv", overwrite=True)

print(f"Results saved to {output_filename}")


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# First subplot (without zoom)
scatter1 = ax1.scatter(dt["tau"], dt['A_n'], c=dt["logM_total"].value, cmap="viridis", edgecolors="k")
ax1.set_yscale("log")
ax1.set_xlim(-1e10, 1e10)  # Adjust x-axis limits if needed
ax1.set_ylim(0, 1e31)  # Adjust y-axis limits if needed
ax1.set_xlabel(r"$\tau$ [Gyr]")
ax1.set_ylabel(r"$A_{del}\, [M_{\odot}]$")
ax1.grid(True, linestyle="--", alpha=0.5)

# Second subplot (with zoom)
scatter2 = ax2.scatter(dt["tau"], dt['A_n'], c=dt["logM_total"].value, cmap="viridis", edgecolors="k")
ax2.set_yscale("log")
ax2.set_xlim(0, 16)  # Adjust x-axis limits for zoom if needed
ax2.set_ylim(0, 10**13)  # Adjust y-axis limits for zoom
ax2.set_xlabel(r"$\tau$ [Gyr]")
ax2.grid(True, linestyle="--", alpha=0.5)

# Add a single colorbar for both subplots
cbar = fig.colorbar(scatter2, location="right", label=r"$\log(M_*/M_\odot)$")

# Tight layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig("NR/tau_A_double_plot.png")

# Show the plot
plt.close()