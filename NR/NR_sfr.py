import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from joblib import Parallel, delayed
from scipy import optimize

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

        sol = optimize.root_scalar(f, x0=4, fprime=f_prime, method="newton")
        return sol.root if sol.converged else np.nan
    except ValueError:
        return np.nan

# Run the solver in parallel
num_cores = 8  # Use all CPU cores
x_n[:] = Parallel(n_jobs=num_cores)(delayed(solve_x_n)(i) for i in range(len(dt)))

# Compute A_n for all valid x_n (same as original, but vectorized)
valid_x = ~np.isnan(x_n)
A_n[valid_x] = SFR_arr[valid_x] * tsf * np.exp(x_n[valid_x]) / (x_n[valid_x]**2)
A_n[~valid_x] = np.nan  # Keep NaNs consistent

# Add results to the DataFrame (same as original)
dt["x_n"] = x_n
dt['tau_n'] = np.empty(len(dt))
dt["tau_n"][valid_x] = (t_sf/dt["x_n"][valid_x])
dt["tau_n"][~valid_x] = np.nan
dt["A_n"] = A_n * u.solMass  # Add units to A_n

#Calculate the SFR based on the delayed tau model
dt["SFR"] = dt["A_n"]*tsf/(dt["tau_n"]*10**9)**2*np.exp(-dt['x_n'])
#calculate the total SFRD
sum_SFR = np.sum(dt["SFR"][~np.isnan(dt["SFR"])].value)
V = (4/3 * np.pi * 11**3)
SFRD = sum_SFR/V
logSFRD = np.log10(SFRD)
print(f"\nTotal SFRD: {logSFRD:.2e} M_sun/yr/Mpc^3\n")
#calculate the total SFRD for the original SFR
sum_SFR_total = np.sum(dt["SFR_total"][~np.isnan(dt["SFR_total"])].value)
SFRD_total = sum_SFR_total/V
logSFRD_total = np.log10(SFRD_total)
print(f"\nTotal SFRD: {logSFRD_total:.2e} M_sun/yr/Mpc^3\n")

# Print stats (same as original)
print(dt["x_n", "A_n", "tau_n"].info("stats"))

#remove rows with NaN values in the tau_n and A_n columns
dt=dt[~np.isnan(dt["tau_n"])]
dt=dt[~np.isnan(dt["A_n"])]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# First subplot (without zoom)
scatter1 = ax1.scatter(dt["tau_n"], dt['A_n'], c=dt["logM_total"].value, cmap="viridis", edgecolors="k")
ax1.set_yscale("log")
ax1.set_xlim(-1e10, 1e10)  # Adjust x-axis limits if needed
ax1.set_ylim(0, 1e31)  # Adjust y-axis limits if needed
ax1.set_xlabel(r"$\tau$ [Gyr]")
ax1.set_ylabel(r"$A_{del}\, [M_{\odot}]$")
ax1.grid(True, linestyle="--", alpha=0.5)

# Second subplot (with zoom)
scatter2 = ax2.scatter(dt["tau_n"], dt['A_n'], c=dt["logM_total"].value, cmap="viridis", edgecolors="k")
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

# Plot the original SFR vs. the new SFR
plt.scatter(dt["SFR_total"], dt["SFR"])
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"SFR [M$_\odot$/yr] (Original)")
plt.ylabel(r"SFR [M$_\odot$/yr] (New)")
plt.grid(True, linestyle="--", alpha=0.5)

# Tight layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig("NR/SFR_comparison.png")
plt.close()

# Save the updated table with NR results
dt.keep_columns(["tau_n", "x_n", "A_n", "ID", "sSFR", "logM_total", "SFR", "SFR_total"])
output_filename = "NR/filled_with_NR.csv"
dt.write(output_filename, format="ascii.csv", overwrite=True)

print(f"Results saved to {output_filename}")

print("Min, Max, Mean k:", np.min(k_arr), np.max(k_arr), np.mean(k_arr))
print("Min, Max, Mean SFR:", np.min(SFR_arr), np.max(SFR_arr), np.mean(SFR_arr))
print("Min, Max, Mean M:", np.min(M_arr), np.max(M_arr), np.mean(M_arr))