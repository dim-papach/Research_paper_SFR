import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from joblib import Parallel, delayed
from scipy import optimize

# Read the data
dt = QTable.read("../tables/filled.ecsv")

# Define constants
zeta = 1.3  # Mass-loss through stellar evolution
t_sf = 13.6 * u.Gyr  # Star formation timescale
#tsf = t_sf.to(u.yr).value  # Convert t_sf to years
tsf = t_sf.value  # Convert t_sf to years

# Initialize arrays (same as original)
x_n = np.empty(len(dt))
A_n = np.empty(len(dt))

# Extract SFR and M arrays
SFR_arr = np.array(dt["SFR_total"])
M_arr = np.array(dt["logM_total"])


# Function to solve for x using the Newton-Raphson method
def solve_x_n(i):
    SFR, M = SFR_arr[i], M_arr[i]
    if np.isnan(SFR) or np.isnan(M):
        return np.nan  # Handle NaNs early
    try:
    # Define the function and its derivative for the Newton-Raphson method
        def f(x):
            return (-SFR + zeta * M * x**2 / (np.exp(x) - 1 - x) / tsf)

        def f_prime(x):
            return -zeta * M * (x * (np.exp(x) * (x - 2) + x + 2) / (np.exp(x) - x - 1)**2) / t_sf.value
        sol = optimize.root_scalar(f, bracket=[0, 4], x0=3.4, fprime=f_prime, method="newton")
        return sol.root if sol.converged else np.nan
    except ValueError:
        return np.nan

# Run the solver in parallel
num_cores = 12  # Use all CPU cores
x_n[:] = Parallel(n_jobs=num_cores)(delayed(solve_x_n)(i) for i in range(len(dt)))

# Compute A_n for all valid x_n (same as original, but vectorized)
valid_x = ~np.isnan(x_n)
A_n[valid_x] = SFR_arr[valid_x] * tsf * np.exp(x_n[valid_x]) / (x_n[valid_x]**2)
A_n[~valid_x] = np.nan  # Keep NaNs consistent

# Add results to the DataFrame (same as original)
dt["x_n"] = x_n
dt["tau"] = (tsf/dt["x_n"])
dt["A_n"] = A_n*1e9 * u.solMass  # Add units to A_n

# Print stats (same as original)
print(dt["x_n", "A_n", "tau"].info("stats"))




# Scatter plot
plt.figure(figsize=(8, 6))
#add color from M_total
plt.scatter(dt["tau"], dt['A_n'], c=dt["logM_total"].value, cmap="viridis", alpha=0.6, edgecolors="k")
plt.yscale("log")
plt.xlim(0, 6)
plt.ylim(10**9, 10**11)
# Labels and title
plt.xlabel("tau")
plt.ylabel("A_n (Solar Mass)")
plt.title("Scatter Plot of A_n vs tau")

# Grid for readability
plt.grid(True, linestyle="--", alpha=0.5)
plt.colorbar(label="logM_total")

# Show the plot
plt.savefig("tau_A.png")
plt.close()

# hist tau
plt.figure(figsize=(8, 6))
plt.hist(dt["tau"], bins=50, color="skyblue", edgecolor="black")
plt.xlabel("tau")
plt.ylabel("Frequency")
plt.title("Histogram of tau")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("hist_tau.png")
plt.close()

# Save the updated table with NR results
output_filename = "filled_with_NR.csv"
dt.write(output_filename, format="ascii.csv", overwrite=True)

print(f"Results saved to {output_filename}")
