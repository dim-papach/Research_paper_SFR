import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable, MaskedColumn
from joblib import Parallel, delayed
from scipy.optimize import fsolve

# Read the data
dt = QTable.read("../tables/filled.ecsv")

t_sf = 13.6 * u.Gyr
zeta = 1.3


dt["av_SFR_theor"] = 1.3 * dt["logM_total"].value * u.M_sun / t_sf.to(u.yr)

dt["SFR_ratio"] = dt["av_SFR_theor"].value / dt["SFR_total"].value

#log10 of ratio
dt["logSFR_ratio"] = np.log10(dt["SFR_ratio"].value)


ratio_array = np.array(dt["SFR_ratio"])
sfr_array = np.array(dt["SFR_total"])
mass_array = np.array(dt["logM_total"])
av_array = np.array(dt["av_SFR_theor"])

#tsf = t_sf.to(u.yr).value
tsf = t_sf.value

x_f = np.ma.empty(len(dt))
A_f = np.ma.empty(len(dt))


# Function to solve for x using fsolve
def solve_x_f(i):
    SFR, M = sfr_array[i], mass_array[i]
    
    # Handle NaNs early
    if np.isnan(SFR) or np.isnan(M):
        return np.nan, np.nan  
    
    try:
        ratio = ratio_array[i]
        sfr = SFR

        def sfrx(z):
            x, A = z
            f = np.zeros(2)
            f[0] = ratio - (np.exp(x) - x - 1) / x**2
            f[1] = sfr*1e9 - A * x ** 2 * np.exp(-x) / tsf
            return f

        # Solve the equation
        z = fsolve(sfrx, [3, 1e+8])
        return z[0], z[1]   # Scale A back to its correct unit

    except (ValueError, RuntimeError):
        return np.nan, np.nan  # Ensure robustness


# Run the solver in parallel
num_cores = 12  # Use all CPU cores
results = Parallel(n_jobs=num_cores)(delayed(solve_x_f)(i) for i in range(len(dt)))

# Unpack results into arrays
x_f[:], A_f[:] = np.array(results).T  # Transpose to separate x_f and A_f

dt["x_f"] = MaskedColumn(x_f, name = "x")
dt["A_f"] = MaskedColumn(A_f, name = "A", unit = u.solMass)

dt["tau"] = t_sf/dt["x_f"]

# Print stats (same as original)
print(dt["x_f", "A_f", "tau"].info("stats"))

# Scatter plot x-A and next to it add zoom  from x=[-16,16]



# Grid for readability
plt.grid(True, linestyle="--", alpha=0.5)

# Show the plot
plt.savefig("./x_A_fsolve.png")
plt.close()


# Scatter plot = A_f tau
dt["tau"] = t_sf/dt["x_f"]
plt.figure(figsize=(8, 6))
plt.scatter(dt["tau"], dt['A_f'], c=mass_array, alpha=0.6, edgecolors="k")
plt.xlim(-5,5)
#plt.ylim(1e-1,1e11)
plt.yscale("log")
# Labels and title
plt.xlabel("tau")
plt.ylabel("A_f (Solar Mass)")
plt.title("Scatter Plot of A_f vs tau")

# Grid for readability
plt.grid(True, linestyle="--", alpha=0.5)

# Show the plot
plt.savefig("tau_A_fsolve.png")
plt.close()

#histogram for A_f
plt.figure(figsize=(8, 6))
plt.hist(np.log10(dt['A_f'].data), bins=50, alpha=0.6, edgecolor="k")
plt.yscale("log")
plt.xlabel("A_f (Solar Mass)")
plt.ylabel("Frequency")
plt.title("Histogram of A_f")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("hist_A_fsolve.png")
plt.close()




# Save the updated table with NR results
output_filename = "filled_with_fsolve.csv"
dt.write(output_filename, format="ascii.csv", overwrite=True)

print(f"Results saved to {output_filename}")

