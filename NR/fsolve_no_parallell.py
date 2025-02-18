import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import qtable
from joblib import parallel, delayed
from scipy.optimize import fsolve

# read the data
dt = qtable.read("../tables/filled.ecsv")

# define constants
zeta = 1.3  # mass-loss through stellar evolution
t_sf = 13.6 * u.gyr  # star formation timescale
tsf = t_sf.to(u.yr).value  # convert t_sf to years

# initialize arrays (same as original)
x = np.empty(len(dt))
a = np.empty(len(dt))

# extract sfr and m arrays
sfr_arr = np.array(dt["sfr_total"])
m_arr = np.array(dt["logm_total"])
av_sfr = zeta * m_arr / tsf
ratio_sfr = av_sfr / sfr_arr


# function to solve for x using fsolve
def solve_x_f(i):
    sfr, m = sfr_arr[i], m_arr[i]
    
    # handle nans early
    if np.isnan(sfr) or np.isnan(m):
        return np.nan, np.nan  
    
    try:
        ratio = ratio_sfr[i]
        sfr = sfr_arr[i]

        def sfrx(z):
            x, a = z
            f = np.zeros(2)
            f[0] = ratio - (np.exp(x) - x - 1) / x**2
            f[1] = sfr - a * 10**(-9) * x * tsf * np.exp(-x) / x
            return f

        # solve the equation
        z = fsolve(sfrx, [3, 1e+8])
        return z[0], z[1] * 10**9  # scale a back to its correct unit

    except (valueerror, runtimeerror):
        return np.nan, np.nan  # ensure robustness


# run the solver in parallel
num_cores = 12  # use all cpu cores
results = parallel(n_jobs=num_cores)(delayed(solve_x_f)(i) for i in range(len(dt)))

# unpack results into arrays
x_f[:], a_f[:] = np.array(results).t  # transpose to separate x_f and a_f

dt["x_f"] = x_f
dt["a_f"] = a_f
dt["tau"] = t_sf/dt["x_f"]

# print stats (same as original)
print(dt["x_f", "a_f", "tau"].info("stats"))

# scatter plot x-a
plt.figure(figsize=(8, 6))
plt.scatter(x_f, a_f, alpha=0.6, edgecolors="k")

# labels and title
plt.xlabel("x_f")
plt.ylabel("a_f (solar mass)")
plt.title("scatter plot of a_f vs x_f")

# grid for readability
plt.grid(true, linestyle="--", alpha=0.5)

# show the plot
plt.savefig("./x_a_fsolve.png")
plt.close()


# scatter plot = a_f
dt["tau"] = t_sf/dt["x_f"]
plt.figure(figsize=(8, 6))
plt.scatter(dt["tau"], dt['a_f'], alpha=0.6, edgecolors="k")
plt.xscale("log")
plt.yscale("log")
# labels and title
plt.xlabel("tau")
plt.ylabel("a_f (solar mass)")
plt.title("scatter plot of a_f vs tau")

# grid for readability
plt.grid(true, linestyle="--", alpha=0.5)

# show the plot
plt.savefig("tau_a_fsolve.png")
plt.close()

# save the updated table with nr results
output_filename = "filled_with_fsolve.csv"
dt.write(output_filename, format="ascii.csv", overwrite=true)

print(f"results saved to {output_filename}")