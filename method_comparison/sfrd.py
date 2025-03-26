import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate  # Ensure scipy is imported
from astropy.cosmology import Planck18 as cosmo  # Using Planck 2018 cosmology
import astropy.units as u


# Import data
data_uni = pd.read_csv('r_mcmc_uni/joined_data.csv')
data_norm = pd.read_csv('r_mcmc_normal/joined_data.csv')
data_skew = pd.read_csv('r_mcmc_skew/joined_data.csv')
data_nr = pd.read_csv('NR/filled_with_NR.csv')
T0 = 13.8*1e9  # Age of the Universe in years

# Define the A and tau values for each dataset
A_values_uni = data_uni['A_up'].values
tau_values_uni = (data_uni['tau_up']*u.Gyr.to(u.yr)).values
tsf_values_uni = (data_uni['t_sf_up']*u.Gyr.to(u.yr)).values
tstart_values_uni = T0 - tsf_values_uni
sigma_A_values_uni = data_uni['A_sigma_up'].values
sigma_tau_values_uni = (data_uni['tau_sigma_up']*u.Gyr.to(u.yr)).values
sigma_tsf_values_uni = (data_uni['t_sf_sigma_up']*u.Gyr.to(u.yr)).values
sigma_tstart_values_uni = sigma_tsf_values_uni


A_values_norm = data_norm['A_np'].values
tau_values_norm = (data_norm['tau_np']*u.Gyr.to(u.yr)).values
tsf_values_norm = (data_norm['t_sf_np']*u.Gyr.to(u.yr)).values
tstart_values_norm = T0 - tsf_values_norm
sigma_A_values_norm = data_norm['A_sigma_np'].values
sigma_tau_values_norm = (data_norm['tau_sigma_np']*u.Gyr.to(u.yr)).values
sigma_tsf_values_norm = (data_norm['t_sf_sigma_np']*u.Gyr.to(u.yr)).values
sigma_tstart_values_norm = sigma_tsf_values_norm

#for skew
A_values_skew = data_skew['A_np'].values
tau_values_skew = (data_skew['tau_np']*u.Gyr.to(u.yr)).values
tsf_values_skew = (data_skew['t_sf_np']*u.Gyr.to(u.yr)).values
tstart_values_skew = T0 - tsf_values_skew
sigma_A_values_skew = data_skew['A_sigma_np'].values
sigma_tau_values_skew = (data_skew['tau_sigma_np']*u.Gyr.to(u.yr)).values
sigma_tsf_values_skew = (data_skew['t_sf_sigma_np']*u.Gyr.to(u.yr)).values
sigma_tstart_values_skew = sigma_tsf_values_skew

A_values_nr = data_nr['A_n'].values
tau_values_nr = (data_nr['tau_n']*u.Gyr.to(u.yr)).values
tsf_values_nr = 13.6*1e9  # Fixed value for NR
tstart_values_nr = T0 - tsf_values_nr  # This is a scalar (e.g. 0.2 Gyr)
# For NR, we want t_start to be an array with one value per galaxy:
tstart_values_nr = np.full_like(A_values_nr, tstart_values_nr)
# Similarly, we assume no uncertainty for t_start in NR:
sigma_tstart_values_nr = np.zeros_like(A_values_nr)

# Volume factor
V = (4/3 * np.pi * 11**3)

# Define the SFR function


def sfr_function(t, A, tau):
    """
    Computes the SFR for a given time (in years), normalization A, and timescale tau (in years).
    """
    return A * (t / tau**2) * np.exp(-t / tau) / V

# Define the error propagation function (only for datasets with sigmas)

# New sfr_error including uncertainty in t_start.
def sfr_error(T_current, t_start, A, tau, sigma_A, sigma_tau, sigma_t_start):
    """
    Compute SFR uncertainty with time defined as t = T_current - t_start.
    
    Parameters:
        T_current (float): Age of the Universe at redshift z (in Gyr).
        t_start (array): Formation start times of galaxies (in Gyr).
        A (array): Star formation amplitude values.
        tau (array): Star formation timescales (in years).
        sigma_A (array): Uncertainties in A.
        sigma_tau (array): Uncertainties in tau.
        sigma_t_start (array): Uncertainties in t_start (in Gyr).
    
    Returns:
        sigma_SFR (array): Uncertainty in the SFR.
    """
    # Compute the galaxy age (in Gyr) at this redshift:
    t_Gyr = T_current - t_start
    # Convert t to years (1 Gyr = 1e9 years)
    t_years = t_Gyr * 1e9

    # Partial derivatives
    dSFR_dA = (t_years / tau**2) * np.exp(-t_years / tau) / V
    dSFR_dtau = A * np.exp(-t_years / tau) * ((2 * t_years / tau**3) - (t_years**2 / tau**4)) / V
    # Derivative with respect to t (and chain rule: dt/dt_start = -1)
    dSFR_dt = -A * np.exp(-t_years / tau) * ((1 / tau) - (t_years / tau**2)) / V

    # Convert sigma_t_start from Gyr to years:
    sigma_t_start_years = sigma_t_start * 1e9

    sigma_SFR = np.sqrt((dSFR_dA * sigma_A)**2 +
                        (dSFR_dtau * sigma_tau)**2 +
                        (dSFR_dt * sigma_t_start_years)**2)
    return sigma_SFR


def compute_total_sfr_and_error(z_array, t_start, sigma_t_start, A_values, tau_values, sigma_A_values=None, sigma_tau_values=None):
    """
    Compute total SFR and its uncertainty for an array of redshifts.
    
    For each redshift, we compute the current age of the Universe (T_current) using cosmo.age(z)
    and then compute the galaxy age as t = T_current - t_start.
    
    Parameters:
        z_array (array): Array of redshifts.
        t_start (array): Formation start times of galaxies (in Gyr) for each galaxy.
        sigma_t_start (array): Uncertainty in t_start (in Gyr) for each galaxy.
        A_values (array): A values for galaxies.
        tau_values (array): Tau values for galaxies (in years).
        sigma_A_values (array, optional): Uncertainty in A.
        sigma_tau_values (array, optional): Uncertainty in tau.
        
    Returns:
        total_sfr (array): Total SFR (summed over galaxies) at each redshift.
        total_sfr_error (array): Combined uncertainty at each redshift.
    """

    total_sfr = np.zeros_like(z_array)
    total_sfr_error = np.zeros_like(z_array)

    for i, z in enumerate(z_array):
        # Get the current age of the Universe at redshift z (in Gyr)
        T_current = cosmo.age(z).value  # in Gyr
        # Compute galaxy age for each galaxy: t = T_current - t_start (in yr)
        t = T_current*10**9 - t_start 
        # Create mask for galaxies with positive age
        valid = t > 0
        
        # Only for valid galaxies, convert t to years
        t_years = t[valid] 
        
        # Compute SFR for valid galaxies
        sfr_i = np.zeros_like(t)
        if np.any(valid):
            sfr_i[valid] = sfr_function(t_years, A_values[valid], tau_values[valid])
        total_sfr[i] = np.sum(sfr_i)
        
        if sigma_A_values is not None and sigma_tau_values is not None:
            sigma_sfr_i = np.zeros_like(t)
            if np.any(valid):
                sigma_sfr_i[valid] = sfr_error(T_current, t_start[valid], A_values[valid],
                                                tau_values[valid], sigma_A_values[valid],
                                                sigma_tau_values[valid], sigma_t_start[valid])
            total_sfr_error[i] = np.sqrt(np.sum(sigma_sfr_i**2))
    return total_sfr, total_sfr_error   
# ------------------------------

# Define redshift grid for plotting
redshifts = np.linspace(0, 10, 100)

# Compute SFR and errors for each dataset:
# Uniform Prior Dataset
total_sfr_uni, total_sfr_error_uni = compute_total_sfr_and_error(
    z_array=redshifts,
    t_start=tstart_values_uni,
    sigma_t_start=sigma_tstart_values_uni,
    A_values=A_values_uni,
    tau_values=tau_values_uni,
    sigma_A_values=sigma_A_values_uni,
    sigma_tau_values=sigma_tau_values_uni
)

# Normal Prior Dataset
total_sfr_norm, total_sfr_error_norm = compute_total_sfr_and_error(
    z_array=redshifts,
    t_start=tstart_values_norm,
    sigma_t_start=sigma_tstart_values_norm,
    A_values=A_values_norm,
    tau_values=tau_values_norm,
    sigma_A_values=sigma_A_values_norm,
    sigma_tau_values=sigma_tau_values_norm
)

#Skew Prior Dataset
total_sfr_skew, total_sfr_error_skew = compute_total_sfr_and_error(
    z_array=redshifts,
    t_start=tstart_values_skew,
    sigma_t_start=sigma_tstart_values_skew,
    A_values=A_values_skew,
    tau_values=tau_values_skew,
    sigma_A_values=sigma_A_values_skew,
    sigma_tau_values=sigma_tau_values_skew
)

# NR Dataset (no uncertainties available)
total_sfr_nr, _ = compute_total_sfr_and_error(
    z_array=redshifts,
    t_start=tstart_values_nr,            # Already an array of shape (n_gal,) for NR
    sigma_t_start=sigma_tstart_values_nr,  # Zero uncertainties for NR
    A_values=A_values_nr,
    tau_values=tau_values_nr
)

# Compute log10 of SFR (and propagate errors)
def compute_log_sfr_and_error(total_sfr, total_sfr_error):
    log_sfr = np.log10(total_sfr)
    log_sfr_error = (total_sfr_error / total_sfr) / np.log(10)
    return log_sfr, log_sfr_error

log_sfr_uni, log_sfr_error_uni = compute_log_sfr_and_error(total_sfr_uni, total_sfr_error_uni)
log_sfr_norm, log_sfr_error_norm = compute_log_sfr_and_error(total_sfr_norm, total_sfr_error_norm)
log_sfr_skew, log_sfr_error_skew = compute_log_sfr_and_error(total_sfr_skew, total_sfr_error_skew)
log_sfr_nr = np.log10(total_sfr_nr)

# Define Lilly-Madau SFRD function for comparison
def lilly_madau(z):
    """Lilly-Madau SFRD function from Madau & Dickinson (2014)."""
    return 0.015 * ((1 + z)**2.7) / (1 + ((1 + z)/2.9)**5.6)

# Calculate Lilly-Madau SFRD at the redshifts
sfrd_lilly_madau = lilly_madau(redshifts)
log_sfrd_lm = np.log10(sfrd_lilly_madau)

# Function to compute co-moving radial distance from redshift


def comoving_distance(z):
    return cosmo.comoving_distance(z).value  # Returns distance in Mpc


# Define redshifts and corresponding lookback times
redshifts = np.array([0, 2, 4, 6, 8, 10])
lookback_times = cosmo.lookback_time(redshifts).value  # Gyr
co_moving_distances = comoving_distance(redshifts) / 1000  # Convert to Gpc

# Compute SFR for plotting
redshifts_interp = np.linspace(0, 10, len(total_sfr_uni))
lookback_interp = cosmo.lookback_time(redshifts_interp).value
comoving_interp = comoving_distance(redshifts_interp) / 1000  # Convert to Gpc
# Calculate Lilly-Madau SFRD at interpolated redshifts
sfrd_lilly_madau = lilly_madau(redshifts_interp)
log_sfrd_lm = np.log10(sfrd_lilly_madau)


# Plot
## set theme for the plot
plt.style.use('bmh')
# Reset color cycle to default
plt.rcParams["axes.prop_cycle"] = plt.matplotlib.rcParamsDefault["axes.prop_cycle"]

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twiny()
ax3 = ax1.twiny()

# Adjust position of third axis
ax3.spines['top'].set_position(('outward', 40))  # Move it slightly upward

# Plot data with log values on linear y-axis
ax1.errorbar(redshifts_interp, log_sfr_uni, yerr=log_sfr_error_uni,
             fmt='o-', capsize=3, label=r"MCMC, Uniform Prior for $\tau$", markersize=3)
ax1.errorbar(redshifts_interp, log_sfr_norm, yerr=log_sfr_error_norm,
             fmt='s-', capsize=3, label=r"MCMC, Normal Prior for $\tau$", markersize=3)
ax1.errorbar(redshifts_interp, log_sfr_skew, yerr=log_sfr_error_skew,
             fmt='v-', capsize=3, label=r"MCMC, Skew Prior for $t_{sf}$, Normal $\tau$", markersize=3)
ax1.plot(redshifts_interp, log_sfr_nr, '^-', label="Newton-Raphson", markersize=3)
# Add Lilly-Madau theoretical curve
ax1.plot(redshifts_interp, log_sfrd_lm, 'k--', linewidth=2, label="Lilly-Madau (2014)")

# Primary x-axis (Redshift)
ax1.set_xlabel("Redshift $z$")
ax1.set_xlim(0, 10)
ax1.set_xticks(redshifts)
ax1.axvline(x=1.86, color='r', linestyle='--', label='z=1.86')
ax1.legend()
ax1.invert_xaxis()  # Match direction in the image

# Secondary x-axis (Lookback Time)
ax2.set_xlabel("Lookback time [Gyr]")
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(redshifts)
ax2.set_xticklabels([f"{t:.1f}" for t in lookback_times])

# Tertiary x-axis (Co-moving Radial Distance)
ax3.set_xlabel("Co-moving radial distance [cGpc]")
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(redshifts)
ax3.set_xticklabels([f"{d:.1f}" for d in co_moving_distances])

# Y-axis (Linear scale with log values)
ax1.set_ylabel(r"$\log_{10}\left(\text{SFRD}\ \left[\text{M}_\odot \text{yr}^{-1} \text{Mpc}^{-3}\right]\right)$")
ax1.axhline(y=-0.88, color='g', linestyle='--', label='log(SFRD) = -0.88')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Invert x-axis for all three
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()

# Save
plt.tight_layout()
plt.savefig('method_comparison/sfrd_comparison_custom_axes_log_values.png', dpi=300)
plt.close()
#-----------------------------------
# Compute residuals (difference in log space)
residual_uni = log_sfrd_lm-log_sfr_uni 
residual_norm =  + log_sfrd_lm-log_sfr_norm
residual_skew =  + log_sfrd_lm-log_sfr_skew
residual_nr =  + log_sfrd_lm-log_sfr_nr

# Propagate errors (same as original errors since LM is a fixed theoretical curve)
residual_error_uni = log_sfr_error_uni
residual_error_norm = log_sfr_error_norm
residual_error_skew = log_sfr_error_skew

# Create figure
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twiny()
ax3 = ax1.twiny()

ax3.spines['top'].set_position(('outward', 40))

# Plot residuals
ax1.errorbar(redshifts_interp, residual_uni, yerr=residual_error_uni,
             fmt='o-', capsize=3, label=r"Uniform Prior", markersize=3)
ax1.errorbar(redshifts_interp, residual_norm, yerr=residual_error_norm,
             fmt='s-', capsize=3, label=r"Normal Prior", markersize=3)
ax1.errorbar(redshifts_interp, residual_skew, yerr=residual_error_skew,
             fmt='v-', capsize=3, label=r"Skew Prior", markersize=3)
ax1.plot(redshifts_interp, residual_nr, '^-', label="Newton-Raphson", markersize=3)

# Add reference line at zero
ax1.axhline(0, color='k', linestyle='--', alpha=0.7, label="Lilly-Madau")

# Axes configuration
ax1.set_xlabel("Redshift $z$")
ax1.set_ylabel(r"$\Delta \log_{10}(\text{SFRD})$" + "\n(Lilly-Madau $-$ Data)")
ax1.axvline(x=1.86, color='r', linestyle='--', label='z=1.86')
ax1.set_xlim(0, 10)
ax1.set_xticks(redshifts)
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.invert_xaxis()

# Secondary x-axis (Lookback Time)
ax2.set_xlabel("Lookback time [Gyr]")
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(redshifts)
ax2.set_xticklabels([f"{t:.1f}" for t in lookback_times])

# Tertiary x-axis (Co-moving Radial Distance)
ax3.set_xlabel("Co-moving radial distance [cGpc]")
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(redshifts)
ax3.set_xticklabels([f"{d:.1f}" for d in co_moving_distances])

# Invert all x-axes
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()

# Adjust y-axis limits for better visualization
ax1.set_ylim(-1.2, 1.2)

plt.tight_layout()
plt.savefig('method_comparison/sfrd_residuals_vs_lm.png', dpi=300)
plt.close()

# add plot of the ratio data/LM
# Compute ratio of data to LM
ratio_uni = sfrd_lilly_madau/total_sfr_uni  
ratio_norm =  sfrd_lilly_madau/total_sfr_norm
ratio_skew =  sfrd_lilly_madau/total_sfr_skew
ratio_nr =  sfrd_lilly_madau/total_sfr_nr

# Compute error propagation for ratios
def ratio_error(total_sfr,total_sfr_error, sfrd_lilly_madau):
    return np.sqrt(np.abs(sfrd_lilly_madau/total_sfr**2*total_sfr_error**2))

ratio_error_uni = ratio_error(total_sfr_uni,total_sfr_error_uni, sfrd_lilly_madau)
ratio_error_norm = ratio_error(total_sfr_norm,total_sfr_error_norm, sfrd_lilly_madau)
ration_error_skew = ratio_error(total_sfr_skew,total_sfr_error_skew, sfrd_lilly_madau)

# Create figure
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twiny()
ax3 = ax1.twiny()

ax3.spines['top'].set_position(('outward', 40))

# Plot ratios
ax1.errorbar(redshifts_interp, ratio_uni, yerr=ratio_error_uni,
             fmt='o-', capsize=3, label=r"Uniform Prior", markersize=3)
ax1.errorbar(redshifts_interp, ratio_norm, yerr=ratio_error_norm,
                fmt='s-', capsize=3, label=r"Normal Prior", markersize=3)
ax1.errorbar(redshifts_interp, ratio_skew, yerr=ration_error_skew,
                fmt='v-', capsize=3, label=r"Skew Prior", markersize=3)
ax1.plot(redshifts_interp, ratio_nr, '^-', label="Newton-Raphson", markersize=3)

# Add reference line at unity
ax1.axhline(1, color='k', linestyle='--', alpha=0.7, label="Lilly-Madau")

# Axes configuration
ax1.set_xlabel("Redshift $z$")
ax1.set_ylabel(r"$\frac{\text{SFRD}_{\text{LM}}}{\text{SFRD}_{\text{Data}}}$")
ax1.axvline(x=1.86, color='r', linestyle='--', label='z=1.86')
ax1.set_xlim(0, 10)
ax1.set_xticks(redshifts)
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.invert_xaxis()

# Secondary x-axis (Lookback Time)
ax2.set_xlabel("Lookback time [Gyr]")
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(redshifts)
ax2.set_xticklabels([f"{t:.1f}" for t in lookback_times])

# Tertiary x-axis (Co-moving Radial Distance)
ax3.set_xlabel("Co-moving radial distance [cGpc]")
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(redshifts)
ax3.set_xticklabels([f"{d:.1f}" for d in co_moving_distances])

# Invert all x-axes
ax1.invert_xaxis()
ax2.invert_xaxis()  
ax3.invert_xaxis()

plt.tight_layout()
plt.savefig('method_comparison/sfrd_ratio_vs_lm.png', dpi=300)
plt.close()

#calculate again and print the ratio for z=1.86
sfrd_uni_186, sfrd_uni_186_error = compute_total_sfr_and_error(
    z_array=[1.86],
    t_start=tstart_values_uni,
    sigma_t_start=sigma_tstart_values_uni,
    A_values=A_values_uni,
    tau_values=tau_values_uni,
    sigma_A_values=sigma_A_values_uni,
    sigma_tau_values=sigma_tau_values_uni
)
sfrd_norm_186, sfrd_norm_186_error  = compute_total_sfr_and_error(
    z_array=[1.86],
    t_start=tstart_values_norm,
    sigma_t_start=sigma_tstart_values_norm,
    A_values=A_values_norm,
    tau_values=tau_values_norm,
    sigma_A_values=sigma_A_values_norm,
    sigma_tau_values=sigma_tau_values_norm
)
sfrd_skew_186, sfrd_skew_186_error  = compute_total_sfr_and_error(
    z_array=[1.86],
    t_start=tstart_values_skew,
    sigma_t_start=sigma_tstart_values_skew,
    A_values=A_values_skew,
    tau_values=tau_values_skew,
    sigma_A_values=sigma_A_values_skew,
    sigma_tau_values=sigma_tau_values_skew
)
sfrd_nr_186,_ = compute_total_sfr_and_error(
    z_array=[1.86],
    t_start=tstart_values_nr,            # Already an array of shape (n_gal,) for NR
    sigma_t_start=sigma_tstart_values_nr,  # Zero uncertainties for NR
    A_values=A_values_nr,
    tau_values=tau_values_nr
)
sfrd_lilly_madau_186 = lilly_madau(1.86)
ratio_uni_186 = sfrd_lilly_madau_186/sfrd_uni_186
ratio_norm_186 = sfrd_lilly_madau_186/sfrd_norm_186
ratio_skew_186 = sfrd_lilly_madau_186/sfrd_skew_186
ratio_nr_186 = sfrd_lilly_madau_186/sfrd_nr_186
print(f"Ratio of SFRD at z=1.86 for Uniform Prior: {ratio_uni_186[0]:.2f}")
print(f"Ratio of SFRD at z=1.86 for Normal Prior: {ratio_norm_186[0]:.2f}")
print(f"Ratio of SFRD at z=1.86 for Skew Prior: {ratio_skew_186[0]:.2f}")
print(f"Ratio of SFRD at z=1.86 for Newton-Raphson: {ratio_nr_186[0]:.2f}")
#-----------------------------------