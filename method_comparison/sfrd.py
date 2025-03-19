import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate  # Ensure scipy is imported
from astropy.cosmology import Planck18 as cosmo  # Using Planck 2018 cosmology
import astropy.units as u


# Import data
data_uni = pd.read_csv('r_mcmc_uni/joined_data.csv')
data_norm = pd.read_csv('r_mcmc_normal/joined_data.csv')
data_nr = pd.read_csv('NR/filled_with_NR.csv')

# Define the A and tau values for each dataset
A_values_uni = data_uni['A_up'].values
tau_values_uni = (data_uni['tau_up']*u.Gyr.to(u.yr)).values
sigma_A_values_uni = data_uni['A_sigma_up'].values
sigma_tau_values_uni = (data_uni['tau_sigma_up']*u.Gyr.to(u.yr)).values

A_values_norm = data_norm['A_np'].values
tau_values_norm = (data_norm['tau_np']*u.Gyr.to(u.yr)).values
sigma_A_values_norm = data_norm['A_sigma_np'].values
sigma_tau_values_norm = (data_norm['tau_sigma_np']*u.Gyr.to(u.yr)).values

A_values_nr = data_nr['A_n'].values
tau_values_nr = (data_nr['tau_n']*u.Gyr.to(u.yr)).values

# Volume factor
V = (4/3 * np.pi * 11**3)

# Define the SFR function


def sfr_function(t, A, tau):
    return A * (t / tau**2) * np.exp(-t / tau) / V

# Define the error propagation function (only for datasets with sigmas)

def sfr_error(t, A, tau, sigma_A, sigma_tau):
    dSFR_dA = (t / tau**2) * np.exp(-t / tau) / V
    dSFR_dtau = A * np.exp(-t / tau) * ((2 * t / tau**3) - (t**2 / tau**4)) / V
    sigma_SFR = np.sqrt((dSFR_dA * sigma_A)**2 + (dSFR_dtau * sigma_tau)**2)
    return sigma_SFR

def lilly_madau(z):
    """Lilly-Madau SFRD function from Madau & Dickinson (2014)."""
    return 0.015 * ((1 + z)**2.7) / (1 + ((1 + z)/2.9)**5.6)
# Define custom time steps and corresponding redshifts
custom_times = np.array([0, 10.5, 12.3, 12.9, 13.2, 13.3])  # Gyr
redshifts = np.array([0, 2, 4, 6, 8, 10])  # Corresponding redshifts

# Interpolate time steps for smooth plotting
time_steps = np.linspace(0, 13.3, 100)  # 100 points for smooth curve

# Function to compute total SFR and its uncertainty (if sigmas are available)


def compute_total_sfr_and_error(time_steps, A_values, tau_values, sigma_A_values=None, 
                                sigma_tau_values=None):
    total_sfr = np.zeros_like(time_steps)
    total_sfr_error = np.zeros_like(time_steps)

    for i, t in enumerate(time_steps):
        sfr_i = sfr_function(t*u.Gyr.to(u.yr), A_values, tau_values)
        total_sfr[i] = np.sum(sfr_i)

        if sigma_A_values is not None and sigma_tau_values is not None:
            sigma_sfr_i = sfr_error(
                t*u.Gyr.to(u.yr), A_values, tau_values, sigma_A_values, sigma_tau_values)
            total_sfr_error[i] = np.sqrt(np.sum(sigma_sfr_i**2))

    return total_sfr, total_sfr_error


# Compute SFR and errors for each dataset
total_sfr_uni, total_sfr_error_uni = compute_total_sfr_and_error(
    time_steps, A_values_uni, tau_values_uni, sigma_A_values_uni, sigma_tau_values_uni)
total_sfr_norm, total_sfr_error_norm = compute_total_sfr_and_error(
    time_steps, A_values_norm, tau_values_norm, sigma_A_values_norm, sigma_tau_values_norm)
total_sfr_nr, _ = compute_total_sfr_and_error(
    time_steps, A_values_nr, tau_values_nr)  # No sigmas for NR

# Compute log10 of SFR and propagate errors
def compute_log_sfr_and_error(total_sfr, total_sfr_error):
    log_sfr = np.log10(total_sfr)
    log_sfr_error = (total_sfr_error / total_sfr) / np.log(10)
    return log_sfr, log_sfr_error

log_sfr_uni, log_sfr_error_uni = compute_log_sfr_and_error(total_sfr_uni, total_sfr_error_uni)
log_sfr_norm, log_sfr_error_norm = compute_log_sfr_and_error(total_sfr_norm, total_sfr_error_norm)
log_sfr_nr = np.log10(total_sfr_nr)

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

