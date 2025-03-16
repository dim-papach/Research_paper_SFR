import numpy as np
import matplotlib.pyplot as plt

# import data from joined_data.csv file
import pandas as pd
data = pd.read_csv('r_mcmc/joined_data.csv')

# Define the A and tau values
A_values = data['A'].values
tau_values = data['tau'].values
sigma_A_values = data['A_sigma'].values
sigma_tau_values = data['tau_sigma'].values
V = (4/3*np.pi*10**3)
# Define the SFR function
def sfr_function(t, A, tau):
    return A * (t / tau**2) * np.exp(-t / tau)/V

# Define the error propagation function
def sfr_error(t, A, tau, sigma_A, sigma_tau):
    dSFR_dA = (t / tau**2) * np.exp(-t / tau)/V
    dSFR_dtau = A * np.exp(-t / tau) * ( (2*t / tau**3) - (t**2 / tau**4) )/V
    
    sigma_SFR = np.sqrt((dSFR_dA * sigma_A)**2 + (dSFR_dtau * sigma_tau)**2)
    return sigma_SFR

# Define time steps from 0 to 13.8 Gyr in 0.5 Gyr intervals
time_steps = np.arange(0.3, 13.8 + 0.5, 0.5)



# Compute total SFR and its uncertainty at each time step
total_sfr = np.zeros_like(time_steps)
total_sfr_error = np.zeros_like(time_steps)

for i, t in enumerate(time_steps):
    sfr_i = sfr_function(t, A_values, tau_values)
    sigma_sfr_i = sfr_error(t, A_values, tau_values, sigma_A_values, sigma_tau_values)
    
    total_sfr[i] = np.sum(sfr_i)  
    total_sfr_error[i] = np.sqrt(np.sum(sigma_sfr_i**2))  # Sum errors in quadrature

# Plot the results with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(13.8-time_steps, total_sfr, yerr=total_sfr_error, fmt='o-', capsize=3, label="Total SFRD")
#log scale for x-y
plt.yscale('log')
plt.xlabel("Lookback Time (Gyr)")
plt.ylabel("Total SFR/Volume")
plt.title("Total SFRD    Over Time with Uncertainty")
plt.legend()
plt.grid()
plt.savefig('r_mcmc/sfrd.png')
plt.close()
