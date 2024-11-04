import pymc as pm
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import QTable
import astropy.units as u
import arviz as az
from tqdm import tqdm
import gc


# Function to prepare the dataset
def prepare_data(file_path):
    # Load your dataset
    dt = QTable(ascii.read(file_path), masked=True)
    
    # Transform to physical units
    dt["SFR_HEC"] = (10**dt["logSFR_HEC"].value * u.solMass/u.yr).to(u.solMass/u.Gyr)  # SFR in Gyr units
    dt["logSFR_HEC_Gyr"] = np.log10(dt["logSFR_HEC"].value) # logSFR in Gyr units
    
    # Convert to DataFrame and drop missing values
    data = dt[["logSFR_HEC_Gyr", "logM_HEC"]].to_pandas().dropna()
    print(data)
    data.to_csv("test_galaxies.csv")
    del dt
    gc.collect()
    return data

# Function to run MCMC for a given galaxy
def run_mcmc_for_galaxy(row):
    with pm.Model() as model:
        # Priors for t_today, logtau, and logA in log space
        t_today = pm.Uniform('t_today', lower=10, upper=15)
        logtau = pm.Uniform('logtau', lower=-3, upper=3)  # tau in log 
        logA = pm.Uniform('logA', lower=-3, upper=12)  # log scale for A
        # Convert tau and  to linear scale

        tau = 10**logtau  # tau in linear scale
        # Assume Mtot and logSFR_today_observed come from the sample
        Mtot = row['logM_HEC']  # M_HEC is already in solMass
        logSFR_today_observed = row['logSFR_HEC_Gyr']  # logSFR in Gyr units

        # Use the corrected logarithmic form of SFR
        logSFR_today = logA + np.log10(t_today) - 2 * np.log10(tau) - (t_today / tau) * np.log10(np.e)

        # Likelihood (assuming Gaussian errors on log(SFR))
        sigma = 0.1  # Example uncertainty in log(SFR)
        SFR_likelihood = pm.Normal('SFR_likelihood', mu=logSFR_today, sigma=sigma, observed=logSFR_today_observed)

        # Sampling using NUTS
        try:
            trace = pm.sample(2000, tune=2000, target_accept=0.95, init='jitter+adapt_diag',
                              chains = 4, cores= 6,
                              return_inferencedata=True, progressbar=True)
        except pm.exceptions.SamplingError:
            print(f"Sampling failed for row {row.name}")
            return None

    # Extract the results for A, tau, and t_today
    summary = az.summary(trace, var_names=['t_today', 'logA', 'logtau'])
    logA_posterior_mean = (summary['mean']['logA'])
    tau_posterior_mean = (summary['mean']['logtau'])
    t_today_posterior_mean = summary['mean']['t_today']

    return Mtot, logA_posterior_mean, tau_posterior_mean, t_today_posterior_mean, logSFR_today_observed

# Main function to run MCMC over the entire sample
def run_mcmc_for_sample(data, sample_fraction=0.2, save_file="mcmc_results.csv"):
    # Take a random sample of the dataset
    sample_data = data.sample(frac=sample_fraction, random_state=42)
    print(sample_data.info())
    # List to store results
    results = []

    # Progress bar for tracking the process
    for idx, row in tqdm(sample_data.iterrows(), total=sample_data.shape[0], desc="Running MCMC"):
        mcmc_result = run_mcmc_for_galaxy(row)
        if mcmc_result:
            results.append(mcmc_result)

    # Convert to DataFrame and save to file
    results_df = pd.DataFrame(results, columns=['Mtot', 'logA_free', 'logtau', 't_today', 'logSFR'])
    results_df["A_free"] = 10**results_df["logA_free"]
    results_df["tau"] = 10**results_df["logtau"]
    results_df["x"] = results_df["t_today"]/results_df["tau"]
    results_df.to_csv(save_file, index=False)
    
    return results_df

# Run the MCMC process
if __name__ == "__main__":
    data = prepare_data("../tables/outer_join.ecsv")
    results_df = run_mcmc_for_sample(data, sample_fraction=0.2)
    print(results_df)
