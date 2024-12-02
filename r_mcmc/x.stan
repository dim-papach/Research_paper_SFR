/*
  Stan model for estimating the star formation rate (SFR) of galaxies
  based on individual parameters: time of star formation (t_sf),
  tau (logtau), and A (logA). The model predicts log SFR and compares
  it with observed data to infer parameter values.

  Key Components:
  - t_sf: Time of star formation in Gyr.
  - logtau: log10(tau) in Gyr.
  - logA: log10(A) (scaling factor).
  - logSFR_today: Modeled logarithmic SFR at the present time.
*/

data {
  int<lower=1> N;                   // Number of data points
  vector[N] logSFR_UNGC_Gyr;        // Observed log SFR in Gyr
  vector[N] id_numbers;                // Row identifiers for the data points
}

parameters {
  vector<lower=10, upper=13.8>[N] t_sf;     // Time of star formation for each galaxy in Gyr
  vector<lower=-3, upper=3>[N] logtau;    // log10(tau) for each galaxy
  vector<lower=-2, upper=12>[N] logA;     // log10(A) for each galaxy
}

transformed parameters {
  /*
    Derived parameters based on transformations:
    - tau: Linear scale of tau (10^logtau).
    - A: Linear scale of A (10^logA).
    - logSFR_today: Modeled log SFR at the present time.

    Includes constant for converting between natural log and base-10 log.
  */
  vector[N] tau = pow(10, logtau);         // Tau in linear scale
  vector[N] A = pow(10, logA);             // A in linear scale
  vector[N] logSFR_today;                  // Modeled log SFR for each data point
  real log10_e = log10(exp(1));            // Constant for ln to log10 conversion (log10(e))

  // Calculate modeled log SFR at the present time for each galaxy
  logSFR_today = logA
                 + log10(t_sf)
                 - 2 * logtau
                 - (t_sf ./ tau) * log10_e; // Subtract exponential decay term
}

model {
  /*
    Bayesian model specification:
    - Priors on parameters (t_sf, logtau, logA).
    - Likelihood: Observed log SFR is modeled as normally distributed around the predicted value.
  */

  // Priors: Uniform distributions for each parameter
  logA ~ uniform(-2, 12);                 // Prior for log10(A)
  logtau ~ uniform(-3, 4);                // Prior for log10(tau)
  t_sf ~ uniform(10, 15);                 // Prior for t_sf (time of star formation)

  // Likelihood: Observed log SFR compared to modeled log SFR
  logSFR_UNGC_Gyr ~ normal(logSFR_today, 0.1); // Assume small measurement error (std = 0.1)
}

generated quantities {
  /*
    Generated quantities:
    - Predicted log SFR for posterior predictive checks.
    - Pass-through of row identifiers for post-modeling analysis.
  */
  vector[N] logSFR_today_pred;            // Predicted log SFR for each data point
  vector[N] id;                              // Row identifiers

  // Predict log SFR using the same formula as in `transformed parameters`
  logSFR_today_pred = logA
                      + log10(t_sf)       // log10(t_sf)
                      - 2 * logtau        // Subtract 2 * log10(tau)
                      - (t_sf ./ tau) * log10_e; // Subtract exponential decay term

  // Assign row identifiers for reference
  id = id_numbers;
}
