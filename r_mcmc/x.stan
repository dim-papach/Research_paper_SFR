data {
  int<lower=1> N;                        // Number of galaxies
  vector[N] ln_SFR_obs;                  // Observed ln(SFR)
  vector[N] ln_M_star;                   // Observed ln(M_*)
  vector[N] sigma_ln_SFR;                // Uncertainties in ln(SFR)
}

parameters {
  vector<lower=0, upper=13.8>[N] t_sf;   // Star formation times (Gyr)
  vector<lower=1e-3, upper=1e3>[N] tau;  // Characteristic timescales (Gyr)
  real<lower=0, upper=2> zeta;           // Mass loss fraction (adjust bounds as appropriate)
}

transformed parameters {
  vector[N] x;                           // Dimensionless parameter x = t_sf / tau
  vector[N] ln_SFR_model;                // Modeled ln(SFR)
  vector[N] temp;                        // Temporary variable for (x + 1) * exp(-x)
  vector[N] log_denominator;             // Log of the denominator
  vector[N] log_numerator;               // Log of the numerator

  x = t_sf ./ tau;                       // Compute x

  // Compute temp = (x + 1) * exp(-x)
  temp = (x + 1) .* exp(-x);
  
  // Compute log_denominator = log(tau) + x + log1m(temp)
  log_denominator = log(tau) + x + log1m(temp);

  // Compute log_numerator = log(zeta) + ln_M_star + log(x)
  log_numerator = log(zeta) + ln_M_star + log(x);

  // Compute ln_SFR_model = log_numerator - log_denominator
  ln_SFR_model = log_numerator - log_denominator;
}

model {
  // Prior for zeta
  zeta ~ normal(1.3, 0.01);              // Adjust prior parameters as appropriate

  // Priors for t_sf and tau
  t_sf ~ uniform(0, 13.8);               // Uniform prior for t_sf between 0 and 13.8 Gyr
  target += -log(tau);                   // Log-uniform prior for tau between 1e-3 and 1e3 Gyr

  // Likelihood
  ln_SFR_obs ~ normal(ln_SFR_model, sigma_ln_SFR);
}

