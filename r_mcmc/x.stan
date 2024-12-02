data {
  int<lower=1> N;                        // Number of galaxies
  vector[N] logSFR_UNGC_Gyr;        // Observed log SFR in Gyr
  vector[N] id_numbers;                
  vector[N] mass;
}
transformed data {
  vector[N] ln_SFR_obs;                  // Natural log of observed SFR
  vector[N] ln_M_star;                   // Natural log of observed M_*
  vector[N] sigma_ln_SFR;                // Natural log uncertainties in SFR
  real ln10 = log(10);                   // Natural logarithm of 10 for base conversion

  // Transform data from log10 to natural log
  ln_SFR_obs = logSFR_UNGC_Gyr * ln10;
  ln_M_star = mass * ln10;
  }

parameters {
  vector<lower=10, upper=13.8>[N] t_sf;   // Star formation times (Gyr)
  vector<lower=1, upper=10>[N] tau;  // Characteristic timescales (Gyr)
  vector<lower=1, upper=2>[N] zeta;           // Mass loss fraction (adjust bounds as appropriate)
}

transformed parameters {
  vector[N] x;                           // Dimensionless parameter x = t_sf / tau
  vector[N] ln_SFR_model;                // Modeled ln(SFR)
  x = t_sf ./ tau;                       // Compute x

  
  // Compute log_denominator = log(tau) + x + log1m(temp)
  // Compute log_numerator = log(zeta) + ln_M_star + log(x)
  // Compute ln_SFR_model = log_numerator - log_denominator
  ln_SFR_model = log(zeta) + ln_M_star + log(x) 
  		- log(tau) - x + log1m((x + 1) .* exp(-x));
}

model {
  // Prior for zeta
  zeta ~ normal(1.3, 0.01);              // Adjust prior parameters as appropriate

  // Priors for t_sf and tau
  t_sf ~ uniform(0, 13.8);               // Uniform prior for t_sf between 0 and 13.8 Gyr
  target += -log(tau);                   // Log-uniform prior for tau between 1e-3 and 1e3 Gyr

  // Likelihood
  ln_SFR_obs ~ normal(ln_SFR_model, 0.1);
}

