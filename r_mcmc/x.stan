data {
  int<lower=1> N;                // number of data points
  vector[N] logSFR_HEC_Gyr;      // observed log SFR in Gyr
  vector[N] logM_HEC;            // total stellar mass formed
}

parameters {
  vector<lower=10, upper=15>[N] t_today;  // t_today for each data point
  vector<lower=-3, upper=3>[N] logtau;    // log10(tau) for each data point
  vector<lower=-3, upper=12>[N] logA;     // log10(A) for each data point
}

transformed parameters {
  vector[N] tau = pow(10, logtau);        // tau in linear scale for each data point
  vector[N] A = pow(10, logA);            // A in linear scale for each data point
  vector[N] logSFR_today;                 // modeled SFR for each data point

  // Calculate logSFR_today for each data point
  logSFR_today = logA + log10(t_today) - 2 * logtau - (t_today ./ tau) * 0.4342945;
}

model {
  // Vectorized priors
  logA ~ normal(0, 10);
  logtau ~ normal(0, 10);
  t_today ~ uniform(10, 15);

  // Likelihood
  logSFR_HEC_Gyr ~ normal(logSFR_today, 0.1);
}

generated quantities {
  vector[N] logSFR_today_pred;

  // Predicted SFR values for each data point
  logSFR_today_pred = logA + log10(t_today) - 2 * logtau - (t_today ./ tau) * 0.4342945;
}

