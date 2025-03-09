data {
  int<lower=1> n;                        // number of galaxies
  vector[n] sfr_obs;                     // observed sfr (not log-transformed)
  vector[n] mass;                        // stellar mass
}

transformed data {
  vector[n] scaled_mass;                 // scaled stellar mass for numerical stability
  real scaling_factor = 1;            // scale factor for mass
  scaled_mass = mass / scaling_factor;   // scale mass to avoid large values

  // validate data
  for (N in 1:n) {
    if (mass[N] <= 0) {
      reject("invalid mass: mass must be positive, found ", mass[n], " for galaxy ", n);
    }
    if (sfr_obs[N] <= 0) {
      reject("invalid sfr: sfr must be positive, found ", sfr_obs[n], " for galaxy ", n);
    }
  }
}

parameters {
  real<lower=1.0, upper=2.0> zeta;        // mass loss fraction
  vector<lower=10, upper=13.8>[n] t_sf;   // time since star formation began (gyr)
  vector<lower=1e-2, upper=1e3>[n] tau;   // characteristic timescale (gyr)
}

transformed parameters {
  vector[n] x;                           // dimensionless parameter x = t_sf / tau
  vector[n] sfr_model;                   // modeled sfr (not log-transformed)

  for (N in 1:n) {
    real denominator;
    real small_approx;
    real large_approx;
    real sigmoid_weight;
    real k = 10;                         // steepness of sigmoid
    real c = 5;                          // center of transition

    // compute x
    x[N] = t_sf[N] / tau[N];

    // compute approximations for the denominator
    small_approx = 0.5 * x[N] * x[N];    // for small x
    large_approx = exp(x[N]);            // for large x

    // smooth transition using sigmoid
    sigmoid_weight = inv_logit(-k * (x[N] - c));
    denominator = sigmoid_weight * small_approx + (1 - sigmoid_weight) * large_approx;

    // compute sfr model
    sfr_model[N] = (zeta * scaled_mass[N] * x[N]) / (tau[N] * denominator);

    // ensure denominator is positive
    if (denominator <= 0 || is_nan(denominator)) {
      print("invalid denominator: denominator = ", denominator, " for galaxy ", N);
      reject("denominator must be positive and finite");
    }
  }
}

model {
  // priors
  zeta ~ normal(1.3, 0.01);              // tight prior for zeta
  t_sf ~ uniform(10, 13.8);              // uniform prior for t_sf
  tau ~ lognormal(log(1), 0.5);          // adjusted prior for tau to prevent very small values

  // likelihood
  sfr_obs ~ normal(sfr_model, 0.1 * mean(sfr_obs));  // scale the error term
}
