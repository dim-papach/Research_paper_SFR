# functions.R

# Function to load and prepare the data
prepare_data <- function(data_csv) {
  dt <- data_csv %>%
    mutate(
      SFR_HEC = 10^logSFR_HEC_Gyr
    ) %>%
    drop_na(logSFR_HEC_Gyr, logM_HEC)
  return(dt)
}

# Function to define initial values for the Stan model
iiinit_function <- function() {
  list( 
   list(
      t_today = 13.6,            # Initial guess for t_today
      logtau = log10(6.3),       # Initial guess for log(tau)
      logA = log10(5.268e+10)    # Initial guess for log(A)
    )
  )
}
init_function <-function(chains, N) {
  lapply(1:chains, function(i) list(
    t_today = rep(13.6, N),
    logtau = rep(log10(6.3), N),
    logA = rep(log10(5.268e+10), N)
  ))
}


# Function to extract posterior predictions and compute residuals
extract_posterior_predictions <- function(stan_fit, sfr_data) {
  # Access the fit object (assuming the first fit object)
  fit <- stan_fit
  
  # Extract draws for logSFR_today_pred
  draws <- fit$draws(variables = "logSFR_today_pred")
  
  # Convert to data frame
  draws_df <- as_draws_df(draws)
  
  # Gather data into long format
  pred_summaries <- draws_df %>%
    pivot_longer(
      cols = starts_with("logSFR_today_pred"),
      names_to = "variable",
      values_to = "value"
    ) %>%
    group_by(variable) %>%
    summarise(
      mean = mean(value),
      sd = sd(value)
    )
  
  # Extract the index from variable names
  pred_summaries <- pred_summaries %>%
    mutate(
      index = as.integer(str_extract(variable, "\\d+"))
    )
  
  # Prepare observed data with index
  observed_data <- sfr_data %>%
    mutate(index = row_number())
  
  # Join predictions with observed data
  pred_summaries <- pred_summaries %>%
    inner_join(observed_data, by = "index") %>%
    mutate(
      residual = logSFR_HEC_Gyr - mean,
      standardized_residual = residual / sd
    )
  
  return(pred_summaries)
}
