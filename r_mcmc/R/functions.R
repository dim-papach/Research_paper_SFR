# functions.R

# Function to load and prepare the data
prepare_data <- function(data_csv) {
  dt <- data_csv %>%
    mutate(id_number = row_number()) %>%
    select(logSFR_UNGC_Gyr, id_number) %>%
    filter(
      logSFR_UNGC_Gyr >=-3,
      !is.na(logSFR_UNGC_Gyr),
      !is.nan(logSFR_UNGC_Gyr),
      is.finite(logSFR_UNGC_Gyr),
      )
  return(dt)
}

# Function to define initial values for the Stan model
init_function <-function(chains, N) {
  lapply(1:chains, function(i) list(
    t_sf = rep(13.6, N),
    logtau = rep(log10(4), N), # MS galaxies 3.5<tau<4.5
    logA = rep(10, N)
  ))
}

