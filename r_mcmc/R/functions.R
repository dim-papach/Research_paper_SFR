# functions.R

# Function to load and prepare the data
prepare_data <- function(data_csv) {
  dt <- data_csv %>%
    mutate(id_number = row_number(),
           logM_HEC = ifelse(is.na(logM_HEC), 0.822 * K, logM_HEC), # Replace missing logM_HEC with 0.8 * K
           logM_HEC = ifelse(is.na(logM_HEC), 0.822 * KLum, logM_HEC) # Replace missing logM_HEC with 0.8 * KLum
           ) %>%
    select(logSFR_UNGC_Gyr, id_number, logM_HEC) %>%
    filter(
      logSFR_UNGC_Gyr >=-3,
      !is.na(logSFR_UNGC_Gyr),
      !is.nan(logSFR_UNGC_Gyr),
      is.finite(logSFR_UNGC_Gyr),
      !is.na(logM_HEC),
      !is.nan(logM_HEC),
      is.finite(logM_HEC)
      )
  return(dt)
}

# Function to define initial values for the Stan model
init_function <-function(chains, N) {
  lapply(1:chains, function(i) list(
    t_sf = rep(13.6, N),
    logtau = rep(log10(4), N), # MS galaxies 3.5<tau<4.5
    logA = rep(5.5, N),
    tau = rep(4,N),
    zeta = rep(1.3,N)
  ))
}

