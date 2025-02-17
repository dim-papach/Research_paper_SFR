# functions.R

# Function to load and prepare the data
prepare_data <- function(data_csv) {
  dt <- data_csv %>%
    mutate(
      logM_HEC = ifelse(is.na(logM_HEC), 0.822 * K, logM_HEC), # Replace missing logM_HEC with 0.8 * K
      logM_HEC = ifelse(is.na(logM_HEC), 0.822 * KLum, logM_HEC) # Replace missing logM_HEC with 0.8 * KLum
    ) %>%
    select(logSFR_total, id_number, logM_HEC) %>%
    filter(
      # iogSFR_total >=-3,
      !is.na(logSFR_total),
      !is.nan(logSFR_total),
      is.finite(logSFR_total),
      !is.na(logM_HEC),
      !is.nan(logM_HEC),
      is.finite(logM_HEC)
    )
  return(dt)
}

# Function to define initial values for the Stan model
init_function <- function(chains, N) {
  lapply(1:chains, function(i) {
    list(
      t_sf = rep(13.6, N),
      # x = rep(3, N),
      logtau = rep(log10(4), N), # MS galaxies 3.5<tau<4.5
      # logA = rep(5.5, N),
      tau = rep(4, N),
      zeta = rep(1.3, N)
    )
  })
}

mean_vs_median <- function(summary_data, summary_variables) {
  ggplot(summary_data, aes(x = mean, y = median)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(
      title = paste("Mean vs. Median of", summary_variables),
      x = "Mean",
      y = "Median"
    )
  ggsave(sprintf("plots/Mean-Median/%s.png", summary_variables))
}

# Function to create histograms of the mcmc results, with mean value highlighted
plot_histograms <- function(summary_data, summary_variables, summary_names) {
  max_count <- max(ggplot_build(
    ggplot(summary_data, aes(x = mean)) +
      geom_histogram(bins = 30)
  )$data[[1]]$count, na.rm = TRUE)

  p <- ggplot(summary_data, aes(x = mean)) +
    geom_histogram(alpha = 0.7, bins = 40, fill = "blue", color = "black") +
    geom_vline(aes(xintercept = mean(mean)), color = "red", linetype = "dashed", size = 1) +
    geom_text(
      aes(
        x = mean(mean),
        label = format(mean(mean), scientific = TRUE, digits = 2) # Scientific notation, 2 decimal points
      ),
      y = max_count + 1, # Place the text slightly above the maximum bar
      inherit.aes = FALSE,
      vjust = 0, # Above the bar
      hjust = 0.5 # Centered above the mean line
    ) +
    labs(
      title = "Histogram of the mean values from the MCMC run",
      x = summary_names,
      y = "Number of galaxies"
    ) +
    ggeasy::easy_center_title()

  ggsave(sprintf("plots/Hists/%s.png", summary_variables), p)
}

# Function to create histograms for the metrics
plot_metrics <- function(summary_data, summary_variables, metrics) {
  mean_value <- mean(summary_data[[metrics]], na.rm = TRUE) # Calculate mean dynamically
  fname <- paste(summary_variables, metrics, sep = "_")

  ggplot(data = summary_data, aes_string(x = metrics)) +
    geom_histogram(bins = 30, fill = "blue", alpha = 0.6) +
    geom_vline(
      aes(xintercept = mean_value),
      color = "red", linetype = "dashed", size = 1
    ) +
    annotate(
      "text",
      x = mean_value, y = Inf,
      label = sprintf("Mean: %.2f", mean_value),
      vjust = 2, color = "red", size = 4
    ) +
    scale_x_continuous(
      trans = "log10",
      breaks = trans_breaks("log10", function(x) 10^x),
      labels = trans_format("log10", math_format(10^.x))
    ) +
    labs(
      title = sprintf("Histogram of %s", fname),
      x = metrics,
      y = "Frequency"
    ) +
    ggeasy::easy_center_title()

  ggsave(sprintf("plots/metrics/%s.png", fname))
}
