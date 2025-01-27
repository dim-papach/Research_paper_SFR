# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes) # Load other packages as needed.
library(stantargets)
library(crew)
library(future)

# Set target options:
tar_option_set(
  packages = c("readr", "tidyverse","tibble", "stringr", "posterior","bayesplot", "loo","ggplot2", "scales"),
  # format = "qs", # Optionally set the default storage format. qs is fast.
  #
  # Pipelines that take a long time to run may benefit from
  # optional distributed computing. To use this capability
  # in tar_make(), supply a {crew} controller
  # as discussed at https://books.ropensci.org/targets/crew.html.
  # Choose a controller that suits your needs. For example, the following
  # sets a controller that scales up to a maximum of two workers
  # which run as local R processes. Each worker launches when there is work
  # to do and exits if 60 seconds pass with no tasks to run.
  #
  #controller = crew::crew_controller_local(workers = 2)
  #
  # Alternatively, if you want workers to run on a high-performance computing
  # cluster, select a controller from the {crew.cluster} package.
  # For the cloud, see plugin packages like {crew.aws.batch}.
  # The following example is a controller for Sun Grid Engine (SGE).
  # 
  #   controller = crew.cluster::crew_controller_sge(
  #     # Number of workers that the pipeline can scale up to:
  #     workers = 10,
  #     # It is recommended to set an idle time so workers can shut themselves
  #     # down if they are not running tasks.
  #     seconds_idle = 120,
  #     # Many clusters install R as an environment module, and you can load it
  #     # with the script_lines argument. To select a specific verison of R,
  #     # you may need to include a version string, e.g. "module load R/4.3.2".
  #     # Check with your system administrator if you are unsure.
  #     script_lines = "module load R"
  #   )
  #
  # Set other options as needed.
)

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
# tar_source("other_functions.R") # Source other scripts as needed.

# Replace the target list below with your own:
# Define the pipeline
#list(
tar_plan(
  # Step 1: Load and prepare data
  tar_target(
    csv_file_path,
    "filled.csv",
    format = "file"
  ),
  tar_target(
    read_data,
    read.csv(csv_file_path)
  ),
  tar_target(
    sfr_data,
    prepare_data(read_data)
  ),
  
  # Step 2: Fit the Stan model
  tar_stan_mcmc(
    stan_fit,
    stan_files = "x.stan",
    #stdout = "out.txt",
    #stderr = "error.txt",
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number,
      M_star = sfr_data$logM_HEC
    ),
    chains = 7,
    parallel_chains = 8,
    iter_sampling = 4500,
    iter_warmup = 3000,
    init = init_function(7,1761),
    
  ),

  tar_stan_summary(
    id_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "id"
  ),

  tar_stan_summary(
    sfr_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "logSFR_today"
  ),

  tar_stan_summary(
    t_sf_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "t_sf"),
  
    tar_stan_summary(
    log_tsf_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "log_tsf"),

  tar_stan_summary(
    tau_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "tau"
  ),

  tar_stan_summary(
    logtau_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "logtau"
  ),
  tar_stan_summary(
    A_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "logA"
  ),
  
  tar_stan_summary(
    x_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      id_numbers = sfr_data$id_number
    ),
    variables = "x"
  ),
  tar_target(
    divergences,
    stan_fit_diagnostics_x$divergent__
  ),

  # Define the list of variables
  tar_target(
    summary_variables,
    c("logSFR_today","log_tsf", "t_sf", "tau", "logA", "logtau") # List of variables
  ),

  # Compute plots dynamically for mean vs median
  tar_target(
    mean_median_plots,
    {
      # Extract summary data for the variable
      summary_data <- switch(summary_variables,
                             "logA" = A_summary,
                             "tau" = tau_summary,
                             "t_sf" = t_sf_summary,
                             "log_tsf" = log_tsf_summary,
                             "logSFR_today" = sfr_summary,
                             "logtau" = logtau_summary)

      # Generate the plot
      ggplot(summary_data, aes(x = mean, y = median)) +
        geom_point() +
        labs(
          title = paste("Mean vs. Median of", summary_variables),
          x = "Mean",
          y = "Median"
        )
      ggsave(sprintf("plots/Mean-Median/%s.png", summary_variables))
    },
    pattern = map(summary_variables) # Branch over variables
  ),

  # Compute plots dynamically for mean vs median
  tar_target(
      histograms,
    {
      # Extract summary data for the variable
      summary_data <- switch(summary_variables,
                             "logA" = A_summary,
                             "tau" = tau_summary,
                             "t_sf" = t_sf_summary,
                             "log_tsf" = log_tsf_summary,
                             "logSFR_today" = sfr_summary,
                             "logtau" = logtau_summary)
      max_count <- max(ggplot_build(
        ggplot(summary_data, aes(x = mean)) +
          geom_histogram(bins = 30)
      )$data[[1]]$count, na.rm = TRUE)

      ggplot(summary_data, aes(x = mean)) +
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
          title = paste("Histogram of Mean for", summary_variables),
          x = "Mean",
          y = "Count"
        )


      ggsave(sprintf("plots/Hists/%s.png", summary_variables))
     },

    pattern = map(summary_variables) # Branch over variables
  ),


  # Prepare the metric data

  tar_target(
    metrics,
    c("rhat", "ess_bulk", "ess_tail")
  ),
  
  tar_target(
    hist_metrics,
    {
      mean_value <- mean(stan_fit_summary_x[[metrics]], na.rm = TRUE)  # Calculate mean dynamically
      
      ggplot(data = stan_fit_summary_x, aes_string(x = metrics)) +
        geom_histogram(bins = 30, fill = "blue", alpha = 0.6) +
        geom_vline(
          aes(xintercept = mean_value),
          color = "red", linetype = "dashed", size = 1
        ) +
        annotate(
          "text", x = mean_value, y = Inf, 
          label = sprintf("Mean: %.2f", mean_value), 
          vjust = 2, color = "red", size = 4
        ) +
        scale_x_continuous(trans='log10',
                           breaks=trans_breaks('log10', function(x) 10^x),
                           labels=trans_format('log10', math_format(10^.x))) +
        labs(
          title = sprintf("Histogram of %s", metrics),
          x = metrics,
          y = "Frequency"
        )
      
      ggsave(sprintf("plots/%s.png", metrics))
    },
    pattern = map(metrics),
    format = "file"
  )

)
