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
  packages = c("readr", "tidyverse", "tibble", "stringr", "posterior", "bayesplot", "loo", "ggplot2", "scales", "latex2exp", "ggeasy"),
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
  # controller = crew::crew_controller_local(workers = 6)
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
# list(
tar_plan(

  #### Load and prepare data####
  tar_target(
    csv_file_path,
    "../tables/filled.csv",
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

  #### Fit the Stan model####
  tar_stan_mcmc(
    stan_fit,
    stan_files = "x.stan",
    # stdout = "out.txt",
    # stderr = "error.txt",
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      ID = sfr_data$ID,
      M_star = sfr_data$M_total
    ),
    chains = 6,
    parallel_chains = 8,
    iter_sampling = 5500,
    iter_warmup = 2500,
    init = init_function(6, 1761),
  ),

  ######## summaries#######

  tar_stan_summary(
    id_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      ID = sfr_data$ID
    ),
    variables = "id"
  ),
  tar_stan_summary(
    sfr_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_today = sfr_data$logSFR_today,
      ID = sfr_data$ID
    ),
    variables = "logSFR_today"
  ),
  tar_stan_summary(
    t_sf_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      ID = sfr_data$ID
    ),
    variables = "t_sf"
  ),
  tar_stan_summary(
    tau_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      ID = sfr_data$ID
    ),
    variables = "tau"
  ),
  tar_stan_summary(
    A_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      ID = sfr_data$ID
    ),
    variables = "A"
  ),
  tar_stan_summary(
    x_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_total = sfr_data$logSFR_total,
      ID = sfr_data$ID
    ),
    variables = "x"
  ),
  tar_target(
    divergences,
    stan_fit_diagnostics_x$divergent__
  ),

  ######## Define the list of variables########
  tar_target(
    summary_variables,
    c("logSFR_pred", "t_sf", "tau", "A") # List of variables
  ),
  tar_target(
    summary_data,
    {
      summary_data <- switch(summary_variables,
        "A" = A_summary,
        "tau" = tau_summary,
        "t_sf" = t_sf_summary,
        "logSFR_pred" = sfr_summary,
        # "logSFR_total" = sfr_summary_t,
      )

      summary_data
    },
    pattern = map(summary_variables) # Branch over variables
  ),
  tar_target(
    summary_names, # in latex2exp format
    {
      switch(summary_variables,
        logSFR_pred = expression(log[10] * bgroup("[", frac(SFR[obs], M["\u2299"] / yr), "]")),
        log_tsf = expression(log[10] * bgroup("[", frac(t[sf], "Gyr"), "]")),
        t_sf = expression(t[sf] * " [Gyr]"),
        tau = expression(tau * " [Gyr]"),
        logA = expression(log * bgroup("[", frac(A[del], M["\u2299"] / yr), "]")),
        logtau = expression(log[10] * bgroup("[", frac(tau, "Gyr"), "]"))
      )
    },
    pattern = map(summary_variables) # Branch over variables
  ),


  ######## plots########
  #### Compute plots dynamically for mean vs median ####
  tar_target(
    mean_median_plots,
    {
      # Extract summary data for the variable

      # Generate the plot
      mean_vs_median(summary_data, summary_variables)
    },
    pattern = map(summary_variables, summary_data) # Branch over variables
  ),

  # Compute histograms dynamically
  tar_target(
    histograms,
    {
      plot_histograms(summary_data, summary_variables, summary_names)
    },
    pattern = map(summary_data, summary_variables, summary_names) # Branch over variables
  ),


  # Prepare the metric data

  tar_target(
    metrics,
    c("rhat", "ess_bulk", "ess_tail")
  ),
  tar_target(
    hist_metrics,
    plot_metrics(summary_data, summary_variables, metrics),
    pattern = cross(map(summary_data, summary_variables), metrics),
    format = "file"
  ),

  #### join the data####
  tar_target(
    all_means,
    {
      # Create a tibble with mean values from each summary, and their corresponding sigma values
      summary_means <- tibble::tibble(
        ID = id_summary$mean, # Assuming id is the identifier
        logSFR_pred = sfr_summary$mean,
        logSFR_pred_sigma = sfr_summary$sd,
        t_sf = t_sf_summary$mean,
        t_sf_sigma = t_sf_summary$sd,
        tau = tau_summary$mean,
        tau_sigma = tau_summary$sd,
        A = A_summary$mean,
        A_sigma = A_summary$sd,
        x = x_summary$mean,
        x_sigma = x_summary$sd
      )


      # Return the table
      summary_means
    }
  ),
  # Join with sfr_data based on 'id'
  tar_target(
    joined_data,
    {
      # Join the data
      joined_data <- dplyr::left_join(sfr_data, all_means, by = "ID")

      # Return the joined data
      joined_data
    }
  ),


  #### Comparisons####

  # SFRs should be almost equal. Target to flag the rows with big differences
  tar_target(
    sfr_diff,
    {
      # Flag the rows with big differences
      joined_data %>%
        mutate(
          diff = logSFR_pred - logSFR_total,
          flag = ifelse(abs(diff) < 2, "Keep", "Discard")
        ) %>%
        # return joined with all the columns + the new column SFR_flag
        select(everything())
    }
  ),
  # plot the all SFRs and color based on flag
  tar_target(
    sfr_diff_plot,
    sfr_comparison_plot(sfr_diff, logSFR_pred, logSFR_total)
  ),
  tar_target(
    save_joined_data,
    {
      # Save the joined data
      write.csv(sfr_diff, "joined_data.csv")
    }
  )
)
