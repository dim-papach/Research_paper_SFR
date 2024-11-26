# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes) # Load other packages as needed.
library(stantargets)

# Set target options:
tar_option_set(
  packages = c("readr", "dplyr", "tidyr", "stringr", "posterior","bayesplot", "loo","ggplot2"),
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
list(
  # Step 1: Load and prepare data
  tar_target(
    csv_file_path,
    "outer_join.csv",
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
    data = list(
      N = nrow(sfr_data),
      logSFR_UNGC_Gyr = sfr_data$logSFR_UNGC_Gyr,
      id_numbers = sfr_data$id_number
    ),
    chains = 5,
    parallel_chains = 5,
    iter_sampling = 3000,
    iter_warmup = 2500,
    init = init_function(5,1137)
  ),
  
  tar_stan_summary(
    id_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_UNGC_Gyr = sfr_data$logSFR_UNGC_Gyr,
      id_numbers = sfr_data$id_number
    ),
    variables = "id"
  ),
  
  tar_stan_summary(
    sfr_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_UNGC_Gyr = sfr_data$logSFR_UNGC_Gyr,
      id_numbers = sfr_data$id_number
    ),
    variables = "logSFR_today"
  ),

  tar_stan_summary(
    t_sf_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_UNGC_Gyr = sfr_data$logSFR_UNGC_Gyr,
      id_numbers = sfr_data$id_number
    ),
    variables = "t_sf"),
    
  tar_stan_summary(
    tau_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_UNGC_Gyr = sfr_data$logSFR_UNGC_Gyr,
      id_numbers = sfr_data$id_number
    ),
    variables = "tau"
  ),
  
  tar_stan_summary(
    A_summary,
    fit = stan_fit_mcmc_x,
    data = list(
      N = nrow(sfr_data),
      logSFR_UNGC_Gyr = sfr_data$logSFR_UNGC_Gyr,
      id_numbers = sfr_data$id_number
    ),
    variables = "A"
  ),
  
  tar_target(
    divergences,
    stan_fit_diagnostics_x$divergent__
  )
)
