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
  packages = c("readr", "dplyr", "tidyr", "stringr", "posterior"),
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
  #   controller = crew::crew_controller_local(workers = 2, seconds_idle = 60)
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
    "test_galaxies.csv",
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
  tar_target(
    print_rows,
    print(nrow(sfr_data))
  ),
  
  # Step 2: Fit the Stan model
  tar_stan_mcmc(
    stan_fit,
    stan_files = "x.stan",
    data = list(
      N = nrow(sfr_data),
      logSFR_HEC_Gyr = sfr_data$logSFR_HEC_Gyr,
      logM_HEC = sfr_data$logM_HEC
    ),
    chains = 5,
    parallel_chains = 5,
    iter_sampling = 2500,
    iter_warmup = 1500,
    init = init_function(5,24)
  ),
  
  # Step 3: Extract summary statistics for key parameters
  tar_stan_summary(
    name = parameter_estimates,
    fit = stan_fit_mcmc_x,
    variables = c("t_today", "logA", "logtau"),
    summaries = list(
      mean = ~mean(.x),
      sd = ~sd(.x)
    )
  ),
  
  # Step 4: Compute residuals and standardized residuals for each data point
  tar_target(
    posterior_predictions,
    extract_posterior_predictions(stan_fit_mcmc_x, sfr_data),
  )
)