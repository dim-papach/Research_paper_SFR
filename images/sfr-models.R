library(ggplot2)
library(dplyr)
library(latex2exp)


# Define time values (positive times)
t <- seq(0.01, 12, length.out = 200)

# Define SFR models
exponential_decline <- function(t, tau) exp(-t / tau)
delayed_exponential <- function(t, tau) t * exp(-t / tau)
log_normal <- function(t, tau, T0) (1 / tau) * exp(-((log(t) - log(T0))^2) / (2 * tau^2))
double_power_law <- function(t, tau, alpha, beta) 1 / ((t / tau)^alpha + (t / tau)^(-beta))

# Generate data for each model
data <- data.frame(
  t = rep(t, 4),
  SFR = c(
    exponential_decline(t, tau = 2),
    delayed_exponential(t, tau = 2),
    log_normal(t, tau = 2, T0 = 10),
    double_power_law(t, tau = 2, alpha = 2, beta = 5)
  ),
  Model = rep(c("\nExponential Decline\n(τ=2 Gyr)", "\nDelayed Exponential\n(τ=2 Gyr)", "\nLog-Normal\n(τ = 2 Gyr, To = 10 Gyr)", "\nDouble Power Law\n(τ = 2 Gyr, α = 2, β = 5)"), each = length(t))
)

# Plot all models
ggplot(data, aes(x = t, y = SFR, color = Model)) +
  geom_line(size = 1) +
  labs(
    title = "Star Formation Rate Models",
    x = TeX("Star Formation time ( $t_{sf}$ ) [Gyr]"),
    y = TeX("SFR($t_{sf}$) [ M$_*$/yr ]"),
    color = "Model",
  ) +
  scale_x_continuous(limits = c(0, 12)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("./images/sfr-models.png", width = 10, height = 6, dpi = 300)
