# R script for Homework 1
#
# Author: Juanwu Lu
# Date:   2024-08-24
library(moments)

# Control the global seed
set.seed(42)

# Define the helper functions
ar_loglik <- function(rho, log_sig) {
  y <- .GlobalEnv$y
  n <- length(y)
  sig <- exp(log_sig)
  rho <- rep(as.numeric(rho), times = n - 1)
  denom <- n * log(2 * pi) + 2 * n * log_sig
  log_lik <- -0.5 * (
    denom + (y[1]^2 + sum((y[2:n] - rho * y[1:(n - 1)])^2)) / sig^2
  )
  return(log_lik)
}
ar_logpost <- function(rho, log_sig) {
  log_prior <- -log(2) - 0.5 * (log(2 * pi) + log(100) + log_sig^2 / 100)
  log_post <- ar_loglik(rho, log_sig) + log_prior
  return(log_post)
}
ar_post_predictive <- function(rho, log_sig) {
  # Sample from the posterior predictive distribution
  y <- .GlobalEnv$y
  n <- length(y)
  new_y <- rep(0.0, times = n)
  sig <- exp(log_sig)
  for (i in 2:n) {
    new_y[i] <- rnorm(n = 1, mean = rho * new_y[i - 1], sd = sig)
  }
  return(new_y)
}

# ================================================================
# Entry point for problem 1 - Synthetic data
# Read the data
if (file.exists("data/computation_data_hw_1.csv")) {
  data <- read.csv("data/computation_data_hw_1.csv")
  y <- data[["x"]]
} else {
  stop("File not found: 'computation_data_hw_1.csv' at ", getwd())
}
# --- problem 1.2 ---
rho <- seq(-0.99, 0.99, length = 100)
log_sig <- seq(-1.0, 0.0, length = 100)
log_lik <- outer(rho, log_sig, Vectorize(ar_loglik))
contour(
  x = rho,
  y = log_sig,
  z = log_lik,
  xlab = expression(rho),
  ylab = expression(log(sigma)),
  nlevels = 20,
  axes = FALSE
)
axis(side = 1, at = seq(-1.0, 1.0, by = 0.25))
axis(side = 2, at = seq(-1.0, 0.0, by = 0.10))

# --- problem 1.3 ---
log_post <- outer(rho, log_sig, Vectorize(ar_logpost))
contour(
  x = rho,
  y = log_sig,
  z = log_post,
  xlab = expression(rho),
  ylab = expression(log(sigma)),
  nlevels = 20,
  axes = FALSE,
)
axis(side = 1, at = seq(-1.0, 1.0, by = 0.25))
axis(side = 2, at = seq(-1.0, 0.0, by = 0.10))

# --- problem 1.4 - 1.5 ---
rho_grid <- seq(0.25, 0.75, length = 100)
log_sig_grid <- seq(-0.9, 0.4, length = 100)
grid_log_post <- outer(rho_grid, log_sig_grid, Vectorize(ar_logpost))
# Calculate the normalized probability density
probs <- exp(grid_log_post - max(grid_log_post))
probs <- probs / sum(probs)
# Randomly sample from the grid
indices <- sample(
  x = seq_len(length(as.vector(probs))),
  size = 1000,
  replace = TRUE,
  prob = probs
)
rho_sample <- rho_grid[((indices - 1) %% nrow(probs)) + 1]
log_sig_sample <- log_sig_grid[((indices - 1) %/% nrow(probs)) + 1]
# Calculate summaries for rho
print(quantile(rho_sample, probs = c(0.025, 0.25, 0.5, 0.75, 0.975)))
sprintf("Mean: %.4f", mean(rho_sample))
sprintf("Standard Deviation: %.4f", sd(rho_sample))
sprintf("Skewnewss: %.4f", skewness(rho_sample))
sprintf("Kurtosis: %.4f", kurtosis(rho_sample))
# Calculate summaries for log(sigma)
print(quantile(log_sig_sample, probs = c(0.025, 0.25, 0.5, 0.75, 0.975)))
sprintf("Mean: %.4f", mean(log_sig_sample))
sprintf("Standard Deviation: %.4f", sd(log_sig_sample))
sprintf("Skewness: %.4f", skewness(log_sig_sample))
sprintf("Kurtosis: %.4f", kurtosis(log_sig_sample))

# --- problem 1.6 - 1.7 ---
params <- data.frame(rho = rho_sample, log_sig = log_sig_sample)
samples <- matrix(NA, nrow = nrow(params), ncol = length(y))
for (i in seq_len(nrow(params))) {
  samples[i, ] <- ar_post_predictive(params[i, "rho"], params[i, "log_sig"])
}
plot(
  x = seq_len(length(y)),
  y = samples[1, ],
  col = "gray",
  type = "l",
  lwd = 0.75,
  xlab = "Time",
  ylab = "Value",
  ylim = c(-3.0, 3.0)
)
for (i in 2:100) {
  lines(x = seq_len(length(y)), y = samples[i, ], col = "gray", lwd = 0.75)
}
lines(x = seq_len(length(y)), y = y, col = "blue", lwd = 1.0)
legend(
  "topright",
  legend = c("Observed", "Posterior Predictive Samples"),
  col = c("blue", "gray"),
  lwd = c(1, 0.5),
  bg = "white",
)

# ================================================================
# Entry point for problem 2 - Real data
# Read the data
if (file.exists("data/covid_us.txt")) {
  data <- read.table("data/covid_us.txt", header = TRUE, sep = ",")
  y <- data[["x"]]
} else {
  stop("File not found: 'computation_data_hw_1.csv' at ", getwd())
}
