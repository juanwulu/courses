# R script for Homework 1
#
# Author: Juanwu Lu
# Date:   2024-08-24
library(moments)

# Control the global seed
set.seed(42)

# Read the data
if (file.exists("computation_data_hw_1.csv")) {
    data <- read.csv("computation_data_hw_1.csv")
    y <- data[['x']]
} else {
    stop("File not found: 'computation_data_hw_1.csv' at ", getwd())
}

# Define the helper functions
ar_loglik <- function(rho, log_sig) {
    n <- length(y)
    sig <- exp(log_sig)
    rho <- rep(as.numeric(rho), times = n - 1)
    log_lik <- -0.5 * (
        n * log(2 * pi)
        + 2 * n * log_sig
        + (y[1]^2 + sum((y[2:n] - rho * y[1:(n-1)])^2)) / sig^2
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
    n <- length(y)
    new_y <- rep(0.0, times = n)
    sig <- exp(log_sig)
    for (i in 2:n) {
        new_y[i] <- rnorm(n=1, mean=rho * new_y[i-1], sd=sig)
    }
    return(new_y)
}

# Entry point for problem 1 - Synthetic data
# --- problem 1.2 ---
rho <- seq(-0.99, 0.99, length=100)
log_sig <- seq(-1.0, 0.0, length=100)
log_lik <- outer(rho, log_sig, Vectorize(ar_loglik))
contour(
    x=rho,
    y=log_sig,
    z=log_lik,
    xlab=expression(rho),
    ylab=expression(log(sigma)),
    nlevels=20,
)

# --- problem 1.3 ---
log_post <- outer(rho, log_sig, Vectorize(ar_logpost))
contour(
    x=rho,
    y=log_sig,
    z=log_post,
    xlab=expression(rho),
    ylab=expression(log(sigma)),
    nlevels=20,
)

# --- problem 1.4 - 1.5 ---
rho_grid <- sample(x=seq(0.0, 1.0, length=5000), size=1000)
log_sig_grid <- sample(x=seq(-0.9, 0.0, length=5000), size=1000)
# Calculate summaries for rho
print(quantile(rho_grid, probs=c(0.025, 0.25, 0.5, 0.75, 0.975)))
sprintf("Mean: %.4f", mean(rho_grid))
sprintf("Standard Deviation: %.4f", sd(rho_grid))
sprintf("Skewnewss: %.4f", skewness(rho_grid))
sprintf("Kurtosis: %.4f", kurtosis(rho_grid))
# Calculate summaries for log(sigma)
print(quantile(log_sig_grid, probs=c(0.025, 0.25, 0.5, 0.75, 0.975)))
sprintf("Mean: %.4f", mean(log_sig_grid))
sprintf("Standard Deviation: %.4f", sd(log_sig_grid))
sprintf("Skewness: %.4f", skewness(log_sig_grid))
sprintf("Kurtosis: %.4f", kurtosis(log_sig_grid))

# --- problem 1.6 - 1.7 ---
params <- data.frame(rho=rho_grid, log_sig=log_sig_grid)
samples <- matrix(NA, nrow=nrow(params), ncol=length(y))
for (i in 1:nrow(params)) {
    samples[i,] <- ar_post_predictive(params[i, 'rho'], params[i, 'log_sig'])
}
plot(
    x=1:length(y),
    y=samples[1,],
    col='gray',
    type='l',
    lwd=0.75,
    xlab='Time',
    ylab='Value',
    ylim=c(-3.0, 3.0)
)
for (i in 2:100) {
    lines(x=1:length(y), y=samples[i,], col='gray', lwd=0.75)
}
lines(x=1:length(y), y=y, col='blue', lwd=1.0)
legend(
    'topright',
    legend=c('Observed', 'Posterior Predictive Samples'),
    col=c('blue', 'gray'),
    lwd=c(1, 0.5),
    bg='white',
)
