# !/usr/bin/python
# Copyright (c) 2024, Juanwu Lu
# All rights reserved.
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import pandas as pd


def main() -> None:
    global DATA_ROOT, IMG_ROOT, GENERATOR

    dat_us = pd.read_csv(DATA_ROOT.joinpath("covid_us.txt"), sep=",")
    dat_us["date"] = pd.to_datetime(dat_us["date"], format="%Y-%m-%d")

    # Visualize the data
    history = dat_us.loc[dat_us["date"] <= pd.to_datetime("2020-06-30")]
    future = dat_us.loc[
        (dat_us["date"] > pd.to_datetime("2020-06-30"))
        & (dat_us["date"] <= pd.to_datetime("2020-08-25"))
    ]
    y_history = history["cases"].values
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    history.plot(
        x="date", y="cases", ax=axes[0], title="Historical Cases", marker="."
    )
    history.plot(
        x="date",
        y="cases",
        ax=axes[1],
        title="Historical Cases (Log-Scale)",
        marker=".",
        logy=True,
    )
    omega = np.zeros(*y_history.shape)
    omega[1:] = np.diff(y_history, n=1) / 1e5
    axes[2].plot(
        omega[:-1],
        omega[1:],
        marker="o",
        mec="blue",
        mfc="lightblue",
    )
    axes[2].set_xlabel(r"$\omega_{t-1}$")
    axes[2].set_ylabel(r"$\omega_{t}$")
    axes[2].set_title("Log-Difference")
    fig.tight_layout()
    fig.savefig(IMG_ROOT.joinpath("historical_cases.png"))

    # Visualize the contour of log-likelihood and log-posterior
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    rho = np.linspace(0.01, 2.0, 100)
    log_sigma = np.linspace(-5.0, 0.0, 100)
    RHO, LOG_SIGMA = np.meshgrid(rho, log_sigma)
    log_likelihood: npt.NDArray = -0.5 * np.sum(
        np.log(2 * np.pi)
        + 2 * LOG_SIGMA[None, :, :]
        + np.power(
            omega[1:, None, None] - RHO[None, :, :] * omega[:-1, None, None], 2
        )
        / np.exp(2 * LOG_SIGMA[None, :, :]),
        axis=0,
    )
    cbar = axes[0].contourf(
        RHO, LOG_SIGMA, log_likelihood, levels=20, cmap="plasma"
    )
    fig.colorbar(cbar, ax=axes[0])
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"$\log(\sigma)$")
    axes[0].set_title("Log-Likelihood")

    log_posterior = log_likelihood - RHO - 0.5 * np.power(LOG_SIGMA, 2)
    cbar = axes[1].contourf(
        RHO, LOG_SIGMA, log_posterior, levels=20, cmap="plasma"
    )
    fig.colorbar(cbar, ax=axes[1])
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel(r"$\log(\sigma)$")
    axes[1].set_title("Unnormalized Log-Posterior Density")

    fig.tight_layout()
    fig.savefig(IMG_ROOT.joinpath("log_density.png"))

    # Simulate from posterior predictive distribution
    n_sim = 1000
    probs: npt.NDArray = np.exp(log_posterior - np.max(log_posterior))
    probs /= np.sum(probs)
    indices = np.random.choice(
        np.arange(np.prod(log_posterior.shape)),
        size=n_sim,
        replace=True,
        p=probs.flatten(),
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    cbar = axes[0].contourf(
        RHO, LOG_SIGMA, log_likelihood, levels=20, cmap="plasma"
    )
    fig.colorbar(cbar, ax=axes[0])
    axes[0].scatter(
        RHO.flatten()[indices],
        LOG_SIGMA.flatten()[indices],
        s=21,
        ec="red",
        fc="pink",
        label="Simulated Parameters",
    )
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"$\log(\sigma)$")
    axes[0].legend()

    rho_sim = RHO.flatten()[indices]
    log_sigma_sim = LOG_SIGMA.flatten()[indices]
    timestep = np.arange(omega.shape[0])
    omega_sim = np.zeros((n_sim, *omega.shape))
    for i in range(n_sim):
        print(rho_sim[i], np.exp(log_sigma_sim[i]))
        for j in range(omega.shape[0]):
            omega_sim[i, j] = GENERATOR.normal(
                loc=rho_sim[i] * omega[j],
                scale=np.exp(log_sigma_sim[i]),
            )
    axes[1].plot(timestep, omega, label="Observed")
    sim_data = pd.DataFrame.from_dict(
        {
            "timestep": np.tile(timestep, n_sim),
            "omega": omega_sim.flatten(),
            "sim_id": np.repeat(np.arange(n_sim), omega.shape[0]),
        }
    )
    sns.lineplot(
        x="timestep",
        y="omega",
        data=sim_data,
        ax=axes[1],
        alpha=0.5,
        label="Simulated",
    )

    fig.tight_layout()
    fig.savefig(IMG_ROOT.joinpath("posterior_predictive.png"))


if __name__ == "__main__":
    GENERATOR = np.random.default_rng(seed=42)
    DATA_ROOT = Path(__file__).parent.joinpath("data")
    IMG_ROOT = Path(__file__).parent.joinpath("img")
    main()
