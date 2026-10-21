"""Distortion magnitude and duration figures from saved whale-experiment trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distortion import (
    calibrate_band,
    distortion_metrics,
    normalised_pull,
    plot_distortion_figures,
    smoothed_absolute_error,
    summarize_distortion,
)


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)
TRAJECTORIES = SIM_DIR / "whale_convergence_trajectories_300_periods.csv"
PRE_PERIOD = 50
WHALE_VALUATION = 0.70
RHO = "ρ (whale capital share)"


def whale_metrics():
    """Per-run distortion statistics and the ordinary band from the saved whale runs."""
    trajectories = pd.read_csv(TRAJECTORIES).sort_values(["whale_share", "run", "period"])
    paths = dict(list(trajectories.groupby(["whale_share", "run"])))
    smoothed = {
        key: smoothed_absolute_error(path["midquote"], path["fundamental"])
        for key, path in paths.items()
    }
    band = calibrate_band(np.concatenate(
        [error[:PRE_PERIOD] for (share, _), error in smoothed.items() if share == 0]))

    rows = []
    for (share, run), error in smoothed.items():
        rows.append({
            RHO: share,
            "run": run,
            **distortion_metrics(error, PRE_PERIOD, band),
            "normalised_pull": normalised_pull(
                paths[share, run]["midquote"], paths[share, run]["fundamental"],
                WHALE_VALUATION, PRE_PERIOD,
            ),
        })
    return pd.DataFrame(rows), band


def main():
    metrics, band = whale_metrics()
    summary = summarize_distortion(metrics, group=RHO)
    summary.to_csv(SIM_DIR / "whale_distortion_summary.csv", index=False)
    print(f"Ordinary band (95th pct of smoothed control pre-period error): {band:.3f}")
    print(summary.to_string(index=False))

    figure = plot_distortion_figures(metrics, group=RHO)
    figure.savefig(FIGURES_DIR / "whale_distortion.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "whale_distortion.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
