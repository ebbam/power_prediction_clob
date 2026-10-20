"""Distortion magnitude and duration figures from saved whale-experiment trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distortion import (
    calibrate_band,
    distortion_metrics,
    plot_distortion_duration,
    plot_distortion_magnitude,
    smoothed_absolute_error,
    summarize_distortion,
)


OUTPUT_DIR = Path(__file__).resolve().parent
TRAJECTORIES = OUTPUT_DIR / "whale_convergence_trajectories_300_periods.csv"
PRE_PERIOD = 50


def main():
    trajectories = pd.read_csv(TRAJECTORIES).sort_values(["whale_share", "run", "period"])
    smoothed = {
        key: smoothed_absolute_error(path["midquote"], path["fundamental"])
        for key, path in trajectories.groupby(["whale_share", "run"])
    }
    band = calibrate_band(np.concatenate(
        [error[:PRE_PERIOD] for (share, _), error in smoothed.items() if share == 0]))

    rows = []
    for (share, run), error in smoothed.items():
        rows.append({
            "whale share": f"{share:.0%}",
            "run": run,
            **distortion_metrics(error, PRE_PERIOD, band),
        })
    metrics = pd.DataFrame(rows)

    summary = summarize_distortion(metrics, group="whale share")
    summary["ordinary_band"] = band
    summary.to_csv(OUTPUT_DIR / "whale_distortion_summary.csv", index=False)
    print(f"Ordinary band (95th pct of smoothed control pre-period error): {band:.3f}")
    print(summary.to_string(index=False))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    plot_distortion_magnitude(metrics, group="whale share", ax=left)
    plot_distortion_duration(metrics, group="whale share", ax=right)
    figure.savefig(OUTPUT_DIR / "whale_distortion.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "whale_distortion.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
