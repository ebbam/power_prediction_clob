"""Price, fundamental and smoothed error for a few matched whale runs, at every ρ."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distortion import calibrate_band, distortion_metrics, smoothed_absolute_error


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)
TRAJECTORIES = SIM_DIR / "whale_convergence_trajectories_300_periods.csv"
PRE_PERIOD = 50
WHALE_VALUATION = 0.70
PRICE_COLOR, FUNDAMENTAL_COLOR, ERROR_COLOR = "#0072B2", "#222222", "#D55E00"


def pick_runs(trajectories):
    """Runs whose fundamental ends up closest to, typical of, and furthest from 0.70."""
    post = trajectories.loc[(trajectories["whale_share"] == 0)
                            & (trajectories["period"] >= PRE_PERIOD)]
    gap = (post["fundamental"] - WHALE_VALUATION).abs().groupby(post["run"]).mean().sort_values()
    return {
        "Fundamental drifts toward 0.70": gap.index[0],
        "Typical fundamental path": gap.index[len(gap) // 2],
        "Fundamental drifts away from 0.70": gap.index[-1],
    }


def main():
    trajectories = pd.read_csv(TRAJECTORIES).sort_values(["whale_share", "run", "period"])
    trajectories["price"] = trajectories.groupby(["whale_share", "run"])["midquote"].ffill()
    trajectories["smoothed_error"] = np.concatenate([
        smoothed_absolute_error(path["midquote"], path["fundamental"])
        for _, path in trajectories.groupby(["whale_share", "run"])
    ])
    band = calibrate_band(trajectories.loc[
        (trajectories["whale_share"] == 0) & (trajectories["period"] < PRE_PERIOD),
        "smoothed_error",
    ])
    shares = sorted(trajectories["whale_share"].unique())
    runs = pick_runs(trajectories)

    figure, axes = plt.subplots(
        2 * len(runs), len(shares), figsize=(4.2 * len(shares), 3.0 * len(runs) + 1),
        sharex=True, gridspec_kw={"height_ratios": [2, 1] * len(runs)},
        constrained_layout=True,
    )
    for row, (description, run) in enumerate(runs.items()):
        for column, share in enumerate(shares):
            path = trajectories.loc[(trajectories["whale_share"] == share)
                                    & (trajectories["run"] == run)]
            metrics = distortion_metrics(path["smoothed_error"].to_numpy(), PRE_PERIOD, band)
            peak = PRE_PERIOD + metrics["peak_time_after_shock"]
            price_ax, error_ax = axes[2 * row, column], axes[2 * row + 1, column]

            price_ax.plot(path["period"], path["fundamental"], color=FUNDAMENTAL_COLOR,
                          linewidth=1.4, label="Fundamental (true probability)")
            price_ax.plot(path["period"], path["price"], color=PRICE_COLOR, linewidth=1.2,
                          label="Market price (midquote)")
            if share > 0:
                price_ax.axhline(WHALE_VALUATION, color="#888888", linewidth=0.8,
                                 linestyle="--", label="Whale valuation")
            price_ax.set(ylim=(0, 1))
            price_ax.set_title(f"ρ = {share:g}", fontsize=10)
            if column == 0:
                price_ax.set_ylabel(f"{description}\n(run {run})\nprice", fontsize=9)

            outside = path["smoothed_error"] > band
            error_ax.fill_between(path["period"], band, path["smoothed_error"], where=outside,
                                  color=ERROR_COLOR, alpha=0.25, linewidth=0,
                                  label="Outside band (P3)")
            error_ax.plot(path["period"], path["smoothed_error"], color=ERROR_COLOR,
                          linewidth=1.2, label="Smoothed |error|")
            error_ax.axhline(band, color="#888888", linewidth=0.8, linestyle="--",
                             label="Ordinary band")
            error_ax.plot(peak, metrics["max_error"], "v", color=ERROR_COLOR, markersize=7,
                          label="Maximum error (P1)")
            if metrics["reverted"]:
                error_ax.axvline(peak + metrics["reversion_time"], color=ERROR_COLOR,
                                 linewidth=1, linestyle=":", label="Reversion (P2)")
            error_ax.set(ylim=(0, 0.85))
            if column == 0:
                error_ax.set_ylabel("|error|", fontsize=9)

            for ax in (price_ax, error_ax):
                ax.axvline(PRE_PERIOD, color="#BBBBBB", linewidth=0.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Period (whale enters at 50)", fontsize=9)
    handles, labels = [], []
    for ax in (axes[0, -1], axes[1, -1]):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle), labels.append(label)
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False,
                  fontsize=9)
    figure.savefig(FIGURES_DIR / "whale_sample_runs.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
