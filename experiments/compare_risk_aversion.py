"""Whale experiment: risk aversion drawn as before vs every agent at r = 1 (log utility).

Row 1 uses the saved whale runs (ordinary agents r ~ U(0, 1), whale r = 0.5). Row 2 reruns the
same seeds, populations and fundamental paths with every agent, whale included, at r = 1; the
whale's r is set inside run_market_lob, so it is overridden here through the bettor class
rather than by editing bettor.py. The r = 1 runs (~4 min) are cached in
sim_output/whale_log_utility_runs.csv; pass --rerun to redo them.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bettor as bettor_module
import experiments.whale_convergence_experiment as experiment
from distortion import (
    calibrate_band,
    distortion_metrics,
    normalised_pull,
    plot_max_error,
    plot_normalised_pull,
    plot_time_outside_band,
    smoothed_absolute_error,
)
from paper_figures import WHALE_SHARE_COLORS
from experiments.whale_convergence_experiment import RHO, whale_metrics


OUTPUT_DIR = Path(__file__).resolve().parents[1]   # repository root
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)
LOG_UTILITY_CSV = SIM_DIR / "whale_log_utility_runs.csv"
ROWS = ("Risk aversion as before: agents r ~ U(0, 1), whale r = 0.5",
        "All agents, whale included, r = 1")


class LogUtilityBettor(bettor_module.bettor):
    def __init__(self, *args, **kwargs):
        kwargs["risk_av"] = 1.0
        super().__init__(*args, **kwargs)


def run_log_utility():
    original = bettor_module.bettor
    # run_market_lob builds the whale, and make_population the agents, from these names.
    bettor_module.bettor = experiment.bettor = LogUtilityBettor
    try:
        smoothed, paths = {}, {}
        for share in experiment.WHALE_SHARES:
            print(f"r = 1, ρ = {share:g}")
            for run in range(experiment.N_RUNS):
                seed = experiment.BASE_SEED + run
                path = experiment.make_election_path(seed)
                record = bettor_module.run_market_lob(
                    experiment.N_BETTORS, experiment.SIMULATION_HORIZON,
                    experiment.INITIAL_PRICE, experiment.OUTCOME_UNCERTAINTY,
                    experiment.make_population(seed, share),
                    market_error=experiment.MARKET_ERROR, tick=experiment.TICK,
                    ttl=experiment.TTL, max_markup=experiment.MAX_MARKUP,
                    rng=np.random.default_rng(seed), gen_el=path, whale_share=share,
                    whale_start=experiment.PRE_PERIOD,
                    whale_valuation=experiment.WHALE_VALUATION,
                )
                prices = np.asarray(record["midquote"], dtype=float)
                paths[share, run] = (prices, path[:experiment.SIMULATION_HORIZON])
                smoothed[share, run] = smoothed_absolute_error(*paths[share, run])
    finally:
        bettor_module.bettor = experiment.bettor = original

    band = calibrate_band(np.concatenate(
        [error[:experiment.PRE_PERIOD] for (share, _), error in smoothed.items() if share == 0]))
    metrics = pd.DataFrame([
        {RHO: share, "run": run,
         **distortion_metrics(error, experiment.PRE_PERIOD, band),
         "normalised_pull": normalised_pull(*paths[share, run], experiment.WHALE_VALUATION,
                                            experiment.PRE_PERIOD)}
        for (share, run), error in smoothed.items()
    ])
    metrics.to_csv(LOG_UTILITY_CSV, index=False)
    return metrics


def main():
    current, _ = whale_metrics()
    log_utility = (pd.read_csv(LOG_UTILITY_CSV)
                   if LOG_UTILITY_CSV.exists() and "--rerun" not in sys.argv
                   else run_log_utility())

    figure, axes = plt.subplots(2, 3, figsize=(18, 9.6), constrained_layout=True)
    for row, (label, metrics) in enumerate(zip(ROWS, (current, log_utility))):
        for ax, plot in zip(axes[row], (plot_max_error, plot_normalised_pull,
                                        plot_time_outside_band)):
            plot(metrics, RHO, ax, palette=WHALE_SHARE_COLORS)
            ax.set_title(f"{label}\n{ax.get_title()}", fontsize=10)
    for column in range(3):
        left = min(ax.get_xlim()[0] for ax in axes[:, column])
        right = max(ax.get_xlim()[1] for ax in axes[:, column])
        top = max(ax.get_ylim()[1] for ax in axes[:, column])
        for ax in axes[:, column]:
            ax.set(xlim=(left, right), ylim=(0, top))
    figure.savefig(FIGURES_DIR / "risk_aversion_comparison.png", dpi=200, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "risk_aversion_comparison.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
