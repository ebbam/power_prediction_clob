"""Distortion magnitude and duration from a whale shock: Walrasian market vs CLOB.

Matched seeds give both mechanisms the same fundamental path and non-whale population.
Each mechanism's ordinary band is calibrated on its own no-whale control runs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bettor import add_whale, fundamental_path, run_market_lob
from experiments.whale_convergence_experiment import make_population
from metrics import (
    calibrate_band,
    distortion_metrics,
    normalised_pull,
    plot_distortion_figures,
    smoothed_absolute_error,
    summarize_distortion,
)
from walrasian import run_walrasian_market


N_RUNS = 30
HORIZON = 300
SHOCK_TIME = 0
WHALE_SHARE = 0.5
WHALE_VALUATION = 0.70
INITIAL_PRICE = 0.50
OUTCOME_UNCERTAINTY = 0.05
BASE_SEED = 2042
OUTPUT_DIR = Path(__file__).resolve().parents[1]   # repository root
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)


def election_path(seed):
    return fundamental_path(INITIAL_PRICE, HORIZON, OUTCOME_UNCERTAINTY,
                            np.random.default_rng(seed))


def population(seed, whale_share):
    """The whale experiment's bettors, joined from the start by a whale holding whale_share."""
    bettors = make_population(seed, whale_share)
    add_whale(bettors, whale_share, WHALE_VALUATION)
    return bettors


def run_walrasian(seed, path, whale_share):
    record = run_walrasian_market(population(seed, whale_share), HORIZON, path, INITIAL_PRICE)
    return np.asarray(record["price_history"][:HORIZON], dtype=float)


def run_clob(seed, path, whale_share):
    record = run_market_lob(population(seed, whale_share), HORIZON, path, INITIAL_PRICE,
                            rng=np.random.default_rng(seed))
    return np.asarray(record["midquote"], dtype=float)


MODELS = {"Walrasian": run_walrasian, "CLOB": run_clob}


def main():
    smoothed, pull = {}, {}
    for model, run in MODELS.items():
        for share in (0.0, WHALE_SHARE):
            print(f"{model}, whale share {share:.0%}")
            for i in range(N_RUNS):
                seed = BASE_SEED + i
                path = election_path(seed)
                prices = run(seed, path, share)
                smoothed[model, share, i] = smoothed_absolute_error(prices, path[:HORIZON])
                pull[model, share, i] = normalised_pull(prices, path[:HORIZON],
                                                        WHALE_VALUATION, SHOCK_TIME)

    rows = []
    for model in MODELS:
        band = calibrate_band(np.concatenate(
            [smoothed[model, 0.0, i] for i in range(N_RUNS)]))
        for share, label in ((0.0, "no whale"), (WHALE_SHARE, f"{WHALE_SHARE:.0%} whale")):
            for i in range(N_RUNS):
                rows.append({"model": f"{model}, {label}", "run": i,
                             "normalised_pull": pull[model, share, i],
                             **distortion_metrics(smoothed[model, share, i], SHOCK_TIME, band)})
    metrics = pd.DataFrame(rows)

    summary = summarize_distortion(metrics, group="model")
    summary.to_csv(SIM_DIR / "mechanism_distortion_summary.csv", index=False)
    print(summary.to_string(index=False))

    figure = plot_distortion_figures(metrics, group="model")
    figure.savefig(FIGURES_DIR / "mechanism_distortion.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "mechanism_distortion.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
