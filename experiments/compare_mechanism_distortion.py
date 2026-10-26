"""Distortion magnitude and duration from a whale shock: Walrasian market vs CLOB.

Matched seeds give both mechanisms the same fundamental path and non-whale population.
Each mechanism's ordinary band is calibrated on its own no-whale control runs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bettor import bettor, run_market_fixed_el, run_market_lob
from metrics import (
    calibrate_band,
    distortion_metrics,
    normalised_pull,
    plot_distortion_figures,
    smoothed_absolute_error,
    summarize_distortion,
)


N_RUNS = 30
N_BETTORS = 100
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
    rng = np.random.default_rng(seed)
    path = [INITIAL_PRICE]
    for _ in range(HORIZON):
        path.append(np.clip(path[-1] + rng.normal(0, OUTCOME_UNCERTAINTY), 0, 1))
    return np.asarray(path)


def population(seed, whale_share):
    """Non-whale capital scaled so the whale holds `whale_share` of the total."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    return [
        bettor(budget=rng.uniform(100, 1000) * (1 - whale_share),
               market_valuation=np.clip(rng.normal(0.50, 0.05), 0.01, 0.99))
        for _ in range(N_BETTORS)
    ]


def run_walrasian(seed, path, whale_share):
    bettors = population(seed, whale_share)
    if whale_share > 0:
        capital = sum(b.budget for b in bettors)
        bettors.append(bettor(budget=capital * whale_share / (1 - whale_share),
                              market_valuation=WHALE_VALUATION, risk_av=0.5, whale=True))
    # n_bettors must cover the whale, or its fills are never booked.
    record = run_market_fixed_el(len(bettors), HORIZON, INITIAL_PRICE, OUTCOME_UNCERTAINTY,
                                 bettors, path)
    return np.asarray(record["price_history"][:HORIZON], dtype=float)


def run_clob(seed, path, whale_share):
    record = run_market_lob(N_BETTORS, HORIZON, INITIAL_PRICE, OUTCOME_UNCERTAINTY,
                            population(seed, whale_share), gen_el=path,
                            rng=np.random.default_rng(seed), whale_share=whale_share,
                            whale_start=SHOCK_TIME, whale_valuation=WHALE_VALUATION)
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
