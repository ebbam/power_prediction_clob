"""
Repeated CLOB simulations without a whale, summarising per run: price accuracy, submitted
interest, executed volume, outstanding liquidity, bid/ask spread and first-passage
convergence time (also in units of a reference timescale: the Kaplan-Meier median
first-passage time across runs). The market opens MARKET_ERROR above the fundamental.

Run from the repository root with `python3 -m experiments.run`. Saves
sim_output/clob_simulation_summary.csv and four figures of the first run.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bettor import Bettor, fundamental_path, run_market_lob
from metrics import first_passage_convergence, km_median

N_RUNS = 100
N_BETTORS = 100
N_PERIODS = 100
INITIAL_PRICE = 0.50
OUTCOME_UNCERTAINTY = 0.05
MARKET_ERROR = 0.10
CONVERGENCE_TOLERANCE = 0.02
CONVERGENCE_WINDOW = 5
BASE_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[1]   # repository root
FIGURES_DIR = OUTPUT_DIR / "figures"
SIM_DIR = OUTPUT_DIR / "sim_output"


def run_one(seed):
    """One market. Budgets and valuations come from rng; Bettor and update_belief draw the
    other traits and the signals from np.random, so both are seeded."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    bettors = [Bettor(budget=rng.uniform(100, 1000),
                      market_valuation=np.clip(rng.normal(0.50, 0.05), 0.01, 0.99))
               for _ in range(N_BETTORS)]
    path = fundamental_path(INITIAL_PRICE, N_PERIODS, OUTCOME_UNCERTAINTY)
    return run_market_lob(bettors, N_PERIODS, path, INITIAL_PRICE + MARKET_ERROR, rng=rng)


def convergence(record):
    return first_passage_convergence(record["midquote"], record["fundamental"][:N_PERIODS],
                                     CONVERGENCE_TOLERANCE, CONVERGENCE_WINDOW)


def summarise(record, timescale):
    """One row of run statistics; errors and spreads only over two-sided quotes."""
    conv = convergence(record)
    error = conv["absolute_error"]
    spreads = record["spread"][np.isfinite(record["spread"])]
    first = conv["first_passage_time"]
    submitted, executed = record["submitted_volume"].sum(), record["traded_volume"].sum()
    return {
        "converged": conv["converged"],
        "first_passage_time": first,
        "confirmation_time": conv["confirmation_time"],
        "dimensionless_convergence_time": first / timescale,
        "executed_volume_to_convergence": (
            np.cumsum(record["traded_volume"])[first] if conv["converged"] else np.nan),
        "mean_desired_trades_dropped_per_period": np.mean(record["n_desired_trade_dropped"]),
        "mean_absolute_error": np.nanmean(error),
        "rmse": np.sqrt(np.nanmean(error ** 2)),
        "total_submitted_volume": submitted,
        "total_executed_volume": executed,
        "execution_rate": executed / submitted if submitted else np.nan,
        "total_trades": record["n_trades"].sum(),
        "mean_bid_depth": np.mean(record["bid_depth"]),
        "mean_ask_depth": np.mean(record["ask_depth"]),
        "mean_total_depth": np.mean(record["total_depth"]),
        "mean_spread": spreads.mean() if spreads.size else np.nan,
        "two_sided_quote_fraction": np.mean(np.isfinite(record["midquote"])),
    }


def reference_timescale(records):
    """Kaplan-Meier median first-passage time, with unconverged runs right-censored at
    N_PERIODS (dropping them would bias it downwards); NaN if under half the runs converge."""
    passages = [convergence(record) for record in records]
    converged = np.array([p["converged"] for p in passages])
    times = np.where(converged, [p["first_passage_time"] for p in passages], N_PERIODS)
    return km_median(times, converged, N_PERIODS), converged.mean()


def plot_example(record, timescale):
    """Four diagnostics of one run, in units of the reference timescale when identified."""
    identified = np.isfinite(timescale)
    time = np.arange(N_PERIODS) / (timescale if identified else 1.0)
    label = r"Dimensionless time $t/\tau_{ref}$" if identified else "Simulation period"
    fundamental = record["fundamental"][:N_PERIODS]
    plots = {
        "clob_price_discovery": ("CLOB Price Discovery", "Probability",
                                 {"Fundamental probability": fundamental,
                                  "CLOB midquote": record["midquote"]}),
        "clob_pricing_error": ("Pricing Error Through Time", "Absolute pricing error",
                               {"Absolute error": np.abs(record["midquote"] - fundamental)}),
        "clob_market_activity": ("Submitted and Executed Trading Activity", "Contracts",
                                 {"Submitted volume": record["submitted_volume"],
                                  "Executed volume": record["traded_volume"]}),
        "clob_liquidity": ("CLOB Outstanding Liquidity", "Outstanding contracts",
                           {"Bid depth": record["bid_depth"],
                            "Ask depth": record["ask_depth"]}),
    }
    for name, (title, ylabel, series) in plots.items():
        figure, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
        for series_label, values in series.items():
            ax.plot(time, values, label=series_label)
        if name == "clob_pricing_error":
            ax.axhline(CONVERGENCE_TOLERANCE, linestyle="--", label="Convergence tolerance")
        ax.set(xlabel=label, ylabel=ylabel, title=title)
        ax.legend()
        figure.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
        plt.close(figure)


def main():
    print("Running baseline simulations...")
    records = [run_one(BASE_SEED + run) for run in range(N_RUNS)]
    timescale, converged_share = reference_timescale(records)
    if np.isfinite(timescale):
        print(f"Reference timescale (Kaplan-Meier median): {timescale:.2f} periods")
    else:
        print(f"Reference timescale not identified: {converged_share:.0%} of runs converged "
              "and the Kaplan-Meier median needs at least half. Plots use periods.")

    summary = pd.DataFrame([{"run": run, **summarise(record, timescale)}
                            for run, record in enumerate(records)])
    print(f"\nRuns: {N_RUNS}")
    for column, label, fmt in (
        ("converged", "Convergence rate", ".1%"),
        ("mean_absolute_error", "Mean absolute pricing error", ".4f"),
        ("rmse", "Mean RMSE", ".4f"),
        ("total_submitted_volume", "Mean submitted volume", ".2f"),
        ("total_executed_volume", "Mean executed volume", ".2f"),
        ("execution_rate", "Mean execution rate", ".2%"),
        ("mean_spread", "Mean spread", ".4f"),
        ("two_sided_quote_fraction", "Mean two-sided quote availability", ".1%"),
    ):
        print(f"{label}: {summary[column].mean():{fmt}}")
    print("Median executed volume to convergence (converged runs): "
          f"{summary['executed_volume_to_convergence'].median():.0f}")

    SIM_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    summary.to_csv(SIM_DIR / "clob_simulation_summary.csv", index=False)
    print("\nSaved: sim_output/clob_simulation_summary.csv")
    plot_example(records[0], timescale)


if __name__ == "__main__":
    main()
