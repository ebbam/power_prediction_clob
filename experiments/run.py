"""
run.py

Run repeated CLOB prediction-market simulations and collect:
    - price accuracy
    - submitted trading interest
    - executed volume
    - outstanding liquidity
    - bid/ask spread
    - first-passage convergence time
    - dimensionless convergence time
"""

import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bettor import bettor, run_market_lob
from metrics import kaplan_meier_median

os.makedirs("figures", exist_ok=True)
os.makedirs("sim_output", exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

N_RUNS = 100

N_BETTORS = 100
T_ELECTION = 100

INITIAL_PRICE = 0.50
OUTCOME_UNCERTAINTY = 0.05

MARKET_ERROR = 0.10

# CLOB parameters
TICK = 0.01
TTL = 5
MAX_MARKUP = 0.05

# Convergence parameters
CONVERGENCE_TOLERANCE = 0.02
CONVERGENCE_WINDOW = 5

# Reproducibility
BASE_SEED = 42


# ============================================================
# CREATE AGENTS
# ============================================================

def create_bettors(n, rng):
    """
    Create a population of bettors.

    Budget and initial valuation are drawn here from rng; risk aversion,
    stubbornness and expertise are drawn per bettor inside bettor.__init__
    from np.random, which run_one seeds.
    """

    bettors = []

    for _ in range(n):

        b = bettor(
            budget=rng.uniform(100, 1000),
            market_valuation=np.clip(
                rng.normal(0.50, 0.05),
                0.01,
                0.99,
            ),
        )

        bettors.append(b)

    return bettors


# ============================================================
# RUN ONE SIMULATION
# ============================================================

def run_one(seed, reference_timescale=None):

    rng = np.random.default_rng(seed)

    # bettor.__init__ and bettor.update_belief draw from the global
    # np.random generator, so seed it too for reproducible runs.
    np.random.seed(seed)

    bettors = create_bettors(
        N_BETTORS,
        rng,
    )

    results = run_market_lob(
        n_bettors=N_BETTORS,
        t_election=T_ELECTION,
        initial_price=INITIAL_PRICE,
        outcome_uncertainty=OUTCOME_UNCERTAINTY,
        bettors=bettors,
        market_error=MARKET_ERROR,
        tick=TICK,
        ttl=TTL,
        max_markup=MAX_MARKUP,
        rng=rng,
        convergence_tolerance=CONVERGENCE_TOLERANCE,
        convergence_window=CONVERGENCE_WINDOW,
        reference_timescale=reference_timescale,
    )

    return results


# ============================================================
# FIRST PASS:
# ESTIMATE REFERENCE TIMESCALE
# ============================================================

print("Running baseline simulations...")

baseline_results = []

for run in range(N_RUNS):

    seed = BASE_SEED + run

    result = run_one(
        seed,
        reference_timescale=None,
    )

    baseline_results.append(result)


# Reference timescale: Kaplan-Meier median first-passage time over ALL
# runs. Runs that never converged are right-censored at T_ELECTION rather
# than dropped; using only converged runs would bias tau_ref downwards.
time_to_event = np.array([
    r["convergence"]["first_passage_time"] if r["convergence"]["converged"] else T_ELECTION
    for r in baseline_results
])
converged = np.array([r["convergence"]["converged"] for r in baseline_results])

REFERENCE_TIMESCALE = kaplan_meier_median(time_to_event, converged)

TIMESCALE_IDENTIFIED = np.isfinite(REFERENCE_TIMESCALE) and REFERENCE_TIMESCALE > 0

print()
if TIMESCALE_IDENTIFIED:
    print(f"Reference timescale (Kaplan-Meier median): {REFERENCE_TIMESCALE:.2f} periods")
    TIME_SCALE, TIME_LABEL = REFERENCE_TIMESCALE, r"Dimensionless time $t/\tau_{ref}$"
else:
    print(
        f"Reference timescale not identified: {converged.mean():.0%} of runs converged and the "
        "Kaplan-Meier median needs at least half. Dimensionless times are NaN; plots use periods."
    )
    TIME_SCALE, TIME_LABEL = 1.0, "Simulation period"


# ============================================================
# BUILD RUN-LEVEL SUMMARY
# ============================================================

summary_rows = []


for run, result in enumerate(baseline_results):

    conv = result["convergence"]

    mid = np.asarray(
        result["midquote"],
        dtype=float,
    )

    fundamental = np.asarray(
        result["gen_el"][:len(mid)],
        dtype=float,
    )

    error = np.abs(
        mid - fundamental
    )

    # Only calculate over valid two-sided quotes
    valid_error = error[
        np.isfinite(error)
    ]

    mean_absolute_error = (
        np.mean(valid_error)
        if len(valid_error) > 0
        else np.nan
    )

    rmse = (
        np.sqrt(
            np.mean(valid_error ** 2)
        )
        if len(valid_error) > 0
        else np.nan
    )

    first_passage = (
        conv["first_passage_time"]
        if conv["converged"]
        else np.nan
    )

    dimensionless_convergence = (
        first_passage / REFERENCE_TIMESCALE
        if conv["converged"] and TIMESCALE_IDENTIFIED
        else np.nan
    )

    valid_spreads = np.asarray(
        result["spread"],
        dtype=float,
    )

    valid_spreads = valid_spreads[
        np.isfinite(valid_spreads)
    ]

    # Activity-based clock: contracts executed up to and including the
    # first-passage period (NaN if the run never converged).
    executed_volume_to_convergence = (
        np.cumsum(result["traded_volume"])[conv["first_passage_time"]]
        if conv["converged"]
        else np.nan
    )

    summary_rows.append({

        "run": run,

        "executed_volume_to_convergence":
            executed_volume_to_convergence,

        "mean_desired_trades_dropped_per_period":
            np.mean(result["n_desired_trade_dropped"]),

        "converged":
            conv["converged"],

        "first_passage_time":
            first_passage,

        "confirmation_time":
            conv["confirmation_time"],

        "dimensionless_convergence_time":
            dimensionless_convergence,

        "mean_absolute_error":
            mean_absolute_error,

        "rmse":
            rmse,

        "total_submitted_volume":
            np.sum(
                result["submitted_volume"]
            ),

        "total_executed_volume":
            np.sum(
                result["traded_volume"]
            ),

        "total_trades":
            np.sum(
                result["n_trades"]
            ),

        "mean_bid_depth":
            np.nanmean(
                result["bid_depth"]
            ),

        "mean_ask_depth":
            np.nanmean(
                result["ask_depth"]
            ),

        "mean_total_depth":
            np.nanmean(
                result["total_depth"]
            ),

        "mean_spread":
            (
                np.mean(valid_spreads)
                if len(valid_spreads) > 0
                else np.nan
            ),

        "two_sided_quote_fraction":
            np.mean(
                np.isfinite(mid)
            ),
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# ADD EXECUTION RATE
# ============================================================

summary["execution_rate"] = (
    summary["total_executed_volume"]
    /
    summary["total_submitted_volume"].replace(
        0,
        np.nan,
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================")
print("CLOB SIMULATION SUMMARY")
print("==============================\n")

print(
    f"Runs: {N_RUNS}"
)

print(
    f"Convergence rate: "
    f"{summary['converged'].mean():.1%}"
)

print(
    f"Median executed volume to convergence (converged runs): "
    f"{summary['executed_volume_to_convergence'].median():.0f}"
)

print(
    f"Mean absolute pricing error: "
    f"{summary['mean_absolute_error'].mean():.4f}"
)

print(
    f"Mean RMSE: "
    f"{summary['rmse'].mean():.4f}"
)

print(
    f"Mean submitted volume: "
    f"{summary['total_submitted_volume'].mean():.2f}"
)

print(
    f"Mean executed volume: "
    f"{summary['total_executed_volume'].mean():.2f}"
)

print(
    f"Mean execution rate: "
    f"{summary['execution_rate'].mean():.2%}"
)

print(
    f"Mean spread: "
    f"{summary['mean_spread'].mean():.4f}"
)

print(
    f"Mean two-sided quote availability: "
    f"{summary['two_sided_quote_fraction'].mean():.1%}"
)


# ============================================================
# SAVE RUN-LEVEL RESULTS
# ============================================================

summary.to_csv(
    "sim_output/clob_simulation_summary.csv",
    index=False,
)

print(
    "\nSaved: sim_output/clob_simulation_summary.csv"
)


# ============================================================
# PLOT 1:
# EXAMPLE PRICE DISCOVERY
# ============================================================

example = baseline_results[0]

mid = np.asarray(
    example["midquote"],
    dtype=float,
)

fundamental = np.asarray(
    example["gen_el"][:len(mid)],
    dtype=float,
)

time = np.arange(len(mid))

plot_time = time / TIME_SCALE


plt.figure(figsize=(9, 5))

plt.plot(
    plot_time,
    fundamental,
    label="Fundamental probability",
)

plt.plot(
    plot_time,
    mid,
    label="CLOB midquote",
)

plt.xlabel(TIME_LABEL)

plt.ylabel("Probability")

plt.title(
    "CLOB Price Discovery"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/clob_price_discovery.png",
    dpi=300,
)

plt.show()


# ============================================================
# PLOT 2:
# PRICING ERROR
# ============================================================

error = np.abs(
    mid - fundamental
)


plt.figure(figsize=(9, 5))

plt.plot(
    plot_time,
    error,
)

plt.axhline(
    CONVERGENCE_TOLERANCE,
    linestyle="--",
    label="Convergence tolerance",
)

plt.xlabel(TIME_LABEL)

plt.ylabel(
    "Absolute pricing error"
)

plt.title(
    "Pricing Error Through Time"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/clob_pricing_error.png",
    dpi=300,
)

plt.show()


# ============================================================
# PLOT 3:
# SUBMITTED VS EXECUTED VOLUME
# ============================================================

submitted = np.asarray(
    example["submitted_volume"]
)

executed = np.asarray(
    example["traded_volume"]
)


plt.figure(figsize=(9, 5))

plt.plot(
    plot_time,
    submitted,
    label="Submitted volume",
)

plt.plot(
    plot_time,
    executed,
    label="Executed volume",
)

plt.xlabel(TIME_LABEL)

plt.ylabel("Contracts")

plt.title(
    "Submitted and Executed Trading Activity"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/clob_market_activity.png",
    dpi=300,
)

plt.show()


# ============================================================
# PLOT 4:
# OUTSTANDING LIQUIDITY
# ============================================================

bid_depth = np.asarray(
    example["bid_depth"]
)

ask_depth = np.asarray(
    example["ask_depth"]
)


plt.figure(figsize=(9, 5))

plt.plot(
    plot_time,
    bid_depth,
    label="Bid depth",
)

plt.plot(
    plot_time,
    ask_depth,
    label="Ask depth",
)

plt.xlabel(TIME_LABEL)

plt.ylabel(
    "Outstanding contracts"
)

plt.title(
    "CLOB Outstanding Liquidity"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/clob_liquidity.png",
    dpi=300,
)

plt.show()

plt.figure(figsize=(9, 5))

plt.plot(
    plot_time,
    example["submitted_buy_volume"],
    label="Submitted buy volume",
)

plt.plot(
    plot_time,
    example["submitted_sell_volume"],
    label="Submitted sell volume",
)

plt.xlabel(TIME_LABEL)
plt.ylabel("Contracts")
plt.title("Submitted Buy and Sell Interest")
plt.legend()
plt.tight_layout()
plt.show()


print("\nAnalysis complete.")