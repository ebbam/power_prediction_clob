"""One-shot whale: a single very large buy at an extreme valuation, then no further trading.

A transient-shock counterpart to whale_convergence_experiment.py, which uses a whale that
keeps trading for the rest of the run. Uses the same populations, fundamental paths and
seeds, and the existing run_market_lob unchanged: the whale sits in the population from
period 0 but stays idle until SHOT_PERIOD, submits one buy order at its valuation, and then
never trades again. Any unfilled part of the order rests in the book until the whale's next
activation (the following period), when the market cancels it.

The ρ = 0 control carries the same whale with no money, so every ρ sees identical random
draws up to the shot.

Two conditions are compared: ordinary traders who update beliefs from the fundamental only,
and traders who also move their belief part of the way toward the market price each period
(bettor.market_imitation), so the whale's price spike carries information. Each condition's
ordinary band is calibrated on its own ρ = 0 runs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bettor import bettor, run_market_lob
from distortion import (
    calibrate_band,
    distortion_metrics,
    plot_max_error,
    plot_reversion_time,
    plot_time_outside_band,
    smoothed_absolute_error,
    summarize_distortion,
)
from whale_convergence_experiment import (
    BASE_SEED,
    INITIAL_PRICE,
    MARKET_ERROR,
    MAX_MARKUP,
    N_BETTORS,
    OUTCOME_UNCERTAINTY,
    TICK,
    TTL,
    make_election_path,
    make_population,
)


N_RUNS = 50
RHOS = (0.0, 0.25, 0.50, 0.75)
WHALE_VALUATION = 0.95
WHALE_RISK_AVERSION = 0.5
SHOT_PERIOD = 50
HORIZON = SHOT_PERIOD + 300
SAMPLE_RUNS = (3, 35, 38)
RHO = "ρ (whale capital share)"
CONDITIONS = {"No market imitation": 0.0, "Market imitation = 0.3": 0.3}
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)


class OneShotWhale(bettor):
    """A whale that trades exactly once, in period `shot_period`, and is idle otherwise."""

    def __init__(self, budget, market_valuation, shot_period):
        super().__init__(budget=budget, market_valuation=market_valuation,
                         risk_av=WHALE_RISK_AVERSION, whale=True)
        self.shot_period = shot_period
        self.period = 0

    def _shooting(self):
        return self.period == self.shot_period and self.budget > 0

    def trade(self, m):
        return super().trade(m) if self._shooting() else 0

    def limit_order(self, ref_price, ob, rng, max_markup=0.05):
        if not self._shooting():
            return None
        return super().limit_order(ref_price, ob, rng, max_markup)

    def update_belief(self, *args, **kwargs):
        # Called once per period after trading: use it as the whale's clock. Beliefs stay fixed.
        self.period += 1


def run_one(run, rho, imitation):
    seed = BASE_SEED + run
    path = make_election_path(seed)
    population = make_population(seed, rho)
    for agent in population:
        agent.market_imitation = imitation
    capital = sum(agent.budget for agent in population)
    whale = OneShotWhale(capital * rho / (1 - rho), WHALE_VALUATION, SHOT_PERIOD)
    population.append(whale)
    whale_budget = whale.budget

    record = run_market_lob(
        N_BETTORS, HORIZON, INITIAL_PRICE, OUTCOME_UNCERTAINTY, population,
        market_error=MARKET_ERROR, tick=TICK, ttl=TTL, max_markup=MAX_MARKUP,
        rng=np.random.default_rng(seed), gen_el=path,
    )
    trajectory = pd.DataFrame({
        "run": run, "seed": seed, "whale_share": rho, "market_imitation": imitation,
        "period": np.arange(HORIZON),
        "midquote": record["midquote"], "fundamental": path[:HORIZON],
    })
    shot = {
        "whale_budget": whale_budget,
        "contracts_bought": whale.n_contracts,
        "cash_spent": whale_budget - whale.budget,
        "share_of_budget_spent": (whale_budget - whale.budget) / whale_budget if rho else np.nan,
    }
    return trajectory, shot


def plot_sample_runs(trajectories, metrics, band, filename, title):
    """Price, fundamental and smoothed error around the shot for a few matched runs."""
    figure, axes = plt.subplots(
        2 * len(SAMPLE_RUNS), len(RHOS), figsize=(4.2 * len(RHOS), 3.0 * len(SAMPLE_RUNS) + 1),
        sharex=True, gridspec_kw={"height_ratios": [2, 1] * len(SAMPLE_RUNS)},
        constrained_layout=True,
    )
    for row, run in enumerate(SAMPLE_RUNS):
        for column, rho in enumerate(RHOS):
            path = trajectories.loc[(trajectories["whale_share"] == rho)
                                    & (trajectories["run"] == run)]
            result = metrics.loc[(metrics[RHO] == rho) & (metrics["run"] == run)].iloc[0]
            peak = SHOT_PERIOD + result["peak_time_after_shock"]
            price_ax, error_ax = axes[2 * row, column], axes[2 * row + 1, column]

            price_ax.plot(path["period"], path["fundamental"], color="#222222", linewidth=1.4,
                          label="Fundamental (true probability)")
            price_ax.plot(path["period"], path["midquote"].ffill(), color="#0072B2",
                          linewidth=1.2, label="Market price (midquote)")
            price_ax.set(ylim=(0, 1))
            price_ax.set_title(f"ρ = {rho:g}, {result['contracts_bought']:,.0f} contracts bought",
                               fontsize=10)
            if column == 0:
                price_ax.set_ylabel(f"Run {run}\nprice", fontsize=9)

            error = path["smoothed_error"]
            error_ax.fill_between(path["period"], band, error, where=error > band,
                                  color="#D55E00", alpha=0.25, linewidth=0,
                                  label="Outside band (P3)")
            error_ax.plot(path["period"], error, color="#D55E00", linewidth=1.2,
                          label="Smoothed |error|")
            error_ax.axhline(band, color="#888888", linewidth=0.8, linestyle="--",
                             label="Ordinary band")
            error_ax.plot(peak, result["max_error"], "v", color="#D55E00", markersize=7,
                          label="Maximum error (P1)")
            if result["reverted"]:
                error_ax.axvline(peak + result["reversion_time"], color="#D55E00",
                                 linewidth=1, linestyle=":", label="Reversion (P2)")
            error_ax.set(ylim=(0, 0.85))
            if column == 0:
                error_ax.set_ylabel("|error|", fontsize=9)

            for ax in (price_ax, error_ax):
                ax.axvline(SHOT_PERIOD, color="#BBBBBB", linewidth=0.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelsize=8)

    for ax in axes[-1]:
        ax.set_xlabel(f"Period (whale trades once at {SHOT_PERIOD})", fontsize=9)
    handles, labels = [], []
    for ax in (axes[0, -1], axes[1, -1]):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle), labels.append(label)
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    figure.suptitle(title, fontsize=12)
    figure.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyse(trajectories):
    """Smoothed errors, the condition's ordinary band, and P1-P3 per run."""
    trajectories = trajectories.copy()
    trajectories["smoothed_error"] = np.concatenate([
        smoothed_absolute_error(path["midquote"], path["fundamental"])
        for _, path in trajectories.groupby(["whale_share", "run"], sort=False)
    ])
    band = calibrate_band(trajectories.loc[
        (trajectories["whale_share"] == 0) & (trajectories["period"] < SHOT_PERIOD),
        "smoothed_error",
    ])
    rows = [
        {RHO: rho, "run": run,
         **distortion_metrics(path["smoothed_error"].to_numpy(), SHOT_PERIOD, band)}
        for (rho, run), path in trajectories.groupby(["whale_share", "run"])
    ]
    return trajectories, pd.DataFrame(rows), band


def main():
    all_trajectories, all_metrics, summaries = [], [], []
    figure, axes = plt.subplots(2, 3, figsize=(18, 9.6), constrained_layout=True)

    for row, (condition, imitation) in enumerate(CONDITIONS.items()):
        trajectories, shots = [], []
        for rho in RHOS:
            print(f"{condition}, ρ = {rho:g}")
            for run in range(N_RUNS):
                trajectory, shot = run_one(run, rho, imitation)
                trajectories.append(trajectory)
                shots.append({RHO: rho, "run": run, **shot})
        trajectories, metrics, band = analyse(pd.concat(trajectories, ignore_index=True))
        metrics = metrics.merge(pd.DataFrame(shots), on=[RHO, "run"])

        summary = summarize_distortion(metrics, group=RHO).merge(
            metrics.groupby(RHO)[["contracts_bought", "share_of_budget_spent"]].median()
            .add_prefix("median_").reset_index(),
            on=RHO,
        )
        print(f"Ordinary band: {band:.3f}")
        print(summary.round(3).to_string(index=False))

        for ax, plot in zip(axes[row], (plot_max_error, plot_reversion_time,
                                        plot_time_outside_band)):
            plot(metrics, RHO, ax)
            ax.set_title(f"{condition}\n{ax.get_title()}", fontsize=11)

        suffix = "" if imitation == 0 else "_imitation"
        plot_sample_runs(trajectories, metrics, band,
                         f"one_shot_whale{suffix}_sample_runs.png", condition)
        all_trajectories.append(trajectories)
        all_metrics.append(metrics.assign(market_imitation=imitation))
        summaries.append(summary.assign(market_imitation=imitation))

    # Same x-range in both rows so the conditions can be compared by eye.
    for column in range(3):
        right = max(ax.get_xlim()[1] for ax in axes[:, column])
        for ax in axes[:, column]:
            ax.set_xlim(right=right)
    figure.savefig(FIGURES_DIR / "one_shot_whale_distortion.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "one_shot_whale_distortion.pdf", bbox_inches="tight")
    plt.close(figure)

    pd.concat(all_trajectories).to_csv(SIM_DIR / "one_shot_whale_trajectories.csv", index=False)
    pd.concat(all_metrics).to_csv(SIM_DIR / "one_shot_whale_runs.csv", index=False)
    pd.concat(summaries).to_csv(SIM_DIR / "one_shot_whale_summary.csv", index=False)


if __name__ == "__main__":
    main()
