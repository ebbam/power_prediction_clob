"""Matched-seed experiment of price recovery by fixed total whale-capital share.

The simulations (~a few minutes) are cached in sim_output/whale_convergence_*_300_periods.csv;
pass --rerun to redo them. Figures, drawn from the saved trajectories:
    whale_convergence_time_300_periods   recovery from peak error (drawn only on --rerun)
    whale_distortion                     P1, P1b, P2 and P3 by ρ (formerly plot_whale_distortion.py)
    whale_sample_runs                    three example runs at every ρ (formerly
                                         plot_whale_sample_runs.py)
Run from the repository root with `python3 -m experiments.whale_convergence_experiment`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from bettor import Bettor, add_whale, fundamental_path, run_market_lob
from metrics import (
    calibrate_band,
    distortion_metrics,
    first_sustained,
    kaplan_meier_curve,
    km_median,
    normalised_pull,
    plot_distortion_figures,
    smoothed_absolute_error,
    summarize_distortion,
)


N_RUNS = 50
N_BETTORS = 100
WHALE_SHARES = (0.0, 0.25, 0.50, 0.75)
WHALE_VALUATION = 0.70
WHALE_RISK_AVERSION = 0.5
PRE_PERIOD = 50
POST_PERIOD = 300
SIMULATION_HORIZON = PRE_PERIOD + POST_PERIOD
CONVERGENCE_WINDOW = 5
INITIAL_PRICE = 0.50
OUTCOME_UNCERTAINTY = 0.05
TICK = 0.01
TTL = 5
MAX_MARKUP = 0.05
BASE_SEED = 1042
OUTPUT_DIR = Path(__file__).resolve().parents[1]   # repository root
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)
ORDINARY_ERROR_QUANTILE = 0.95
CONTROL_PEAK_QUANTILE = 0.95
TRAJECTORIES = SIM_DIR / f"whale_convergence_trajectories_{POST_PERIOD}_periods.csv"
RHO = "ρ (whale capital share)"
PRICE_COLOR, FUNDAMENTAL_COLOR, ERROR_COLOR = "#0072B2", "#222222", "#D55E00"


def make_election_path(seed):
    """One common fundamental path for every share condition in a seed block."""
    return fundamental_path(INITIAL_PRICE, SIMULATION_HORIZON, OUTCOME_UNCERTAINTY,
                            np.random.default_rng(seed))


def make_population(seed, whale_share=0.0, **traits):
    """
    N_BETTORS bettors with budgets U(100, 1000) scaled by 1 - whale_share (so a whale added
    with that share leaves total money unchanged) and valuations N(0.5, 0.05) clipped to
    [0.01, 0.99]. Other traits are drawn by Bettor unless given in traits, each as one value
    for everyone or a sequence with one value per bettor.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    population = []
    for i in range(N_BETTORS):
        own = {key: value[i] if np.ndim(value) else value for key, value in traits.items()}
        population.append(Bettor(budget=rng.uniform(100, 1000) * (1 - whale_share),
                                 market_valuation=np.clip(rng.normal(0.50, 0.05), 0.01, 0.99),
                                 **own))
    return population


def run_whale_market(seed, share, population, whale_risk_av=WHALE_RISK_AVERSION):
    """
    One run of this design: `population` (from make_population(seed, share)), joined at
    PRE_PERIOD by a whale holding `share` of all money with valuation WHALE_VALUATION.
    Returns the market record, the fundamental path and the whale (None when share = 0).
    """
    path = make_election_path(seed)
    whale = add_whale(population, share, WHALE_VALUATION, risk_av=whale_risk_av,
                      entry_time=PRE_PERIOD)
    record = run_market_lob(population, SIMULATION_HORIZON, path, INITIAL_PRICE, tick=TICK,
                            ttl=TTL, max_markup=MAX_MARKUP, rng=np.random.default_rng(seed))
    return record, path, whale


def recovery_after_peak(absolute_error, threshold, peak_cutoff, window):
    """Peak distortion, and the first sustained return to the control error band after it."""
    absolute_error = np.asarray(absolute_error, dtype=float)
    result = {
        "peak_absolute_error": np.nan,
        "peak_time": np.nan,
        "unusual_peak": False,
        "recovered": False,
        "recovery_time_after_peak": np.nan,
        "recovery_confirmation_time_after_peak": np.nan,
    }
    valid = np.isfinite(absolute_error)
    if not valid.any():
        return result
    peak_time = int(np.nanargmax(absolute_error))
    result.update(peak_absolute_error=float(absolute_error[peak_time]), peak_time=peak_time,
                  unusual_peak=bool(absolute_error[peak_time] > peak_cutoff))
    if not result["unusual_peak"]:
        return result
    with np.errstate(invalid="ignore"):
        inside = valid & (absolute_error <= threshold)
    start = first_sustained(inside, window, start=peak_time + 1)
    if start is not None:
        result.update(recovered=True, recovery_time_after_peak=start - peak_time,
                      recovery_confirmation_time_after_peak=start + window - 1 - peak_time)
    return result


def plot_convergence_curves(run_frame, error_band, peak_cutoff):
    """Plot cumulative sustained-recovery probability for unusual peak runs."""
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    colors = ("#222222", "#0072B2", "#D55E00", "#CC79A7")
    for share, color in zip(WHALE_SHARES, colors):
        group = run_frame.loc[
            (run_frame["whale_share"] == share)
            & run_frame["peak_exceeds_control_cutoff"]
        ]
        if group.empty:
            continue
        durations = np.where(
            group["recovered"],
            group["recovery_time_after_peak"],
            POST_PERIOD - group["peak_time_post"],
        )
        times, probability, lower, upper = kaplan_meier_curve(
            durations,
            group["recovered"].to_numpy(),
            POST_PERIOD,
        )
        count = int(group["recovered"].sum())
        distorted_count = len(group)
        axis.step(
            times,
            probability,
            where="post",
            color=color,
            linewidth=1.9,
            label=(
                f"{share:.0%} share "
                f"({count}/{distorted_count} recovered; "
                f"{distorted_count}/{int((run_frame['whale_share'] == share).sum())} unusual peaks)"
            ),
        )
        axis.fill_between(
            times,
            lower,
            upper,
            step="post",
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    axis.set(
        xlim=(0, POST_PERIOD),
        ylim=(0, 1),
        xlabel="Periods from peak error to sustained recovery",
        ylabel="Cumulative probability of recovery among unusual-peak runs",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No post-intervention peaks exceeded the control cutoff",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.text(
        0.01,
        -0.02,
        "Shading: 95% pointwise Greenwood log-log confidence intervals. "
        "Recovery: five consecutive observations within the no-whale pre-period error band. "
        "Unrecovered unusual-peak runs are right-censored; other runs are excluded.",
        ha="left",
        fontsize=8,
    )
    axis.set_title(
        f"Peak-error recovery by whale share; error band = {error_band:.3f}, "
        f"control peak cutoff = {peak_cutoff:.3f}",
        fontsize=11,
    )
    figure.savefig(
        FIGURES_DIR / f"whale_convergence_time_{POST_PERIOD}_periods.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURES_DIR / f"whale_convergence_time_{POST_PERIOD}_periods.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def apply_recovery_metrics(run_frame, trajectory_frame, error_band, peak_cutoff):
    """Post-intervention peak error and recovery from that peak, one row per run."""
    post = trajectory_frame.loc[trajectory_frame["phase"] == "post"]
    errors = post.groupby(["whale_share", "run"])["absolute_error"]
    recovery = pd.DataFrame(
        [recovery_after_peak(errors.get_group((row.whale_share, row.run)).to_numpy(),
                             error_band, peak_cutoff, CONVERGENCE_WINDOW)
         for row in run_frame.itertuples()],
        index=run_frame.index,
    ).rename(columns={"peak_absolute_error": "peak_absolute_error_post",
                      "peak_time": "peak_time_post",
                      "unusual_peak": "peak_exceeds_control_cutoff"})
    return run_frame.join(recovery)


def summarize_groups(run_frame):
    """Unusual-peak frequency and recovery times by whale share, with censoring."""
    rows = []
    for share in WHALE_SHARES:
        group = run_frame.loc[run_frame["whale_share"] == share]
        unusual = group.loc[group["peak_exceeds_control_cutoff"]]
        recovered = unusual.loc[unusual["recovered"], "recovery_time_after_peak"]
        durations = np.where(unusual["recovered"], unusual["recovery_time_after_peak"],
                             POST_PERIOD - unusual["peak_time_post"])
        median = km_median(durations, unusual["recovered"].to_numpy(), POST_PERIOD)
        rows.append({
            "whale_share": share,
            "runs": len(group),
            "unusual_peak_runs": len(unusual),
            "unusual_peak_rate": float(group["peak_exceeds_control_cutoff"].mean()),
            "recovered_runs": int(unusual["recovered"].sum()),
            "recovery_rate_among_unusual_peaks": (
                float(unusual["recovered"].mean()) if not unusual.empty else np.nan),
            "mean_post_peak_error": float(group["peak_absolute_error_post"].mean()),
            "median_post_peak_error": float(group["peak_absolute_error_post"].median()),
            "median_recovery_time_km": median,
            "median_recovery_time_km_identified": bool(np.isfinite(median)),
            "median_recovery_time_recovered_only": (
                float(recovered.median()) if not recovered.empty else np.nan),
            "mean_recovery_time_recovered_only": (
                float(recovered.mean()) if not recovered.empty else np.nan),
        })
    return pd.DataFrame(rows)


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


def plot_whale_distortion():
    """Distortion magnitude and duration figures from the saved whale trajectories."""
    metrics, band = whale_metrics()
    summary = summarize_distortion(metrics, group=RHO)
    summary.to_csv(SIM_DIR / "whale_distortion_summary.csv", index=False)
    print(f"Ordinary band (95th pct of smoothed control pre-period error): {band:.3f}")
    print(summary.to_string(index=False))

    figure = plot_distortion_figures(metrics, group=RHO)
    figure.savefig(FIGURES_DIR / "whale_distortion.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "whale_distortion.pdf", bbox_inches="tight")
    plt.close(figure)


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


def draw_run(price_ax, error_ax, path, metrics, band, event_period, whale_valuation=None):
    """One run's fundamental and price (price_ax, with the whale's valuation if given) and its
    smoothed error against the ordinary band with P1-P3 marked (error_ax).
    path has columns period, fundamental, price and smoothed_error."""
    peak = event_period + metrics["peak_time_after_shock"]
    price_ax.plot(path["period"], path["fundamental"], color=FUNDAMENTAL_COLOR, linewidth=1.4,
                  label="Fundamental (true probability)")
    price_ax.plot(path["period"], path["price"], color=PRICE_COLOR, linewidth=1.2,
                  label="Market price (midquote)")
    if whale_valuation is not None:
        price_ax.axhline(whale_valuation, color="#888888", linewidth=0.8, linestyle="--",
                         label="Whale valuation")
    price_ax.set(ylim=(0, 1))

    error = path["smoothed_error"]
    error_ax.fill_between(path["period"], band, error, where=error > band, color=ERROR_COLOR,
                          alpha=0.25, linewidth=0, label="Outside band (P3)")
    error_ax.plot(path["period"], error, color=ERROR_COLOR, linewidth=1.2,
                  label="Smoothed |error|")
    error_ax.axhline(band, color="#888888", linewidth=0.8, linestyle="--", label="Ordinary band")
    error_ax.plot(peak, metrics["max_error"], "v", color=ERROR_COLOR, markersize=7,
                  label="Maximum error (P1)")
    if metrics["reverted"]:
        error_ax.axvline(peak + metrics["reversion_time"], color=ERROR_COLOR, linewidth=1,
                         linestyle=":", label="Reversion (P2)")
    error_ax.set(ylim=(0, 0.85))

    for ax in (price_ax, error_ax):
        ax.axvline(event_period, color="#BBBBBB", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)


def shared_legend(figure, axes, ncol):
    """One legend under the figure with every distinct label on the given axes."""
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False,
                  fontsize=9)


def plot_sample_runs():
    """Price, fundamental and smoothed error for a few matched whale runs, at every ρ."""
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
            price_ax, error_ax = axes[2 * row, column], axes[2 * row + 1, column]
            draw_run(price_ax, error_ax, path, metrics, band, PRE_PERIOD,
                     WHALE_VALUATION if share > 0 else None)
            price_ax.set_title(f"ρ = {share:g}", fontsize=10)
            if column == 0:
                price_ax.set_ylabel(f"{description}\n(run {run})\nprice", fontsize=9)
                error_ax.set_ylabel("|error|", fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel(f"Period (whale enters at {PRE_PERIOD})", fontsize=9)
    shared_legend(figure, (axes[0, -1], axes[1, -1]), ncol=4)
    figure.savefig(FIGURES_DIR / "whale_sample_runs.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _positive_quantile(values, quantile, description):
    """A quantile of the finite values, which must exist and be positive."""
    values = pd.Series(values).dropna()
    if values.empty:
        raise ValueError(f"No valid {description} available for calibration.")
    result = float(values.quantile(quantile))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"The calibration from {description} must be finite and positive.")
    return result, len(values)


def run_experiment():
    """Run matched simulations, calibrate the common thresholds, and save results."""
    print(f"Running {N_RUNS} matched seed blocks x {len(WHALE_SHARES)} shares "
          f"= {N_RUNS * len(WHALE_SHARES)} simulations.")
    periods = np.arange(SIMULATION_HORIZON)
    post = periods >= PRE_PERIOD
    trajectories, run_rows = [], []
    for share in WHALE_SHARES:
        print(f"  Whale initial-capital share: {share:.0%}")
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            population = make_population(seed, share)
            nonwhale_capital = sum(agent.budget for agent in population)
            record, path, whale = run_whale_market(seed, share, population)
            whale_capital = whale.initial_budget if whale else 0.0
            prices = np.asarray(record["midquote"], dtype=float)
            signed_error = prices - path[:-1]
            abs_error = np.abs(signed_error)
            trajectories.append(pd.DataFrame({
                "run": run, "seed": seed, "whale_share": share, "period": periods,
                "phase": np.where(post, "post", "pre"), "midquote": prices,
                "fundamental": path[:-1], "signed_error": signed_error,
                "absolute_error": abs_error,
            }))
            run_rows.append({
                "run": run, "seed": seed, "whale_share": share,
                "initial_nonwhale_market_capital": nonwhale_capital,
                "whale_initial_budget": whale_capital if share else np.nan,
                "market_cash_after_whale": nonwhale_capital + whale_capital,
                "mean_signed_error_post": np.nanmean(signed_error[post]),
                "mae_post": np.nanmean(abs_error[post]),
                "rmse_post": np.sqrt(np.nanmean(abs_error[post] ** 2)),
            })
    trajectory_frame = pd.concat(trajectories, ignore_index=True)
    run_frame = pd.DataFrame(run_rows)

    control = trajectory_frame.loc[trajectory_frame["whale_share"] == 0]
    error_band, n_errors = _positive_quantile(
        control.loc[control["phase"] == "pre", "absolute_error"], ORDINARY_ERROR_QUANTILE,
        "no-whale pre-period errors")
    peak_cutoff, n_peaks = _positive_quantile(
        control.loc[control["phase"] == "post"].groupby("run")["absolute_error"].max(),
        CONTROL_PEAK_QUANTILE, "no-whale post-period peaks")

    settings = {
        "ordinary_error_band": error_band, "ordinary_error_quantile": ORDINARY_ERROR_QUANTILE,
        "control_peak_cutoff": peak_cutoff, "control_peak_quantile": CONTROL_PEAK_QUANTILE,
        "pre_period": PRE_PERIOD, "post_period": POST_PERIOD,
        "convergence_window": CONVERGENCE_WINDOW,
    }
    run_frame = apply_recovery_metrics(run_frame, trajectory_frame, error_band, peak_cutoff)
    aggregate_frame = summarize_groups(run_frame)

    suffix = f"{POST_PERIOD}_periods"
    outputs = {
        f"whale_convergence_runs_{suffix}.csv": run_frame.assign(**settings),
        f"whale_convergence_trajectories_{suffix}.csv": trajectory_frame.assign(
            ordinary_error_band=error_band, control_peak_cutoff=peak_cutoff),
        f"whale_convergence_summary_{suffix}.csv": aggregate_frame.assign(**settings),
    }
    for name, frame in outputs.items():
        frame.to_csv(SIM_DIR / name, index=False)
    plot_convergence_curves(run_frame, error_band, peak_cutoff)

    print(f"\nOrdinary-error band: {error_band:.4f} (q{ORDINARY_ERROR_QUANTILE:.0%} of "
          f"{n_errors} valid no-whale pre-period errors)")
    print(f"Control peak cutoff: {peak_cutoff:.4f} (q{CONTROL_PEAK_QUANTILE:.0%} of "
          f"{n_peaks} no-whale post-period run peaks)")
    print(aggregate_frame.to_string(index=False))
    for name in outputs:
        print(f"Saved sim_output/{name}")
    print(f"Saved figures/whale_convergence_time_{suffix}.png and .pdf")


def main():
    if "--rerun" in sys.argv or not TRAJECTORIES.exists():
        run_experiment()
    plot_whale_distortion()
    plot_sample_runs()


if __name__ == "__main__":
    main()
