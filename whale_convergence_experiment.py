"""Matched-seed experiment of price recovery by fixed total whale-capital share."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from bettor import bettor, run_market_lob


N_RUNS = 50
N_BETTORS = 100
WHALE_SHARES = (0.0, 0.25, 0.50, 0.75)
WHALE_VALUATION = 0.70
PRE_PERIOD = 50
POST_PERIOD = 300
SIMULATION_HORIZON = PRE_PERIOD + POST_PERIOD
CONVERGENCE_WINDOW = 5
INITIAL_PRICE = 0.50
OUTCOME_UNCERTAINTY = 0.05
MARKET_ERROR = 0.0   # market starts at INITIAL_PRICE, as in the methods section
TICK = 0.01
TTL = 5
MAX_MARKUP = 0.05
BASE_SEED = 1042
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)
ORDINARY_ERROR_QUANTILE = 0.95
CONTROL_PEAK_QUANTILE = 0.95


def make_election_path(seed):
    """Create one common fundamental path for every share condition in a seed block."""
    rng = np.random.default_rng(seed)
    path = np.empty(SIMULATION_HORIZON + 1, dtype=float)
    path[0] = INITIAL_PRICE
    for period in range(SIMULATION_HORIZON):
        path[period + 1] = np.clip(
            path[period] + rng.normal(0, OUTCOME_UNCERTAINTY),
            0,
            1,
        )
    return path


def make_population(seed, whale_share):
    """Allocate only non-whale capital to the market before intervention."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    population = [
        bettor(
            budget=rng.uniform(100, 1000),
            market_valuation=np.clip(rng.normal(0.50, 0.05), 0.01, 0.99),
        )
        for _ in range(N_BETTORS)
    ]
    for agent in population:
        agent.budget *= 1 - whale_share
    return population


def kaplan_meier_curve(durations, observed, horizon):
    """Return cumulative convergence and Greenwood log-log confidence intervals."""
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    event_times = np.unique(durations[observed])

    survival = 1.0
    greenwood_sum = 0.0
    times = [0.0]
    probability = [0.0]
    lower = [0.0]
    upper = [0.0]
    z_975 = 1.959963984540054

    for event_time in event_times:
        at_risk = int(np.sum(durations >= event_time))
        events = int(np.sum(observed & (durations == event_time)))
        survival *= 1 - events / at_risk

        if events == at_risk:
            greenwood_sum = math.inf
        elif math.isfinite(greenwood_sum):
            greenwood_sum += events / (at_risk * (at_risk - events))

        if survival <= 0:
            survival_lower = survival_upper = 0.0
        elif survival >= 1 or greenwood_sum == 0:
            survival_lower = survival_upper = survival
        else:
            log_neg_log = math.log(-math.log(survival))
            standard_error = math.sqrt(greenwood_sum) / abs(math.log(survival))
            survival_lower = math.exp(-math.exp(log_neg_log + z_975 * standard_error))
            survival_upper = math.exp(-math.exp(log_neg_log - z_975 * standard_error))

        times.append(float(event_time))
        probability.append(1 - survival)
        lower.append(1 - survival_upper)
        upper.append(1 - survival_lower)

    if times[-1] < horizon:
        times.append(float(horizon))
        probability.append(probability[-1])
        lower.append(lower[-1])
        upper.append(upper[-1])

    return times, probability, lower, upper


def recovery_after_peak(absolute_error, threshold, peak_cutoff, window):
    """Measure peak distortion and sustained return to the control error band."""
    absolute_error = np.asarray(absolute_error, dtype=float)
    valid = np.isfinite(absolute_error)
    if not valid.any():
        return {
            "peak_absolute_error": np.nan,
            "peak_time": np.nan,
            "unusual_peak": False,
            "recovered": False,
            "recovery_time_after_peak": np.nan,
            "recovery_confirmation_time_after_peak": np.nan,
        }

    peak_time = int(np.nanargmax(absolute_error))
    peak_absolute_error = float(absolute_error[peak_time])
    unusual_peak = peak_absolute_error > peak_cutoff
    if not unusual_peak:
        return {
            "peak_absolute_error": peak_absolute_error,
            "peak_time": peak_time,
            "unusual_peak": False,
            "recovered": False,
            "recovery_time_after_peak": np.nan,
            "recovery_confirmation_time_after_peak": np.nan,
        }

    for start in range(peak_time + 1, len(absolute_error) - window + 1):
        end = start + window
        if np.all(
            np.isfinite(absolute_error[start:end])
            & (absolute_error[start:end] <= threshold)
        ):
            return {
                "peak_absolute_error": peak_absolute_error,
                "peak_time": peak_time,
                "unusual_peak": True,
                "recovered": True,
                "recovery_time_after_peak": start - peak_time,
                "recovery_confirmation_time_after_peak": end - 1 - peak_time,
            }

    return {
        "peak_absolute_error": peak_absolute_error,
        "peak_time": peak_time,
        "unusual_peak": True,
        "recovered": False,
        "recovery_time_after_peak": np.nan,
        "recovery_confirmation_time_after_peak": np.nan,
    }


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


def apply_recovery_metrics(
    run_frame,
    trajectory_frame,
    error_band,
    peak_cutoff,
):
    """Calculate post-intervention peak error and recovery from that peak."""
    run_frame = run_frame.copy()
    run_frame["peak_absolute_error_post"] = np.nan
    run_frame["peak_time_post"] = np.nan
    run_frame["peak_exceeds_control_cutoff"] = False
    run_frame["recovered"] = False
    run_frame["recovery_time_after_peak"] = np.nan
    run_frame["recovery_confirmation_time_after_peak"] = np.nan

    for row_index, row in run_frame.iterrows():
        trajectory = trajectory_frame.loc[
            (trajectory_frame["run"] == row["run"])
            & (trajectory_frame["whale_share"] == row["whale_share"])
        ]
        post = trajectory.loc[trajectory["phase"] == "post"]
        recovery = recovery_after_peak(
            post["absolute_error"].to_numpy(),
            error_band,
            peak_cutoff,
            CONVERGENCE_WINDOW,
        )
        run_frame.loc[row_index, "peak_absolute_error_post"] = recovery[
            "peak_absolute_error"
        ]
        run_frame.loc[row_index, "peak_time_post"] = recovery["peak_time"]
        run_frame.loc[row_index, "peak_exceeds_control_cutoff"] = recovery[
            "unusual_peak"
        ]
        run_frame.loc[row_index, "recovered"] = recovery["recovered"]
        run_frame.loc[row_index, "recovery_time_after_peak"] = recovery[
            "recovery_time_after_peak"
        ]
        run_frame.loc[
            row_index, "recovery_confirmation_time_after_peak"
        ] = recovery[
            "recovery_confirmation_time_after_peak"
        ]

    for column in ("peak_exceeds_control_cutoff", "recovered"):
        run_frame[column] = run_frame[column].astype(bool)
    return run_frame


def summarize_groups(run_frame):
    """Summarize unusual peak frequency and recovery times with censoring."""
    event_rows = []
    for share in WHALE_SHARES:
        group = run_frame.loc[run_frame["whale_share"] == share]
        unusual = group.loc[group["peak_exceeds_control_cutoff"]]
        recovered = unusual.loc[
            unusual["recovered"], "recovery_time_after_peak"
        ]
        durations = np.where(
            unusual["recovered"],
            unusual["recovery_time_after_peak"],
            POST_PERIOD - unusual["peak_time_post"],
        )
        curve_times, curve_probability, _, _ = kaplan_meier_curve(
            durations,
            unusual["recovered"].to_numpy(),
            POST_PERIOD,
        )
        km_median = next(
            (
                time
                for time, probability in zip(curve_times, curve_probability)
                if probability >= 0.5
            ),
            np.nan,
        )
        event_rows.append(
            {
                "whale_share": share,
                "runs": len(group),
                "unusual_peak_runs": len(unusual),
                "unusual_peak_rate": float(group["peak_exceeds_control_cutoff"].mean()),
                "recovered_runs": int(unusual["recovered"].sum()),
                "recovery_rate_among_unusual_peaks": (
                    float(unusual["recovered"].mean())
                    if not unusual.empty
                    else np.nan
                ),
                "mean_post_peak_error": float(
                    group["peak_absolute_error_post"].mean()
                ),
                "median_post_peak_error": float(
                    group["peak_absolute_error_post"].median()
                ),
                "median_recovery_time_km": km_median,
                "median_recovery_time_km_identified": bool(
                    np.isfinite(km_median)
                ),
                "median_recovery_time_recovered_only": (
                    float(recovered.median()) if not recovered.empty else np.nan
                ),
                "mean_recovery_time_recovered_only": (
                    float(recovered.mean()) if not recovered.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(event_rows)


def main():
    """Run matched simulations, calibrate the common threshold, and save results."""
    trajectories = []
    run_rows = []
    fundamental_paths = {
        run: make_election_path(BASE_SEED + run)
        for run in range(N_RUNS)
    }

    print(
        f"Running {N_RUNS} matched seed blocks x {len(WHALE_SHARES)} shares "
        f"= {N_RUNS * len(WHALE_SHARES)} simulations."
    )
    for share in WHALE_SHARES:
        print(f"  Whale initial-capital share: {share:.0%}")
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            population = make_population(seed, share)
            initial_nonwhale_capital = sum(
                agent.budget for agent in population
            )
            results = run_market_lob(
                n_bettors=N_BETTORS,
                t_election=SIMULATION_HORIZON,
                initial_price=INITIAL_PRICE,
                outcome_uncertainty=OUTCOME_UNCERTAINTY,
                bettors=population,
                market_error=MARKET_ERROR,
                tick=TICK,
                ttl=TTL,
                max_markup=MAX_MARKUP,
                rng=np.random.default_rng(seed),
                gen_el=fundamental_paths[run],
                whale_share=share,
                whale_start=PRE_PERIOD,
                whale_valuation=WHALE_VALUATION,
            )

            prices = np.asarray(results["midquote"], dtype=float)
            fundamentals = fundamental_paths[run][:-1]
            signed_error = prices - fundamentals
            abs_error = np.abs(signed_error)

            for period, (price, fundamental, error, absolute_error) in enumerate(
                zip(prices, fundamentals, signed_error, abs_error)
            ):
                trajectories.append(
                    {
                        "run": run,
                        "seed": seed,
                        "whale_share": share,
                        "period": period,
                        "phase": "pre" if period < PRE_PERIOD else "post",
                        "midquote": price,
                        "fundamental": fundamental,
                        "signed_error": error,
                        "absolute_error": absolute_error,
                    }
                )

            run_rows.append(
                {
                    "run": run,
                    "seed": seed,
                    "whale_share": share,
                    "peak_absolute_error_post": np.nan,
                    "peak_time_post": np.nan,
                    "peak_exceeds_control_cutoff": False,
                    "recovered": False,
                    "recovery_time_after_peak": np.nan,
                    "recovery_confirmation_time_after_peak": np.nan,
                    "initial_nonwhale_market_capital": initial_nonwhale_capital,
                    "whale_initial_budget": results["whale_initial_budget"],
                    "market_cash_before_whale": results[
                        "market_cash_before_whale"
                    ],
                    "market_cash_after_whale": results[
                        "market_cash_after_whale"
                    ],
                    "mean_signed_error_post": np.nanmean(
                        signed_error[PRE_PERIOD:]
                    ),
                    "mae_post": np.nanmean(abs_error[PRE_PERIOD:]),
                    "rmse_post": np.sqrt(
                        np.nanmean(abs_error[PRE_PERIOD:] ** 2)
                    ),
                }
            )

    trajectory_frame = pd.DataFrame(trajectories)
    run_frame = pd.DataFrame(run_rows)

    control_pre_errors = trajectory_frame.loc[
        (trajectory_frame["whale_share"] == 0)
        & (trajectory_frame["phase"] == "pre"),
        "absolute_error",
    ].dropna()
    if control_pre_errors.empty:
        raise ValueError("No valid no-whale pre-period errors available for calibration.")

    error_band = float(
        control_pre_errors.quantile(ORDINARY_ERROR_QUANTILE)
    )
    if not np.isfinite(error_band) or error_band <= 0:
        raise ValueError("The calibrated ordinary-error band must be finite and positive.")

    control_post_peaks = (
        trajectory_frame.loc[
            (trajectory_frame["whale_share"] == 0)
            & (trajectory_frame["phase"] == "post")
        ]
        .groupby("run")["absolute_error"]
        .max()
        .dropna()
    )
    if control_post_peaks.empty:
        raise ValueError("No valid no-whale post-period peaks available for calibration.")
    peak_cutoff = float(control_post_peaks.quantile(CONTROL_PEAK_QUANTILE))
    if not np.isfinite(peak_cutoff) or peak_cutoff <= 0:
        raise ValueError("The calibrated control peak cutoff must be finite and positive.")

    run_frame = apply_recovery_metrics(
        run_frame,
        trajectory_frame,
        error_band,
        peak_cutoff,
    )
    run_frame["ordinary_error_band"] = error_band
    run_frame["ordinary_error_quantile"] = ORDINARY_ERROR_QUANTILE
    run_frame["control_peak_cutoff"] = peak_cutoff
    run_frame["control_peak_quantile"] = CONTROL_PEAK_QUANTILE
    run_frame["pre_period"] = PRE_PERIOD
    run_frame["post_period"] = POST_PERIOD
    run_frame["convergence_window"] = CONVERGENCE_WINDOW

    aggregate_frame = summarize_groups(run_frame)

    run_suffix = f"{POST_PERIOD}_periods"
    run_frame.to_csv(
        SIM_DIR / f"whale_convergence_runs_{run_suffix}.csv",
        index=False,
    )
    trajectory_frame.assign(
        ordinary_error_band=error_band,
        control_peak_cutoff=peak_cutoff,
    ).to_csv(
        SIM_DIR / f"whale_convergence_trajectories_{run_suffix}.csv",
        index=False,
    )
    aggregate_frame.assign(
        ordinary_error_band=error_band,
        ordinary_error_quantile=ORDINARY_ERROR_QUANTILE,
        control_peak_cutoff=peak_cutoff,
        control_peak_quantile=CONTROL_PEAK_QUANTILE,
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        convergence_window=CONVERGENCE_WINDOW,
    ).to_csv(
        SIM_DIR / f"whale_convergence_summary_{run_suffix}.csv",
        index=False,
    )

    plot_convergence_curves(run_frame, error_band, peak_cutoff)

    print(
        f"\nOrdinary-error band: {error_band:.4f} "
        f"(q{ORDINARY_ERROR_QUANTILE:.0%} of {len(control_pre_errors)} valid "
        "no-whale pre-period errors)"
    )
    print(
        f"Control peak cutoff: {peak_cutoff:.4f} "
        f"(q{CONTROL_PEAK_QUANTILE:.0%} of {len(control_post_peaks)} "
        "no-whale post-period run peaks)"
    )
    print(aggregate_frame.to_string(index=False))
    print(f"\nSaved sim_output/whale_convergence_runs_{run_suffix}.csv")
    print(f"Saved sim_output/whale_convergence_trajectories_{run_suffix}.csv")
    print(f"Saved sim_output/whale_convergence_summary_{run_suffix}.csv")
    print(f"Saved figures/whale_convergence_time_{run_suffix}.png and .pdf")


if __name__ == "__main__":
    main()
