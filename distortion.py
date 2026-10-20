"""Model-agnostic measures of the magnitude and duration of a price distortion.

Both statistics are computed for every run, distorted or not, from three inputs any
market model can supply: the price path, the fundamental path and the shock time.

    smoothed error  e_t = trailing mean of |price - fundamental| over `window` periods
    ordinary band   b   = a high quantile of e_t in undisturbed control periods

    Magnitude: peak excess error   M = max(0, max_{t >= shock} e_t - b)
        How far, in probability points, the price moved beyond ordinary noise.
        M = 0 means the shock never pushed the error outside the ordinary band.

    Duration: distortion duration  D = first d >= 0 such that e_t <= b for the
        `window` consecutive periods starting at shock + d.
        How long the market stayed mispriced. Runs that never recover are
        right-censored at the horizon and handled with Kaplan-Meier.

Measuring both from the shock, rather than from a noisy peak, and over all runs rather
than only "distorted" ones, keeps the numbers comparable across models and horizons.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DEFAULT_WINDOW = 5
DEFAULT_BAND_QUANTILE = 0.95
GROUP_COLORS = ("#222222", "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00")


def smoothed_absolute_error(prices, fundamentals, window=DEFAULT_WINDOW):
    """Trailing rolling mean of |price - fundamental|, over fewer periods at the very start.

    A missing price (e.g. a one-sided book has no midquote) carries the last quoted price
    forward: that is the price the market was still showing.
    """
    prices = np.asarray(prices, dtype=float)
    fundamentals = np.asarray(fundamentals, dtype=float)
    if prices.shape != fundamentals.shape or prices.ndim != 1:
        raise ValueError("Prices and fundamentals must be 1D arrays of equal length")
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("Window must be a positive integer")
    prices = pd.Series(prices).ffill().to_numpy()
    error = pd.Series(np.abs(prices - fundamentals))
    return error.rolling(window, min_periods=1).mean().to_numpy()


def calibrate_band(control_errors, quantile=DEFAULT_BAND_QUANTILE):
    """Ordinary-error band: a quantile of smoothed errors from undisturbed periods."""
    control_errors = np.asarray(control_errors, dtype=float)
    control_errors = control_errors[np.isfinite(control_errors)]
    if control_errors.size == 0:
        raise ValueError("No valid control errors available for calibration")
    band = float(np.quantile(control_errors, quantile))
    if band <= 0:
        raise ValueError("The calibrated ordinary-error band must be positive")
    return band


def distortion_metrics(smoothed_error, shock_time, band, window=DEFAULT_WINDOW):
    """Peak excess error and distortion duration for one run (see module docstring)."""
    post = np.asarray(smoothed_error, dtype=float)[shock_time:]
    horizon = len(post)
    if horizon == 0:
        raise ValueError("shock_time must fall inside the simulation")

    valid = np.isfinite(post)
    if valid.any():
        peak_time = int(np.nanargmax(post))
        peak_excess = max(0.0, float(post[peak_time]) - band)
    else:
        peak_time, peak_excess = np.nan, np.nan

    duration, recovered = float(horizon), False
    inside = valid & (post <= band)
    for start in range(horizon - window + 1):
        if inside[start:start + window].all():
            duration, recovered = float(start), True
            break

    return {
        "peak_excess_error": peak_excess,
        "peak_time_after_shock": peak_time,
        "distortion_duration": duration,
        "recovered": recovered,
        "horizon_after_shock": horizon,
    }


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

        if event_time == 0:
            # Recovery at the shock itself: replace the origin point instead of adding a step.
            times.pop(), probability.pop(), lower.pop(), upper.pop()
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


def km_median(durations, observed, horizon):
    """Kaplan-Meier median duration; NaN when fewer than half the runs recover."""
    times, probability, _, _ = kaplan_meier_curve(durations, observed, horizon)
    return next((t for t, p in zip(times, probability) if p >= 0.5), np.nan)


def summarize_distortion(metrics, group="group"):
    """One row per group: the two headline statistics plus their uncertainty."""
    rows = []
    for name, frame in metrics.groupby(group, sort=False):
        magnitude = frame["peak_excess_error"].dropna()
        horizon = int(frame["horizon_after_shock"].min())
        rows.append({
            group: name,
            "runs": len(frame),
            "median_peak_excess_error": magnitude.median(),
            "peak_excess_error_q25": magnitude.quantile(0.25),
            "peak_excess_error_q75": magnitude.quantile(0.75),
            "share_with_excess_error": float((magnitude > 0).mean()),
            "median_distortion_duration_km": km_median(
                frame["distortion_duration"], frame["recovered"], horizon
            ),
            "recovered_by_horizon": float(frame["recovered"].mean()),
            "horizon_after_shock": horizon,
        })
    return pd.DataFrame(rows)


def _group_colors(names):
    if len(names) > len(GROUP_COLORS):
        raise ValueError(f"At most {len(GROUP_COLORS)} groups can be plotted together")
    return dict(zip(names, GROUP_COLORS))


def _style(axis):
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_distortion_magnitude(metrics, group="group", ax=None, title=None):
    """Peak excess error per run by group: every run as a dot, median and IQR as a bar."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    names = list(pd.unique(metrics[group]))
    colors = _group_colors(names)
    jitter = np.random.default_rng(0)

    for position, name in enumerate(names):
        values = metrics.loc[metrics[group] == name, "peak_excess_error"].dropna()
        if values.empty:
            continue
        x = position + jitter.uniform(-0.18, 0.18, len(values))
        ax.scatter(x, values, s=14, color=colors[name], alpha=0.45, linewidths=0)
        q25, median, q75 = values.quantile([0.25, 0.5, 0.75])
        ax.vlines(position + 0.3, q25, q75, color=colors[name], linewidth=2)
        ax.plot(position + 0.3, median, "o", color=colors[name], markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{median:.2f}", (position + 0.36, median), va="center",
                    fontsize=8, color="#333333")

    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(range(len(names)), [str(name).replace(", ", "\n") for name in names])
    ax.set(xlim=(-0.5, len(names) - 0.3), xlabel=group,
           ylabel="Peak excess error (probability points)")
    ax.set_title(title or "Magnitude: peak error beyond the ordinary band", fontsize=11)
    _style(ax)
    return ax


def plot_distortion_duration(metrics, group="group", ax=None, title=None):
    """Kaplan-Meier share of runs recovered against periods since the shock, by group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    names = list(pd.unique(metrics[group]))
    colors = _group_colors(names)
    horizon_max = 0

    for name in names:
        frame = metrics.loc[metrics[group] == name]
        horizon = int(frame["horizon_after_shock"].min())
        horizon_max = max(horizon_max, horizon)
        times, probability, lower, upper = kaplan_meier_curve(
            frame["distortion_duration"], frame["recovered"], horizon
        )
        median = km_median(frame["distortion_duration"], frame["recovered"], horizon)
        median_label = f"median {median:.0f}" if np.isfinite(median) else "median not reached"
        ax.step(times, probability, where="post", color=colors[name], linewidth=2,
                label=f"{name} ({median_label}; {frame['recovered'].mean():.0%} recovered)")
        ax.fill_between(times, lower, upper, step="post", color=colors[name],
                        alpha=0.12, linewidth=0)

    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.set(xlim=(0, horizon_max), ylim=(0, 1), xlabel="Periods since shock",
           ylabel="Share of runs back inside the ordinary band")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(title or "Duration: time until sustained return to the band", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _style(ax)
    return ax
