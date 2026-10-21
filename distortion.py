"""Model-agnostic measures of the size and persistence of a price distortion.

Every statistic is computed for every run, distorted or not, from three inputs any
market model can supply: the price path, the fundamental path and the shock time.

    smoothed error  e_t = trailing mean of |price - fundamental| over `window` periods
    ordinary band   b   = a high quantile of e_t in undisturbed control periods

    P1  Maximum error        M = max_{t >= shock} e_t
        The worst mispricing after the shock, in probability points; compare with b.

    P1b Normalised pull      P = least-squares slope of (price - fundamental) on
        (target - fundamental) after the shock, with no intercept. The share of the gap
        between fundamental and a distorting target (e.g. the whale's valuation) that the
        price closes: 0 = price tracks the fundamental, 1 = price sits on the target.
        Unlike M, it does not depend on how far the fundamental happens to wander.

    P2  Reversion time       R = periods from the time of M until e_t <= b for
        `window` consecutive periods (0 when M <= b). Runs that never revert are
        right-censored at the horizon and summarised with Kaplan-Meier.

    P3  Time outside band    O = share of post-shock periods with e_t > b
        Total persistence, counting gradual onsets and repeated episodes. About
        1 - quantile in undisturbed runs, by construction.

Timing reversion from the maximum, not the shock, stops a gradual shock (one that leaves
the price inside the band for its first few periods) from counting as an instant recovery.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DEFAULT_WINDOW = 5
DEFAULT_BAND_QUANTILE = 0.95
MIN_AT_RISK = 5
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
    """P1-P3 for one run (see module docstring)."""
    post = np.asarray(smoothed_error, dtype=float)[shock_time:]
    horizon = len(post)
    if horizon == 0:
        raise ValueError("shock_time must fall inside the simulation")
    valid = np.isfinite(post)
    if not valid.any():
        raise ValueError("No valid smoothed errors after the shock")

    peak_time = int(np.nanargmax(post))
    max_error = float(post[peak_time])

    reversion_time, reverted = float(horizon - peak_time), False
    inside = valid & (post <= band)
    if max_error <= band:
        reversion_time, reverted = 0.0, True
    else:
        for start in range(peak_time, horizon - window + 1):
            if inside[start:start + window].all():
                reversion_time, reverted = float(start - peak_time), True
                break

    return {
        "max_error": max_error,
        "ordinary_band": band,
        "peak_time_after_shock": peak_time,
        "reversion_time": reversion_time,
        "reverted": reverted,
        "share_outside_band": float(np.mean(valid & (post > band))),
        "horizon_after_shock": horizon,
    }


def normalised_pull(prices, fundamentals, target, shock_time):
    """P1b for one run (see module docstring); missing prices carry the last quote forward."""
    prices = pd.Series(np.asarray(prices, dtype=float)).ffill().to_numpy()[shock_time:]
    fundamentals = np.asarray(fundamentals, dtype=float)[shock_time:]
    if prices.shape != fundamentals.shape:
        raise ValueError("Prices and fundamentals must be 1D arrays of equal length")
    gap = target - fundamentals
    valid = np.isfinite(prices) & np.isfinite(gap)
    if not valid.any() or not np.any(gap[valid]):
        return np.nan
    return float(np.sum((prices - fundamentals)[valid] * gap[valid])
                 / np.sum(gap[valid] ** 2))


def kaplan_meier_curve(durations, observed, horizon, min_at_risk=1):
    """Return cumulative convergence and Greenwood log-log confidence intervals.

    The curve stops once fewer than `min_at_risk` runs remain under observation, so a tail
    estimated from one or two late runs is not drawn.
    """
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    event_times = np.unique(durations[observed])
    if min_at_risk > 1:
        if len(durations) < min_at_risk:
            return [0.0], [0.0], [0.0], [0.0]
        horizon = min(horizon, float(np.sort(durations)[-min_at_risk]))
        event_times = event_times[event_times <= horizon]

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


def km_median(durations, observed, horizon, min_at_risk=1):
    """Kaplan-Meier median duration; NaN when fewer than half the runs recover."""
    times, probability, _, _ = kaplan_meier_curve(durations, observed, horizon, min_at_risk)
    return next((t for t, p in zip(times, probability) if p >= 0.5), np.nan)


def summarize_distortion(metrics, group="group"):
    """One row per group: medians and interquartile ranges of P1-P3."""
    rows = []
    for name, frame in metrics.groupby(group, sort=False):
        row = {group: name, "runs": len(frame),
               "ordinary_band": frame["ordinary_band"].iloc[0]}
        for column in ("max_error", "normalised_pull", "share_outside_band"):
            if column not in frame:
                continue
            q25, median, q75 = frame[column].quantile([0.25, 0.5, 0.75])
            row.update({f"median_{column}": median, f"{column}_q25": q25,
                        f"{column}_q75": q75})
        row["median_reversion_time_km"] = km_median(
            frame["reversion_time"], frame["reverted"], frame["horizon_after_shock"].max(),
            MIN_AT_RISK,
        )
        row["reverted_by_horizon"] = float(frame["reverted"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _group_colors(names, palette=None):
    palette = GROUP_COLORS if palette is None else palette
    if len(names) > len(palette):
        raise ValueError(f"At most {len(palette)} groups can be plotted together")
    return dict(zip(names, palette))


def _style(axis):
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_ridges(metrics, column, group, ax, xlabel, title, label_format, limits, palette,
                 bin_width, show_band):
    """One histogram per group in its own row on a shared x-axis, first group at the top.

    Each row is scaled to its own tallest bar, so rows show shape and position, not counts.
    """
    names = list(pd.unique(metrics[group]))
    colors = _group_colors(names, palette)
    edges = np.arange(limits[0], limits[1] + bin_width / 2, bin_width)
    height = 0.85
    ticks, tick_labels = [], []

    for row, name in enumerate(names):
        base = len(names) - 1 - row
        frame = metrics.loc[metrics[group] == name]
        values = frame[column].dropna()
        counts, _ = np.histogram(values, bins=edges)
        scaled = base + height * counts / max(counts.max(), 1)
        ax.stairs(scaled, edges, baseline=base, fill=True, color=colors[name], alpha=0.3)
        ax.stairs(scaled, edges, baseline=base, color=colors[name], linewidth=1.4)
        ax.axhline(base, color="#BBBBBB", linewidth=0.6)
        median = values.median()
        ax.vlines(median, base, base + height, color=colors[name], linewidth=1.6)
        if show_band:
            ax.vlines(frame["ordinary_band"].iloc[0], base, base + height, color="#888888",
                      linewidth=1, linestyle="--", label="Ordinary band" if row == 0 else None)
        ticks.append(base + height / 2)
        tick_labels.append(f"{name}\nmedian {label_format(median)}")

    ax.set_yticks(ticks, tick_labels, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set(xlim=limits, ylim=(0, len(names)), xlabel=xlabel, ylabel=group)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if show_band:
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    return ax


def _plot_distribution(metrics, column, group, ax, xlabel, title, label_format, limits,
                       palette=None,
                       bin_width=0.025, show_band=False, layout="overlay"):
    """Histograms of one statistic by group, each median marked.

    layout="overlay" draws them on top of each other; layout="ridge" gives each its own row.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    if layout == "ridge":
        return _plot_ridges(metrics, column, group, ax, xlabel, title, label_format, limits,
                            palette, bin_width, show_band)
    names = list(pd.unique(metrics[group]))
    colors = _group_colors(names, palette)
    edges = np.arange(limits[0], limits[1] + bin_width / 2, bin_width)

    for name in names:
        values = metrics.loc[metrics[group] == name, column].dropna()
        median = values.median()
        ax.hist(values, bins=edges, color=colors[name], alpha=0.22)
        ax.hist(values, bins=edges, histtype="step", color=colors[name], linewidth=1.6,
                label=f"{name} (median {label_format(median)})")
        ax.axvline(median, color=colors[name], linewidth=1.2, linestyle=":")

    if show_band:
        bands = pd.unique(metrics["ordinary_band"])
        for position, band in enumerate(bands):
            ax.axvline(band, color="#888888", linewidth=1, linestyle="--",
                       label=("Ordinary band" if len(bands) == 1 else "Ordinary bands")
                       if position == 0 else None)

    ax.set(xlim=limits, xlabel=xlabel, ylabel="Runs")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=8, title=group, title_fontsize=8)
    _style(ax)
    return ax


def plot_max_error(metrics, group="group", ax=None, title=None, palette=None,
                   layout="overlay"):
    """P1: distribution of the maximum post-shock error across runs, by group."""
    upper = 0.05 * np.ceil(metrics["max_error"].max() / 0.05 + 1)
    return _plot_distribution(
        metrics, "max_error", group, ax, palette=palette,
        xlabel="Maximum market error (probability points)",
        title=title or "P1: Maximum market error",
        label_format=lambda value: f"{value:.2f}",
        limits=(0, upper), show_band=True, layout=layout,
    )


def plot_normalised_pull(metrics, group="group", ax=None, title=None, palette=None,
                         layout="overlay"):
    """P1b: distribution of the normalised pull toward the distorting target, by group."""
    values = metrics["normalised_pull"]
    lower = min(-0.1, 0.05 * np.floor(values.min() / 0.05 - 1))
    upper = max(1.0, 0.05 * np.ceil(values.max() / 0.05 + 1))
    ax = _plot_distribution(
        metrics, "normalised_pull", group, ax, palette=palette,
        xlabel="Normalised pull (0 = tracks fundamental, 1 = at target)",
        title=title or "P1b: Pull toward the whale's valuation",
        label_format=lambda value: f"{value:.2f}",
        limits=(lower, upper), layout=layout,
    )
    for reference in (0, 1):
        ax.axvline(reference, color="#BBBBBB", linewidth=0.8)
    return ax


def plot_reversion_time(metrics, group="group", ax=None, title=None, min_at_risk=MIN_AT_RISK,
                        palette=None):
    """P2: Kaplan-Meier share of runs reverted against periods since their maximum error.

    Each curve ends once fewer than `min_at_risk` runs remain under observation.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    names = list(pd.unique(metrics[group]))
    colors = _group_colors(names, palette)
    horizon = int(metrics["horizon_after_shock"].max())

    for name in names:
        frame = metrics.loc[metrics[group] == name]
        times, probability, lower, upper = kaplan_meier_curve(
            frame["reversion_time"], frame["reverted"], horizon, min_at_risk
        )
        median = km_median(frame["reversion_time"], frame["reverted"], horizon, min_at_risk)
        median_label = f"median {median:.0f}" if np.isfinite(median) else "median not reached"
        ax.step(times, probability, where="post", color=colors[name], linewidth=2,
                label=f"{name} ({median_label}; {frame['reverted'].mean():.0%} reverted)")
        ax.fill_between(times, lower, upper, step="post", color=colors[name],
                        alpha=0.12, linewidth=0)

    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.set(xlim=(0, horizon), ylim=(0, 1), xlabel="Periods since maximum error",
           ylabel="Share of runs back inside the ordinary band")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(title or "P2: Market error reversion time", fontsize=11)
    ax.text(0.01, 0.99, f"Curves end when fewer than {min_at_risk} runs remain",
            transform=ax.transAxes, va="top", fontsize=8, color="#555555")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _style(ax)
    return ax


def plot_time_outside_band(metrics, group="group", ax=None, title=None,
                           band_quantile=DEFAULT_BAND_QUANTILE, palette=None,
                           layout="overlay"):
    """P3: distribution of the share of post-shock periods spent outside the band."""
    bands = pd.unique(metrics["ordinary_band"])
    band_value = f" = {bands[0]:.3f}" if len(bands) == 1 else " (set per model)"
    ax = _plot_distribution(
        metrics, "share_outside_band", group, ax, palette=palette,
        xlabel="Share of post-shock periods with error above the band",
        title=title or (
            "P3: Time with pricing error above ordinary noise\n"
            f"Band{band_value}: {band_quantile:.0%} of control-run errors fall below it"
        ),
        label_format=lambda value: f"{value:.0%}",
        limits=(0, 1), layout=layout,
    )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    return ax


def plot_distortion_figures(metrics, group="group", title=None):
    """P1-P3 on one figure, with P1b beside P1 when `normalised_pull` is present."""
    if "normalised_pull" in metrics:
        figure, axes = plt.subplots(2, 2, figsize=(13, 9.6), constrained_layout=True)
        plot_max_error(metrics, group, axes[0, 0])
        plot_normalised_pull(metrics, group, axes[0, 1])
        plot_reversion_time(metrics, group, axes[1, 0])
        plot_time_outside_band(metrics, group, axes[1, 1])
    else:
        figure, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
        plot_max_error(metrics, group, axes[0])
        plot_reversion_time(metrics, group, axes[1])
        plot_time_outside_band(metrics, group, axes[2])
    if title:
        figure.suptitle(title, fontsize=12)
    return figure
