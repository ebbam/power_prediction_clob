"""Paper figures 1-3, rerun with the CLOB market (run_market_lob) in place of the original market.

Figure 1
    A  Market price, election outcome and average valuation in one example run
    B  Parameter sweeps: mean squared error and peak price-fundamental cross-correlation
    C  Contracts traded per period and capital locked in open contracts, same run
    D  Misclassification probability heatmap (analytic, from Dec25Plots.ipynb cell 24)
    Panels A-D follow the co-author's Dec25Plots.ipynb (cells 19, 22 and 24), including its
    colours. Panel D does not depend on the market model. Panel B's right-hand
    column replaces the Granger "dominant lag" with the peak cross-correlation between price
    and fundamental changes.

Figure 2 (whale capital share ρ; the whale_convergence_experiment design, 500 runs per ρ)
    A  P1, maximum market error
    B  P3, time with pricing error above ordinary noise

Figure 3 (herding)
    A  P1 and B  P3 by herding strength h (bettor.market_imitation): a whale with ρ = 0.3 and
    valuation 1 trades from period 0 against 100 bettors with expertise 0.9 and herding h,
    500 runs per h. Each h is judged against the band of its own no-whale runs.

Figures 2 and 3 are drawn twice: overlaid histograms (figure2, figure3) and one row per group
(figure2b, figure3b). Their simulations run on N_WORKERS = 4 processes.

Simulations are cached in sim_output/ (figure1_sweep.csv ~4 min; figure2_runs.csv and
figure3_runs.csv ~10 min each on 4 processes); pass --rerun to redo them. Figures go to figures/.
"""

import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, PercentFormatter

from bettor import bettor, run_market_lob
from metrics import (
    _plot_distribution,
    calibrate_band,
    distortion_metrics,
    normalised_pull,
    plot_normalised_pull,
    plot_max_error,
    plot_time_outside_band,
    smoothed_absolute_error,
)
import figure_style as style
import experiments.whale_convergence_experiment as whale
from experiments.whale_convergence_experiment import RHO, make_election_path


style.apply()

N_BETTORS = 100
T_MAX = 100
INITIAL_PRICE = 0.5
RW_VARIANCE = 0.05
N_ITER = 30
MAX_LAG = 5
DELAY_MAX_LAG = 10
MIN_RESPONSE_LAG = 2  # price responds to news no sooner than 2 periods later, by timing
PRESSURE_SCALE = 15
EXAMPLE_SEED = 4  # the example run in panels A and C
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SIM_DIR = OUTPUT_DIR / "sim_output"
SIM_DIR.mkdir(exist_ok=True)
SWEEP_CSV = SIM_DIR / "figure1_sweep.csv"
LINE_COLOR = "#B22222"
POINT_COLOR = "#457B9D"
ELECTION_COLOR = "#E63946"
VALUATION_COLOR = "#B788E5"
# Panel D: market price and outcome as correlated normals (co-author's parameters).
MISCLASS_SIGMA_PRICE = 0.07
MISCLASS_SIGMA_OUTCOME = 0.05
MISCLASS_CORRELATION = 0.6
MISCLASS_GRID = 201
HERDING_CSV = SIM_DIR / "figure3_runs.csv"
HERDING_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
HERDING_RUNS = 500
WHALE_RUNS = 500
WHALE_CSV = SIM_DIR / "figure2_runs.csv"
# Figure 2C: the original design (Dec25Plots cells 27-31) in the CLOB.
WHALE_VALUATIONS = (0.6, 0.7, 0.8)
VALUATION_COLORS = ("#457B9D", "#339966", "#F4A261")
VALUATION_WHALE_SHARE = 0.5
VALUATION_BUDGET = 100
VALUATION_EXPERTISE = 0.9
VALUATION_RUNS = 500
VALUATION_SEED = 20_000
VALUATION_CSV = SIM_DIR / "figure2c_paths.csv"
# Figure 2 D-E: profit experiment (expertise spread 0-1, ρ = 0, 0.1, ..., 0.9).
PROFIT_SHARES = tuple(round(0.1 * k, 1) for k in range(10))
PROFIT_RUNS = 500
PROFIT_DIR = SIM_DIR / "profit_experiment"
PROFIT_SHOWN_SHARES = PROFIT_SHARES            # ρ values in the full-width wealth figures
FIGURE2_D_SHARES = (0.0, 0.3, 0.6, 0.9)        # ρ values in Figure 2D (one column)
EXPERTISE_GROUPS = (0.0, 0.5, 0.8, 0.9, 1.0)   # panel D's expertise groups
RISK_AVERSION_GROUPS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEALTH_FLOOR = 0.01                            # log-axis floor: a 99% loss
N_WORKERS = 4  # processes for the simulations; kept low to leave the laptop usable
HERDING_WHALE_SHARE = 0.3
HERDING_WHALE_VALUATION = 1.0
HERDING_EXPERTISE = 0.9
HERDING_BUDGET = 100
HERDING = "Herding strength (h)"
STABILITY_ALPHAS = (0.10, 1.50)   # Figure 3 C and D, as in Dec25Plots.ipynb cell 46
# Position of the h = 1 ABM setup on the phase diagrams, read off the published Figure 3;
# the notebook does not record how it was computed.
HERDING_ABM_POINT = (0.34, 0.0)

# The power_prediction palette (cell 3 of the original notebooks), with the orders the
# published figures use.
PALETTE = {
    "lavender": "#CC99FF", "navy": "#000080", "green": "#339966", "orange": "#F4A261",
    "ruby": "#B22222", "violet": "#6A4C93", "indigo": "#1D3557", "teal": "#2A9D8F",
    "steel": "#457B9D", "amber": "#FFB347", "crimson": "#E63946",
}
# Figure 2's ρ groups: the first four colours of the project palette, in its order
# (cell 1 of Dec25Plots.ipynb).
WHALE_SHARE_COLORS = [PALETTE[name] for name in ("navy", "green", "orange", "ruby")]
HERDING_COLORS = [PALETTE[name] for name in ("teal", "ruby", "crimson", "steel", "violet")]


def _budget_population(variance):
    budgets = np.clip(np.random.normal(500, variance, N_BETTORS), 0, None)
    return [bettor(budget=budget) for budget in budgets]


# Parameter name -> (values, population builder), as in the original sweep.
SWEEPS = {
    "Expertise": (np.arange(0, 1.1, 0.1),
                  lambda e: [bettor(expertise=e) for _ in range(N_BETTORS)]),
    "Bias": (np.arange(-0.3, 0.3, 0.05),
             lambda b: [bettor(bias=b) for _ in range(N_BETTORS)]),
    "Risk Aversion": (np.arange(0, 1.1, 0.1),
                      lambda r: [bettor(risk_av=r) for _ in range(N_BETTORS)]),
    "Stubbornness": (np.arange(0.05, 1, 0.05),
                     lambda s: [bettor(stubbornness=s) for _ in range(N_BETTORS)]),
    "Budget Variance": (np.arange(0, 500, 50), _budget_population),
}


def lagged_correlations(prices, fundamentals, max_lag):
    """corr(Δprice_t, Δfundamental_{t-k}) for k = 0..max_lag (NaN where undefined)."""
    price_changes = np.diff(np.asarray(prices, dtype=float))
    fundamental_changes = np.diff(np.asarray(fundamentals, dtype=float))
    correlations = []
    for lag in range(max_lag + 1):
        x = price_changes[lag:]
        y = fundamental_changes[:len(fundamental_changes) - lag]
        if x.std() == 0 or y.std() == 0:
            correlations.append(np.nan)
        else:
            correlations.append(np.corrcoef(x, y)[0, 1])
    return np.asarray(correlations)


def peak_cross_correlation(correlations, max_lag=MAX_LAG):
    """Largest correlation over lags 0..max_lag, and the lag k at which it occurs."""
    correlations = np.asarray(correlations)[:max_lag + 1]
    if np.all(np.isnan(correlations)):
        return np.nan, np.nan
    lag = int(np.nanargmax(correlations))
    return float(correlations[lag]), lag


def average_response_delay(correlations, n_changes):
    """Correlation-weighted mean lag beyond the 2-period minimum, using only lags whose
    correlation exceeds what noise reaches by chance (1.96 / sqrt(n)); NaN if none does."""
    correlations = np.nan_to_num(np.asarray(correlations, dtype=float))
    weights = np.where(correlations > 1.96 / math.sqrt(n_changes), correlations, 0.0)
    if weights.sum() == 0:
        return np.nan
    return float(np.sum(np.arange(len(weights)) * weights) / weights.sum() - MIN_RESPONSE_LAG)


def sweep_run(sweep_index, value_index, iteration):
    """One sweep run: MSE and price-fundamental correlations at lags 0..DELAY_MAX_LAG."""
    parameter, (values, make_population) = list(SWEEPS.items())[sweep_index]
    value = values[value_index]
    seed = 100_000 * sweep_index + 1_000 * value_index + iteration
    np.random.seed(seed)
    record = run_market_lob(
        N_BETTORS, T_MAX, INITIAL_PRICE, RW_VARIANCE, make_population(value),
        rng=np.random.default_rng(seed),
    )
    prices = np.asarray(record["price_history"], dtype=float)
    fundamentals = np.asarray(record["gen_el"], dtype=float)
    correlations = lagged_correlations(prices, fundamentals, DELAY_MAX_LAG)
    correlation, lag = peak_cross_correlation(correlations)
    return {
        "parameter": parameter, "value": round(float(value), 10),
        "iteration": iteration, "seed": seed,
        "mse": float(np.mean((prices - fundamentals) ** 2)),
        "peak_cross_correlation": correlation, "peak_lag": lag,
        "response_delay": average_response_delay(correlations, len(prices) - 1),
        **{f"correlation_lag_{k}": c for k, c in enumerate(correlations)},
    }


def run_sweep():
    tasks = [(sweep_index, value_index, iteration)
             for sweep_index, (values, _) in enumerate(SWEEPS.values())
             for value_index in range(len(values))
             for iteration in range(N_ITER)]
    print(f"Parameter sweep: {len(tasks)} runs on {N_WORKERS} processes")
    sweep = pd.DataFrame(parallel_map(sweep_run, *zip(*tasks)))
    sweep.to_csv(SWEEP_CSV, index=False)
    return sweep


def example_run(seed=None):
    """One run of the standard market, without a whale: 100 bettors with budget 100."""
    seed = EXAMPLE_SEED if seed is None else seed
    np.random.seed(seed)
    bettors = [bettor(budget=100) for _ in range(N_BETTORS)]
    return run_market_lob(N_BETTORS, T_MAX, INITIAL_PRICE, RW_VARIANCE, bettors,
                          rng=np.random.default_rng(seed))


def plot_market_price(record, ax):
    """Panel A, as plot_returns(step=True) in the original figure."""
    prices = np.asarray(record["price_history"], dtype=float)
    pressure = np.asarray(record["market_pressure"], dtype=float)[1:]
    periods = np.arange(len(prices) - 1)
    upward = prices[:-1] + np.where(pressure > 0, pressure, 0) / PRESSURE_SCALE
    downward = prices[:-1] + np.where(pressure < 0, pressure, 0) / PRESSURE_SCALE

    ax.plot(prices, color="black", label="Market price", drawstyle="steps")
    ax.fill_between(periods, downward, prices[:-1], color="dodgerblue", alpha=0.5,
                    step="pre", label="Downward Pressure")
    ax.plot(record["gen_el"], color=ELECTION_COLOR, label="Election outcome", drawstyle="steps")
    ax.fill_between(periods, upward, prices[:-1], color="darkorange", alpha=0.5,
                    step="pre", label="Upward Pressure")
    ax.plot(record["beliefs"], color=VALUATION_COLOR, linewidth=1.2, label="Average market valuation",
            drawstyle="steps")
    ax.set(ylim=(0, 1.02), xlabel="Time", ylabel="Result", title="Market Price")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=2, frameon=False)


def plot_volume(record, ax_volume, ax_capital, volume="all"):
    """Panel C: contracts traded each period (the first difference of cumulative volume) and,
    below it, the capital locked in open contracts as a share of all traders' money.

    Each contract locks $1 in total: its price p paid by the YES holder and 1 - p of collateral
    from the NO side, so the two stacked areas sum to the open interest in dollars.

    volume="all" plots every contract traded each period; volume="new" plots the net change in
    contracts outstanding (positions opened minus positions closed).
    """
    if volume == "new":
        outstanding = np.asarray(record["vol_history"], dtype=float) / 2
        ax_volume.plot(np.diff(outstanding), color="#1D3557", drawstyle="steps")
        ax_volume.axhline(0, color="#BBBBBB", linewidth=0.6)
        ax_volume.set(ylabel="New contracts\nper period (net)",
                      title="New Contracts & Capital Locked")
    else:
        ax_volume.plot(record["traded_volume"], color="#1D3557", drawstyle="steps")
        ax_volume.set(ylabel="Traded per\nperiod",
                      title="Trading Volume & Capital Locked", ylim=(0, None))
    ax_volume.tick_params(axis="x", labelbottom=False)
    ax_volume.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))

    open_interest = np.asarray(record["vol_history"][1:], dtype=float) / 2
    prices = np.asarray(record["price_history"][1:], dtype=float)
    total = float(np.sum(record["final_budgets"]))     # cash only moves between traders
    yes_stake = open_interest * prices / total
    no_stake = open_interest * (1 - prices) / total
    periods = np.arange(len(open_interest))
    ax_capital.fill_between(periods, 0, yes_stake, step="post", color=PALETTE["green"],
                            alpha=0.85, linewidth=0, label="Yes Contracts")
    ax_capital.fill_between(periods, yes_stake, yes_stake + no_stake, step="post",
                            color=PALETTE["ruby"], alpha=0.75, linewidth=0,
                            label="No Contracts")
    ax_capital.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncols=2,
                      frameon=False)
    ax_capital.set(xlabel="Time", ylabel="% Capital\nLocked", ylim=(0, None))
    ax_capital.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_capital.tick_params(axis="x", labelrotation=45)
    ax_volume.figure.align_ylabels([ax_volume, ax_capital])


def plot_lag_column(sweep, parameter, ax):
    """Best-matching lag per run (jittered), with the median and interquartile range."""
    data = sweep.loc[sweep["parameter"] == parameter]
    spread = np.ptp(data["value"]) * 0.01
    jitter = np.random.default_rng(0).uniform(-spread, spread, len(data))
    ax.scatter(data["value"] + jitter, data["peak_lag"], s=4, alpha=0.6, color=POINT_COLOR)
    stats = data.groupby("value")["peak_lag"].quantile([0.25, 0.5, 0.75]).unstack()
    ax.plot(stats.index, stats[0.5], color=LINE_COLOR, linewidth=1)
    ax.fill_between(stats.index, stats[0.25], stats[0.75], color=LINE_COLOR, alpha=0.25,
                    linewidth=0)
    ax.set(xlabel=parameter, ylabel="Best-matching lag", ylim=(-0.5, MAX_LAG + 0.5),
           yticks=range(MAX_LAG + 1))


def plot_sweep_column(sweep, parameter, column, ylabel, ax):
    """Every run as a dot, with the mean and its 95% confidence interval across runs."""
    data = sweep.loc[sweep["parameter"] == parameter]
    ax.scatter(data["value"], data[column], s=4, alpha=0.6, color=POINT_COLOR)
    stats = data.groupby("value")[column].agg(["mean", "std", "count"])
    half_width = 1.96 * stats["std"] / np.sqrt(stats["count"])
    ax.plot(stats.index, stats["mean"], color=LINE_COLOR, linewidth=1)
    ax.fill_between(stats.index, stats["mean"] - half_width, stats["mean"] + half_width,
                    color=LINE_COLOR, alpha=0.25, linewidth=0)
    ax.set(xlabel=parameter, ylabel=ylabel)


def _normal_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2)))


def bivariate_normal_cdf(a, b, correlation, n_nodes=4001):
    """P(X <= a, Y <= b) for standard normals with the given correlation, on an a-by-b grid.

    Integrates phi(x) * Phi((b - correlation * x) / sqrt(1 - correlation^2)) over x <= a.
    """
    nodes = np.linspace(-9, 9, n_nodes)
    density = np.exp(-0.5 * nodes ** 2) / math.sqrt(2 * math.pi)
    scale = math.sqrt(1 - correlation ** 2)
    inner = _normal_cdf((np.asarray(b)[:, None] - correlation * nodes[None, :]) / scale)
    integrand = density[None, :] * inner
    cumulative = np.concatenate([np.zeros((len(b), 1)), np.cumsum(
        0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(nodes), axis=1)], axis=1)
    return np.array([np.interp(a, nodes, row) for row in cumulative])


def misclassification_grid():
    """P((price - 0.5)(outcome - 0.5) < 0): price and outcome on opposite sides of 0.5."""
    centres = np.linspace(0, 1, MISCLASS_GRID)
    a = (0.5 - centres) / MISCLASS_SIGMA_PRICE      # one per market-price centre (columns)
    b = (0.5 - centres) / MISCLASS_SIGMA_OUTCOME    # one per true-outcome centre (rows)
    joint = bivariate_normal_cdf(a, b, MISCLASS_CORRELATION)
    return centres, _normal_cdf(a)[None, :] + _normal_cdf(b)[:, None] - 2 * joint


def plot_misclassification(ax, figure):
    """Panel D, as cell 24 of Dec25Plots.ipynb."""
    centres, probability = misclassification_grid()
    image = ax.imshow(probability, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                      cmap="magma", vmin=0, vmax=1)
    ax.plot([0, 1], [0, 1], color="white", linestyle="--", linewidth=1)
    ax.set(xlabel="Market price", ylabel="True outcome")
    figure.colorbar(image, ax=ax, location="bottom", fraction=0.06, pad=0.03).set_label(
        "Misclassification probability")


def _style_axis(ax):
    """Figure 1's look: a full frame around the plot and no gridlines."""
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)


def _legend_below(ax, ncols=3, offset=-0.2):
    """Move an axis's legend below it, without a frame, as in Figure 1."""
    legend = ax.get_legend()
    if legend is None:
        return
    title = legend.get_title().get_text()
    handles, labels = ax.get_legend_handles_labels()
    legend.remove()
    ax.legend(handles, labels, title=title or None, loc="upper center",
              bbox_to_anchor=(0.5, offset), ncols=ncols,
              frameon=False)


def label_panels(axes_and_letters, x=None, y=None):
    """Panel letters; see figure_style.label_panels (x and y are accepted for old callers)."""
    style.label_panels(axes_and_letters)


def save(figure, name):
    style.save(figure, FIGURES_DIR / name, name)
    plt.close(figure)


def figure1(rerun, right_column="cross_correlation", name="figure1_clob", volume="all"):
    """right_column: "cross_correlation" (peak value), "lag" (the lag at which it peaks) or
    "delay" (average response delay)."""
    sweep = pd.read_csv(SWEEP_CSV) if SWEEP_CSV.exists() and not rerun else run_sweep()
    record = example_run()

    # Layout of the original figure: a wide left column (short panel A, C, D) and the
    # parameter sweeps on the right, each plot with its own y-axis label.
    figure = plt.figure(figsize=(style.PAGE_WIDTH, 7.4), layout="constrained")
    left_fig, right_fig = figure.subfigures(1, 2, width_ratios=[1.15, 1], wspace=0.03)
    left = left_fig.add_gridspec(4, 1, height_ratios=[1.0, 0.65, 0.65, 1.75])
    ax_a = left_fig.add_subplot(left[0])
    ax_c = left_fig.add_subplot(left[1])
    ax_capital = left_fig.add_subplot(left[2], sharex=ax_c)
    ax_d = left_fig.add_subplot(left[3])
    plot_market_price(record, ax_a)
    plot_volume(record, ax_c, ax_capital, volume)
    plot_misclassification(ax_d, left_fig)

    right_titles = {"lag": "Best-Matching Lag", "delay": "Response Delay",
                    "cross_correlation": "Peak Cross-Corr."}
    sweep_axes = []
    axes = right_fig.subplots(len(SWEEPS), 2)
    for row, parameter in enumerate(SWEEPS):
        ax_mse, ax_right = axes[row]
        plot_sweep_column(sweep, parameter, "mse", "", ax_mse)
        if right_column == "lag":
            plot_lag_column(sweep, parameter, ax_right)
            ax_right.set_ylabel("")
        elif right_column == "delay":
            plot_sweep_column(sweep, parameter, "response_delay", "", ax_right)
            ax_right.set_ylim(bottom=-0.2)
        else:
            plot_sweep_column(sweep, parameter, "peak_cross_correlation", "", ax_right)
            ax_right.set_ylim(top=1)
        for ax in (ax_mse, ax_right):
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        sweep_axes.append(ax_mse)
    # What each column measures, once at its top, instead of a y-label on every row.
    axes[0, 0].set_title("Mean Squared Error")
    axes[0, 1].set_title(right_titles[right_column])

    label_panels(((ax_a, "A"), (sweep_axes[0], "B"), (ax_c, "C"), (ax_d, "D")), x=-0.2, y=1.05)
    save(figure, name)


def parallel_map(function, *iterables):
    """function over the zipped iterables on N_WORKERS processes, results in input order."""
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        return list(pool.map(function, *iterables, chunksize=8))


def plot_nonwhale_return(metrics, group, ax, palette, layout=None):
    """Box plots, one per group, of the non-whale traders' total profit as a share of their
    starting capital (one value per run), as in the original Figure 2B.

    Positions are valued at their expected payout given the final true probability. Trading
    is zero-sum, so this is exactly 0 without a whale and is otherwise the whale's loss.
    layout is accepted for a common signature; both figure versions use box plots.
    """
    names = list(pd.unique(metrics[group]))
    data = [metrics.loc[metrics[group] == name, "nonwhale_return"].dropna() for name in names]
    boxes = ax.boxplot(data, positions=range(len(names)), widths=0.6, patch_artist=True,
                       flierprops=dict(marker="o", markersize=2, alpha=0.5),
                       medianprops=dict(color="#333333"))
    for box, color in zip(boxes["boxes"], palette):
        box.set(facecolor=color, alpha=0.75, edgecolor="#333333")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(names)), [f"{name:g}" for name in names])
    ax.set(xlabel=group.replace("ρ (whale capital share)", "ρ (Whale Capital Share)"),
           ylabel="Return on Starting Capital", title="Non-Whale Traders' Return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    _style_axis(ax)
    return ax


def _shared_legend(ax_band, axes, group, layout, ax_below):
    """One legend, under ax_below, in place of the per-panel legends: the group colours
    (without per-panel medians, which the dotted lines mark) and the ordinary band."""
    handles, labels = ax_band.get_legend_handles_labels()
    entries = {}
    for handle, label in zip(handles, labels):
        name = label.split(" (median")[0]
        if name == "Ordinary band":
            name = "Ordinary noise band (A)"
        entries.setdefault(name, handle)
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    title = {"ρ (whale capital share)": "ρ (Whale Capital Share)",
             "Herding strength (h)": "Herding Strength (h)"}.get(group, group)
    # Both layouts get the full key: group colours (ridge rows carry no labels), then the band.
    names = sorted(entries, key=lambda name: name.startswith("Ordinary"))
    handles = [entries[name] for name in names]
    # Group colours as filled boxes, which are easier to read than thin line samples.
    from matplotlib.patches import Patch
    handles = [handle if name.startswith("Ordinary") else
               Patch(facecolor=handle.get_edgecolor() if hasattr(handle, "get_edgecolor")
                     else handle.get_color(), edgecolor="none")
               for name, handle in zip(names, handles)]
    holder = ax_below.get_figure()          # the row's subfigure when rows are separate
    if holder is not ax_below.get_figure(root=True):
        holder.legend(handles, names, title=title, loc="outside lower center",
                      ncols=len(names), frameon=False, handlelength=1.4)
    else:
        ax_below.legend(handles, names, title=title, loc="upper center",
                        bbox_to_anchor=(0.5, -0.35), ncols=3, frameon=False)


def distribution_figures(metrics, group, palette, name, pull=False, bottom=None,
                         bottom_height=2.2, layouts=("overlay",), full_width=False,
                         top_height=None, name_suffix=""):
    """A = maximum market error (P1), B = time above the band (P3) and, with pull,
    C = normalised pull toward the whale's valuation (P1b), with one shared legend: stacked in
    one column, or side by side across the page when full_width. bottom(figure, slot) draws the remaining panels in the slot below and returns
    them as (axis, letter) pairs. Saved as `name` (overlaid histograms) and `name`b (one row
    per group; in one column only short figures fit a page, so layouts defaults to overlay).
    """
    for layout in layouts:
        suffix = "" if layout == "overlay" else "b"
        panel_height = 1.35 if layout == "overlay" else 0.3 * metrics[group].nunique() + 0.75
        n_dist = 3 if pull else 2
        if full_width:
            heights = [top_height or panel_height + 0.45] + ([bottom_height] if bottom else [])
            figure = plt.figure(figsize=(style.PAGE_WIDTH, sum(heights) + 0.5),
                                layout="constrained")
            # Rows as separate subfigures, so each row is laid out on its own.
            rows = figure.subfigures(len(heights), 1, height_ratios=heights)
            rows = list(rows) if len(heights) > 1 else [rows]
            axes = list(rows[0].subplots(1, n_dist))
            slot = rows[1] if bottom else None
        else:
            heights = [panel_height] * n_dist + ([bottom_height] if bottom else [])
            figure = plt.figure(figsize=(style.COLUMN_WIDTH, sum(heights) + 0.9),
                                layout="constrained")
            grid = figure.add_gridspec(len(heights), 1, height_ratios=heights)
            axes = [figure.add_subplot(grid[k]) for k in range(n_dist)]
            slot = grid[n_dist] if bottom else None
        plot_max_error(metrics, group, axes[0], palette=palette, layout=layout,
                       row_labels=False)
        plot_time_outside_band(metrics, group, axes[1], palette=palette, layout=layout,
                               row_labels=False)
        axes[0].set(title="Maximum Error", xlabel="Max |Price − Outcome|")
        axes[1].set(title="Time Above Noise", xlabel="Share of Periods Above Band")
        if pull:
            plot_normalised_pull(metrics, group, axes[2], palette=palette, layout=layout,
                                 row_labels=False)
            axes[2].set(title="Whale Pull",
                        xlabel="Normalised Pull")
        for ax in axes:
            _style_axis(ax)
            ax.set_ylabel("Runs" if layout == "overlay" else "")
        _shared_legend(axes[0], axes, group, layout, axes[len(axes) // 2] if full_width
                       else axes[-1])
        panels = list(zip(axes, "ABC"))
        if bottom:
            panels += bottom(figure, slot)
        label_panels(panels, x=-0.16 if layout == "overlay" else -0.24, y=1.02)
        save(figure, name + suffix + name_suffix)


def whale_run(share, run):
    """One run of the whale capital-share experiment.

    Returns the smoothed error path, the normalised pull toward the whale's valuation, and the
    non-whale traders' returns. Positions are valued at their expected payout given the final
    true probability: wealth = cash + contracts x p_final.
    """
    seed = whale.BASE_SEED + run
    path = whale.make_election_path(seed)
    population = whale.make_population(seed, share)
    starting = np.array([agent.budget for agent in population])
    record = run_market_lob(
        whale.N_BETTORS, whale.SIMULATION_HORIZON, whale.INITIAL_PRICE,
        whale.OUTCOME_UNCERTAINTY, population,
        market_error=whale.MARKET_ERROR, tick=whale.TICK, ttl=whale.TTL,
        max_markup=whale.MAX_MARKUP, rng=np.random.default_rng(seed), gen_el=path,
        whale_share=share, whale_start=whale.PRE_PERIOD, whale_valuation=whale.WHALE_VALUATION,
    )
    fundamentals = path[:whale.SIMULATION_HORIZON]
    final_probability = path[-1]
    traders = [agent for agent in population if not agent.whale]
    wealth = np.array([agent.budget + agent.n_contracts * final_probability
                       for agent in traders])
    profit = wealth - starting
    returns = {
        "nonwhale_return": profit.sum() / starting.sum(),
        "median_trader_return": float(np.median(profit / starting)),
    }
    return (smoothed_absolute_error(record["midquote"], fundamentals),
            normalised_pull(record["midquote"], fundamentals, whale.WHALE_VALUATION,
                            whale.PRE_PERIOD),
            returns)


def run_whale_shares():
    """The whale experiment's design with WHALE_RUNS runs per ρ (its first 50 seeds included)."""
    tasks = [(share, run) for share in whale.WHALE_SHARES for run in range(WHALE_RUNS)]
    print(f"Whale experiment: {len(tasks)} runs on {N_WORKERS} processes")
    results = dict(zip(tasks, parallel_map(whale_run, *zip(*tasks))))
    band = calibrate_band(np.concatenate(
        [error[:whale.PRE_PERIOD] for (share, _), (error, *_) in results.items()
         if share == 0]))
    metrics = pd.DataFrame([
        {RHO: share, "run": run, "seed": whale.BASE_SEED + run,
         **distortion_metrics(error, whale.PRE_PERIOD, band), "normalised_pull": pull,
         **returns}
        for (share, run), (error, pull, returns) in results.items()
    ])
    metrics.to_csv(WHALE_CSV, index=False)
    return metrics


def valuation_run(seed, valuation):
    """One run of the original Figure 2C design; valuation None is the no-whale control.

    Returns the price - fundamental path. The last quote is carried over one-sided books, and
    the opening price fills any periods before the first quote.
    """
    path = make_election_path(seed)
    np.random.seed(seed)
    bettors = [bettor(budget=VALUATION_BUDGET, expertise=VALUATION_EXPERTISE)
               for _ in range(N_BETTORS)]
    if valuation is not None:
        capital = VALUATION_BUDGET * N_BETTORS
        bettors.append(bettor(
            budget=capital * VALUATION_WHALE_SHARE / (1 - VALUATION_WHALE_SHARE),
            market_valuation=valuation, risk_av=0.5, whale=True,
        ))
    record = run_market_lob(N_BETTORS, T_MAX, INITIAL_PRICE, RW_VARIANCE, bettors,
                            gen_el=path, rng=np.random.default_rng(seed))
    prices = pd.Series(record["midquote"]).ffill().fillna(INITIAL_PRICE).to_numpy()
    return prices - path[:T_MAX]


def run_valuations():
    """Signed error paths for each whale valuation and the no-whale control (matched seeds)."""
    groups = [None, *WHALE_VALUATIONS]
    tasks = [(VALUATION_SEED + run, valuation) for valuation in groups
             for run in range(VALUATION_RUNS)]
    print(f"Whale valuation experiment: {len(tasks)} runs on {N_WORKERS} processes")
    paths = parallel_map(valuation_run, *zip(*tasks))
    frame = pd.DataFrame(np.vstack(paths), columns=[f"t{t}" for t in range(T_MAX)])
    frame.insert(0, "valuation", [np.nan if v is None else v for _, v in tasks])
    frame.insert(1, "run", [seed - VALUATION_SEED for seed, _ in tasks])
    frame.to_csv(VALUATION_CSV, index=False)
    return frame


def valuation_summary(paths):
    """Band from the no-whale runs, and P1/P3 for each whale run, by valuation."""
    errors = paths.filter(regex=r"^t\d+$").to_numpy()
    control = paths["valuation"].isna().to_numpy()
    smoothed = np.vstack([smoothed_absolute_error(row, np.zeros(T_MAX)) for row in errors])
    band = calibrate_band(smoothed[control].ravel())
    metrics = pd.DataFrame([
        {"valuation": valuation, **distortion_metrics(error, 0, band)}
        for valuation, error in zip(paths["valuation"][~control], smoothed[~control])
    ])
    return band, metrics


def plot_valuation_paths(paths, ax):
    """Mean signed price error (price - true outcome) over time for each whale valuation, with
    95% confidence bands, the no-whale runs' mean as a reference, and the expected shift ρΔ.

    Signed, because ρΔ predicts the direction and size of the whale's push on the price. The
    absolute-error measures (maximum error, time above the ordinary-noise band) are in A-B.
    """
    times = np.arange(T_MAX)
    control = paths.loc[paths["valuation"].isna()].filter(regex=r"^t\d+$").to_numpy()
    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.plot(times, control.mean(axis=0), color="#777777", linewidth=1.2, label="No whale")
    for valuation, color in zip(WHALE_VALUATIONS, VALUATION_COLORS):
        errors = paths.loc[paths["valuation"] == valuation].filter(regex=r"^t\d+$").to_numpy()
        mean = errors.mean(axis=0)
        half_width = 1.96 * errors.std(axis=0, ddof=1) / np.sqrt(len(errors))
        expected = VALUATION_WHALE_SHARE * (valuation - INITIAL_PRICE)
        ax.plot(times, mean, color=color, linewidth=1.2, label=f"Δ = {valuation - INITIAL_PRICE:.1f}")
        ax.fill_between(times, mean - half_width, mean + half_width, color=color, alpha=0.2,
                        linewidth=0)
        ax.axhline(expected, color=color, linestyle="--", linewidth=1)
    ax.plot([], [], in_layout=False, color="black", linestyle="--", label="Expected shift ρΔ")
    ax.set(xlim=(0, T_MAX), xlabel="Time Since Shock",
           ylabel="Price − Outcome (mean)")
    ax.set_title(f"Price Error After a Whale Shock (ρ = {VALUATION_WHALE_SHARE:g})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncols=3, frameon=False,
              handlelength=1.3, columnspacing=0.8)
    _style_axis(ax)


def profit_population(seed, share):
    """The whale experiment's population, with expertise spread evenly from 0 to 1."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    population = [
        bettor(budget=rng.uniform(100, 1000) * (1 - share),
               market_valuation=np.clip(rng.normal(0.50, 0.05), 0.01, 0.99),
               expertise=expertise)
        for expertise in np.linspace(0, 1, whale.N_BETTORS)
    ]
    return population


def profit_run(share, run):
    """One run of the profit experiment, recording everything needed for later statistics.

    Returns (run summary, per-trader rows, price path, fundamental path). Positions are valued
    at their expected payout given the final true probability: wealth = cash + contracts x p.
    """
    seed = whale.BASE_SEED + run
    path = whale.make_election_path(seed)
    population = profit_population(seed, share)
    traders = [dict(expertise=1 - agent.expertise, budget0=agent.budget,
                    belief0=agent.market_valuation, stubbornness=agent.stubbornness,
                    risk_aversion=agent.risk_av) for agent in population]
    record = run_market_lob(
        whale.N_BETTORS, whale.SIMULATION_HORIZON, whale.INITIAL_PRICE,
        whale.OUTCOME_UNCERTAINTY, population, market_error=whale.MARKET_ERROR,
        tick=whale.TICK, ttl=whale.TTL, max_markup=whale.MAX_MARKUP,
        rng=np.random.default_rng(seed), gen_el=path, whale_share=share,
        whale_start=whale.PRE_PERIOD, whale_valuation=whale.WHALE_VALUATION,
    )
    final_p = float(path[-1])
    for index, (row, agent) in enumerate(zip(traders, population)):
        wealth = agent.budget + agent.n_contracts * final_p
        row.update(trader=index, final_cash=agent.budget,
                   final_contracts=agent.n_contracts, final_belief=agent.market_valuation,
                   profit=wealth - row["budget0"], trader_return=wealth / row["budget0"] - 1)
    whale_agent = population[-1] if share > 0 else None
    starting = sum(row["budget0"] for row in traders)
    summary = dict(
        final_probability=final_p,
        nonwhale_return=sum(row["profit"] for row in traders) / starting,
        whale_budget0=record["whale_initial_budget"],
        whale_final_cash=whale_agent.budget if whale_agent else np.nan,
        whale_final_contracts=whale_agent.n_contracts if whale_agent else np.nan,
        traded_volume=float(np.sum(record["traded_volume"])),
    )
    return summary, traders, np.asarray(record["midquote"], float), path


def run_profit_experiment():
    """PROFIT_RUNS runs per ρ in PROFIT_SHARES; saves runs, traders and paths to sim_output."""
    tasks = [(share, run) for share in PROFIT_SHARES for run in range(PROFIT_RUNS)]
    print(f"Profit experiment: {len(tasks)} runs on {N_WORKERS} processes")
    results = parallel_map(profit_run, *zip(*tasks))
    runs, traders, prices, fundamentals = [], [], [], []
    for (share, run), (summary, rows, price, path) in zip(tasks, results):
        runs.append({RHO: share, "run": run, "seed": whale.BASE_SEED + run, **summary})
        traders += [{RHO: share, "run": run, **row} for row in rows]
        prices.append(price)
        fundamentals.append(path)
    runs, traders = pd.DataFrame(runs), pd.DataFrame(traders)
    PROFIT_DIR.mkdir(exist_ok=True)
    runs.to_csv(PROFIT_DIR / "runs.csv", index=False)
    traders.to_csv(PROFIT_DIR / "traders.csv.gz", index=False)
    np.savez_compressed(PROFIT_DIR / "paths.npz", midquote=np.vstack(prices),
                        fundamental=np.vstack(fundamentals),
                        whale_share=runs[RHO].to_numpy(), run=runs["run"].to_numpy())
    return runs, traders


def load_profit_experiment(rerun):
    if (PROFIT_DIR / "traders.csv.gz").exists() and not rerun:
        return pd.read_csv(PROFIT_DIR / "runs.csv"), pd.read_csv(PROFIT_DIR / "traders.csv.gz")
    return run_profit_experiment()


def wealth_multiple(traders):
    """Final wealth as a multiple of starting wealth, floored at WEALTH_FLOOR for a log axis."""
    return np.maximum(traders["trader_return"] + 1, WEALTH_FLOOR)


def _wealth_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(WEALTH_FLOOR * 0.8, None)
    ax.axhline(1, color="#888888", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.set_ylabel("Wealth Multiple (log scale)")


def plot_trader_returns(traders, ax, column="expertise", groups=None, xlabel="Trader Expertise",
                        title="Trader Wealth by Expertise and Whale Share", shares=None):
    """Box plots of trader wealth multiples, grouped by bands of `column`, one box per ρ."""
    groups = EXPERTISE_GROUPS if groups is None else groups
    shares = list(PROFIT_SHOWN_SHARES if shares is None else shares)
    colors = profit_colors(len(shares))
    labels = [f"{low:g}–{high:g}" for low, high in zip(groups[:-1], groups[1:])]
    bands = pd.cut(traders[column], groups, include_lowest=True, labels=labels)
    width = 0.8 / len(shares)
    multiples = wealth_multiple(traders)
    for k, (share, color) in enumerate(zip(shares, colors)):
        in_share = np.isclose(traders[RHO], share)
        data = [multiples[in_share & (bands == label)] for label in labels]
        positions = np.arange(len(labels)) + (k - (len(shares) - 1) / 2) * width
        boxes = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True,
                           showfliers=False, medianprops=dict(color="#333333"))
        for box in boxes["boxes"]:
            box.set(facecolor=color, edgecolor="#333333")
        ax.plot([], [], in_layout=False, color=color, linewidth=6, label=f"{share:g}")
    ax.set_xticks(range(len(labels)), labels)
    ax.set(xlabel=xlabel, title=title)
    _wealth_axis(ax)
    ax.legend(title="ρ (Whale Capital Share)", loc="upper center", bbox_to_anchor=(0.5, -0.3),
              ncols=5, frameon=False, handlelength=1.2, columnspacing=1.0)
    _style_axis(ax)


def trader_wealth_by_risk_aversion(rerun=False):
    """Supplementary figure: panel D's layout with trader risk aversion on the x-axis."""
    _, traders = load_profit_experiment(rerun)
    figure, ax = plt.subplots(figsize=(style.PAGE_WIDTH, 3.0), constrained_layout=True)
    plot_trader_returns(traders, ax, column="risk_aversion", groups=RISK_AVERSION_GROUPS,
                        xlabel="Trader Risk Aversion",
                        title="Trader Wealth by Risk Aversion and Whale Share")
    save(figure, "trader_wealth_by_risk_aversion")


def plot_return_by_expertise(traders, ax, bins=10, sample_runs=10):
    """Trader wealth multiple against expertise for the shown ρ: points from a few runs, with
    the median in each expertise bin across all runs as a line."""
    shares = list(PROFIT_SHOWN_SHARES)
    edges = np.linspace(0, 1, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    multiples = wealth_multiple(traders)
    for share, color in zip(shares, profit_colors(len(shares))):
        in_share = np.isclose(traders[RHO], share)
        group, values = traders.loc[in_share], multiples[in_share]
        sample = (group["run"] < sample_runs).to_numpy()
        ax.scatter(group["expertise"][sample], values[sample], s=3, color=color, alpha=0.25,
                   linewidths=0)
        medians = values.groupby(pd.cut(group["expertise"], edges, include_lowest=True),
                                 observed=False).median()
        ax.plot(centres, medians.to_numpy(), color=color, linewidth=2, label=f"{share:g}")
    ax.set(xlabel="Trader Expertise", title="Trader Wealth by Expertise", xlim=(0, 1))
    _wealth_axis(ax)
    ax.legend(title="ρ (Whale Capital Share)", loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncols=len(shares), frameon=False)
    _style_axis(ax)


def profit_ramp():
    """Sequential colour ramp for ordered ρ levels, between two project-palette colours."""
    return LinearSegmentedColormap.from_list("rho", [PALETTE["lavender"], PALETTE["navy"]])


def profit_colors(n):
    return [profit_ramp()(i / max(n - 1, 1)) for i in range(n)]


def rho_colorbar(figure_or_subfigure, ax, shares, label="ρ (Whale Capital Share)"):
    """A continuous ρ colour bar under `ax`, in place of a legend of discrete swatches."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    mappable = ScalarMappable(norm=Normalize(min(shares), max(shares)), cmap=profit_ramp())
    bar = figure_or_subfigure.colorbar(mappable, ax=ax, location="bottom", fraction=0.05,
                                       pad=0.02, aspect=40)
    bar.set_label(label)
    bar.set_ticks(shares[::3] if len(shares) > 5 else shares)
    return bar


def figure2(rerun):
    """Full page width, 2 x 2: A maximum error | B time above band; C pull toward the whale |
    D trader wealth by expertise and ρ (all 10 whale shares). Panel E (price error after a
    whale shock) is saved separately as figure2e."""
    cached = WHALE_CSV.exists() and not rerun
    metrics = pd.read_csv(WHALE_CSV) if cached else run_whale_shares()
    if "nonwhale_return" not in metrics:
        metrics = run_whale_shares()
    runs, traders = load_profit_experiment(rerun)
    paths = (pd.read_csv(VALUATION_CSV) if VALUATION_CSV.exists() and not rerun
             else run_valuations())

    figure = plt.figure(figsize=(style.PAGE_WIDTH, 6.6), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[1, 1.1])
    ax_a, ax_b = top.subplots(1, 2)
    ax_c, ax_d = bottom.subplots(1, 2)
    for ax, plot in ((ax_a, plot_max_error), (ax_b, plot_time_outside_band),
                     (ax_c, plot_normalised_pull)):
        plot(metrics, RHO, ax, palette=WHALE_SHARE_COLORS)
        _style_axis(ax)
        ax.set_ylabel("Runs")
    ax_a.set(title="Maximum Market Error", xlabel="Max |Price − Outcome|")
    ax_b.set(title="Time Above Ordinary Noise", xlabel="Share of Periods Above Band")
    ax_c.set(title="Pull Toward the Whale", xlabel="Normalised Pull (0 = outcome, 1 = whale)")

    # A-C share colours: one legend, under the top row.
    entries = {}
    for handle, label in zip(*ax_a.get_legend_handles_labels()):
        name = label.split(" (median")[0]
        entries.setdefault("Ordinary noise band (A)" if name == "Ordinary band" else name,
                           handle)
    for ax in (ax_a, ax_b, ax_c):
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    top.legend(list(entries.values()), list(entries), title="ρ (Whale Capital Share), A–C",
               loc="outside lower center", ncols=len(entries), frameon=False)

    plot_trader_returns(traders, ax_d, title="Trader Wealth by Expertise and Whale Share")
    ax_d.set_ylabel("Wealth Multiple (log scale)")
    rho_colorbar(bottom, ax_d, list(PROFIT_SHOWN_SHARES))
    style.label_panels(((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")))
    save(figure, "figure2")

    figure, ax_e = plt.subplots(figsize=(style.PAGE_WIDTH, 3.4), layout="constrained")
    plot_valuation_paths(paths, ax_e)
    style.label_panels(((ax_e, "E"),))
    save(figure, "figure2e")


def herding_run(seed, h, with_whale, theory=False):
    """One herding run: the smoothed |price - fundamental| path and, with a whale, the pull
    toward its valuation (NaN without one).

    theory=True uses the CLOB theory's assumptions: whale valuation 0.7 and log utility (r = 1)
    for everyone, the whale included. Otherwise the original design: whale at 1, agents'
    risk aversion drawn from U(0, 1) and the whale's fixed at 0.5.
    """
    valuation = THEORY_WHALE_VALUATION if theory else HERDING_WHALE_VALUATION
    extra = {"risk_av": 1.0} if theory else {}
    path = make_election_path(seed)
    np.random.seed(seed)
    bettors = [bettor(budget=HERDING_BUDGET, expertise=HERDING_EXPERTISE, market_imitation=h,
                      **extra) for _ in range(N_BETTORS)]
    if with_whale:
        capital = HERDING_BUDGET * N_BETTORS
        bettors.append(bettor(
            budget=capital * HERDING_WHALE_SHARE / (1 - HERDING_WHALE_SHARE),
            market_valuation=valuation, risk_av=1.0 if theory else 0.5, whale=True,
        ))
    record = run_market_lob(N_BETTORS, T_MAX, INITIAL_PRICE, RW_VARIANCE, bettors,
                            gen_el=path, rng=np.random.default_rng(seed))
    pull = (normalised_pull(record["midquote"], path[:T_MAX], valuation, 0)
            if with_whale else np.nan)
    return smoothed_absolute_error(record["midquote"], path[:T_MAX]), pull


def run_herding(theory=False):
    """Herding experiment; theory=True saves to its own file (HERDING_THEORY_CSV)."""
    tasks = [(10_000 * index + run, h, with_whale, theory)
             for index, h in enumerate(HERDING_STRENGTHS)
             for with_whale in (False, True)
             for run in range(HERDING_RUNS)]
    print(f"Herding experiment{' (theory setup)' if theory else ''}: "
          f"{len(tasks)} runs on {N_WORKERS} processes")
    results = {task[:3]: result
               for task, result in zip(tasks, parallel_map(herding_run, *zip(*tasks)))}
    rows = []
    for index, h in enumerate(HERDING_STRENGTHS):
        seeds = [10_000 * index + run for run in range(HERDING_RUNS)]
        band = calibrate_band(np.concatenate([results[seed, h, False][0] for seed in seeds]))
        for run, seed in enumerate(seeds):
            error, pull = results[seed, h, True]
            rows.append({HERDING: h, "run": run, "seed": seed,
                         **distortion_metrics(error, 0, band), "normalised_pull": pull})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(HERDING_THEORY_CSV if theory else HERDING_CSV, index=False)
    return metrics


def stability_regions(alpha, h_sum, s_sum):
    """Spectral radius and region (0 stable monotone, 1 stable oscillatory, 2 unstable) of the
    two-root linearised error dynamics, as in cell 46 of Dec25Plots.ipynb."""
    trace = (1.0 - alpha) + s_sum
    determinant = (1.0 - alpha) * s_sum - alpha * h_sum
    root = np.sqrt((trace ** 2 - 4.0 * determinant).astype(np.complex128))
    r1, r2 = 0.5 * (trace + root), 0.5 * (trace - root)
    radius = np.maximum(np.abs(r1), np.abs(r2)).real
    region = np.zeros_like(radius, dtype=int)
    oscillatory = (np.abs(r1.imag) > 1e-9) | (np.abs(r2.imag) > 1e-9)
    unstable = radius >= 1.0 - 1e-12
    region[oscillatory & ~unstable] = 1
    region[unstable] = 2
    return radius, region


def plot_stability(ax, alpha, resolution=301):
    """One phase diagram: stable (error decays) vs unstable, with the h = 1 ABM setup marked."""
    grid = np.linspace(0, 1, resolution)
    h_sum, s_sum = np.meshgrid(grid, grid, indexing="xy")
    radius, region = stability_regions(alpha, h_sum, s_sum)
    ax.contourf(grid, grid, (region == 0).astype(int), levels=[0.5, 1.5],
                colors=[PALETTE["teal"]])
    ax.contourf(grid, grid, (region >= 1).astype(int), levels=[0.5, 1.5],
                colors=[PALETTE["orange"]])
    ax.contour(grid, grid, radius, levels=[1.0], colors="white", linewidths=1.2,
               linestyles="--")
    for code, text in ((0, "stable"), (2, "unstable")):
        rows, cols = np.where(region == code)
        if rows.size:
            ax.text(np.clip(grid[int(np.median(cols))], 0.25, 0.75),
                    np.clip(grid[int(np.median(rows))], 0.3, 0.85), text,
                    ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))
    ax.scatter(*HERDING_ABM_POINT, s=110, color=HERDING_COLORS[-1], edgecolor="black",
               zorder=10, clip_on=False)
    ax.text(HERDING_ABM_POINT[0] + 0.07, HERDING_ABM_POINT[1] + 0.02, r"$h_i = 1$", va="bottom", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r"$\sum_i w_i h_i$",
           ylabel=r"$\sum_i w_i (1 - h_i) s_i$")
    ax.set_title(rf"$\alpha = {alpha:.2f}$")


def theory_kappa(h, s, rho):
    """CLOB convergence factor under fast clearing: errors shrink by κ each period."""
    return (1 - h) * s + h * (1 - rho)


def theory_pull(h, rho, s):
    """Steady-state pull toward the whale under fast clearing (Eq. 13)."""
    return rho * ((1 - h) * (1 - s) + h) / ((1 - h) * (1 - s) + h * rho)


THEORY_SETUPS = {
    # name: (whale valuation, everyone log utility)
    "Figure 3 setup (whale valuation 1, mixed risk aversion)": (HERDING_WHALE_VALUATION, False),
    "Theory's assumptions (whale valuation 0.7, log utility)": (0.7, True),
}
THEORY_LEGEND = {
    "Figure 3 setup (whale valuation 1, mixed risk aversion)":
        "Sim, as in A–C\n(whale at 1, mixed risk aversion)",
    "Theory's assumptions (whale valuation 0.7, log utility)":
        "Sim, theory's assumptions\n(whale at 0.7, log utility)",
}
THEORY_CSV = SIM_DIR / "figure3_theory_pull.csv"
THEORY_WHALE_VALUATION = 0.7
HERDING_THEORY_CSV = SIM_DIR / "figure3_runs_theory.csv"   # separate from HERDING_CSV
THEORY_RUNS = 100


def theory_pull_run(setup, h, run):
    """One herding run with a ρ = 0.3 whale; returns its normalised pull."""
    valuation, log_utility = THEORY_SETUPS[setup]
    seed = 30_000 + run
    path = make_election_path(seed)
    np.random.seed(seed)
    extra = {"risk_av": 1.0} if log_utility else {}
    bettors = [bettor(budget=HERDING_BUDGET, expertise=HERDING_EXPERTISE, market_imitation=h,
                      **extra) for _ in range(N_BETTORS)]
    capital = HERDING_BUDGET * N_BETTORS
    bettors.append(bettor(budget=capital * HERDING_WHALE_SHARE / (1 - HERDING_WHALE_SHARE),
                          market_valuation=valuation, risk_av=1.0 if log_utility else 0.5,
                          whale=True))
    record = run_market_lob(N_BETTORS, T_MAX, INITIAL_PRICE, RW_VARIANCE, bettors,
                            gen_el=path, rng=np.random.default_rng(seed))
    return normalised_pull(record["midquote"], path[:T_MAX], valuation, 0)


def run_theory_pull():
    tasks = [(setup, h, run) for setup in THEORY_SETUPS for h in HERDING_STRENGTHS
             for run in range(THEORY_RUNS)]
    print(f"Theory comparison: {len(tasks)} runs on {N_WORKERS} processes")
    pulls = parallel_map(theory_pull_run, *zip(*tasks))
    frame = pd.DataFrame([{"setup": setup, "h": h, "run": run, "pull": pull}
                          for (setup, h, run), pull in zip(tasks, pulls)])
    frame.to_csv(THEORY_CSV, index=False)
    return frame


def plot_kappa_map(ax, figure, rho=HERDING_WHALE_SHARE, s_runs=0.3):
    """Panel D: convergence factor κ = (1-h)s + h(1-ρ) over herding and stubbornness."""
    grid = np.linspace(0, 1, 301)
    h, s = np.meshgrid(grid, grid, indexing="xy")
    ramp = LinearSegmentedColormap.from_list("kappa", ["#FFFFFF", PALETTE["teal"],
                                                       PALETTE["indigo"]])
    image = ax.imshow(theory_kappa(h, s, rho), origin="lower", extent=[0, 1, 0, 1],
                      aspect="auto", cmap=ramp, vmin=0, vmax=1)
    lines = ax.contour(grid, grid, theory_kappa(h, s, rho), levels=[0.25, 0.5, 0.75, 0.9],
                       colors="white", linewidths=0.6)
    ax.clabel(lines, fmt="%.2f", fontsize=style.SMALL)
    ax.axhline(1 - rho, color="black", linestyle="--", linewidth=0.8)
    ax.text(0.98, 1 - rho + 0.02, "s = 1 − ρ", ha="right", va="bottom")
    for h_value, color in zip(HERDING_STRENGTHS, HERDING_COLORS):
        ax.plot(h_value, s_runs, "o", color=color, markeredgecolor="black", markersize=6,
                clip_on=False, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Herding Strength h", ylabel="Stubbornness s",
           title=f"Convergence Factor κ (ρ = {rho:g})")
    bar = figure.colorbar(image, ax=ax, location="right", fraction=0.05, pad=0.02)
    bar.set_label("κ (error kept each period)")


def plot_pull_theory(ax, sims, rho=HERDING_WHALE_SHARE, s=0.3, setups=None):
    """Panel E: Eq. 13's predicted pull against simulated medians (bars: middle 50%)."""
    hs = np.linspace(0, 1, 200)
    ax.plot(hs, theory_pull(hs, rho, s), color="black", linewidth=1.2,
            label=f"Theory (ρ = {rho:g}, s = {s:g})")
    markers = {0: ("o", "white"), 1: ("o", None)}
    if setups is not None:
        sims = sims[sims["setup"].isin(setups)]
    for k, (setup, group) in enumerate(sims.groupby("setup", sort=False)):
        k = list(THEORY_SETUPS).index(setup)
        stats = group.groupby("h").pull.quantile([0.25, 0.5, 0.75]).unstack()
        offset = (k - 0.5) * 0.02
        for h_value, color in zip(HERDING_STRENGTHS, HERDING_COLORS):
            q25, median, q75 = stats.loc[h_value]
            face = "white" if k == 0 else color
            ax.errorbar(h_value + offset, median, yerr=[[median - q25], [q75 - median]],
                        fmt="o", color=color, markerfacecolor=face, markersize=5,
                        elinewidth=0.8, capsize=0)
        ax.plot([], [], "o", color="#555555", markerfacecolor="white" if k == 0 else "#555555",
                in_layout=False, label=THEORY_LEGEND[setup])
    ax.set(xlim=(-0.05, 1.05), ylim=(0, 1.1), xlabel="Herding Strength h",
           ylabel="Normalised Pull", title="Pull: Theory vs Simulation")
    ax.legend(loc="upper left", frameon=False)
    _style_axis(ax)


THEORY_CHECKS_DIR = OUTPUT_DIR / "theory_checks"
THEORY_DELTA = 0.2            # whale valuation 0.7 minus true outcome 0.5 in those checks
THEORY_STUBBORNNESS = 0.3


def load_theory_checks():
    """The co-author's CLOB theory-check simulations (theory_checks/figsec.py, section AB):
    log utility for all, whale valuation 0.7, true outcome fixed at 0.5, 100 agents."""
    with open(THEORY_CHECKS_DIR / "figdata_AB.json") as handle:
        data = json.load(handle)
    a = pd.DataFrame(data["A"], columns=["rho", "noise", "error"])
    b = pd.DataFrame(data["B"], columns=["rho", "h", "pass_through"])
    return a, b


def plot_theory_whale_share(ax, a):
    """Steady-state price error against whale share: theory ρΔ against simulation."""
    rho = np.linspace(0, 0.65, 100)
    ax.plot(rho, rho * THEORY_DELTA, color="black", linewidth=1.2, label="Theory: ρΔ")
    exact, noisy = a[a["noise"] == 0], a[a["noise"] > 0]
    ax.plot(exact["rho"], exact["error"], "o", color=PALETTE["navy"], markersize=5,
            label="Simulated, exact signals")
    ax.plot(noisy["rho"], noisy["error"], "s", color=PALETTE["navy"], markerfacecolor="white",
            markersize=5, label="Simulated, noisy signals")
    ax.set(xlim=(0, 0.65), ylim=(0, None), xlabel="Whale's Share of Capital (ρ)",
           ylabel="Steady State Price Error",
           title=f"Whale Distortion Without Herding (Δ = {THEORY_DELTA:g})")
    ax.legend(loc="upper left", frameon=False)
    _style_axis(ax)


def plot_theory_herding(ax, b):
    """Share of the whale's misvaluation passed into the price, against herding strength."""
    h = np.linspace(0, 1, 200)
    for rho, color in zip(sorted(b["rho"].unique()), (PALETTE["navy"], PALETTE["orange"])):
        ax.plot(h, theory_pull(h, rho, THEORY_STUBBORNNESS), color=color, linewidth=1.2)
        points = b[b["rho"] == rho]
        ax.plot(points["h"], points["pass_through"], "o", color=color, markersize=5,
                label=f"ρ = {rho:g}")
    ax.plot([], [], color="#555555", linewidth=1.2, in_layout=False, label="Theory")
    ax.plot([], [], "o", color="#555555", markersize=5, in_layout=False, label="Simulated")
    ax.set(xlim=(-0.03, 1.03), ylim=(0, 1.08), xlabel="Herding Strength (h)",
           ylabel="Pass-Through of Whale Error",
           title=f"Herding Amplifies the Whale (s = {THEORY_STUBBORNNESS:g})")
    ax.legend(loc="upper left", frameon=False)
    _style_axis(ax)


def figure3(rerun, theory=False):
    """Ridgeline A-C (one row per herding level) above the CLOB theory panels D-E.

    theory=False: the original design (whale at 1, mixed risk aversion), saved as figure3b.
    theory=True: the theory's assumptions (whale at 0.7, log utility), from its own runs, saved
    as figure3b_theory.
    """
    csv = HERDING_THEORY_CSV if theory else HERDING_CSV
    metrics = pd.read_csv(csv) if csv.exists() and not rerun else run_herding(theory)
    if "normalised_pull" not in metrics:
        metrics = run_herding(theory)  # cached runs predate the pull
    sims = (pd.read_csv(THEORY_CSV) if THEORY_CSV.exists() and not rerun
            else run_theory_pull())
    shown = [list(THEORY_SETUPS)[1]] if theory else None

    checks_a, checks_b = load_theory_checks()

    def bottom(figure, slot):
        # D-E: the co-author's CLOB theory checks (theory_checks/), restyled.
        ax_d, ax_e = slot.subplots(1, 2)
        plot_theory_whale_share(ax_d, checks_a)
        plot_theory_herding(ax_e, checks_b)
        return [(ax_d, "D"), (ax_e, "E")]

    distribution_figures(metrics, HERDING, HERDING_COLORS, "figure3", pull=True,
                         name_suffix="_theory" if theory else "",
                         bottom=bottom, bottom_height=3.0, full_width=True,
                         layouts=("ridge",), top_height=2.9)


def main():
    rerun = "--rerun" in sys.argv
    figure1(rerun)
    # Alternative right-hand columns for Figure 1B (same sweep; drawn from the cache).
    figure1(False, right_column="delay", name="figure1_response_delay_clob")
    figure1(False, right_column="lag", name="figure1_best_lag_clob")
    figure2(rerun)
    figure3(rerun)
    figure3(rerun, theory=True)


if __name__ == "__main__":
    main()
