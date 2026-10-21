"""Paper figures 1-3, rerun with the CLOB market (run_market_lob) in place of the original market.

Figure 1
    A  Market price, election outcome and average valuation in one example run
    B  Parameter sweeps: mean squared error and peak price-fundamental cross-correlation
    C  Contract volume and net supply in the same example run
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

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from bettor import bettor, run_market_lob
from distortion import (
    calibrate_band,
    distortion_metrics,
    plot_max_error,
    plot_time_outside_band,
    smoothed_absolute_error,
)
import whale_convergence_experiment as whale
from plot_whale_distortion import RHO
from whale_convergence_experiment import make_election_path


N_BETTORS = 100
T_MAX = 100
INITIAL_PRICE = 0.5
RW_VARIANCE = 0.05
N_ITER = 30
MAX_LAG = 5
DELAY_MAX_LAG = 10
MIN_RESPONSE_LAG = 2  # price responds to news no sooner than 2 periods later, by timing
PRESSURE_SCALE = 15
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
N_WORKERS = 4  # processes for the simulations; kept low to leave the laptop usable
HERDING_WHALE_SHARE = 0.3
HERDING_WHALE_VALUATION = 1.0
HERDING_EXPERTISE = 0.9
HERDING_BUDGET = 100
HERDING = "Herding strength (h)"

# The power_prediction palette (cell 3 of the original notebooks), with the orders the
# published figures use.
PALETTE = {
    "lavender": "#CC99FF", "navy": "#000080", "green": "#339966", "orange": "#F4A261",
    "ruby": "#B22222", "violet": "#6A4C93", "indigo": "#1D3557", "teal": "#2A9D8F",
    "steel": "#457B9D", "amber": "#FFB347", "crimson": "#E63946",
}
WHALE_SHARE_COLORS = [PALETTE[name] for name in ("indigo", "steel", "green", "orange")]
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


def example_run():
    """One run with the original example population: 100 bettors and a whale valuing YES at 1."""
    np.random.seed(0)
    bettors = ([bettor(budget=100) for _ in range(N_BETTORS)]
               + [bettor(budget=200, market_valuation=1, whale=True)])
    return run_market_lob(N_BETTORS, T_MAX, INITIAL_PRICE, RW_VARIANCE, bettors,
                          rng=np.random.default_rng(0))


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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=3, fontsize=8, frameon=False)


def plot_volume(record, ax_volume, ax_supply):
    """Panel C, as plot_returns(step=True) in the original figure, split into two stacked plots.

    Contracts traded each period (the first difference of cumulative volume) replaces
    vol_history (total open positions). Net supply gets its own axis so neither series
    flattens the other.
    """
    ax_volume.plot(record["traded_volume"], color="#1D3557", drawstyle="steps")
    ax_volume.set(ylabel="Traded", title="Contract Volume & Net Supply", ylim=(0, None))
    ax_volume.tick_params(axis="x", labelbottom=False)
    ax_supply.plot(record["net_supply"], color="#339966", linestyle="--", drawstyle="steps")
    ax_supply.axhline(0, color="#BBBBBB", linewidth=0.6)
    ax_supply.set(xlabel="Time", ylabel="Net supply")
    ax_supply.tick_params(axis="x", labelrotation=45)
    thousands = FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0")
    for ax in (ax_volume, ax_supply):
        ax.yaxis.set_major_formatter(thousands)
        ax.yaxis.label.set_fontsize(8)
        ax.yaxis.set_label_coords(-0.12, 0.5)


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
    figure.colorbar(image, ax=ax).set_label("Misclassification probability")


def _style_axis(ax):
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_panels(axes_and_letters, x=-0.12, y=1.12):
    for ax, letter in axes_and_letters:
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="bottom")


def save(figure, name):
    figure.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def figure1(rerun, right_column="cross_correlation", name="figure1"):
    """right_column: "cross_correlation" (peak value), "lag" (the lag at which it peaks) or
    "delay" (average response delay)."""
    sweep = pd.read_csv(SWEEP_CSV) if SWEEP_CSV.exists() and not rerun else run_sweep()
    record = example_run()

    figure = plt.figure(figsize=(13, 10.5))
    outer = GridSpec(1, 2, figure=figure, width_ratios=[1.05, 1], wspace=0.28)
    left = outer[0].subgridspec(3, 1, height_ratios=[1, 0.8, 1.7], hspace=0.85)
    right = outer[1].subgridspec(len(SWEEPS), 2, hspace=0.55, wspace=0.45)

    ax_a = figure.add_subplot(left[0])
    volume = left[1].subgridspec(2, 1, height_ratios=[1.4, 1], hspace=0.12)
    ax_c = figure.add_subplot(volume[0])
    ax_supply = figure.add_subplot(volume[1], sharex=ax_c)
    ax_d = figure.add_subplot(left[2])
    plot_market_price(record, ax_a)
    plot_volume(record, ax_c, ax_supply)
    plot_misclassification(ax_d, figure)

    sweep_axes = []
    for row, parameter in enumerate(SWEEPS):
        ax_mse = figure.add_subplot(right[row, 0])
        ax_right = figure.add_subplot(right[row, 1])
        plot_sweep_column(sweep, parameter, "mse", "Mean Squared Error", ax_mse)
        if right_column == "lag":
            plot_lag_column(sweep, parameter, ax_right)
        elif right_column == "delay":
            plot_sweep_column(sweep, parameter, "response_delay",
                              "Response delay\n(periods)", ax_right)
            ax_right.set_ylim(bottom=-0.2)
        else:
            plot_sweep_column(sweep, parameter, "peak_cross_correlation",
                              "Peak cross-correlation", ax_right)
            ax_right.set_ylim(top=1)
        sweep_axes.append(ax_mse)

    label_panels(((ax_a, "A"), (sweep_axes[0], "B"), (ax_c, "C"), (ax_d, "D")))
    save(figure, name)


def parallel_map(function, *iterables):
    """function over the zipped iterables on N_WORKERS processes, results in input order."""
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        return list(pool.map(function, *iterables, chunksize=8))


def distribution_figures(metrics, group, palette, name, p3_title=None, bottom=None):
    """P1 and P3 side by side: overlaid histograms as `name`, one row per group as `name`b.

    bottom, if given, draws a panel C spanning the width below them: bottom(ax).
    """
    p3_kwargs = {"title": p3_title} if p3_title else {}
    for layout, suffix in (("overlay", ""), ("ridge", "b")):
        top_height = 4.6 if layout == "overlay" else 1.25 * metrics[group].nunique() + 1.4
        bottom_height = 4.6 if bottom else 0
        figure = plt.figure(figsize=(12, top_height + bottom_height), constrained_layout=True)
        grid = figure.add_gridspec(2 if bottom else 1, 2,
                                   height_ratios=[top_height, bottom_height] if bottom else None)
        ax_a, ax_b = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
        plot_max_error(metrics, group, ax_a, title="Maximum market error", palette=palette,
                       layout=layout)
        plot_time_outside_band(metrics, group, ax_b, palette=palette, layout=layout,
                               **p3_kwargs)
        ax_b.set_title(ax_b.get_title().replace("P3: ", ""), fontsize=11)
        x = -0.1 if layout == "overlay" else -0.18
        label_panels(((ax_a, "A"), (ax_b, "B")), x=x, y=1.04)
        if bottom:
            ax_c = figure.add_subplot(grid[1, :])
            bottom(ax_c)
            label_panels(((ax_c, "C"),), x=-0.05, y=1.04)
        save(figure, name + suffix)


def whale_run(share, run):
    """One run of the whale capital-share experiment; returns the smoothed error path."""
    seed = whale.BASE_SEED + run
    path = whale.make_election_path(seed)
    record = run_market_lob(
        whale.N_BETTORS, whale.SIMULATION_HORIZON, whale.INITIAL_PRICE,
        whale.OUTCOME_UNCERTAINTY, whale.make_population(seed, share),
        market_error=whale.MARKET_ERROR, tick=whale.TICK, ttl=whale.TTL,
        max_markup=whale.MAX_MARKUP, rng=np.random.default_rng(seed), gen_el=path,
        whale_share=share, whale_start=whale.PRE_PERIOD, whale_valuation=whale.WHALE_VALUATION,
    )
    return smoothed_absolute_error(record["midquote"], path[:whale.SIMULATION_HORIZON])


def run_whale_shares():
    """The whale experiment's design with WHALE_RUNS runs per ρ (its first 50 seeds included)."""
    tasks = [(share, run) for share in whale.WHALE_SHARES for run in range(WHALE_RUNS)]
    print(f"Whale experiment: {len(tasks)} runs on {N_WORKERS} processes")
    errors = dict(zip(tasks, parallel_map(whale_run, *zip(*tasks))))
    band = calibrate_band(np.concatenate(
        [error[:whale.PRE_PERIOD] for (share, _), error in errors.items() if share == 0]))
    metrics = pd.DataFrame([
        {RHO: share, "run": run, "seed": whale.BASE_SEED + run,
         **distortion_metrics(error, whale.PRE_PERIOD, band)}
        for (share, run), error in errors.items()
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
    """Panel C: mean price - fundamental over time for each whale valuation (95% bands), the
    no-whale ordinary band around 0, the expected ρΔ, P1 medians and P3 in the legend."""
    band, metrics = valuation_summary(paths)
    times = np.arange(T_MAX)
    ax.axhspan(-band, band, color="#E5E5E5", linewidth=0,
               label=f"Ordinary noise (no-whale band, ±{band:.2f})")
    for valuation, color in zip(WHALE_VALUATIONS, VALUATION_COLORS):
        errors = paths.loc[paths["valuation"] == valuation].filter(regex=r"^t\d+$").to_numpy()
        mean = errors.mean(axis=0)
        half_width = 1.96 * errors.std(axis=0, ddof=1) / np.sqrt(len(errors))
        group = metrics.loc[metrics["valuation"] == valuation]
        expected = VALUATION_WHALE_SHARE * (valuation - INITIAL_PRICE)
        ax.plot(times, mean, color=color, linewidth=1.5,
                label=(f"ρΔ = {expected:.2f}: median max error "
                       f"{group['max_error'].median():.2f}, "
                       f"{group['share_outside_band'].median():.0%} of time above band"))
        ax.fill_between(times, mean - half_width, mean + half_width, color=color, alpha=0.2,
                        linewidth=0)
        ax.axhline(expected, color="black", linestyle="--", linewidth=1, alpha=0.8)
        ax.plot(T_MAX + 2.5, group["max_error"].median(), marker="<", color=color,
                markersize=9)
    ax.plot([], [], color="black", linestyle="--", label="Expected value (ρΔ)")
    ax.plot([], [], marker="<", color="#555555", linestyle="none",
            label="Median maximum error (right edge)")
    ax.set(xlim=(0, T_MAX + 4), xlabel="Time Since Shock",
           ylabel="Market Price − True Election Outcome")
    ax.set_title(f"Whale holding ρ = {VALUATION_WHALE_SHARE:g} of capital, "
                 "misvaluing by Δ = valuation − 0.5", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, fontsize=8,
              frameon=False)
    _style_axis(ax)


def figure2(rerun):
    metrics = (pd.read_csv(WHALE_CSV) if WHALE_CSV.exists() and not rerun
               else run_whale_shares())
    paths = (pd.read_csv(VALUATION_CSV) if VALUATION_CSV.exists() and not rerun
             else run_valuations())
    distribution_figures(metrics, RHO, WHALE_SHARE_COLORS, "figure2",
                         bottom=lambda ax: plot_valuation_paths(paths, ax))


def herding_run(seed, h, with_whale):
    """One herding run; returns the smoothed |price - fundamental| path."""
    path = make_election_path(seed)
    np.random.seed(seed)
    bettors = [bettor(budget=HERDING_BUDGET, expertise=HERDING_EXPERTISE, market_imitation=h)
               for _ in range(N_BETTORS)]
    if with_whale:
        capital = HERDING_BUDGET * N_BETTORS
        bettors.append(bettor(
            budget=capital * HERDING_WHALE_SHARE / (1 - HERDING_WHALE_SHARE),
            market_valuation=HERDING_WHALE_VALUATION, risk_av=0.5, whale=True,
        ))
    record = run_market_lob(N_BETTORS, T_MAX, INITIAL_PRICE, RW_VARIANCE, bettors,
                            gen_el=path, rng=np.random.default_rng(seed))
    return smoothed_absolute_error(record["midquote"], path[:T_MAX])


def run_herding():
    tasks = [(10_000 * index + run, h, with_whale)
             for index, h in enumerate(HERDING_STRENGTHS)
             for with_whale in (False, True)
             for run in range(HERDING_RUNS)]
    print(f"Herding experiment: {len(tasks)} runs on {N_WORKERS} processes")
    errors = dict(zip(tasks, parallel_map(herding_run, *zip(*tasks))))
    rows = []
    for index, h in enumerate(HERDING_STRENGTHS):
        seeds = [10_000 * index + run for run in range(HERDING_RUNS)]
        band = calibrate_band(np.concatenate([errors[seed, h, False] for seed in seeds]))
        for run, seed in enumerate(seeds):
            rows.append({HERDING: h, "run": run, "seed": seed,
                         **distortion_metrics(errors[seed, h, True], 0, band)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(HERDING_CSV, index=False)
    return metrics


def figure3(rerun):
    metrics = (pd.read_csv(HERDING_CSV) if HERDING_CSV.exists() and not rerun
               else run_herding())
    distribution_figures(
        metrics, HERDING, HERDING_COLORS, "figure3",
        p3_title=("Time with pricing error above ordinary noise\n"
                  "Band: 95% of each herding level's no-whale errors fall below it"),
    )


def main():
    rerun = "--rerun" in sys.argv
    figure1(rerun)
    figure2(rerun)
    figure3(rerun)


if __name__ == "__main__":
    main()
