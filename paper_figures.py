"""Simulations and plotting functions for the manuscript figures, which are assembled and saved
in manuscript_figures.ipynb. The market is the CLOB (run_market_lob).

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

Figure 3 (herding), one row per herding strength h (Bettor.market_imitation)
    A  P1, B  P3 and C  P1b: a whale with ρ = 0.3 and valuation 1 trades from period 0
       against 100 bettors with expertise 0.9 and herding h, 500 runs per h. Each h is judged
       against the band of its own no-whale runs. Saved as figure3b, and as figure3b_theory
       with the CLOB theory's assumptions (whale at 0.7, log utility for everyone).
    D-E  The co-author's CLOB theory checks (theory_checks/figdata_AB.json).

Supplementary: trader_wealth_by_risk_aversion (Figure 2D's layout by risk aversion).
The simulations run on N_WORKERS = 4 processes.

Simulations are cached in sim_output/ (figure1_sweep.csv ~4 min; figure2_runs.csv and
figure3_runs.csv ~10 min each on 4 processes): the load_* functions read the cache, or rerun
the simulations when asked to or when the cache is missing. Figures go to figures/.
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, PercentFormatter

from bettor import Bettor, add_whale, fundamental_path, run_market_lob
from metrics import (
    calibrate_band,
    distortion_metrics,
    normalised_pull,
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
# Figures 2E and 3 use the original design: N_BETTORS bettors with equal budgets and
# expertise, plus a whale (Dec25Plots cells 27-31).
DESIGN_BUDGET = 100
DESIGN_EXPERTISE = 0.9
# Figure 2E: whale valuations at a fixed whale share.
WHALE_VALUATIONS = (0.6, 0.7, 0.8)
VALUATION_WHALE_SHARE = 0.5
VALUATION_RUNS = 500
VALUATION_SEED = 20_000
VALUATION_CSV = SIM_DIR / "figure2c_paths.csv"
# Figure 2 D-E: profit experiment (expertise spread 0-1, ρ = 0, 0.1, ..., 0.9).
PROFIT_SHARES = tuple(round(0.1 * k, 1) for k in range(10))
PROFIT_RUNS = 500
PROFIT_DIR = SIM_DIR / "profit_experiment"
PROFIT_SHOWN_SHARES = PROFIT_SHARES            # ρ values in the full-width wealth figures
EXPERTISE_GROUPS = (0.0, 0.5, 0.8, 0.9, 1.0)   # panel D's expertise groups
RISK_AVERSION_GROUPS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEALTH_FLOOR = 0.01                            # log-axis floor: a 99% loss
N_WORKERS = 4  # processes for the simulations; kept low to leave the laptop usable
HERDING_WHALE_SHARE = 0.3
HERDING_WHALE_VALUATION = 1.0
HERDING = "Herding strength (h)"
HERDING_THEORY_CSV = SIM_DIR / "figure3_runs_theory.csv"   # theory's assumptions
THEORY_WHALE_VALUATION = 0.7
THEORY_CHECKS_DIR = OUTPUT_DIR / "theory_checks"
THEORY_DELTA = 0.2            # whale valuation 0.7 minus true outcome 0.5 in those checks
THEORY_STUBBORNNESS = 0.3

# Colours by role, as names in figure_style.PALETTE (looked up when a figure is drawn).
COLORS = {
    "whale_share": ("navy", "green", "orange", "ruby"),       # Figure 2 A-C, by ρ
    "herding": ("teal", "ruby", "crimson", "steel", "violet"),  # Figure 3 A-C, by h
    "whale_valuation": ("steel", "green", "orange"),          # Figure 2E, by valuation
    "rho_ramp": ("lavender", "navy"),                         # Figure 2D, ρ from low to high
    "theory": ("navy", "orange"),                             # Figure 3 D-E
    "price_up": "tangerine", "price_down": "sky",             # Figure 1A market pressure
    "outcome": "crimson", "mean_valuation": "lilac",          # Figure 1A
    "volume": "indigo", "yes": "green", "no": "ruby",         # Figure 1C
    "sweep_line": "ruby", "sweep_point": "steel",             # Figure 1B
}


def role_color(role):
    """The colour, or list of colours, for a role in COLORS."""
    return style.color(COLORS[role])


def _budget_population(variance):
    budgets = np.clip(np.random.normal(500, variance, N_BETTORS), 0, None)
    return [Bettor(budget=budget) for budget in budgets]


def _uniform_population(**traits):
    return [Bettor(**traits) for _ in range(N_BETTORS)]


# Parameter name -> (values, population builder), as in the original sweep.
SWEEPS = {
    "Expertise": (np.arange(0, 1.1, 0.1), lambda e: _uniform_population(expertise=e)),
    "Bias": (np.arange(-0.3, 0.3, 0.05), lambda b: _uniform_population(bias=b)),
    "Risk Aversion": (np.arange(0, 1.1, 0.1), lambda r: _uniform_population(risk_av=r)),
    "Stubbornness": (np.arange(0.05, 1, 0.05), lambda s: _uniform_population(stubbornness=s)),
    "Budget Variance": (np.arange(0, 500, 50), _budget_population),
}


def standard_market(seed, population, fundamentals=None):
    """A T_MAX-period market opening at INITIAL_PRICE. Without fundamentals, a random walk
    with step sd RW_VARIANCE is drawn from np.random (seeded, like the population, with seed)."""
    if fundamentals is None:
        fundamentals = fundamental_path(INITIAL_PRICE, T_MAX, RW_VARIANCE)
    return run_market_lob(population, T_MAX, fundamentals, INITIAL_PRICE,
                          rng=np.random.default_rng(seed))


def design_population(seed, whale_share, whale_valuation, whale_risk_av=0.5, **traits):
    """The original design's bettors (np.random seeded with seed), plus a whale holding
    whale_share of all money; whale_valuation None means no whale."""
    np.random.seed(seed)
    population = _uniform_population(budget=DESIGN_BUDGET, expertise=DESIGN_EXPERTISE, **traits)
    if whale_valuation is not None:
        add_whale(population, whale_share, whale_valuation, risk_av=whale_risk_av)
    return population


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
    record = standard_market(seed, make_population(value))
    prices = np.asarray(record["price_history"], dtype=float)
    fundamentals = record["fundamental"]
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


def example_run(seed=EXAMPLE_SEED):
    """One run of the standard market, without a whale: 100 bettors with budget 100."""
    np.random.seed(seed)
    return standard_market(seed, _uniform_population(budget=100))


def plot_market_price(record, ax):
    """Panel A, as plot_returns(step=True) in the original figure."""
    prices = np.asarray(record["price_history"], dtype=float)
    pressure = np.asarray(record["market_pressure"], dtype=float)[1:]
    periods = np.arange(len(prices) - 1)
    upward = prices[:-1] + np.where(pressure > 0, pressure, 0) / PRESSURE_SCALE
    downward = prices[:-1] + np.where(pressure < 0, pressure, 0) / PRESSURE_SCALE

    ax.plot(prices, color="black", label="Market price", drawstyle="steps")
    ax.fill_between(periods, downward, prices[:-1], color=role_color("price_down"), alpha=0.5,
                    step="pre", label="Downward Pressure")
    ax.plot(record["fundamental"], color=role_color("outcome"), label="Election outcome", drawstyle="steps")
    ax.fill_between(periods, upward, prices[:-1], color=role_color("price_up"), alpha=0.5,
                    step="pre", label="Upward Pressure")
    ax.plot(record["beliefs"], color=role_color("mean_valuation"), linewidth=1.2, label="Average market valuation",
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
        ax_volume.plot(np.diff(outstanding), color=role_color("volume"), drawstyle="steps")
        ax_volume.axhline(0, color="#BBBBBB", linewidth=0.6)
        ax_volume.set(ylabel="New contracts\nper period (net)",
                      title="New Contracts & Capital Locked")
    else:
        ax_volume.plot(record["traded_volume"], color=role_color("volume"), drawstyle="steps")
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
    ax_capital.fill_between(periods, 0, yes_stake, step="post", color=role_color("yes"),
                            alpha=0.85, linewidth=0, label="Yes Contracts")
    ax_capital.fill_between(periods, yes_stake, yes_stake + no_stake, step="post",
                            color=role_color("no"), alpha=0.75, linewidth=0,
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
    ax.scatter(data["value"] + jitter, data["peak_lag"], s=4, alpha=0.6, color=role_color("sweep_point"))
    stats = data.groupby("value")["peak_lag"].quantile([0.25, 0.5, 0.75]).unstack()
    ax.plot(stats.index, stats[0.5], color=role_color("sweep_line"), linewidth=1)
    ax.fill_between(stats.index, stats[0.25], stats[0.75], color=role_color("sweep_line"), alpha=0.25,
                    linewidth=0)
    ax.set(xlabel=parameter, ylabel="Best-matching lag", ylim=(-0.5, MAX_LAG + 0.5),
           yticks=range(MAX_LAG + 1))


def plot_sweep_column(sweep, parameter, column, ylabel, ax):
    """Every run as a dot, with the mean and its 95% confidence interval across runs."""
    data = sweep.loc[sweep["parameter"] == parameter]
    ax.scatter(data["value"], data[column], s=4, alpha=0.6, color=role_color("sweep_point"))
    stats = data.groupby("value")[column].agg(["mean", "std", "count"])
    half_width = 1.96 * stats["std"] / np.sqrt(stats["count"])
    ax.plot(stats.index, stats["mean"], color=role_color("sweep_line"), linewidth=1)
    ax.fill_between(stats.index, stats["mean"] - half_width, stats["mean"] + half_width,
                    color=role_color("sweep_line"), alpha=0.25, linewidth=0)
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


def style_axis(ax):
    """Figure 1's look: a full frame around the plot and no gridlines."""
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)


def save(figure, name):
    """Save to figures/ as PDF and PNG, after figure_style's font and layout checks."""
    style.save(figure, FIGURES_DIR / name, name)


def parallel_map(function, *iterables):
    """function over the zipped iterables on N_WORKERS processes, results in input order."""
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        return list(pool.map(function, *iterables, chunksize=8))


def group_legend_entries(axes):
    """Legend handles of axes[0] by group name (per-panel medians dropped, the band renamed);
    removes every axis's own legend."""
    entries = {}
    for handle, label in zip(*axes[0].get_legend_handles_labels()):
        name = label.split(" (median")[0]
        entries.setdefault("Ordinary noise band (A)" if name == "Ordinary band" else name,
                           handle)
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    return entries


def whale_run(share, run):
    """One run of the whale capital-share experiment.

    Returns the smoothed error path, the normalised pull toward the whale's valuation, and the
    non-whale traders' returns. Positions are valued at their expected payout given the final
    true probability: wealth = cash + contracts x p_final.
    """
    seed = whale.BASE_SEED + run
    population = whale.make_population(seed, share)
    record, path, _ = whale.run_whale_market(seed, share, population)
    traders = [agent for agent in population if not agent.whale]
    starting = np.array([agent.initial_budget for agent in traders])
    profit = np.array([wealth(agent, path[-1]) for agent in traders]) - starting
    returns = {
        "nonwhale_return": profit.sum() / starting.sum(),
        "median_trader_return": float(np.median(profit / starting)),
    }
    fundamentals = path[:whale.SIMULATION_HORIZON]
    return (smoothed_absolute_error(record["midquote"], fundamentals),
            normalised_pull(record["midquote"], fundamentals, whale.WHALE_VALUATION,
                            whale.PRE_PERIOD),
            returns)


def wealth(agent, probability):
    """Cash plus the expected payout of the agent's contracts."""
    return agent.budget + agent.n_contracts * probability


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
    """One run of the Figure 2E design; valuation None is the no-whale control.

    Returns the price - fundamental path. The last quote is carried over one-sided books, and
    the opening price fills any periods before the first quote.
    """
    path = make_election_path(seed)
    population = design_population(seed, VALUATION_WHALE_SHARE, valuation)
    record = standard_market(seed, population, path)
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
    for valuation, color in zip(WHALE_VALUATIONS, role_color("whale_valuation")):
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
    style_axis(ax)


def profit_run(share, run):
    """One run of the profit experiment, recording everything needed for later statistics:
    the whale experiment's design with expertise spread evenly from 0 to 1.

    Returns (run summary, per-trader rows, price path, fundamental path). Positions are valued
    at their expected payout given the final true probability: wealth = cash + contracts x p.
    """
    seed = whale.BASE_SEED + run
    traders = whale.make_population(seed, share, expertise=np.linspace(0, 1, whale.N_BETTORS))
    record, path, whale_agent = whale.run_whale_market(seed, share, list(traders))
    final_p = float(path[-1])
    rows = [dict(trader=index, expertise=agent.expertise, budget0=agent.initial_budget,
                 belief0=agent.market_valuation, stubbornness=agent.stubbornness,
                 risk_aversion=agent.risk_av, final_cash=agent.budget,
                 final_contracts=agent.n_contracts, final_belief=agent.market_valuation,
                 profit=wealth(agent, final_p) - agent.initial_budget,
                 trader_return=wealth(agent, final_p) / agent.initial_budget - 1)
            for index, agent in enumerate(traders)]
    summary = dict(
        final_probability=final_p,
        nonwhale_return=sum(row["profit"] for row in rows) / sum(row["budget0"] for row in rows),
        whale_budget0=whale_agent.initial_budget if whale_agent else np.nan,
        whale_final_cash=whale_agent.budget if whale_agent else np.nan,
        whale_final_contracts=whale_agent.n_contracts if whale_agent else np.nan,
        traded_volume=float(np.sum(record["traded_volume"])),
    )
    return summary, rows, np.asarray(record["midquote"], float), path


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


def cached(path, simulate, rerun=False, needs=None):
    """The CSV at path, or simulate() (which saves it) on rerun, when the file is missing, or
    when it lacks the column `needs` (it predates that statistic)."""
    if path.exists() and not rerun:
        frame = pd.read_csv(path)
        if needs is None or needs in frame:
            return frame
    return simulate()


def load_sweep(rerun=False):
    return cached(SWEEP_CSV, run_sweep, rerun)


def load_whale_shares(rerun=False):
    return cached(WHALE_CSV, run_whale_shares, rerun, needs="nonwhale_return")


def load_valuations(rerun=False):
    return cached(VALUATION_CSV, run_valuations, rerun)


def load_herding(rerun=False, theory=False):
    return cached(HERDING_THEORY_CSV if theory else HERDING_CSV, lambda: run_herding(theory),
                  rerun, needs="normalised_pull")


def load_profit_experiment(rerun=False):
    """(runs, traders) frames of the profit experiment."""
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
    style_axis(ax)


def profit_ramp():
    """Sequential colour ramp for ordered ρ levels, between two project-palette colours."""
    return LinearSegmentedColormap.from_list("rho", role_color("rho_ramp"))


def profit_colors(n):
    return [profit_ramp()(i / max(n - 1, 1)) for i in range(n)]


def rho_colorbar(figure_or_subfigure, ax, shares, label="ρ (Whale Capital Share)"):
    """A continuous ρ colour bar under `ax`, in place of a legend of discrete swatches."""
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    mappable = ScalarMappable(norm=Normalize(min(shares), max(shares)), cmap=profit_ramp())
    bar = figure_or_subfigure.colorbar(mappable, ax=ax, location="bottom", fraction=0.05,
                                       pad=0.02, aspect=40)
    bar.set_label(label)
    bar.set_ticks(shares[::3] if len(shares) > 5 else shares)
    return bar


def herding_run(seed, h, with_whale, theory=False):
    """One herding run: the smoothed |price - fundamental| path and, with a whale, the pull
    toward its valuation (NaN without one).

    theory=True uses the CLOB theory's assumptions: whale valuation 0.7 and log utility (r = 1)
    for everyone, the whale included. Otherwise the original design: whale at 1, agents'
    risk aversion drawn from U(0, 1) and the whale's fixed at 0.5.
    """
    valuation = THEORY_WHALE_VALUATION if theory else HERDING_WHALE_VALUATION
    log_utility = {"risk_av": 1.0} if theory else {}
    path = make_election_path(seed)
    population = design_population(seed, HERDING_WHALE_SHARE,
                                   valuation if with_whale else None,
                                   whale_risk_av=1.0 if theory else 0.5,
                                   market_imitation=h, **log_utility)
    record = standard_market(seed, population, path)
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


def theory_pull(h, rho, s):
    """Steady-state pull toward the whale under fast clearing (Eq. 13)."""
    return rho * ((1 - h) * (1 - s) + h) / ((1 - h) * (1 - s) + h * rho)


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
    ax.plot(exact["rho"], exact["error"], "o", color=role_color("theory")[0], markersize=5,
            label="Simulated, exact signals")
    ax.plot(noisy["rho"], noisy["error"], "s", color=role_color("theory")[0], markerfacecolor="white",
            markersize=5, label="Simulated, noisy signals")
    ax.set(xlim=(0, 0.65), ylim=(0, None), xlabel="Whale's Share of Capital (ρ)",
           ylabel="Steady State Price Error",
           title=f"Whale Distortion Without Herding (Δ = {THEORY_DELTA:g})")
    ax.legend(loc="upper left", frameon=False)
    style_axis(ax)


def plot_theory_herding(ax, b):
    """Share of the whale's misvaluation passed into the price, against herding strength."""
    h = np.linspace(0, 1, 200)
    for rho, color in zip(sorted(b["rho"].unique()), role_color("theory")):
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
    style_axis(ax)
