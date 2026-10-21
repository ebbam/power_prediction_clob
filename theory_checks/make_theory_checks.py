"""
Generate clob_theory_checks.png: the theoretical results of the manuscript
(Theoretical Exploration and Supplementary Material) compared with simulations
of the order-book model.

Panels
  A  steady-state whale distortion without herding        delta_S = rho * Delta_S
  B  steady-state distortion with herding                 eq. herding_steady_state
  C  price error after a whale enters
  D  recovery from a common belief shock (no whale)       delta_t ~ delta_0 * kappa^t
  E  simulated first-passage times vs the fast-clearing benchmark
  F  convergence factor kappa against herding weight      d kappa / d h = 1 - rho - s

All agents have log utility (r_i = 1), so the theoretical results are exact up to
the fast-clearing approximation. Run from anywhere:  python3 make_theory_checks.py
Takes about three minutes.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repository root, for bettor.py
from bettor import bettor, run_market_lob

# ---------------------------------------------------------------- parameters
N_AGENTS = 100          # non-whale agents
ETA = 0.5               # true outcome, held fixed
W = 0.7                 # whale valuation
DELTA = W - ETA         # whale misvaluation Delta_S
S_DEFAULT = 0.3         # stubbornness used in panels B-D
TOLERANCE = 0.02        # first-passage tolerance epsilon (panel E)
SHOCK = -0.15           # common shock to non-whale valuations (panels E-F)


# ---------------------------------------------------------------- theory
def steady_state_error(rho, h, s):
    """delta_S for a whale with wealth share rho and homogeneous non-whales (h, s)."""
    return rho * DELTA * ((1 - h) * (1 - s) + h) / ((1 - h) * (1 - s) + h * rho)


def convergence_factor(rho, h, s):
    """kappa = (1-h) s + h (1-rho)."""
    return (1 - h) * s + h * (1 - rho)


# ---------------------------------------------------------------- simulation helpers
def make_population(seed, s=S_DEFAULT, h=0.0, noise=0.0, valuation=ETA):
    """Non-whale agents with log utility; noise is the signal standard deviation (1 - expertise)."""
    rng = np.random.default_rng(seed)
    return [bettor(budget=rng.uniform(100, 1000), market_valuation=valuation, risk_av=1.0,
                   stubbornness=s, expertise=1 - noise, market_imitation=h)
            for _ in range(N_AGENTS)]


def add_whale(population, rho):
    """Append a whale holding a share rho of total wealth."""
    non_whale_wealth = sum(b.budget for b in population)
    whale = bettor(budget=non_whale_wealth * rho / (1 - rho), market_valuation=W, risk_av=1.0, whale=True)
    return population + [whale]


def simulate(population, periods, rng, initial_price=ETA):
    """Run the order-book market with eta fixed; return the end-of-period price error m_t - eta."""
    result = run_market_lob(len(population), periods, initial_price, 0.0, population,
                            gen_el=np.full(periods, ETA), rng=rng)
    return np.array(result["price_history"][1:]) - ETA


def seeded(seed):
    """Seed both random sources used by the model (np.random for signals, a Generator for orders)."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def first_passage(error, tolerance=TOLERANCE, window=3):
    """First period from which |error| <= tolerance for `window` consecutive periods."""
    inside = np.abs(error) <= tolerance
    for t in range(len(inside) - window + 1):
        if inside[t:t + window].all():
            return t
    return np.nan


def recovery_after_shock(rho, s, h, seed, settle=50, after=40):
    """
    Let the market settle (with a whale if rho > 0), then shift every non-whale
    valuation by SHOCK and return the price error relative to the settled level.
    The second run starts a new order book at the settled price; holdings, cash
    and the random stream carry over.
    """
    rng = seeded(seed)
    population = make_population(seed, s=s, h=h)
    if rho > 0:
        population = add_whale(population, rho)
    settled_error = simulate(population, settle, rng)[-1]
    for agent in population:
        if not agent.whale:
            agent.market_valuation += SHOCK
    return simulate(population, after, rng, initial_price=ETA + settled_error) - settled_error


def shock_experiment(rho, s, h, seeds=range(3)):
    """
    Compare recovery after a common shock with the fast-clearing benchmark.
    kappa is estimated by a pooled log-linear fit across seeds, using only errors
    larger than 1.5 ticks so that the price grid does not dominate the fit.
    """
    errors = [recovery_after_shock(rho, s, h, seed) for seed in seeds]
    t_fit, log_fit = [], []
    for error in errors:
        above_floor = np.flatnonzero(np.abs(error) > 0.015)[:8]
        t_fit += above_floor.tolist()
        log_fit += np.log(np.abs(error[above_floor])).tolist()
    kappa_fit = float(np.exp(np.polyfit(t_fit, log_fit, 1)[0])) if len(set(t_fit)) >= 2 else np.nan
    kappa = convergence_factor(rho, h, s)
    initial_gap = (1 - rho) * SHOCK                 # fast-clearing price jump at the shock
    return {"rho": rho, "s": s, "h": h, "kappa": kappa, "kappa_fit": kappa_fit,
            "first_passage": float(np.nanmean([first_passage(e) for e in errors])),
            "benchmark": float(np.log(TOLERANCE / abs(initial_gap)) / np.log(kappa))}


# ---------------------------------------------------------------- experiments
def run_experiments():
    data = {}
    print("A: whale distortion without herding")
    data["A"] = [{"rho": rho, "noise": noise,
                  "error": float(np.mean([simulate(add_whale(make_population(seed, noise=noise), rho), 120, seeded(seed))[40:].mean()
                                          for seed in range(2)]))}
                 for rho in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] for noise in [0.0, 0.1]]
    print("B: distortion with herding")
    data["B"] = [{"rho": rho, "h": h,
                  "error": float(simulate(add_whale(make_population(0, h=h), rho), 120, seeded(0))[-40:].mean())}
                 for rho in [0.2, 0.4] for h in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    print("C: whale entry paths")
    data["C"] = {h: simulate(add_whale(make_population(0, h=h), 0.3), 50, seeded(0)).tolist() for h in [0.0, 0.5, 0.9]}
    print("D: belief-shock recovery")
    data["D"] = {h: simulate(make_population(1, h=h, valuation=0.7), 40, seeded(1)).tolist() for h in [0.0, 0.5, 0.9]}
    print("E-F: shock experiments")
    data["EF"] = [shock_experiment(rho, s, h)
                  for rho, s in [(0.0, 0.1), (0.0, 0.3), (0.0, 0.6), (0.3, 0.2), (0.3, 0.9)]
                  for h in [0.0, 0.3, 0.6]]
    return data


# ---------------------------------------------------------------- figure
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
HERDING_SHADES = {0.0: "#86b6ef", 0.5: "#2a78d6", 0.9: "#104281"}
DOT = dict(marker="o", ms=7, mec=SURFACE, mew=1.2, ls="none")
SQUARE = dict(DOT, marker="s")


def plot(data, path):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    a, b, c, d, e, f = axes.ravel()

    # A
    rho_grid = np.linspace(0.05, 0.65, 50)
    a.plot(rho_grid, rho_grid * DELTA, color=MUTED, label=r"theory $\delta_S=\rho\Delta_S$")
    for noise, color, style, label in [(0.0, BLUE, DOT, "sim, no signal noise"), (0.1, ORANGE, SQUARE, "sim, signal s.d. 0.1")]:
        rows = [r for r in data["A"] if r["noise"] == noise]
        a.plot([r["rho"] for r in rows], [r["error"] for r in rows], color=color, label=label, **style)
    a.set(title="A  Whale distortion without herding", xlabel=r"whale wealth share $\rho$", ylabel=r"steady-state error $\delta_S$")
    a.legend(loc="upper left")

    # B
    h_grid = np.linspace(0, 1, 60)
    for rho, color, offset in [(0.2, BLUE, -14), (0.4, ORANGE, 8)]:
        b.plot(h_grid, steady_state_error(rho, h_grid, S_DEFAULT) / DELTA, color=color, alpha=0.55)
        rows = [r for r in data["B"] if r["rho"] == rho]
        b.plot([r["h"] for r in rows], [r["error"] / DELTA for r in rows], color=color, **DOT)
        b.annotate(rf"$\rho={rho}$", (0.0, rho), xytext=(8, offset), textcoords="offset points", color=color, fontsize=9, fontweight="bold")
    b.plot([], [], color=MUTED, alpha=0.6, label=r"theory (homogeneous $h$, $s$)")
    b.plot([], [], color=MUTED, label="sim", **DOT)
    b.set(title=rf"B  Herding amplifies the whale distortion ($s={S_DEFAULT}$)", xlabel=r"herding weight $h$",
          ylabel=r"$\delta_S/\Delta_S$", ylim=(0, 1.08), xlim=(-0.03, 1.13))
    b.legend(loc="upper left")

    # C
    for h, path_c in data["C"].items():
        h = float(h)
        c.plot(path_c, color=HERDING_SHADES[h], label=f"h = {h:g}")
        c.axhline(steady_state_error(0.3, h, S_DEFAULT), color=HERDING_SHADES[h], lw=1, ls="--")
    c.set(title=r"C  Error after a whale enters ($\rho=0.3$)", xlabel="period", ylabel=r"price error $m_t-\eta$")
    c.legend(loc="lower right", title=r"dashed = predicted $\delta_S$", title_fontsize=8)

    # D
    for h, path_d in data["D"].items():
        h = float(h)
        path_d = np.abs(np.array(path_d))
        kappa = convergence_factor(0.0, h, S_DEFAULT)
        t = np.arange(len(path_d))
        d.plot(t, path_d + 1e-4, color=HERDING_SHADES[h], label=rf"h = {h:g} ($\kappa$={kappa:.2f})")
        d.plot(t, path_d[0] * kappa ** t, color=HERDING_SHADES[h], lw=1, ls="--")
    d.axhline(0.01, color=MUTED, lw=1, ls=":")
    d.text(39, 0.0112, "1 tick", ha="right", fontsize=8, color=MUTED)
    d.set(yscale="log", ylim=(1e-3, 0.3), title="D  Recovery from a belief shock (no whale)",
          xlabel="period", ylabel=r"$|m_t-\eta|$ (log scale)")
    d.legend(loc="upper right", title=r"dashed = $\delta_0\,\kappa^t$", title_fontsize=8)

    # E
    limit = 17
    e.plot([0, limit], [0, limit], color=MUTED, lw=1, ls="--")
    e.text(16.5, 15.2, "sim = benchmark", color=MUTED, fontsize=8, ha="right")
    for has_whale, color, style, label in [(False, BLUE, DOT, "no whale"), (True, ORANGE, SQUARE, r"whale, $\rho=0.3$")]:
        rows = [r for r in data["EF"] if (r["rho"] > 0) == has_whale]
        e.plot([r["benchmark"] for r in rows], [r["first_passage"] for r in rows], color=color, label=label, **style)
    e.set(title="E  First passage vs fast-clearing benchmark", xlabel=r"benchmark $\ln(\epsilon/|\delta_0-\delta_S|)/\ln\kappa$",
          ylabel="simulated first-passage time (mean of 3 runs)", xlim=(0, limit), ylim=(0, limit))
    e.legend(loc="upper left", title=rf"belief shock ${SHOCK}$; $\epsilon={TOLERANCE}$", title_fontsize=8)

    # F
    h_grid = np.linspace(0, 0.65, 40)
    for s, color, style, label in [(0.2, BLUE, DOT, r"$s=0.2<1-\rho$: herding slows"),
                                   (0.9, ORANGE, SQUARE, r"$s=0.9>1-\rho$: herding speeds")]:
        f.plot(h_grid, convergence_factor(0.3, h_grid, s), color=color, alpha=0.55)
        rows = [r for r in data["EF"] if r["rho"] == 0.3 and r["s"] == s and np.isfinite(r["kappa_fit"])]
        f.plot([r["h"] for r in rows], [r["kappa_fit"] for r in rows], color=color, label=label, **style)
    f.set(title=r"F  $\partial\kappa/\partial h = 1-\rho-s$ ($\rho=0.3$)", xlabel=r"herding weight $h$",
          ylabel=r"convergence factor $\kappa$ (line theory, marks sim)", ylim=(0, 1))
    f.legend(loc="center right")

    fig.suptitle(rf"CLOB theory vs simulation  (log utility $r_i=1$; {N_AGENTS} agents; $\eta={ETA}$; whale $W={W}$, $\Delta={DELTA:.1f}$)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    data = run_experiments()
    with open(os.path.join(HERE, "clob_theory_checks.json"), "w") as fh:
        json.dump(data, fh, indent=1)
    plot(data, os.path.join(HERE, "clob_theory_checks.png"))
    print("saved clob_theory_checks.png and clob_theory_checks.json")
