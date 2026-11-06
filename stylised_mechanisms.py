"""Panels of the stylised mechanisms figure (assembled in manuscript_figures.ipynb):
how the bettor parameters shape an agent's decisions.

    A  Risk aversion, for a bettor whose internal valuation is V_i = 0.5:
       top, the order it submits at each limit price in the CLOB's window around its
       reservation price (bettor.limit_order, which sizes orders with bettor.trade);
       bottom, how its reservation price moves once it holds a position
       (bettor.reservation_price).
    B  Stubbornness: the expected belief path from V_i,0 = 0.1 toward the true outcome eta_t
       under the update in Bettor.update_belief, V_t = eta + (V_0 - eta) s^t.
    C  Expertise and bias: distribution of the signals a bettor receives about eta_t = 0.5.

Rebuilt from mechanism_experiment_plots.ipynb (power_prediction repo, scratch_plots branch,
commit 06b6404), with the combined layout of the published figure.
"""

import matplotlib.pyplot as plt
import numpy as np

import figure_style as style
from bettor import Bettor

style.apply()

# Colours by role, as names in figure_style.PALETTE (looked up when a figure is drawn).
COLORS = {
    # The colours of Figure 3's whale shares, so the figures read as one set.
    "levels": ("lavender", "navy", "green", "orange", "ruby"),   # one per parameter value
    "bias": ("navy", "green", "orange"),
    "buy": "green", "sell": "ruby",                     # as Yes and No contracts in Figure 2
}


def role_color(role):
    """The colour, or list of colours, for a role in COLORS."""
    return style.color(COLORS[role])


RISK_AVERSIONS = [0.1, 0.25, 0.5, 0.75, 0.9]
STUBBORNNESS = [0.1, 0.25, 0.5, 0.75, 0.9]
EXPERTISE = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
BOX_ALPHA = 0.75          # box fill, as in Figure 3C
BIASES = [-0.1, 0.0, 0.1]
VALUATION = 0.5          # V_i in panel A
TRUE_VALUE = 0.5         # eta_t in panels B and C
INITIAL_BELIEF = 0.1     # V_i,0 in panel B
SIGNAL_EXPERTISE = 0.9   # signal noise in panel B: sd = 1 - expertise
BUDGET = 100
WIDTH = style.PAGE_WIDTH   # the 2 x 2 grid needs the full page width
MAX_MARKUP = 0.05      # CLOB limit prices lie within this distance of the reservation price
N_STEPS = 50
N_SIGNALS = 100


def label_right(ax, y, text):
    ax.text(1.015, y, text, transform=ax.get_yaxis_transform(), va="center")


def optimal_order(prices, valuation, risk_aversion, budget):
    """Continuous utility-maximising order size for a bettor with no position.

    bettor.trade maximises v u(W + (1 - p) c) + (1 - v) u(W - p c) over whole contracts c in
    [-W, W], with u(w) = w^(1 - r) / (1 - r) (log for r = 1). The first-order condition gives
    (W + (1 - p) c) / (W - p c) = k with k = [v (1 - p) / ((1 - v) p)]^(1 / r), so
    c = W (k - 1) / (k p + 1 - p). The model rounds this to whole contracts.
    """
    prices = np.asarray(prices, dtype=float)
    k = (valuation * (1 - prices) / ((1 - valuation) * prices)) ** (1 / risk_aversion)
    order = budget * (k - 1) / (k * prices + 1 - prices)
    return np.clip(order, -budget, budget)


def plot_order_size(ax):
    """A (top): the order a bettor with no position submits at each limit price. In the CLOB a
    bettor sets its limit price within MAX_MARKUP of its reservation price (V_i with no position), buying
    below it and selling above it, and the quantity is the utility-maximising trade there
    (drawn as the continuous optimum; the model rounds it to whole contracts)."""
    prices = np.linspace(VALUATION - MAX_MARKUP, VALUATION + MAX_MARKUP, 201)
    # Only two quadrants occur: buy orders below V_i and sell orders above it.
    top = BUDGET * 1.15
    ax.fill_between([VALUATION - MAX_MARKUP, VALUATION], 0, top, color=role_color("buy"), alpha=0.15,
                    linewidth=0)
    ax.fill_between([VALUATION, VALUATION + MAX_MARKUP], -top, 0, color=role_color("sell"), alpha=0.15,
                    linewidth=0)
    for color, r in zip(role_color("levels"), RISK_AVERSIONS):
        ax.plot(prices, optimal_order(prices, VALUATION, r, BUDGET), color=color, label=f"{r}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(VALUATION, color="gray", linestyle="--")
    ax.text(VALUATION - MAX_MARKUP * 0.1, BUDGET * 0.95, "Buy", color=role_color("buy"),
            ha="right", va="top")
    ax.text(VALUATION + MAX_MARKUP * 0.1, -BUDGET * 0.95, "Sell", color=role_color("sell"),
            ha="left", va="bottom")
    ax.text(VALUATION - 0.001, -BUDGET * 0.5, r"$V_i$", ha="right", va="center")
    ax.set(xlim=(VALUATION - MAX_MARKUP, VALUATION + MAX_MARKUP),
           ylim=(-BUDGET * 1.15, BUDGET * 1.15), xlabel="Limit Price",
           ylabel="Order Size (+ buy, − sell)", title="Order Size at Each Limit Price")


def plot_reservation(ax):
    """A (bottom): the reservation price (the price at which the bettor is indifferent between
    buying and selling one more contract) after it has built up a position. It buys below this
    price and sells above it, so a bettor holding Yes contracts sells even below V_i and one
    holding No contracts buys even
    above it.
    """
    positions = np.linspace(-BUDGET * 0.9, BUDGET * 0.9, 181)
    for color, r in zip(role_color("levels"), RISK_AVERSIONS):
        prices = np.array([Bettor(budget=BUDGET - VALUATION * q, market_valuation=VALUATION,
                                  risk_av=r, n_contracts=q).reservation_price()
                           for q in positions])
        ax.plot(positions, prices, color=color)
    ax.axhline(VALUATION, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8)
    label_right(ax, VALUATION, r"$V_i$")
    ax.set(xlim=(positions[0], positions[-1]), ylim=(0.25, 0.75),
           xlabel="Contracts Held (− No, + Yes)", ylabel="Reservation Price",
           title="Reservation Price by Position Held")


def plot_stubbornness(ax):
    """Expected belief path for each stubbornness s: with unbiased signals, the belief closes a
    share 1 - s of its gap to the true outcome every step."""
    steps = np.arange(N_STEPS + 1)
    for line_color, s in zip(role_color("levels"), STUBBORNNESS):
        ax.plot(steps, TRUE_VALUE + (INITIAL_BELIEF - TRUE_VALUE) * s ** steps,
                color=line_color, label=f"{s}")
    ax.axhline(TRUE_VALUE, color="gray", linestyle="--")
    ax.axhline(INITIAL_BELIEF, color="gray", linestyle=":")
    label_right(ax, TRUE_VALUE, r"$\eta_t$")
    label_right(ax, INITIAL_BELIEF, r"$V_{i,0}$")
    ax.set(xlim=(0, N_STEPS), ylim=(0, 0.6), xlabel="Time step", ylabel="Market Valuation",
           title="Stubbornness: Belief Updating")
    ax.legend(title="Stubbornness", loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=5)


def plot_signals(ax, rng):
    width = 0.27
    for e_index, e in enumerate(EXPERTISE):
        base = rng.normal(TRUE_VALUE, max(1e-9, 1 - e), N_SIGNALS)
        for b_index, (b, color) in enumerate(zip(BIASES, role_color("bias"))):
            box = ax.boxplot(base - b, positions=[e_index + (b_index - 1) * width],
                             widths=width * 0.95, patch_artist=True, whis=1.5,
                             flierprops=dict(marker="o", markersize=2),
                             medianprops=dict(color="#333333"))
            box["boxes"][0].set(facecolor=color, alpha=BOX_ALPHA, edgecolor="#333333")
    ax.axhspan(1, 2, color="#E5E5E5", zorder=0)
    ax.axhspan(-1, 0, color="#E5E5E5", zorder=0)
    ax.axhline(TRUE_VALUE, color="#555555", linestyle="--")
    label_right(ax, TRUE_VALUE, r"$\eta_t$")
    ax.set_xticks(range(len(EXPERTISE)), [f"{e:.1f}" for e in EXPERTISE])
    ax.set(xlim=(-0.5, len(EXPERTISE) - 0.5), ylim=(-0.1, 1.1), xlabel="Expertise",
           ylabel="Signal", title="Expertise and Bias: Signal Quality")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=BOX_ALPHA, edgecolor="#333333")
               for c in role_color("bias")]
    ax.legend(handles, ["−0.1", "0", "+0.1"], title="Bias", loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncols=3)
