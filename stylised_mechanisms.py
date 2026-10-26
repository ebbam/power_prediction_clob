"""Stylised figure of how the bettor parameters shape an agent's decisions.

    A  Risk aversion, for a bettor whose internal valuation is V_i = 0.5:
       top, the order it submits at each limit price in the CLOB's window around its
       reservation price (bettor.limit_order, which sizes orders with bettor.trade);
       bottom, how its reservation price moves once it holds a position
       (bettor.reservation_price).
    B  Stubbornness: belief updating toward a noisy signal of the true outcome eta_t,
       starting from V_i,0 = 0.1 (the update in bettor.update_belief).
    C  Expertise and bias: distribution of the signals a bettor receives about eta_t = 0.5.

Rebuilt from mechanism_experiment_plots.ipynb (power_prediction repo, scratch_plots branch,
commit 06b6404), with the combined layout of the published figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as style
from bettor import bettor

style.apply()


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Line colours, in the order of the original notebook's palette.
COLORS = ["#CC99FF", "#000080", "#339966", "#F4A261", "#B22222"]
BIAS_COLORS = ["royalblue", "lightseagreen", "slateblue"]
SELL_COLOR, BUY_COLOR = "#8E3B8E", "#FFA500"
SELL_LABEL_COLOR, BUY_LABEL_COLOR = "#5B1F7A", "#C45A00"

RISK_AVERSIONS = [0.1, 0.25, 0.5, 0.75, 0.9]
STUBBORNNESS = [0.1, 0.25, 0.5, 0.75, 0.9]
EXPERTISE = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
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
    top = BUDGET * 1.05
    ax.fill_between([VALUATION - MAX_MARKUP, VALUATION], 0, top, color=BUY_COLOR, alpha=0.2,
                    linewidth=0)
    ax.fill_between([VALUATION, VALUATION + MAX_MARKUP], -top, 0, color=SELL_COLOR, alpha=0.25,
                    linewidth=0)
    for color, r in zip(COLORS, RISK_AVERSIONS):
        ax.plot(prices, optimal_order(prices, VALUATION, r, BUDGET), color=color, label=f"{r}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(VALUATION, color="gray", linestyle="--")
    ax.text(VALUATION - MAX_MARKUP * 0.1, BUDGET * 0.95, "Buy", color=BUY_LABEL_COLOR,
            ha="right", va="top")
    ax.text(VALUATION + MAX_MARKUP * 0.1, -BUDGET * 0.95, "Sell", color=SELL_LABEL_COLOR,
            ha="left", va="bottom")
    ax.text(VALUATION - 0.001, -BUDGET * 0.5, r"$V_i$", ha="right", va="center")
    ax.set(xlim=(VALUATION - MAX_MARKUP, VALUATION + MAX_MARKUP),
           ylim=(-BUDGET * 1.05, BUDGET * 1.05), xlabel="Limit Price",
           ylabel="Order Size (+ buy, − sell)", title="Order Size at Each Limit Price")


def plot_reservation(ax):
    """A (bottom): the reservation price (the price at which the bettor is indifferent between
    buying and selling one more contract) after it has built up a position. It buys below this
    price and sells above it, so a bettor holding Yes contracts sells even below V_i and one
    holding No contracts buys even
    above it.
    """
    positions = np.linspace(-BUDGET * 0.9, BUDGET * 0.9, 181)
    for color, r in zip(COLORS, RISK_AVERSIONS):
        prices = np.array([bettor(budget=BUDGET - VALUATION * q, market_valuation=VALUATION,
                                  risk_av=r, n_contracts=q).reservation_price()
                           for q in positions])
        ax.plot(positions, prices, color=color)
    ax.axhline(VALUATION, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8)
    label_right(ax, VALUATION, r"$V_i$")
    ax.set(xlim=(positions[0], positions[-1]), ylim=(0.25, 0.75),
           xlabel="Contracts Held (− No, + Yes)", ylabel="Reservation Price",
           title="Reservation Price by Position Held")


def plot_stubbornness(ax, rng):
    signals = rng.normal(TRUE_VALUE, 1 - SIGNAL_EXPERTISE, N_STEPS)
    ax.plot(range(1, N_STEPS + 1), np.clip(signals, 0, 1), color="gray", alpha=0.6,
            linewidth=2)
    for color, s in zip(COLORS, STUBBORNNESS):
        beliefs = [INITIAL_BELIEF]
        for signal in signals:
            beliefs.append(np.clip(s * beliefs[-1] + (1 - s) * signal, 0, 1))
        ax.plot(range(N_STEPS + 1), beliefs, color=color, label=f"{s}")
    ax.axhline(TRUE_VALUE, color="gray", linestyle="--")
    ax.axhline(INITIAL_BELIEF, color="gray", linestyle=":")
    label_right(ax, TRUE_VALUE, r"$\eta_t$")
    label_right(ax, INITIAL_BELIEF, r"$V_{i,0}$")
    ax.set(xlim=(0, N_STEPS), ylim=(0, 0.9), xlabel="Time step", ylabel="Market Valuation",
           title="Stubbornness: Belief Updating")
    ax.legend(title="Stubbornness", loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=5,
              frameon=False, handlelength=1.2, columnspacing=0.8)


def plot_signals(ax, rng):
    width = 0.27
    for e_index, e in enumerate(EXPERTISE):
        base = rng.normal(TRUE_VALUE, max(1e-9, 1 - e), N_SIGNALS)
        for b_index, (b, color) in enumerate(zip(BIASES, BIAS_COLORS)):
            box = ax.boxplot(base - b, positions=[e_index + (b_index - 1) * width],
                             widths=width * 0.95, patch_artist=True, whis=1.5,
                             flierprops=dict(marker="o", markersize=2),
                             medianprops=dict(color="#333333"))
            box["boxes"][0].set(facecolor=color, edgecolor="#333333")
    ax.axhspan(1, 2, color="#E5E5E5", zorder=0)
    ax.axhspan(-1, 0, color="#E5E5E5", zorder=0)
    ax.axhline(TRUE_VALUE, color="#555555", linestyle="--")
    label_right(ax, TRUE_VALUE, r"$\eta_t$")
    ax.set_xticks(range(len(EXPERTISE)), [f"{e:.1f}" for e in EXPERTISE])
    ax.set(xlim=(-0.5, len(EXPERTISE) - 0.5), ylim=(-0.1, 1.1), xlabel="Expertise",
           ylabel="Signal", title="Expertise and Bias: Signal Quality")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#333333")
               for c in BIAS_COLORS]
    ax.legend(handles, ["−0.1", "0", "+0.1"], title="Bias", loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncols=3, frameon=False, handlelength=1.2)


def main():
    rng = np.random.default_rng(42)
    # 2 x 2 grid: A (risk aversion: order size | reservation price) on top, B | C below.
    figure = plt.figure(figsize=(WIDTH, 6.4), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[1.05, 1])
    ax_a, ax_a2 = top.subplots(1, 2)
    ax_b, ax_c = bottom.subplots(1, 2)
    plot_order_size(ax_a)
    plot_reservation(ax_a2)
    top.suptitle("Risk Aversion", fontsize=style.TITLE)
    for ax in (ax_a, ax_a2):
        ax.title.set_fontsize(style.LABEL)            # subtitles under the shared title
    top.legend(*ax_a.get_legend_handles_labels(), title="Risk Aversion",
               loc="outside lower center", ncols=len(RISK_AVERSIONS), frameon=False,
               handlelength=1.2, columnspacing=1.0)
    plot_stubbornness(ax_b, rng)
    plot_signals(ax_c, rng)
    ax_b.set_title("Stubbornness")
    ax_c.set_title("Expertise and Bias")
    ax_b.set_ylabel("Market Valuation")
    style.label_panels(((ax_a, "A"), (ax_b, "B"), (ax_c, "C")))
    style.save(figure, FIGURES_DIR / "stylised_mechanisms_clob", "stylised_mechanisms_clob")
    plt.close(figure)


if __name__ == "__main__":
    main()
