"""Parallel, reproducible parameter sweep for the original LOB market model.

Run from a directory containing bettor.py, orderbook.py and convergence.py:
    python whale_herding_parallel.py --reps 200 --workers 8
"""
from whale_setup import scheduled_whale

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bettor import bettor, run_market_lob

ETA = 0.5
W = 0.7
DELTA = W - ETA
N = 100
T = 100
TAIL = 50
RHO_VALUES = np.arange(0.1, 0.91, 0.1)
H_VALUES = np.linspace(0, 0.9, 10)
S_VALUES = np.linspace(0, 0.9, 10)
S_FIXED = 0.1
RHO_FIXED = 0.3


def theoretical_error(rho, h, s, delta=DELTA):
    numerator = (1 - h) * (1 - s) + h
    denominator = (1 - h) * (1 - s) + h * rho
    return rho * delta * numerator / denominator


def simulate(rho, h, s, seed, B=100):
    # The original model uses BOTH NumPy's legacy global RNG and Generator.
    # This is safe because every worker is a separate PROCESS, not a thread.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    agents = [
        bettor(
            budget=B, market_valuation=ETA, n_contracts=0,
            risk_av=1.0, stubbornness=s, expertise=1.0,
            market_imitation=h, bias=0,
        )
        for _ in range(N)
    ]
    schedule, _ = scheduled_whale(agents, rho, 0, W)
    record = run_market_lob(
        n_bettors=N, t_election=T, initial_price=ETA,
        outcome_uncertainty=0, bettors=agents,
        gen_el=np.full(T, ETA), scheduled_agents=schedule, tick=0.01, max_markup=0.05, rng=rng,
    )
    price = np.asarray(record['reference_price'][1:], dtype=float)
    midquote = np.asarray(record['midquote'], dtype=float)
    return (
        np.mean(price[-TAIL:] - ETA),
        np.max(np.abs(price - ETA)),
        np.mean(np.isfinite(midquote)),
    )


def run_condition(task):
    # Parallelise CONDITIONS rather than individual replications to minimise
    # interprocess communication and preserve original reduction order.
    panel, rho, h, s, reps, budget = task
    results = [simulate(rho, h, s, seed=10000 + rep, B=budget)
               for rep in range(reps)]
    steady = [r[0] for r in results]
    maximum = [r[1] for r in results]
    coverage = [r[2] for r in results]
    return {
        'panel': panel, 'rho': rho, 'h': h, 's': s,
        'steady_error': np.mean(steady),
        'steady_se': np.std(steady, ddof=1) / np.sqrt(reps),
        'max_error': np.mean(maximum),
        'midquote_coverage': np.mean(coverage),
    }


def make_tasks(reps, budget):
    # Preserve exactly the original order and floating-point grid values.
    return (
        [('A', rho, h, S_FIXED, reps, budget)
         for h in H_VALUES for rho in RHO_VALUES]
        + [('B', RHO_FIXED, h, s, reps, budget)
           for s in S_VALUES for h in H_VALUES]
    )


def plot_figure(df, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(H_VALUES)))
    ax = axes[0]
    rho_dense = np.linspace(0.01, 0.65, 300)
    for h, color in zip(H_VALUES, colors):
        ax.plot(rho_dense, theoretical_error(rho_dense, h, S_FIXED),
                color=color, lw=2, label=fr'$h={h:g}$')
        subset = df[(df.panel == 'A') & (np.isclose(df.h, h))].sort_values('rho')
        ax.errorbar(subset.rho, subset.steady_error,
                    yerr=1.96 * subset.steady_se, fmt='o', color=color,
                    capsize=2, markersize=4, linestyle='none')
    ax.set(title='A  Steady-state distortion',
           xlabel=r'Initial whale capital share $\rho_0$',
           ylabel=r'Steady-state price error $p_S-\eta$',
           xlim=(0, 0.65), ylim=(0, 0.22))
    ax.legend(title='Herding strength', frameon=False, fontsize=8)
    ax = axes[1]
    subset = df[df.panel == 'B']
    heatmap = subset.pivot(index='s', columns='h', values='max_error')
    im = ax.imshow(heatmap.values, origin='lower', aspect='auto',
                   extent=[min(H_VALUES)-0.125, max(H_VALUES)+0.125,
                           min(S_VALUES)-0.05, max(S_VALUES)+0.05], cmap='viridis')
    ax.set(title='B  Maximum market error', xlabel=r'Herding strength $h$',
           ylabel=r'Stubbornness $s$')
    ax.set_xticks(H_VALUES)
    fig.colorbar(im, ax=ax, label=r'Maximum $|p_t-\eta|$')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig('figure4_whale_herding.pdf', bbox_inches='tight')
    fig.savefig('figure4_whale_herding.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--budget', type=float, default=100)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.reps < 2:
        parser.error('--reps must be >= 2 for the original standard-error calculation')
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    tasks = make_tasks(args.reps, args.budget)
    print(f'Running {len(tasks)} conditions x {args.reps} replications '
          f'= {len(tasks)*args.reps} simulations with {args.workers} processes', flush=True)
    if args.workers == 1:
        rows = [run_condition(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(run_condition, tasks, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv('whale_herding_results.csv', index=False)
    plot_figure(df, show=args.show)
    print(df.groupby('panel')['midquote_coverage'].describe())


if __name__ == '__main__':
    main()
