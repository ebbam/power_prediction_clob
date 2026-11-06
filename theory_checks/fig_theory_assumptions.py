"""Appendix: six diagnostics for the approximations behind the CLOB theory.

Place this file in power_prediction_clob/theory_checks/ and run:
    python fig_theory_assumptions.py

Requires the repository-root bettor.py and NumPy/Matplotlib. Independent
simulation paths are cached in .simulation_cache, separately from plotting.
Panel F reports first-passage times to a fixed theoretical steady-state
price p_S; right-censored recovery diagnostics are saved in
recovery_diagnostics.json. The previous run-specific terminal-price target
is retained only as a diagnostic for comparison.

IMPORTANT timing: the market's price_history[t+1] is an end-of-period price.
The callback below samples budgets, positions, beliefs and reservation prices
AFTER trades, BEFORE the end-of-period belief update. This avoids combining
prices with next-period beliefs.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bettor import Bettor, add_whale, run_market_lob
import figure_style as style

TICK = .01
ETA = .5
W = .7
N = 100
BLUE = '#2a78d6'
ORANGE = '#eb6834'
GREEN = '#299d80'
INK = '#272725'
RISK_COLORS = ['#86b6ef', '#2a78d6', '#104281']
# Discrete ordered palette supplied for this figure: purple/indigo through
# blue, teal and green to yellow. Numeric square keys identify every h value.
HERDING_COLORS = [
    '#482878',  # 0.0 - deep purple
    '#4146A5',  # 0.1 - distinctive indigo
    '#3B528B',  # 0.2 - blue
    '#2C718E',  # 0.3 - blue-teal
    '#26828E',  # 0.4 - teal-blue
    '#01B4AE',  # 0.5 - distinctive turquoise
    '#249F74',  # 0.6 - green-teal
    '#5EC962',  # 0.7 - green
    '#AADC32',  # 0.8 - yellow-green
    '#FDE725',  # 0.9 - bright yellow
]

# Panel F uses the two high-contrast colours supplied for the recovery
# comparison, independently of the ten-value herding palette.
PANEL_F_C1 = '#1E90FF'
PANEL_F_C2 = '#E63946'
HERDING_TICKS = np.round(np.arange(0.0, 1.0, 0.1), 1)
HERDING_BOUNDS = np.arange(-0.05, 1.0, 0.1)
# Only Panel E uses this shorter, independently labelled comparison grid.
D_HERDING = [0.0, 0.5, 0.9]

# Panels B-D all use the same complete default grid. There are no later
# assignments to args.herding that can silently overwrite this setting.
DEFAULT_HERDING = HERDING_TICKS.tolist()
HERDING_CMAP = ListedColormap(HERDING_COLORS, name='discrete_herding')
HERDING_NORM = BoundaryNorm(HERDING_BOUNDS, HERDING_CMAP.N)


def legend_below(ax, handles, *, ncol=2, fontsize=None, columnspacing=1.0):
    """Place a compact plot key immediately below an axis."""
    return ax.legend(
        handles=handles, loc='upper center', bbox_to_anchor=(.5, -.22),
        ncol=ncol, fontsize=fontsize or style.SMALL, frameon=True,
        framealpha=1.0, edgecolor='#BBBBBB', fancybox=False,
        borderpad=.35, handlelength=1.8, columnspacing=columnspacing,
        labelspacing=.3, handletextpad=.5)


def theoretical_clearing_price(budgets, contracts, valuations, risk_aversions):
    """Solve sum_j A_j(p)*(V_j-p) = 0, A_j=(B_j+p*C_j)/r_j.

    Under log utility (r_j=1) AND zero net contracts, this is exactly
        p* = sum_j (B_j+p*C_j)*V_j / sum_j (B_j+p*C_j).
    At other CRRA coefficients it is the linear-demand approximation.
    Uses the ACTUAL end-of-period mark-to-market portfolios, not initial cash.
    We evaluate the quadratic clearing equation by bisection on p in (0,1).
    """
    b, c, v, r = map(lambda x: np.asarray(x, dtype=float),
                     (budgets, contracts, valuations, risk_aversions))
    if np.any(r <= 0):
        raise ValueError('The demand approximation needs strictly positive risk aversion.')
    def excess(p):
        return np.sum((b + p*c)/r * (v - p))
    left, right = 1e-8, 1-1e-8
    if excess(left) <= 0: return left
    if excess(right) >= 0: return right
    for _ in range(52):
        mid = .5*(left+right)
        if excess(mid) > 0: left = mid
        else: right = mid
    return .5*(left+right)


def predicted_steady_error(rho, h, s, delta=W-ETA):
    """Manuscript Eq. (16), constant wealth share and mean belief dynamics."""
    return rho*delta*((1-h)*(1-s)+h)/((1-h)*(1-s)+h*rho)


def predicted_kappa(rho, h, s):
    """Manuscript geometric convergence coefficient."""
    return (1-h)*s + h*(1-rho)


def make_population(seed, rho=.3, h=.3, s=.3, shock=0., noisy=False):
    """Exact log utility, homogeneous non-whale learning and no clipping issues.

    Equal initial cash gives rho its simple initial-budget interpretation.
    The whale is the final agent; all agents enter at t=0.
    When noisy=False expertise=1 means exact private signals in bettor.py.
    """
    np.random.seed(seed)
    agents = [Bettor(budget=500., market_valuation=ETA+shock,
                     risk_av=1., stubbornness=s, expertise=.9 if noisy else 1.,
                     market_imitation=h, bias=0.) for _ in range(N)]
    whale = add_whale(agents, share=rho, valuation=W, risk_av=1.)
    return agents, whale


def simulate_observed(seed, rho=.3, h=.3, s=.3, steps=100,
                      shock=0., noisy=False, markup=.05):
    """Run the real repository CLOB, sampling the end-of-period PRE-UPDATE state.

    The original run_market_lob does not expose valuation or portfolio history.
    Rather than modifying the market implementation, temporarily wrap the first
    non-whale Bettor's update_belief method. It is called after order matching,
    before any other bettor's belief changes. Each entry thus represents a
    single consistent market state. The wrapper is restored afterwards.
    """
    agents, whale = make_population(seed, rho, h, s, shock, noisy)
    snapshots = []
    first = agents[0]
    original_update = first.update_belief
    def record_then_update(true_value, market_price):
        active = [a for a in agents if a.entry_time <= len(snapshots)]
        b = np.array([a.budget for a in active], float)
        c = np.array([a.n_contracts for a in active], float)
        v = np.array([a.market_valuation for a in active], float)
        r = np.array([a.risk_av for a in active], float)
        z = np.array([a.reservation_price() for a in active], float)
        pstar = theoretical_clearing_price(b, c, v, r)
        wealth = b + market_price*c
        whale_indices = np.array([a.whale for a in active], dtype=bool)
        rho_t = wealth[whale_indices].sum() / wealth.sum() if whale_indices.any() else 0.
        snapshots.append(dict(market_price=float(market_price), pstar=pstar,
                              median_reservation_gap=float(np.nanmedian(abs(z-pstar))),
                              p10_reservation_gap=float(np.nanpercentile(abs(z-pstar), 10)),
                              p90_reservation_gap=float(np.nanpercentile(abs(z-pstar), 90)),
                              wealth_share=float(rho_t)))
        return original_update(true_value, market_price)
    first.update_belief = record_then_update
    try:
        path = np.full(steps+1, ETA, float)
        rec = run_market_lob(agents, steps, path, initial_price=ETA,
                             tick=TICK, max_markup=markup,
                             rng=np.random.default_rng(seed+100000))
    finally:
        first.update_belief = original_update
    if len(snapshots) != steps:
        raise RuntimeError(f'Expected {steps} snapshots, got {len(snapshots)}')
    return rec, snapshots


def simulation_job(parameters):
    # Cache simulation states, independently of plotting and fitting choices.
    folder = Path(__file__).resolve().parent / '.simulation_cache'
    folder.mkdir(exist_ok=True)
    signature = ''.join((ROOT/name).read_text() for name in ('bettor.py', 'orderbook.py'))
    signature += inspect.getsource(simulate_observed) + inspect.getsource(make_population)
    signature += inspect.getsource(theoretical_clearing_price)
    key = hashlib.sha256((signature + json.dumps(parameters, sort_keys=True)).encode()).hexdigest()
    path = folder / (key + '.json')
    if path.exists():
        with path.open() as stream:
            item = json.load(stream)
        return item['record'], item['snapshots']
    record, snapshots = simulate_observed(**parameters)
    record = {'price_history': np.asarray(record['price_history']).tolist()}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(record=record, snapshots=snapshots)))
    temporary.replace(path)
    return record, snapshots


def simulation_runs(seeds, **parameters):
    jobs = [dict(seed=seed, **parameters) for seed in seeds]
    return POOL.map(simulation_job, jobs) if POOL is not None else map(simulation_job, jobs)

POOL = None


def recovery_fit(market_prices, tail=20, min_pairs=3):
    """Fit post-shock recovery to a run-specific settled price.

    The imposed initial price is excluded. The last 20 periods must have
    10-90% spread <= two ticks and half-tail medians within one tick.
    Fit only the initial transient before first entering the one-tick band.
    R2 measures the whole geometric trajectory, not just adjacent pairs.
    """
    p = np.asarray(market_prices, float)[1:]
    result = dict(status='unsettled', kappa=None, r2=None, n_pairs=0,
                  target=None, recovery_time=None)
    if len(p) < 2*tail or not np.isfinite(p).all():
        return result
    end = p[-tail:]
    target = float(np.median(end))
    result['target'] = target
    if (np.ptp(np.percentile(end, [10, 90])) > 2*TICK+1e-9
            or abs(np.median(end[:tail//2])-np.median(end[tail//2:])) > TICK+1e-9):
        return result
    error = p-target
    # Require five consecutive periods inside the band for recovery time.
    hits = [t for t in range(len(p)-tail-4)
            if np.all(np.abs(error[t:t+5]) <= TICK+1e-9)]
    result['recovery_time'] = hits[0] if hits else None
    entering = np.flatnonzero(np.abs(error[:-tail]) <= TICK+1e-9)
    stop = int(entering[0]) if len(entering) else len(p)-tail-1
    transient = error[:stop+1]
    x, y = transient[:-1], transient[1:]
    result['n_pairs'] = len(x)
    if len(x) < min_pairs:
        result['status'] = 'too_short'
        return result
    k = float(np.dot(x, y)/np.dot(x, x))
    predicted = transient[0]*k**np.arange(len(transient))
    total = np.sum((transient-transient.mean())**2)
    r2 = float(1-np.sum((transient-predicted)**2)/total) if total > 0 else None
    result.update(status='fitted', kappa=k, r2=r2)
    return result


def fitted_kappa(market_prices, rho=None, h=None, s=None):
    fit = recovery_fit(market_prices)
    return fit['kappa'] if fit['status'] == 'fitted' else np.nan


def theoretical_recovery_time(rho, h, s, shock=-.15, epsilon=TICK):
    """Continuous geometric benchmark from the ideal initial clearing price.

    Initial non-whale valuations are ETA+shock, whale valuation W. The
    theoretical long-run price uses the constant-wealth-share approximation.
    No simulated quantities or integer rounding enter this benchmark.
    """
    p_initial = (1-rho)*(ETA+shock)+rho*W
    p_steady = ETA+predicted_steady_error(rho, h, s)
    distance = np.abs(p_initial-p_steady)
    kappa = predicted_kappa(rho, h, s)
    return np.maximum(0, np.log(epsilon/np.maximum(distance, epsilon))/np.log(kappa))


def first_persistent_recovery_time(market_prices, target, epsilon=TICK, consecutive=5):
    """First start period of `consecutive` prices within epsilon of fixed target.

    `market_prices` must contain post-shock end-of-period prices only. The
    returned time is measured from the first post-shock period (zero-based).
    No run-specific terminal price or settlement filter is used.
    """
    prices = np.asarray(market_prices, dtype=float)
    if len(prices) < consecutive or not np.isfinite(prices).all():
        return None
    inside = np.abs(prices - float(target)) <= epsilon + 1e-12
    for t in range(len(inside) - consecutive + 1):
        if np.all(inside[t:t + consecutive]):
            return int(t)
    return None


def kaplan_meier_quantile(observed_times, events, quantile):
    """Estimate a time-to-recovery quantile with right-censored runs.

    `observed_times` contains the recovery time for events and the final
    observable five-period-window start for censored runs. Event times are
    processed before censoring at the same time. Returns None when the
    requested quantile is not identified within the simulation horizon.
    """
    times = np.asarray(observed_times, dtype=float)
    events = np.asarray(events, dtype=bool)
    valid = np.isfinite(times)
    times, events = times[valid], events[valid]
    if len(times) == 0:
        return None
    survival = 1.0
    for t in np.sort(np.unique(times[events])):
        at_risk = int(np.sum(times >= t))
        failures = int(np.sum((times == t) & events))
        if at_risk:
            survival *= (1.0 - failures / at_risk)
        if (1.0 - survival) >= quantile:
            return float(t)
    return None


def fixed_target_recovery(market_prices, p_steady, epsilon=TICK, consecutive=5):
    """Return fixed-p_S first-passage time and right-censoring information."""
    prices = np.asarray(market_prices, dtype=float)[1:]  # Exclude imposed initial price.
    recovery_time = first_persistent_recovery_time(
        prices, p_steady, epsilon=epsilon, consecutive=consecutive)
    last_start = max(0, len(prices) - consecutive)
    reached = recovery_time is not None
    observed_time = float(recovery_time if reached else last_start)
    return dict(
        target_fixed=float(p_steady),
        recovery_time_fixed=(int(recovery_time) if reached else None),
        reached_fixed=bool(reached),
        censored_fixed=not reached,
        km_observed_time=observed_time,
        km_event=bool(reached),
        final_price_median=float(np.median(prices[-min(20, len(prices)):])) if len(prices) else None,
        final_abs_error_to_fixed_target=(
            float(abs(np.median(prices[-min(20, len(prices)):]) - p_steady))
            if len(prices) else None),
    )


def plot_zero_holdings_demand(ax):
    """Panel A: exact utility-optimal order size vs its local linearisation.

    One agent has B=500, valuation V=0.6 and C=0 contracts. For each risk
    aversion r, the exact series uses Bettor.trade(), which maximises expected
    CRRA utility over feasible integer contract quantities. The dashed curve
    is the first-order approximation from manuscript Eq. (11):

        x_lin(p) = B * (V-p) / [r*p*(1-p)]       (because C=0).

    With C=0, the reservation price is z=V=0.6 for every r. The exact
    optimiser is collateral-constrained and integer-valued; the linearised
    expression is continuous and is only local to p=V.
    """
    prices = np.linspace(.3, .9, 61)
    risk_values = [1.0, .5, .1]
    colors = RISK_COLORS[::-1]
    budget = 500.0
    valuation = .6
    holdings = 0

    for risk_aversion, color in zip(risk_values, colors):
        agent = Bettor(budget=budget, market_valuation=valuation,
                       n_contracts=holdings, risk_av=risk_aversion)
        exact = np.asarray([agent.trade(float(p)) for p in prices], dtype=float)
        linear = (-holdings + (budget + prices*holdings)
                  / (risk_aversion*prices*(1-prices)) * (valuation-prices))
        ax.plot(prices, exact, color=color, lw=1.8)
        ax.plot(prices, linear, color=color, lw=1.15, ls='--')
        # With zero holdings, all three reservation prices equal V=0.6.
        ax.axvline(agent.reservation_price(), color=color, lw=.85, ls=':')

    ax.axhline(0, color=INK, lw=.8)
    ax.set(xlim=(.3, .9), ylim=(-3000, 3000),
           xlabel='Order price, $p$',
           ylabel='Optimal trade (contracts)')

    # Two complementary legend keys: colour identifies r; line style identifies
    # the exact optimiser, its linearisation, and the reservation price.
    risk_handles = [Line2D([], [], color=color, lw=2,
                           label=rf'$r={r:g}$')
                    for r, color in zip(risk_values, colors)]
    style_handles = [
        Line2D([], [], color=INK, lw=1.15, ls='--', label='Linear approximation'),
        Line2D([], [], color=INK, lw=.9, ls=':', label='Reservation price'),
    ]
    # The coloured solid curves already identify the exact optimiser by r.
    legend_below(ax, risk_handles + style_handles, ncol=2, columnspacing=1.8)
    note = dict(fontsize=style.SMALL,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
    ax.text(.97, .96,
            r'$B=500$''\n' r'$V=0.6$''\n''$C=0$'
            '\n' r'$x^*(p)=\frac{B(V-p)}{rp(1-p)}$',
            transform=ax.transAxes, ha='right', va='top', **note)


def build_figure(args):
    style.apply()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    # A slightly wider page accommodates Panel F's conventional vertical
    # right-axis label. The tighter grid reduces unused whitespace.
    fig = plt.figure(figsize=(style.PAGE_WIDTH, 11.5))
    gs = fig.add_gridspec(3, 2, left=.09, right=.90, bottom=.22,
                          top=.955, hspace=.60, wspace=.34)
    # Six panels: A demand approximation; B-D CLOB clearing/wealth diagnostics;
    # E geometric recovery; F recovery-time comparison.
    axa, axb, axc, axd, axe, axf = (
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
    )
    # A discrete colour assignment: each tenth-step of h has its own fixed colour.
    def herding_color(h):
        h_rounded = float(np.round(h, 1))
        return HERDING_CMAP(HERDING_NORM(h_rounded))

    # A: demand approximation at zero holdings, added from the user's figure.
    plot_zero_holdings_demand(axa)

    all_snapshots = {}
    print('Herding grid for panels B-D:', ', '.join(f'{h:.1f}' for h in args.herding))
    print('Herding grid for panel E:', ', '.join(f'{h:.1f}' for h in D_HERDING))
    print('Panel A: holdings=0; risk aversion r=1.0, 0.5, 0.1')

    # B-D: run the actual CLOB; independent repetitions and herding settings.
    a_colors = [herding_color(h) for h in args.herding]
    for h, color in zip(args.herding, a_colors):
        states = []
        full_states = []
        seeds = [args.seed + 100*rep + int(h*1000)
                 for rep in range(args.reps)]
        for _, snap in simulation_runs(
                seeds, rho=args.rho, h=h, s=args.s, steps=args.steps,
                noisy=args.noisy, markup=args.markup):
            states.extend(snap[args.burn:])
            full_states.extend(snap)
        all_snapshots[h] = states

        observed = np.array([x['market_price'] for x in states])
        theoretical = np.array([x['pstar'] for x in states])
        gap = np.array([x['median_reservation_gap'] for x in full_states])
        q10 = np.array([x['p10_reservation_gap'] for x in full_states])
        q90 = np.array([x['p90_reservation_gap'] for x in full_states])
        wealth_share = np.array([x['wealth_share'] for x in full_states])

        # B: observed CLOB price vs implied competitive-equilibrium price.
        axb.scatter(theoretical, observed, s=6, alpha=.4, color=color)

        # C-D: summarise aligned time paths across independent repetitions.
        gaps = gap.reshape(args.reps, args.steps)
        q10s = q10.reshape(args.reps, args.steps)
        q90s = q90.reshape(args.reps, args.steps)
        shares = wealth_share.reshape(args.reps, args.steps)
        tt = np.arange(args.steps)

        # C: median agent reservation-price gap; band is median across runs
        # of the within-run 10th and 90th percentiles across agents.
        center = np.median(gaps, axis=0) / TICK
        low = np.median(q10s, axis=0) / TICK
        high = np.median(q90s, axis=0) / TICK
        # Log axes cannot display an exact zero. Clip only the plotted values
        # to a small visual floor; the underlying diagnostics remain unchanged.
        log_floor = 1e-3
        axc.plot(tt, np.maximum(center, log_floor), color=color, lw=.9)
        if any(np.isclose(h, v) for v in (0, .5, .9)):
            axc.fill_between(tt, np.maximum(low, log_floor),
                             np.maximum(high, log_floor),
                             color=color, alpha=.10, linewidth=0)

        # D: wealth-share deviation from its initial budget share, in pp.
        axd.plot(tt, 100*(np.median(shares, axis=0)-args.rho),
                 color=color, lw=.9)
        axd.fill_between(
            tt, 100*(np.percentile(shares, 10, axis=0)-args.rho),
            100*(np.percentile(shares, 90, axis=0)-args.rho),
            color=color, alpha=.12, linewidth=0)

        dev = np.abs(observed-theoretical) / TICK
        print(f'h={h:g}: |m-p*| in ticks median={np.median(dev):.2f}, '
              f'p95={np.percentile(dev, 95):.2f}; '
              f'median reservation gap={np.median(gap)/TICK:.2f} ticks; '
              f'whale wealth share range={np.min(wealth_share):.3f}-'
              f'{np.max(wealth_share):.3f}')

    prices = np.array(
        [x['pstar'] for states in all_snapshots.values() for x in states]
        + [x['market_price'] for states in all_snapshots.values() for x in states])
    limits = (prices.min()-.01, prices.max()+.01)
    axb.plot(limits, limits, '--', color=INK, lw=.8)
    axb.set(xlabel=r'Clearing price $p_t^*$', ylabel=r'CLOB price $m_t$',
            xlim=limits, ylim=limits)
    # legend_below(axb, [
    #     Line2D([], [], marker='o', color=BLUE, markerfacecolor=BLUE,
    #            markersize=4, ls='none', label='CLOB observations'),
    #     # Line2D([], [], color=INK, ls='--', lw=.9),
    # ], columnspacing=1.8)

    axc.axhline(1, color=INK, ls=':', lw=.8, label='One tick')
    axc.set_yscale('log')
    axc.set(xlabel='Period', ylabel=r'Median $|z_{i,t}-p_t^*|$ (ticks, log scale)',
            ylim=(6e-2, None), xlim=(0, args.steps-1))
    legend_below(axc, [
        Line2D([], [], color=INK, lw=1.4, label='Median across agents and runs'),
        Line2D([], [], color=INK, lw=5, alpha=.18, label='10th-90th percentile across agents'),
        Line2D([], [], color=INK, ls=':', lw=.9, label='One tick'),
    ], ncol=1)

    axd.axhline(0, color=INK, lw=.8, ls='--', label='Initial share')
    axd.set(xlabel='Period', ylabel=r'$100(\rho_t-\rho_0)$ (pp)',
            xlim=(0, args.steps-1))
    # The dashed zero line marks the initial whale wealth share.
    legend_below(axd, [
        Line2D([], [], color=INK, lw=1.4, label='Median wealth-share deviation'),
        Line2D([], [], color=INK, lw=5, alpha=.18, label='10th-90th percentile across runs'),
        Line2D([], [], color=INK, ls='--', lw=.9,
               label=r'Initial share ($\rho_0$)'),
    ], ncol=1)

    # One shared, discrete legend identifies h across Panels B-D.
    # Use square colour keys rather than a continuous-looking colourbar.
    herding_key_handles = [
        Patch(facecolor=HERDING_COLORS[i], edgecolor='none', label=f'{h:.1f}')
        for i, h in enumerate(HERDING_TICKS)
    ]
    herding_legend = fig.legend(
        handles=herding_key_handles,
        title=r'Herding strength $h$',
        loc='lower center', bbox_to_anchor=(.5, .08),
        ncol=10, fontsize=style.SMALL,
        frameon=True, framealpha=1.0, edgecolor='#BBBBBB',
        fancybox=False, borderpad=.35,
        handlelength=.85, handleheight=.85,
        handletextpad=.35, columnspacing=.85,
        labelspacing=.25,
    )
    herding_legend.get_title().set_fontsize(style.SMALL)

    # E: independent common-shock CLOB runs and geometric-theory comparison.
    print(f'E: generating {args.d_reps} independent common-shock simulations per h')
    d_colors = [herding_color(h) for h in D_HERDING]
    e_upper = 0.0
    for h, color in zip(D_HERDING, d_colors):
        trajectories = []
        seeds = [args.seed+777+10000*rep+int(h*1000)
                 for rep in range(args.d_reps)]
        for rec, _ in simulation_runs(
                seeds, rho=0, h=h, s=args.s, steps=args.d_steps,
                shock=args.d_shock, noisy=False, markup=args.markup):
            # Initial price is exogenously set to ETA; the first realised
            # end-of-period price follows the common initial belief shock.
            market_prices = np.asarray(rec['price_history'], float)
            errors = np.abs(market_prices[1:args.d_steps+1]-ETA)
            if len(errors) != args.d_steps:
                raise ValueError('Unexpected price_history length for panel E')
            trajectories.append(errors)
        trajectories = np.asarray(trajectories)
        time = np.arange(args.d_steps)
        med = np.median(trajectories, axis=0)
        lo, hi = np.percentile(trajectories, [10, 90], axis=0)
        e_upper = max(e_upper, float(np.max(hi)))
        k = predicted_kappa(0, h, args.s)
        initial = med[0]
        axe.plot(time, med, color=color, lw=1.7, label=rf'$h={h:.1f}$')
        axe.fill_between(time, lo, hi, color=color, alpha=.19, linewidth=0)
        axe.plot(time, initial*k**time, '--', color=color, lw=1.4)
    axe.axhline(TICK, color=INK, lw=.9, ls=':', label='One tick')
    axe.set(xlabel='Periods after shock', ylabel=r'$|m_t-\eta|$',
            title='Recovery without a whale')
    axe.set_ylim(0, max(.15, 1.25*e_upper))
    # Panel E needs its own explicit h key: it uses only three values from
    # the full ten-value herding grid. Colour identifies h; line/fill style
    # identifies the statistic or theoretical benchmark.
    panel_e_legend = [
        Line2D([], [], color=herding_color(h), lw=2.6,
               label=rf'$h={h:.1f}$')
        for h in D_HERDING
    ]
    panel_e_legend += [
        Line2D([], [], color=INK, lw=1.4, ls='--', label='Theory'),
        Patch(facecolor=INK, edgecolor='none', alpha=.19,
              label='10th-90th percentile across runs'),
        Line2D([], [], color=INK, lw=.9, ls=':', label='One-tick threshold'),
    ]
    legend_below(axe, panel_e_legend, ncol=2)

    # F: first-passage recovery to a FIXED theoretical steady-state price.
    # Unlike the previous version, the target does not depend on each run's
    # final-20-period median. All runs are retained; non-hits are right-censored.
    diagnostics = []
    summaries = []
    f_h_grid = np.round(np.arange(0.0, 0.61, 0.1), 1)
    f_s_values = [(0.2, PANEL_F_C1, 'o'),
                  (0.9, PANEL_F_C2, 's')]
    plot_quantiles = {}
    print(f'F: generating {args.reps} independent simulations per (h, s), '
          f'{args.f_steps} periods each; fixed target p_S')

    for s, color, marker in f_s_values:
        km_q10, km_q50, km_q90 = [], [], []
        reach_rates = []
        for i, h in enumerate(f_h_grid):
            h = float(np.round(h, 1))
            seeds = [args.seed + 9500 + i * 100 + rep
                     for rep in range(args.reps)]
            p_steady = ETA + predicted_steady_error(args.rho, h, s)
            observed_times, events = [], []
            rows = []

            for seed, (record, _) in zip(
                    seeds, simulation_runs(
                        seeds, rho=args.rho, h=h, s=s, steps=args.f_steps,
                        shock=-.15, noisy=False, markup=args.markup)):
                prices = np.asarray(record['price_history'], dtype=float)
                fixed = fixed_target_recovery(prices, p_steady,
                                              epsilon=TICK, consecutive=5)

                # Retain the previous run-specific measure as a diagnostic,
                # not as the target used in Panel F.
                legacy = recovery_fit(prices)
                row = dict(
                    seed=int(seed), s=float(s), h=h,
                    **fixed,
                    legacy_status=legacy['status'],
                    legacy_target=legacy['target'],
                    legacy_recovery_time=legacy['recovery_time'],
                    legacy_kappa=legacy['kappa'],
                    legacy_r2=legacy['r2'],
                    legacy_n_pairs=legacy['n_pairs'],
                )
                rows.append(row)
                diagnostics.append(row)
                observed_times.append(fixed['km_observed_time'])
                events.append(fixed['km_event'])

            q10 = kaplan_meier_quantile(observed_times, events, 0.10)
            q50 = kaplan_meier_quantile(observed_times, events, 0.50)
            q90 = kaplan_meier_quantile(observed_times, events, 0.90)
            reached = int(np.sum(events))
            reach_rate = reached / len(events) if events else np.nan
            reach_rates.append(reach_rate)
            km_q10.append(np.nan if q10 is None else q10)
            km_q50.append(np.nan if q50 is None else q50)
            km_q90.append(np.nan if q90 is None else q90)

            legacy_resolved = [r['legacy_recovery_time'] for r in rows
                               if r['legacy_status'] != 'unsettled'
                               and r['legacy_recovery_time'] is not None]
            summary = dict(
                s=float(s), h=h, runs=len(rows),
                fixed_target_pS=float(p_steady),
                target_reached=reached,
                target_not_reached=len(rows) - reached,
                target_reach_rate=float(reach_rate),
                km_recovery_quantiles_q10_q50_q90=[
                    None if q is None else float(q) for q in (q10, q50, q90)],
                final_median_price_median=float(np.median([
                    r['final_price_median'] for r in rows
                    if r['final_price_median'] is not None])) if rows else None,
                median_final_abs_error_to_pS=float(np.median([
                    r['final_abs_error_to_fixed_target'] for r in rows
                    if r['final_abs_error_to_fixed_target'] is not None])) if rows else None,
                legacy_run_specific_recovery_quantiles=(
                    [float(v) for v in np.quantile(legacy_resolved, [.1, .5, .9])]
                    if legacy_resolved else None),
                legacy_resolved_runs=len(legacy_resolved),
                legacy_unsettled_runs=sum(r['legacy_status'] == 'unsettled' for r in rows),
            )
            summaries.append(summary)
            print('F:', json.dumps(summary), flush=True)

        plot_quantiles[float(s)] = dict(
            q10=np.asarray(km_q10, dtype=float),
            q50=np.asarray(km_q50, dtype=float),
            q90=np.asarray(km_q90, dtype=float),
            reach=np.asarray(reach_rates, dtype=float),
            color=color, marker=marker,
        )

    # Theoretical continuous first-passage benchmark and Kaplan-Meier medians.
    # A right-hand axis reports the fraction of runs that actually reach p_S;
    # this makes right-censoring visible rather than hiding it in the JSON.
    axf_rate = axf.twinx()
    axf_rate.set_ylim(0, 1.2)
    # Conventional vertical right-axis label. The wider figure and reserved
    # right margin keep it inside the page checked by figure_style.py.
    axf_rate.set_ylabel('Share reaching steady state (%)', labelpad=8)
    # Move the vertical label beyond the percentage tick labels while keeping
    # it within the reserved right margin of the 7-inch page.
    axf_rate.yaxis.set_label_coords(1.22, .5)
    axf_rate.set_yticks([0, .25, .50, .75, 1.0])
    axf_rate.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    axf_rate.spines['top'].set_visible(False)

    dense_h = np.linspace(0, .65, 400)
    all_y = []
    for s, color, marker in f_s_values:
        theory = theoretical_recovery_time(args.rho, dense_h, s, shock=-.15)
        axf.plot(dense_h, theory, color=color, lw=1.5,
                 label=rf'Theory, $s={s:g}$')
        q = plot_quantiles[float(s)]
        axf_rate.plot(f_h_grid, q['reach'], color=color, ls=':', lw=.9,
                      marker='x', markersize=3, alpha=.9,
                      label=rf'Reached, $s={s:g}$')
        valid_med = np.isfinite(q['q50'])
        for j, h in enumerate(f_h_grid):
            if not valid_med[j]:
                continue
            med = q['q50'][j]
            q10 = q['q10'][j]
            q90 = q['q90'][j]
            lower = max(0.0, med - q10) if np.isfinite(q10) else 0.0
            upper = max(0.0, q90 - med) if np.isfinite(q90) else 0.0
            axf.errorbar(
                [h], [med],
                yerr=np.asarray([[lower], [upper]]),
                fmt=marker, color=color, markersize=3.5,
                capsize=2, linewidth=.8,
                label=(rf'CLOB KM median (P10-P90), $s={s:g}$' if j == 0 else None))
            all_y.append(med)
            if np.isfinite(q90):
                all_y.append(q90)
        # An upward arrow indicates that the Kaplan-Meier P90 was not reached
        # before the end of the simulation horizon; it is not set to the horizon.

    theory_top = max(float(np.nanmax(theoretical_recovery_time(
        args.rho, dense_h, s, shock=-.15))) for s, _, _ in f_s_values)
    finite_q90 = [float(v) for q in plot_quantiles.values()
                  for v in q['q90'] if np.isfinite(v)]
    ymax = max([25.0, 1.15 * theory_top]
               + ([1.15 * max(all_y)] if all_y else [])
               + ([1.15 * max(finite_q90)] if finite_q90 else []))
    axf.set_ylim(0, ymax)

    # Mark non-estimable survival quantiles and report target-attainment rates.
    for s, color, marker in f_s_values:
        q = plot_quantiles[float(s)]
        for j, h in enumerate(f_h_grid):
            if not np.isfinite(q['q50'][j]):
                axf.scatter([h], [ymax * .94], marker='^', s=22,
                            facecolors='none', edgecolors=[color], zorder=5)
                axf.annotate('median > horizon', (h, ymax * .94),
                             xytext=(0, 4), textcoords='offset points',
                             ha='center', fontsize=style.SMALL, color=color)
            elif not np.isfinite(q['q90'][j]):
                axf.annotate('', xy=(h, ymax * .91), xytext=(h, q['q50'][j]),
                             arrowprops=dict(arrowstyle='-|>', color=color, lw=.9))

    # Fixed-target attainment is printed and stored in the JSON diagnostics;
    # the caption/legend makes clear that the interval is a KM interval with
    # right-censoring, not a percentile interval conditional on success.
    axf.set(xlabel=r'Herding strength $h$',
            ylabel='Periods to reach theoretical steady state',
            xlim=(-.03, .63))
    # Keep Panel F's key deliberately short: the previous labels were too
    # wide for the bottom-row legend and triggered figure_style.check_fonts.
    # Error bars denote the KM P10-P90 interval around the simulated median.
    panel_f_legend = [
        Line2D([], [], color=PANEL_F_C1, lw=1.5,
               label=r'Theory ($s=0.2$)'),
        Line2D([], [], color=PANEL_F_C1, marker='o', ls='none',
               markersize=4, label=r'Simulated ($s=0.2$)'),
        Line2D([], [], color=PANEL_F_C2, lw=1.5,
               label=r'Theory ($s=0.9$)'),
        Line2D([], [], color=PANEL_F_C2, marker='s', ls='none',
               markersize=4, label=r'Simulated ($s=0.9$)'),
        Line2D([], [], color=INK, marker='x', ls=':', lw=.9,
               markersize=3, label='Target reach rate'),
    ]
    # Add a key for the triangle only when the plot actually contains one.
    if any(np.any(~np.isfinite(q['q50'])) for q in plot_quantiles.values()):
        panel_f_legend.append(
            Line2D([], [], color=INK, marker='^', ls='none', markersize=4,
                   label='Median beyond horizon'))
    legend_below(axf, panel_f_legend, ncol=2)

    report = Path(__file__).resolve().parent/'recovery_diagnostics.json'
    report.write_text(json.dumps(dict(
        theory=('Continuous geometric benchmark T_epsilon=max(0, '
                'log(epsilon/abs(p0_star-pS))/log(kappa)); '
                'p0_star=(1-rho)*(ETA-0.15)+rho*W; '
                'pS=ETA+predicted_steady_error(rho,h,s); epsilon=0.01.'),
        definition=('First start of five consecutive post-shock CLOB prices '
                    'within one tick of the fixed theoretical steady-state '
                    'price pS=ETA+predicted_steady_error(rho,h,s).'),
        censoring=('Runs that do not reach the fixed pS band within f_steps '
                   'are right-censored at the last observable five-period '
                   'window start. Panel F reports Kaplan-Meier recovery-time '
                   'quantiles; a quantile that is not identified by the horizon '
                   'is shown as not reached, not replaced by a finite value.'),
        target_attainment=('target_reach_rate is the share of all runs that '
                            'enter and remain in the fixed one-tick band for '
                            'five consecutive periods.'),
        legacy_comparison=('The previous final-20-period-median target method '
                           'is retained per run as legacy_* diagnostics only; '
                           'it does not determine Panel F.'),
        fixed_target_horizon=int(args.f_steps),
        summaries=summaries, runs=diagnostics), indent=2))

    # Keep the equation inside Panel F: placing it above the axes caused it
    # to collide with the legends beneath Panels C and D.
    axf.text(.50, .985,
             r'$p_S=\eta+\delta_S(h,s),\quad '
             r'T_\epsilon:\ |m_{t+j}-p_S|\leq\epsilon,\ j=0,\ldots,4$'
             '\n' r'$\kappa=(1-h)s+h(1-\rho)$',
             transform=axf.transAxes, ha='center', va='top', fontsize=style.SMALL,
             bbox=dict(facecolor='white', edgecolor='none', alpha=.92, pad=1.5),
             clip_on=True)
    axf.set_ylim(0,50)
    

    # Equations and assumptions printed in each panel.
    note = dict(fontsize=style.SMALL,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
    axb.text(.13, .96,
             r'$m_t\approx p_t^*=\frac{\sum_i w_{i,t}V_{i,t}}{\sum_i w_{i,t}}$'
             '\n' r'$w_{i,t}=B_{i,t}+p_t^*C_{i,t}$',
             transform=axb.transAxes, va='top', **note)
    axc.text(.97, .94, r'$z_{i,t}\approx p_t^*$', transform=axc.transAxes,
             ha='right', va='top', **note)
    axe.text(.18, .96, r'$|\delta_t|=|\delta_0|\kappa^t$'
             '\n' r'$\kappa=(1-h)s+h\quad(\rho=0)$',
             transform=axe.transAxes, ha='left', va='top', **note)
    axes = (axa, axb, axc, axd, axe, axf)
    axf_rate.spines['top'].set_visible(False)
    axf_rate.spines['right'].set_visible(True)
    axf_rate.tick_params(axis='y', labelsize=style.SMALL)
    for ax in axes:
        ax.set_title('')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
    for ax, letter in zip(axes, 'ABCDEF'):
        pos = ax.get_position()
        fig.text(pos.x0-.055, pos.y1+.012, letter, fontsize=style.PANEL,
                 fontweight='bold', va='bottom')

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent/output
    output.parent.mkdir(parents=True, exist_ok=True)
    style.save(fig, output.with_suffix(''), 'Theory assumptions')
    print('Saved', output, 'and', output.with_suffix('.pdf'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--burn', type=int, default=20)
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--d-reps', type=int, default=100,
                        help='Independent simulations for D uncertainty bands')
    parser.add_argument('--d-steps', type=int, default=40)
    parser.add_argument('--f-steps', type=int, default=200,
                        help='Observation horizon for Panel F fixed-target recovery')
    parser.add_argument('--d-shock', type=float, default=-.15)
    parser.add_argument('--herding', type=float, nargs='+',
                        default=DEFAULT_HERDING,
                        help='Herding values for B-D; values are rounded to the nearest 0.1 (default: 0.0, 0.1, ..., 0.9)')
    parser.add_argument('--rho', type=float, default=.3)
    parser.add_argument('--s', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--markup', type=float, default=.05)
    parser.add_argument('--noisy', action='store_true')
    parser.add_argument('--no-cache', dest='use_existing', action='store_false',
                        help='Compatibility flag: historical summary files are no longer used')
    parser.set_defaults(use_existing=False)
    parser.add_argument('--jobs', type=int, default=4, help='Concurrent independent simulations')
    parser.add_argument('--output', default='fig_theory_assumptions_6panel_fixed_theory_target.png')
    args = parser.parse_args()
    requested_h = np.asarray(args.herding, dtype=float)
    if not np.isfinite(requested_h).all() or np.any(requested_h < 0) or np.any(requested_h > .9):
        parser.error('--herding values must lie in [0,0.9]')
    # Snap all requested values to one decimal place, then remove duplicates.
    args.herding = sorted({float(np.round(h, 1)) for h in requested_h})
    print('Using rounded herding grid:', ', '.join(f'{h:.1f}' for h in args.herding))
    if args.d_reps < 2: parser.error('--d-reps must be at least 2')
    if args.reps < 1: parser.error('--reps must be at least 1')
    if args.d_steps < 2: parser.error('--d-steps must be at least 2')
    if args.f_steps < 5: parser.error('--f-steps must be at least 5')
    if not 0 <= args.burn < args.steps: parser.error('--burn must be < --steps')
    if not 0 < args.rho < 1: parser.error('--rho must be in (0,1)')
    global POOL
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            POOL = pool
            build_figure(args)
    else:
        build_figure(args)

if __name__ == '__main__': main()
