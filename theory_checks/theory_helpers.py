"""
Helpers for theory_assumptions.ipynb.

Builds agent populations, runs the order-book market (bettor.run_market_lob),
records agent states, and implements the theoretical formulas of Section 3.2 and
Supplementary Section 8 of the manuscript. Experiment functions live here (not in
the notebook) so they can run in parallel worker processes.
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repository root, for bettor.py
from bettor import bettor, run_market_lob          # noqa: E402

N_AGENTS = 100      # non-whale agents
ETA = 0.5           # true outcome when held fixed
W = 0.7             # whale valuation
TICK = 0.01
CACHE_DIR = os.path.join(HERE, "cache")


# ---------------------------------------------------------------- populations
def seeded(seed):
    """Seed both random sources used by the model: np.random (signals) and a Generator (orders)."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def make_population(seed, n=N_AGENTS, r=1.0, s=0.3, h=0.0, noise=0.0, valuation=ETA):
    """
    Non-whale agents with budgets U(100, 1000) and common parameters.
    r is a number, or a function (rng, n) -> array of risk aversions.
    noise is the signal standard deviation, i.e. 1 - expertise.
    """
    rng = np.random.default_rng(seed)
    budgets = rng.uniform(100, 1000, n)
    risk = r(rng, n) if callable(r) else np.full(n, r)
    values = valuation(rng, n) if callable(valuation) else np.full(n, valuation)
    return [bettor(budget=b, market_valuation=v, risk_av=ri, stubbornness=s, expertise=1 - noise, market_imitation=h)
            for b, v, ri in zip(budgets, values, risk)]


def manuscript_population(seed, n=N_AGENTS, h=0.0):
    """Table 1 defaults: B~U(100,1000), V~N(0.5,0.05), s~N(0.3,0.05), e~N(0.9,0.04), r~U(0,1)."""
    rng = np.random.default_rng(seed)
    return [bettor(budget=rng.uniform(100, 1000),
                   market_valuation=np.clip(rng.normal(0.5, 0.05), 0, 1),
                   risk_av=rng.uniform(0, 1),
                   stubbornness=np.clip(rng.normal(0.3, 0.05), 0, 1),
                   expertise=np.clip(rng.normal(0.9, 0.04), 0.8, 1),
                   market_imitation=h)
            for _ in range(n)]


def add_whale(population, rho, valuation=W, r=1.0):
    """Return population plus a whale holding a share rho of total budget."""
    non_whale_budget = sum(b.budget for b in population)
    whale = bettor(budget=non_whale_budget * rho / (1 - rho), market_valuation=valuation, risk_av=r, whale=True)
    return population + [whale]


# ---------------------------------------------------------------- simulation
def simulate(population, periods, rng, eta_path=None, initial_price=ETA, record=False):
    """
    Run the order-book market. eta_path defaults to eta fixed at ETA.
    Returns arrays of the market price m_t (Eq. 2: midquote, else best quote, else last trade),
    the midquote (NaN when one side is empty), the true outcome and the price error.
    With record=True, also returns each agent's reservation price, valuation, cash and holdings
    at the end of every period (after trading, before beliefs update), shape (periods, agents).
    """
    eta_path = np.full(periods, ETA) if eta_path is None else np.asarray(eta_path[:periods])
    records = []
    original_update = bettor.update_belief
    if record:
        def recording_update(self, *args, **kwargs):
            records.append((self.reservation_price(), self.market_valuation, self.budget, self.n_contracts))
            return original_update(self, *args, **kwargs)
        bettor.update_belief = recording_update
    try:
        result = run_market_lob(len(population), periods, initial_price, 0.0, population, gen_el=eta_path, rng=rng)
    finally:
        bettor.update_belief = original_update
    price = np.array(result["price_history"][1:])
    out = {"price": price, "midquote": np.asarray(result["midquote"]), "eta": eta_path,
           "error": price - eta_path, "spread": np.asarray(result["spread"])}
    if record:
        z, v, cash, held = np.array(records).reshape(periods, len(population), 4).transpose(2, 0, 1)
        out.update(reservation=z, valuation=v, cash=cash, holdings=held)
    return out


def random_walk(seed, periods, sd=0.05, start=ETA):
    """True-outcome path of Eq. 1 (steps N(0, sd), clipped to [0, 1]); sd as in the code's gen_election."""
    rng = np.random.default_rng(seed + 10_000)
    path = [start]
    for _ in range(periods - 1):
        path.append(float(np.clip(path[-1] + rng.normal(0, sd), 0, 1)))
    return np.array(path)


# ---------------------------------------------------------------- theory
def steady_state_error(rho, h, s, delta=W - ETA):
    """Eq. 13 (Eq. 11 when h = 0)."""
    return rho * delta * ((1 - h) * (1 - s) + h) / ((1 - h) * (1 - s) + h * rho)


def convergence_factor(rho, h, s):
    """kappa = (1-h) s + h (1-rho)."""
    return (1 - h) * s + h * (1 - rho)


def first_passage_benchmark(gap, kappa, tolerance):
    """T_eps = ceil( ln(eps / |delta_0 - delta_S|) / ln kappa )."""
    return float(np.ceil(np.log(tolerance / abs(gap)) / np.log(kappa)))


def crra_demand(p, V, B, r):
    """Exact optimal trade at price p with no holdings (Section 8.2), overflow-safe for small r."""
    odds = np.log(V * (1 - p) / ((1 - V) * p)) / r
    k = np.exp(np.clip(odds, -700, 700))
    return np.where(k > 1, B * (1 - 1 / k) / ((1 - p) / k + p), B * (k - 1) / (1 - p + p * k))


def walrasian_price(V, B, r):
    """Competitive-equilibrium price: aggregate exact CRRA demand equals zero (bisection)."""
    lo, hi = 1e-6, 1 - 1e-6
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if crra_demand(mid, V, B, r).sum() > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def first_passage(error, tolerance, window=3):
    """First period from which |error| <= tolerance for `window` consecutive periods (NaN if never)."""
    inside = np.abs(error) <= tolerance
    for t in range(len(inside) - window + 1):
        if inside[t:t + window].all():
            return t
    return np.nan


# ---------------------------------------------------------------- utilities
def cached(name, compute):
    """Load result `name` from the cache folder, or compute and save it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    result = compute()
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def parallel_map(function, arguments, workers=None):
    """Apply function to each argument tuple in worker processes, preserving order."""
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        return list(pool.map(function, *zip(*arguments)))


# ---------------------------------------------------------------- experiments
RISK_SETTINGS = {"r = 1": 1.0, "r = 0.5": 0.5,
                 "r ~ U(0.2, 1)": lambda g, n: g.uniform(0.2, 1, n),
                 "r ~ U(0, 1)": lambda g, n: g.uniform(0, 1, n)}


def fixed_belief_market(seed, r_kind, periods=80):
    """
    Beliefs held fixed (stubbornness 1, no signal noise), so only trading moves reservation prices.
    Valuations ~ N(0.6, 0.08). Returns the price path, the dispersion of reservation prices over
    time, the bid-ask spread, and the exact and linearised competitive-equilibrium prices.
    """
    population = make_population(seed, s=1.0, r=RISK_SETTINGS[r_kind],
                                 valuation=lambda g, n: np.clip(g.normal(0.6, 0.08, n), 0.05, 0.95))
    V = np.array([b.market_valuation for b in population])
    B = np.array([b.budget for b in population])
    R = np.array([b.risk_av for b in population])
    out = simulate(population, periods, seeded(seed), record=True)
    z = out["reservation"]
    return {"price": out["price"], "z_spread": z.max(1) - z.min(1),
            "z_iqr": np.subtract(*np.percentile(z, [90, 10], axis=1)),
            "spread": out["spread"], "walras": walrasian_price(V, B, R),
            "linear": float(np.sum(B / R * V) / np.sum(B / R))}


def whale_steady_state(rho, noise, seed, h=0.0, s=0.3, periods=120, burn=40):
    """Time-averaged price error after burn-in, and the whale's mark-to-market wealth share rho_S."""
    population = add_whale(make_population(seed, s=s, h=h, noise=noise), rho)
    out = simulate(population, periods, seeded(seed), record=True)
    p = out["price"][-1]
    wealth = out["cash"][-1] + p * out["holdings"][-1]
    return {"error": float(out["error"][burn:].mean()), "rho_S": float(wealth[-1] / wealth.sum()),
            "error_sd": float(out["error"][burn:].std())}


def whale_entry_path(h, rho=0.3, seed=0, periods=50):
    return simulate(add_whale(make_population(seed, h=h), rho), periods, seeded(seed))["error"]


def belief_shock_path(h, seed=1, periods=40, start=0.7):
    return simulate(make_population(seed, h=h, valuation=start), periods, seeded(seed))["error"]


def recovery_after_shock(rho, s, h, seed, shock=-0.15, settle=50, after=40):
    """Settle (with a whale if rho > 0), shift non-whale valuations by `shock`, return error relative to the settled level."""
    rng = seeded(seed)
    population = make_population(seed, s=s, h=h)
    if rho > 0:
        population = add_whale(population, rho)
    settled = simulate(population, settle, rng)["error"][-1]
    for agent in population:
        if not agent.whale:
            agent.market_valuation += shock
    return simulate(population, after, rng, initial_price=ETA + settled)["error"] - settled


def noise_run(noise, h, seed, periods=220, burn=50):
    """Order-book noise mu_t = m_t - p*_t, where p*_t is the log-utility equilibrium price of the period's valuations."""
    out = simulate(make_population(seed, noise=noise, h=h), periods, seeded(seed), record=True)
    wealth = out["cash"] + out["price"][:, None] * out["holdings"]
    p_eq = (wealth * out["valuation"]).sum(1) / wealth.sum(1)
    mu = (out["price"] - p_eq)[burn:]
    return {"dispersion": float(out["valuation"][burn:].std(1).mean()), "sigma_mu": float(mu.std()),
            "autocorr": float(np.corrcoef(mu[:-1], mu[1:])[0, 1]), "n": len(mu)}


def held_position(active, h, seed=0, rho=0.3, after=40, s=0.3):
    """Whale trades for `active` periods, then stops trading but keeps its position."""
    rng = seeded(seed)
    non_whales = make_population(seed, h=h, s=s)
    population = add_whale(non_whales, rho)
    first = simulate(population, active, rng)
    second = simulate(non_whales, after, rng, initial_price=ETA + first["error"][-1])
    whale = population[-1]
    p = ETA + second["error"][-10:].mean()
    non_whale_wealth = sum(b.budget + p * b.n_contracts for b in non_whales)
    herding_factor = ((1 - h) * (1 - s) + h) / ((1 - h) * (1 - s))
    return {"active": active, "h": h, "position": int(whale.n_contracts),
            "path": np.concatenate([first["error"], second["error"]]),
            "error_after": float(second["error"][-10:].mean()),
            "predicted": float(whale.n_contracts * p * (1 - p) / non_whale_wealth * herding_factor)}


def collaborator_run(rho, seed, population_kind, periods=200, rw_sd=0.05):
    """Whale (valuation W) present from t = 0 in a market whose true outcome follows a random walk."""
    if population_kind == "manuscript":
        population = manuscript_population(seed)
        whale_r = float(np.random.default_rng(seed + 5_000).uniform(0, 1))
    else:
        population = make_population(seed, r=1.0, s=0.3, noise=0.1)
        whale_r = 1.0
    if rho > 0:
        population = add_whale(population, rho, r=whale_r)
    out = simulate(population, periods, seeded(seed), eta_path=random_walk(seed, periods, rw_sd))
    return {"price": out["price"], "eta": out["eta"], "whale_r": whale_r}
