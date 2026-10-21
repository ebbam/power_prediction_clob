"""
Simulation for the position-limit policy app (policy_visualisation.py).

A whale enters after BASELINE periods and buys (or sells, if its valuation W is below the true
outcome) until it holds its target position, never paying more than W per contract. It then holds
the position without trading, as a buy-and-hold manipulator would until the event resolves.
A position limit caps the number of contracts any trader, including the whale, may hold.

Every bettor has log utility (r = 1), where the steady-state theory is exact. The true outcome is
fixed at ETA. Each run reports the price-error path, the whale's position over time and the price
it paid. Money is in simulation units; a contract pays 1 if the event happens.
"""
import numpy as np

from market_functions import bettor, run_market_lob

N_BETTORS = 100
ETA = 0.5
BASELINE, WHALE_PERIODS = 20, 80
STEADY_WINDOW = 30


class LimitedBettor(bettor):
    """An ordinary bettor whose holdings must stay within +/- position_limit contracts."""

    def __init__(self, position_limit=None, **kwargs):
        super().__init__(**kwargs)
        self.position_limit = position_limit

    def trade(self, m):
        lower, upper = -(int(self.budget) + self.n_contracts), int(self.budget)
        if self.position_limit is not None:
            lower = max(lower, -self.position_limit - self.n_contracts)
            upper = min(upper, self.position_limit - self.n_contracts)
        if upper < lower:
            return 0
        c_range = np.arange(lower, upper + 1)
        return c_range[np.argmax(self.exp_utility(m, c_range))]


class BuyAndHoldWhale(bettor):
    """
    Each period, bids (or offers) for the rest of its target position at limit price W, then holds.
    Its cash covers the full position at W, so every order is fully collateralised.
    """

    def __init__(self, target, valuation, position_limit=None):
        self.direction = 1 if valuation >= ETA else -1
        self.target = target if position_limit is None else min(target, position_limit)
        cost_per_contract = valuation if self.direction > 0 else 1 - valuation
        super().__init__(budget=self.target * cost_per_contract + 1.0, market_valuation=valuation,
                         risk_av=1.0, stubbornness=1.0, expertise=1.0, whale=True)
        self.initial_budget = self.budget
        self.position_history = []

    def limit_order(self, ref_price, ob, rng, max_markup=0.05):
        remaining = self.target - self.direction * self.n_contracts
        if remaining <= 0:
            return None
        return self.direction, ob.round_price(self.market_valuation, self.direction), int(remaining)

    def trade(self, m):                     # only used by the market's diagnostics
        return 0

    def update_belief(self, *args, **kwargs):
        self.position_history.append(self.n_contracts)    # called once per period; beliefs stay fixed


def run_once(seed, target, whale_valuation, position_limit):
    """One run: BASELINE periods without the whale, then WHALE_PERIODS with the buy-and-hold whale."""
    np.random.seed(seed)                   # private signals
    rng = np.random.default_rng(seed)      # attributes and order placement
    bettors = [LimitedBettor(position_limit=position_limit,
                             budget=rng.uniform(100, 1000),
                             market_valuation=np.clip(rng.normal(ETA, 0.05), 0, 1),
                             risk_av=1.0,
                             stubbornness=np.clip(rng.normal(0.3, 0.05), 0, 1),
                             expertise=np.clip(rng.normal(0.95, 0.02), 0.8, 1))
               for _ in range(N_BETTORS)]
    others_wealth = sum(b.budget for b in bettors)
    baseline = run_market_lob(N_BETTORS, BASELINE, ETA, 0.0, bettors, gen_el=np.full(BASELINE, ETA), rng=rng)
    whale = BuyAndHoldWhale(target, whale_valuation, position_limit)
    start_price = baseline["price_history"][-1]
    with_whale = run_market_lob(N_BETTORS + 1, WHALE_PERIODS, start_price, 0.0, bettors + [whale],
                                gen_el=np.full(WHALE_PERIODS, ETA), rng=rng)
    held = abs(whale.n_contracts)
    paid = abs(whale.initial_budget - whale.budget)
    return {"error": np.concatenate([baseline["price_history"][1:], with_whale["price_history"][1:]]) - ETA,
            "position": np.concatenate([np.zeros(BASELINE), np.abs(whale.position_history)]),
            "target": whale.target, "held": held, "others_wealth": others_wealth,
            "direction": whale.direction, "valuation": whale_valuation,
            "average_price": paid / held if held else np.nan}


def log_utility_benchmark(held, others_wealth, whale_valuation):
    """
    Steady-state error when the others (log utility) must hold the opposite of the whale's position:
    |p - eta| = Q p (1 - p) / (others' wealth), solved for p, and never beyond the whale's price limit W.
    """
    direction = 1 if whale_valuation >= ETA else -1
    delta = 0.0
    for _ in range(50):
        p = ETA + direction * delta
        delta = held * p * (1 - p) / others_wealth
    return direction * min(delta, abs(whale_valuation - ETA))


def summarise(runs, band, tolerance=0.99):
    """
    band: ordinary noise band, the 95th percentile of |error| before the whale enters (noise_band).
    Over the whale periods, for each run:
      steady    mean error over the last STEADY_WINDOW periods
      max_error maximum |error| (the collaborator's P1)
      above     share of periods with |error| above the band (the collaborator's P3)
      build     periods after entry until the whale holds `tolerance` of its target (NaN if never)
      if_yes, if_no  the whale's profit if the event does / does not happen
      final_above    1 if the final market price m_T is above 1/2, 0 if below, 1/2 if exactly 1/2
                     (prices move in ticks, so m_T = 1/2 is common without a whale)
    """
    errors = np.array([r["error"] for r in runs])
    after = errors[:, BASELINE:]
    build = []
    for r in runs:
        done = np.flatnonzero(r["position"][BASELINE:] >= tolerance * r["target"])
        build.append(float(done[0] + 1) if len(done) else np.nan)
    # A buyer paid `price` per contract and is paid 1 if the event happens; a seller is the reverse
    direction = runs[0]["direction"]
    price = np.array([r["average_price"] if r["held"] else 0.0 for r in runs])
    held = np.array([r["held"] for r in runs])
    if_yes = held * (1 - price) if direction > 0 else -held * (1 - price)
    if_no = -held * price if direction > 0 else held * price
    return {"errors": errors, "positions": np.array([r["position"] for r in runs]),
            "steady": errors[:, -STEADY_WINDOW:].mean(1), "max_error": np.abs(after).max(1),
            "above": (np.abs(after) > band).mean(1), "build": np.array(build),
            "final_above": (np.sign(np.round(errors[:, -1] + ETA - 0.5, 9)) + 1) / 2,
            "held": held, "target": runs[0]["target"], "if_yes": if_yes, "if_no": if_no,
            "others_wealth": float(np.mean([r["others_wealth"] for r in runs])),
            "benchmark": float(np.mean([log_utility_benchmark(r["held"], r["others_wealth"], r["valuation"]) for r in runs]))}


def noise_band(runs):
    """95th percentile of |error| over the baseline periods (skipping the first 5) of all runs."""
    return float(np.percentile(np.abs(np.array([r["error"][5:BASELINE] for r in runs])), 95))
