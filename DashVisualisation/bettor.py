"""
The model: agents (Bettor) trading a binary YES contract in a central limit order book
(run_market_lob).

Every random draw that shapes beliefs comes from the global np.random generator (traits left
unset in Bettor and the signals in update_belief); the order of activation and the price
shading come from the Generator passed to run_market_lob. Seed both for reproducible runs.
"""
from collections import defaultdict

import numpy as np

from orderbook import OrderBook


class Bettor:
    """
    An agent with CRRA utility trading a contract that pays 1 if YES occurs.

    budget            cash (initial_budget keeps the cash the bettor started with)
    n_contracts       YES position; negative is a short (NO) position
    market_valuation  belief that YES occurs
    risk_av           CRRA coefficient r (r = 1 is log utility)
    stubbornness      weight kept on the current belief when a signal arrives
    expertise         signals about the fundamental have standard deviation 1 - expertise
    market_imitation  weight given to the market price in each belief update (herding)
    bias              subtracted from every signal
    whale             a whale never updates its belief
    entry_time        first period in which the agent trades

    Traits left as None are drawn here, per bettor. (Defaults written in the signature would be
    evaluated once, when the class is defined, so every bettor would share the same draw.)
    """

    def __init__(self, budget=None, market_valuation=None, n_contracts=0, risk_av=None,
                 stubbornness=None, expertise=None, market_imitation=0, bias=0, whale=False,
                 entry_time=0):
        if budget is None:
            budget = np.random.uniform(100, 1000)
        if market_valuation is None:
            market_valuation = np.random.normal(0.5, 0.05)
        if risk_av is None:
            risk_av = np.random.uniform(0, 1)
        if stubbornness is None:
            stubbornness = np.clip(np.random.normal(0.3, 0.05), 0, 1)
        if expertise is None:
            expertise = np.clip(np.random.normal(0.9, 0.04), 0.8, 1)
        self.budget = self.initial_budget = budget
        self.market_valuation = market_valuation
        self.n_contracts = n_contracts
        self.risk_av = risk_av
        self.stubbornness = stubbornness
        self.expertise = expertise
        self.market_imitation = market_imitation
        self.bias = bias
        self.whale = whale
        self.entry_time = entry_time

    # ---------- preferences ----------
    def utility(self, w):
        """CRRA utility of wealth w."""
        if self.risk_av == 1:
            return np.log(w)
        return w ** (1 - self.risk_av) / (1 - self.risk_av)

    def exp_utility(self, price, new_c):
        """Expected utility of trading new_c contracts at price (new_c < 0 sells)."""
        cash = self.budget - price * new_c
        return (self.market_valuation * self.utility(cash + self.n_contracts + new_c)
                + (1 - self.market_valuation) * self.utility(cash))

    def trade(self, price):
        """
        Utility-maximising trade at price, over whole contracts. The range searched keeps the
        position collateralised whether the event resolves YES or NO.
        """
        c_range = np.arange(-(int(self.budget) + self.n_contracts), int(self.budget) + 1)
        return c_range[np.argmax(self.exp_utility(price, c_range))]

    def reservation_price(self):
        """
        Price at which a marginal trade leaves expected utility unchanged: the solution of
        d exp_utility / d new_c = 0 at new_c = 0, i.e.
            r = v u'(W_yes) / (v u'(W_yes) + (1 - v) u'(W_no)),  with u'(w) = w^(-risk_av),
        where W_yes = budget + n_contracts and W_no = budget. With no holdings r = market_valuation.
        A long YES position lowers r and a short position raises it, because risk aversion makes
        the agent value an extra dollar less in the state where it is already richer.
        The optimiser trade(p) buys when p < r and sells when p > r (up to integer rounding).
        """
        v = self.market_valuation
        w_yes, w_no = self.budget + self.n_contracts, self.budget
        with np.errstate(divide="ignore", invalid="ignore"):
            return v / (v + (1 - v) * (w_yes / w_no) ** self.risk_av)

    def worst_case_wealth(self, qty=0, price=0.0):
        """
        Terminal wealth in the worse outcome after trading qty contracts at price.
        YES pays 1: W_yes = C + Q. NO pays 0: W_no = C. A position is fully collateralised
        if and only if this is >= 0.
        """
        cash = self.budget - qty * price
        return min(cash + (self.n_contracts + qty), cash)

    # ---------- decisions ----------
    def limit_order(self, ref_price, ob, rng, max_markup=0.05):
        """
        CLOB order decision (Chiarella & Iori 2002 style), re-using the CRRA demand.
        Side: buy if the reservation price exceeds the reference price, sell if below. This is the
        sign of the utility-optimal trade at the reference price, so it accounts for holdings.
        Limit price: reservation price shaded by a random markup k ~ U(0, max_markup), so buyers bid
        below and sellers ask above the price at which they are indifferent.
        Quantity: trade() evaluated AT the limit price, so any fill (always at the limit price or
        better) is one the bettor is happy with.
        Returns (side, limit_price, qty), or None for no order.
        """
        r = self.reservation_price()
        if np.isnan(r) or r == ref_price:
            return None
        side = int(np.sign(r - ref_price))
        price = ob.round_price(r - side * rng.uniform(0, max_markup), side)
        qty = self.trade(price)
        if np.sign(qty) != side:
            return None
        return side, price, int(abs(qty))

    def update_belief(self, true_value, market_price):
        """
        Move the belief toward a noisy, biased signal of true_value by 1 - stubbornness, then,
        with herding, a share market_imitation of the way to the market price. Whales never update.
        """
        if self.whale:
            return
        signal = np.random.normal(true_value, 1 - self.expertise) - self.bias
        self.market_valuation += (1 - self.stubbornness) * (signal - self.market_valuation)
        if self.market_imitation > 0:
            self.market_valuation = ((1 - self.market_imitation) * self.market_valuation
                                     + self.market_imitation * market_price)
        self.market_valuation = np.clip(self.market_valuation, 0, 1)


# ---------- populations and fundamentals ----------
def whale_budget(population, share):
    """Cash that gives a whale `share` of all money once it joins `population`."""
    if not np.isfinite(share) or not 0 <= share < 1:
        raise ValueError("Whale share must be finite and in [0, 1)")
    return sum(b.budget for b in population) * share / (1 - share)


def add_whale(population, share, valuation, risk_av=0.5, entry_time=0, **traits):
    """
    Append a whale holding `share` of all money, valued at `valuation` and trading from
    `entry_time`, to `population`. Returns the whale, or None (and adds nothing) when share = 0.
    """
    if not 0 <= valuation <= 1:
        raise ValueError("Whale valuation must be in [0, 1]")
    if share == 0:
        return None
    whale = Bettor(budget=whale_budget(population, share), market_valuation=valuation,
                   risk_av=risk_av, whale=True, entry_time=entry_time, **traits)
    population.append(whale)
    return whale


def fundamental_path(start, n_steps, sd, rng=np.random):
    """A random walk of n_steps + 1 values, clipped to [0, 1]. rng defaults to np.random."""
    path = [start]
    for _ in range(n_steps):
        path.append(np.clip(path[-1] + rng.normal(0, sd), 0, 1))
    return np.asarray(path)


# ---------- the market ----------
def run_market_lob(bettors, n_periods, fundamentals, initial_price, tick=0.01, ttl=5,
                   max_markup=0.05, price_rule="mid", rng=None):
    """
    Run the prediction market as a persistent central limit order book (CLOB).

    Each period: expire old orders, activate the agents that have entered (entry_time <= t) one
    at a time in random order, record the end-of-period book, then update those agents' beliefs
    from fundamentals[t]. An activated agent cancels its resting order, observes the live book
    (including orders placed earlier in the same period), and may submit one new order. Because
    every agent is activated every period, an order rests for at most one period unless ttl = 1
    removes it sooner.

    initial_price is the reference price before the first trade. price_rule sets the reference
    price carried between periods: "mid" (OrderBook.mid, with its fallbacks) or "last" (the
    last trade). Per-period series (length n_periods) are returned alongside the reference
    price and belief series (length n_periods + 1). The recorded midquote is NaN whenever one
    side of the book is empty; whale_contracts is the whales' net YES position after each
    period (negative when they are short, i.e. hold NO), and whale_order_size the size of the
    orders whales submitted that period (positive to buy, negative to sell).
    """
    if rng is None:
        rng = np.random.default_rng()
    if price_rule not in ("mid", "last"):
        raise ValueError("price_rule must be 'mid' or 'last'")
    if ttl < 1:
        raise ValueError("ttl must be >= 1")
    if not 0 <= initial_price <= 1:
        raise ValueError("initial_price must be in [0, 1]")
    fundamentals = np.asarray(fundamentals, dtype=float)
    if len(fundamentals) < n_periods:
        raise ValueError("fundamentals must contain at least n_periods values")

    ob = OrderBook(tick=tick)
    mkt_price = float(initial_price)
    # Trades only move cash and contracts between agents, so both totals are conserved.
    total_cash = sum(b.budget for b in bettors)
    total_contracts = sum(b.n_contracts for b in bettors)

    def active(t):
        return [i for i, b in enumerate(bettors) if b.entry_time <= t]

    price_history = [mkt_price]
    beliefs, beliefs_weighted, vol_history = [], [], []

    def record_beliefs(t):
        agents = [bettors[i] for i in active(t)]
        valuations = [b.market_valuation for b in agents]
        beliefs.append(np.mean(valuations))
        beliefs_weighted.append(np.average(valuations, weights=[b.budget for b in agents]))
        vol_history.append(sum(abs(b.n_contracts) for b in bettors))

    record_beliefs(0)
    history = defaultdict(list)   # one entry per period for every diagnostic below

    for t in range(n_periods):
        period = _empty_period_counters(n_expired=len(ob.expire(t)))
        agents = np.asarray(active(t))

        for i in agents[rng.permutation(len(agents))]:
            agent = bettors[i]
            # The agent withdraws its resting order and re-decides from scratch, so its only
            # live order reflects its current beliefs and holdings.
            period["n_cancelled"] += len(ob.cancel_trader(i))
            period["n_one_sided_reference"] += ob.best_bid() is None or ob.best_ask() is None
            ref = ob.mid(mkt_price)

            desired = agent.trade(ref)
            period["n_want_buy"] += desired > 0
            period["n_want_sell"] += desired < 0

            order = agent.limit_order(ref, ob, rng, max_markup)
            if order is None:
                period["n_desired_trade_dropped"] += desired != 0
                continue
            side, price, qty = order
            period["n_side_flipped"] += desired != 0 and side != np.sign(desired)
            if agent.worst_case_wealth(side * qty, price) < 0:
                period["n_collateral_rejected"] += 1
                continue

            period["n_buy_orders" if side > 0 else "n_sell_orders"] += 1
            if agent.whale:
                period["whale_order_size"] += side * qty
            period["submitted_buy_volume" if side > 0 else "submitted_sell_volume"] += qty

            for buyer, seller, fill_price, fill_qty in ob.submit(i, side, price, qty, t, ttl):
                bettors[buyer].n_contracts += fill_qty
                bettors[buyer].budget -= fill_qty * fill_price
                bettors[seller].n_contracts -= fill_qty
                bettors[seller].budget += fill_qty * fill_price
                period["traded_volume"] += fill_qty
                period["n_trades"] += 1

        snapshot = ob.market_snapshot()
        assert not snapshot["spread"] <= 0, "book left crossed or locked"
        assert sum(b.n_contracts for b in bettors) == total_contracts, "contracts not conserved"
        assert np.isclose(sum(b.budget for b in bettors), total_cash), "cash not conserved"
        assert min(b.worst_case_wealth() for b in bettors) >= -1e-9, "uncollateralised position"
        snapshot["whale_contracts"] = sum(b.n_contracts for b in bettors if b.whale)
        for key, value in {**period, **snapshot}.items():
            history[key].append(value)

        # Reference price agents see between periods, and the price used in belief updates.
        if price_rule == "mid":
            mkt_price = ob.mid(mkt_price)
        elif np.isfinite(snapshot["last_trade"]):
            mkt_price = snapshot["last_trade"]

        for i in agents:
            bettors[i].update_belief(fundamentals[t], mkt_price)
        price_history.append(mkt_price)
        record_beliefs(t)

    history = {key: np.asarray(values) for key, values in history.items()}
    submitted = history["submitted_buy_volume"] + history["submitted_sell_volume"]
    net_supply = history["submitted_buy_volume"] - history["submitted_sell_volume"]
    with np.errstate(invalid="ignore", divide="ignore"):
        pressure = np.where(submitted > 0, net_supply / submitted, 0.0)

    return {
        "price_history": price_history,
        "beliefs": beliefs,
        "weighted_beliefs": beliefs_weighted,
        "vol_history": vol_history,
        "fundamental": fundamentals,
        "submitted_volume": submitted,
        "net_supply": net_supply,
        "market_pressure": np.concatenate([[0.0], pressure]),
        "final_budgets": [b.budget for b in bettors],
        **history,
    }


def _empty_period_counters(n_expired):
    """Diagnostic counters for one period."""
    return {
        "submitted_buy_volume": 0, "submitted_sell_volume": 0,
        "n_buy_orders": 0, "n_sell_orders": 0,
        "traded_volume": 0, "n_trades": 0,
        "n_cancelled": 0, "n_expired": n_expired,
        # What each agent's utility optimiser wants at the reference price it observes,
        # and how limit_order() translated that into an order.
        "n_want_buy": 0, "n_want_sell": 0,
        "n_desired_trade_dropped": 0,      # optimiser wants to trade, limit_order() returned None
        "n_side_flipped": 0,               # order side opposite to the optimiser's desired trade
        "n_collateral_rejected": 0,        # order would leave worst-case wealth negative
        "n_one_sided_reference": 0,        # activations where no midquote existed
        "whale_order_size": 0,             # signed size of the whales' orders (+ buy, - sell)
    }
