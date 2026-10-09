"""
Tests for the CLOB, agent order generation and collateral accounting.
Run with `python3 -m pytest test_clob.py` or simply `python3 test_clob.py`.
"""
import numpy as np

from bettor import bettor, run_market_lob, worst_case_wealth
from orderbook import OrderBook


def make_bettor(**kwargs):
    defaults = dict(budget=500.0, market_valuation=0.5, n_contracts=0, risk_av=0.5,
                    stubbornness=0.3, expertise=0.9)
    return bettor(**{**defaults, **kwargs})


# ---------------------------------------------------------------- order book

def test_bid_and_ask_price_priority():
    ob = OrderBook()
    for trader, price in enumerate([0.30, 0.50, 0.40]):
        ob.submit(trader, +1, price, 10, t=0, ttl=5)
    for trader, price in enumerate([0.80, 0.60, 0.70], start=10):
        ob.submit(trader, -1, price, 10, t=0, ttl=5)
    assert [o[3] for o in ob.bids] == [0.50, 0.40, 0.30]   # negative-price keys sort highest bid first
    assert [o[3] for o in ob.asks] == [0.60, 0.70, 0.80]
    assert ob.best_bid() == 0.50 and ob.best_ask() == 0.60


def test_time_priority_at_equal_price():
    ob = OrderBook()
    ob.submit(1, -1, 0.60, 5, t=0, ttl=5)
    ob.submit(2, -1, 0.60, 5, t=0, ttl=5)
    fills = ob.submit(3, +1, 0.60, 5, t=0, ttl=5)
    assert fills == [(3, 1, 0.60, 5)]                       # earlier order at the same price fills first
    assert [o[2] for o in ob.asks] == [2]


def test_execution_at_resting_price_and_level_sweep():
    ob = OrderBook()
    ob.submit(1, -1, 0.55, 4, t=0, ttl=5)
    ob.submit(2, -1, 0.58, 4, t=0, ttl=5)
    ob.submit(3, -1, 0.70, 4, t=0, ttl=5)
    fills = ob.submit(9, +1, 0.60, 10, t=0, ttl=5)
    assert fills == [(9, 1, 0.55, 4), (9, 2, 0.58, 4)]      # stops at 0.70 > limit 0.60
    assert ob.best_bid() == 0.60 and ob.depth() == (2, 4)   # 2-contract remainder rests as the best bid
    assert ob.last_price == 0.58


def test_sell_execution_against_bids():
    ob = OrderBook()
    ob.submit(1, +1, 0.45, 3, t=0, ttl=5)
    fills = ob.submit(2, -1, 0.40, 5, t=0, ttl=5)
    assert fills == [(1, 2, 0.45, 3)]                       # seller receives the better (resting) price
    assert ob.best_ask() == 0.40 and ob.best_bid() is None


def test_partial_fill_of_resting_order():
    ob = OrderBook()
    ob.submit(1, +1, 0.50, 10, t=0, ttl=5)
    ob.submit(2, -1, 0.50, 3, t=0, ttl=5)
    assert ob.depth() == (7, 0) and ob.bids[0][2] == 1      # resting order keeps its place with 7 left


def test_book_never_left_crossed_or_locked():
    ob = OrderBook()
    rng = np.random.default_rng(0)
    for trader in range(500):
        side = rng.choice([-1, 1])
        ob.cancel_trader(trader % 50)
        ob.submit(trader % 50, side, ob.round_price(rng.uniform(0.3, 0.7), side), int(rng.integers(1, 20)), t=0, ttl=5)
        if ob.best_bid() is not None and ob.best_ask() is not None:
            assert ob.best_bid() < ob.best_ask()


def test_cancel_and_expiry():
    ob = OrderBook()
    ob.submit(1, +1, 0.40, 5, t=0, ttl=2)                   # rests during periods 0 and 1
    ob.submit(2, -1, 0.60, 5, t=1, ttl=5)
    assert ob.expire(1) == [] and ob.depth() == (5, 5)
    assert [o[2] for o in ob.expire(2)] == [1] and ob.depth() == (0, 5)
    assert [o[2] for o in ob.cancel_trader(2)] == [2] and ob.depth() == (0, 0)


def test_reference_price_fallbacks():
    ob = OrderBook()
    assert ob.mid(fallback=0.5) == 0.5                      # empty book, no trades
    ob.submit(1, +1, 0.40, 5, t=0, ttl=5)
    assert ob.mid(fallback=0.5) == 0.40                     # one-sided: the remaining best quote
    ob.submit(2, -1, 0.60, 5, t=0, ttl=5)
    assert np.isclose(ob.mid(fallback=0.5), 0.50)
    assert np.isnan(OrderBook().market_snapshot()["midquote"])


# ---------------------------------------------------------------- agents

def test_bettors_get_independent_random_traits():
    np.random.seed(0)
    population = [bettor() for _ in range(5)]
    assert len({b.risk_av for b in population}) == 5
    assert len({b.stubbornness for b in population}) == 5


def test_reservation_price_reflects_inventory():
    assert np.isclose(make_bettor(market_valuation=0.6).reservation_price(), 0.6)
    assert make_bettor(market_valuation=0.6, n_contracts=300).reservation_price() < 0.6
    assert make_bettor(market_valuation=0.6, n_contracts=-300).reservation_price() > 0.6


def test_optimiser_buys_below_and_sells_above_reservation_price():
    for q in [-300, 0, 300]:
        b = make_bettor(market_valuation=0.6, n_contracts=q)
        r = b.reservation_price()
        assert b.trade(r - 0.05) > 0 and b.trade(r + 0.05) < 0


def test_limit_order_side_matches_optimiser():
    # Values YES above the reference price but holds a large YES position: utility says sell.
    b = make_bettor(budget=200.0, market_valuation=0.6, n_contracts=600, risk_av=0.8)
    ref = 0.55
    assert b.trade(ref) < 0
    side, price, qty = b.limit_order(ref, OrderBook(), np.random.default_rng(0))
    assert side == -1 and price > b.reservation_price() and qty > 0


# ---------------------------------------------------------------- collateral and short (NO) exposure

def test_selling_from_flat_opens_fully_collateralised_short():
    b = make_bettor(budget=100.0)
    b.n_contracts -= 50
    b.budget += 50 * 0.40                                   # cash received when selling
    assert b.budget + b.n_contracts == 70.0                 # W_yes = C + Q: pays 1 per contract sold
    assert b.budget == 120.0                                # W_no  = C
    assert worst_case_wealth(b) >= 0


def test_trade_is_always_collateralised_at_any_fill_price():
    """trade(p) at the limit price stays feasible for every partial fill at the limit price or better."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = make_bettor(budget=rng.uniform(1, 1000), market_valuation=rng.uniform(0.01, 0.99),
                        risk_av=rng.uniform(0, 1))
        b.n_contracts = int(rng.integers(-int(b.budget), int(b.budget)))   # any position with W_yes >= 0
        assert worst_case_wealth(b) >= 0
        p = round(rng.uniform(0.01, 0.99), 2)
        q = b.trade(p)
        side = int(np.sign(q))
        for filled in [q, q // 2]:
            for fill_price in [p, p - side * 0.01]:              # limit price, or one tick better
                assert worst_case_wealth(b, filled, np.clip(fill_price, 0.01, 0.99)) >= -1e-9


def test_buying_to_cover_and_selling_a_long_position():
    b = make_bettor(budget=100.0, n_contracts=-50, market_valuation=0.9)    # short, now thinks YES likely
    assert b.trade(0.5) > 0                                                  # wants to buy back
    c = make_bettor(budget=100.0, n_contracts=200, market_valuation=0.1)     # long, now thinks YES unlikely
    assert c.trade(0.5) < 0 and worst_case_wealth(c, c.trade(0.5), 0.5) >= 0


# ---------------------------------------------------------------- full simulation invariants

def test_simulation_conserves_cash_and_contracts_and_keeps_one_order_per_trader():
    np.random.seed(3)
    rng = np.random.default_rng(3)
    population = [bettor(budget=rng.uniform(100, 1000), market_valuation=rng.normal(0.5, 0.05)) for _ in range(30)]
    cash_before = sum(b.budget for b in population)
    result = run_market_lob(30, 30, 0.5, 0.05, population, rng=rng)   # asserts conservation every period
    assert np.isclose(sum(b.budget for b in population), cash_before)
    assert sum(b.n_contracts for b in population) == 0
    assert result["traded_volume"].sum() > 0
    assert np.all(result["n_collateral_rejected"] == 0)


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print("passed", test.__name__)
    print(f"all {len(tests)} tests passed")
