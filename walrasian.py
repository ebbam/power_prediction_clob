"""
The original market mechanism, kept for comparison with the CLOB
(experiments/compare_mechanism_distortion.py): every period all bettors submit their
utility-optimal trade at one common price, the short side of the market is filled in full,
the long side is rationed in random order, and the price moves with the order imbalance.
"""
import numpy as np


def clear_orders(orders):
    """
    Fill signed orders (positive buys, negative sells) at one price. The side with less volume
    fills in full; the other side's orders fill in random order until the volume runs out.
    """
    orders = np.asarray(orders)
    buys, sells = orders > 0, orders < 0
    demand, supply = np.sum(orders[buys]), -np.sum(orders[sells])
    filled = orders.astype(float)
    if demand > supply:
        filled[buys] = _ration(orders[buys], supply)
    elif supply > demand:
        filled[sells] = _ration(orders[sells], demand)
    return filled


def _ration(orders, available):
    """Fill same-signed orders in random order until `available` contracts are used up."""
    filled = np.zeros(len(orders))
    queue = np.random.permutation(len(orders))
    cumulative = np.cumsum(np.abs(orders[queue]))
    n_full = int(np.sum(cumulative < available))
    filled[queue[:n_full]] = orders[queue[:n_full]]
    partly_filled = queue[n_full]
    already_filled = cumulative[n_full - 1] if n_full else 0
    filled[partly_filled] = np.sign(orders[partly_filled]) * (available - already_filled)
    return filled


def run_walrasian_market(bettors, n_periods, fundamentals, initial_price, market_lambda=0.1):
    """Run the original market; the price moves by market_lambda x net demand / order volume."""
    price = initial_price
    price_history, market_pressure = [price], [0]
    for t in range(n_periods):
        orders = np.array([b.trade(price) for b in bettors])
        for b, qty in zip(bettors, clear_orders(orders)):
            b.n_contracts += qty
            b.budget -= qty * price
        net, volume = np.sum(orders), np.sum(np.abs(orders))
        market_pressure.append(net / volume if volume else 0)
        if volume > 0:
            price = np.clip(price + market_lambda * net / volume, 0, 1)
        for b in bettors:
            b.update_belief(fundamentals[t], price)
        price_history.append(price)
    return {"price_history": price_history, "market_pressure": market_pressure,
            "fundamental": np.asarray(fundamentals)}
