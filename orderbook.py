"""
Central limit order book (CLOB) for a binary prediction-market contract.

Design follows the continuous-double-auction ABM tradition
(Chiarella & Iori 2002; Chiarella, Iori & Perello 2009; Farmer, Patelli & Zovko 2005):
  * price-time priority; trades execute at the resting order's price
  * discrete tick (prediction markets quote in cents), prices in [tick, 1 - tick]
  * unfilled remainders rest in the book until filled, cancelled, or expired
  * one live order per trader (cancelled whenever the trader is re-activated), which keeps
    every resting order budget-feasible without a separate escrow ledger: a trader's cash and
    position change only through fills of that one order, at its limit price or better
"""
import bisect
from dataclasses import dataclass, field

import numpy as np


@dataclass(order=True)
class Order:
    """A resting limit order. Orders sort by priority: best price first, then oldest first."""
    priority: float                      # -price for bids (highest first), +price for asks
    seq: int                             # submission order, for time priority
    trader: int = field(compare=False)
    price: float = field(compare=False)
    qty: int = field(compare=False)
    expiry: int = field(compare=False)   # first period in which the order no longer rests


class OrderBook:
    def __init__(self, tick=0.01):
        self.tick = tick
        self.bids, self.asks = [], []    # lists of Order, each kept in priority order
        self.seq = 0
        self.last_price = None

    # ---------- quotes ----------
    def best_bid(self):
        return self.bids[0].price if self.bids else None

    def best_ask(self):
        return self.asks[0].price if self.asks else None

    def midquote(self):
        """Midpoint of the best bid and ask; NaN unless both sides have orders."""
        bid, ask = self.best_bid(), self.best_ask()
        return 0.5 * (bid + ask) if bid is not None and ask is not None else np.nan

    def spread(self):
        bid, ask = self.best_bid(), self.best_ask()
        return ask - bid if bid is not None and ask is not None else np.nan

    def mid(self, fallback):
        """
        Reference price for agents: the midquote if both sides exist; otherwise the one remaining
        best quote (the only price anyone can currently trade at); otherwise the last trade;
        otherwise fallback. A stale last trade is used only when the book is empty.
        """
        for price in (self.midquote(), self.best_bid(), self.best_ask(), self.last_price):
            if price is not None and not np.isnan(price):
                return price
        return fallback

    def depth(self):
        return sum(o.qty for o in self.bids), sum(o.qty for o in self.asks)

    # ---------- order management ----------
    def round_price(self, p, side):
        """Buys round down, sells round up (never quote through your own valuation)."""
        n = np.floor(p / self.tick + 1e-9) if side > 0 else np.ceil(p / self.tick - 1e-9)
        return float(np.clip(n * self.tick, self.tick, 1 - self.tick))

    def _remove(self, should_remove):
        """Remove every resting order for which should_remove(order) is True; return them."""
        removed = [o for o in self.bids + self.asks if should_remove(o)]
        self.bids = [o for o in self.bids if not should_remove(o)]
        self.asks = [o for o in self.asks if not should_remove(o)]
        return removed

    def cancel_trader(self, trader):
        return self._remove(lambda o: o.trader == trader)

    def expire(self, t):
        """An order submitted at t0 with time-to-live ttl rests during periods t0, ..., t0 + ttl - 1."""
        return self._remove(lambda o: o.expiry <= t)

    def submit(self, trader, side, price, qty, t, ttl):
        """
        side = +1 buy / -1 sell, price = limit price, qty > 0 (integer contracts).
        Matches against the opposite side while prices cross; the remainder rests.
        Returns the list of fills: (buyer, seller, price, qty).
        """
        fills = []
        opposite = self.asks if side > 0 else self.bids
        crosses = (lambda p: p <= price) if side > 0 else (lambda p: p >= price)
        while qty > 0 and opposite and crosses(opposite[0].price):
            resting = opposite[0]
            q = min(qty, resting.qty)
            buyer, seller = (trader, resting.trader) if side > 0 else (resting.trader, trader)
            fills.append((buyer, seller, resting.price, q))
            self.last_price = resting.price
            qty -= q
            resting.qty -= q
            if resting.qty == 0:
                opposite.pop(0)
        if qty > 0:
            own = self.bids if side > 0 else self.asks
            priority = -price if side > 0 else price
            bisect.insort(own, Order(priority, self.seq, trader, price, qty, t + ttl))
            self.seq += 1
        return fills

    def market_snapshot(self):
        """The current state of the book. Missing quotes are NaN, never stale trade prices."""
        bid_depth, ask_depth = self.depth()
        total_depth = bid_depth + ask_depth
        nan_if_none = lambda x: np.nan if x is None else x
        return {
            "best_bid": nan_if_none(self.best_bid()),
            "best_ask": nan_if_none(self.best_ask()),
            "midquote": self.midquote(),
            "spread": self.spread(),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": total_depth,
            "book_imbalance": (bid_depth - ask_depth) / total_depth if total_depth else np.nan,
            "last_trade": nan_if_none(self.last_price),
        }
