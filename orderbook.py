"""
Central limit order book (CLOB) for a binary prediction-market contract.

Design follows the continuous-double-auction ABM tradition
(Chiarella & Iori 2002; Chiarella, Iori & Perello 2009; Farmer, Patelli & Zovko 2005):
  * price-time priority; trades execute at the resting order's price
  * discrete tick (prediction markets quote in cents), prices in [tick, 1 - tick]
  * unfilled remainders rest in the book until filled, cancelled, or expired
  * one live order per trader (cancel-and-replace on re-activation), which keeps
    every resting order budget-feasible without a separate escrow ledger
"""
import bisect
import numpy as np


class OrderBook:
    def __init__(self, tick=0.01):
        self.tick = tick
        # each side is a list of [sort_key, seq, trader_id, price, qty, expiry]
        # sort_key = -price for bids (highest first), +price for asks (lowest first)
        self.bids, self.asks = [], []
        self.seq = 0
        self.last_price = None

    # ---------- quotes ----------
    def best_bid(self):
        return self.bids[0][3] if self.bids else None

    def best_ask(self):
        return self.asks[0][3] if self.asks else None

    def mid(self, fallback):
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None:
            return 0.5 * (bb + ba)
        return self.last_price if self.last_price is not None else fallback

    def spread(self):
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) if (bb is not None and ba is not None) else np.nan

    def depth(self):
        return sum(o[4] for o in self.bids), sum(o[4] for o in self.asks)

    # ---------- order management ----------
    def round_price(self, p, side):
        """Buys round down, sells round up (never quote through your own valuation)."""
        n = np.floor(p / self.tick + 1e-9) if side > 0 else np.ceil(p / self.tick - 1e-9)
        return float(np.clip(n * self.tick, self.tick, 1 - self.tick))

    def cancel_trader(self, trader_id):
        self.bids = [o for o in self.bids if o[2] != trader_id]
        self.asks = [o for o in self.asks if o[2] != trader_id]

    def expire(self, t):
        self.bids = [o for o in self.bids if o[5] > t]
        self.asks = [o for o in self.asks if o[5] > t]

    def submit(self, trader_id, side, price, qty, t, ttl):
        """
        side = +1 buy / -1 sell, price = limit price, qty > 0 (integer contracts).
        Matches against the opposite side while prices cross; remainder rests.
        Returns list of fills: (buyer_id, seller_id, price, qty).
        """
        fills = []
        book = self.asks if side > 0 else self.bids
        crosses = (lambda p: p <= price) if side > 0 else (lambda p: p >= price)
        while qty > 0 and book and crosses(book[0][3]):
            resting = book[0]
            q = min(qty, resting[4])
            buyer, seller = (trader_id, resting[2]) if side > 0 else (resting[2], trader_id)
            fills.append((buyer, seller, resting[3], q))
            self.last_price = resting[3]
            qty -= q
            resting[4] -= q
            if resting[4] == 0:
                book.pop(0)
        if qty > 0:
            own = self.bids if side > 0 else self.asks
            key = -price if side > 0 else price
            entry = [key, self.seq, trader_id, price, qty, t + ttl]
            bisect.insort(own, entry, key=lambda o: (o[0], o[1]))  # Python >= 3.10
            self.seq += 1
        return fills
