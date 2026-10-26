"""
Small deterministic market showing that buyers and sellers meet in the CLOB.
Run from the repository root with `python3 -m tests.diagnostic_simulation`.

Two optimists (valuation 0.70) and two pessimists (valuation 0.30), all starting flat with the
same cash and risk aversion. Beliefs are frozen (stubbornness = 1) and the random price shading
comes from a seeded generator, so the run is identical every time. (Shading cannot be zero: an
order priced exactly at the reservation price has a utility-optimal quantity of zero.)
    * optimists bid, pessimists ask, from the first period;
    * trades execute at the resting order's price, between 0.30 and 0.70;
    * optimists end long YES, pessimists end short YES (NO exposure), with no one's
      worst-case terminal wealth negative and cash and contracts conserved;
    * as inventories build, reservation prices move towards each other (0.70/0.30 -> about 0.5)
      and trading dies down.
"""
import numpy as np

from bettor import bettor, run_market_lob, worst_case_wealth

valuations = [0.70, 0.70, 0.30, 0.30]
population = [bettor(budget=100.0, market_valuation=v, risk_av=0.5, stubbornness=1.0, expertise=1.0)
              for v in valuations]

result = run_market_lob(n_bettors=4, t_election=6, initial_price=0.5, outcome_uncertainty=0.0,
                        bettors=population, gen_el=np.full(6, 0.5), max_markup=0.10,
                        rng=np.random.default_rng(0))

print("period  buy_orders  sell_orders  traded  best_bid  best_ask  midquote")
for t in range(6):
    print(f"{t:>6}  {result['n_buy_orders'][t]:>10}  {result['n_sell_orders'][t]:>11}  {result['traded_volume'][t]:>6}"
          f"  {result['best_bid'][t]:>8.2f}  {result['best_ask'][t]:>8.2f}  {result['midquote'][t]:>8.3f}")

print("\nvaluation  reservation  cash     YES position  W_yes    W_no")
for b in population:
    print(f"{b.market_valuation:>9.2f}  {b.reservation_price():>11.3f}  {b.budget:>7.2f}  {b.n_contracts:>12}"
          f"  {b.budget + b.n_contracts:>7.2f}  {b.budget:>7.2f}")

optimists, pessimists = population[:2], population[2:]
assert result["n_buy_orders"][0] == 2 and result["n_sell_orders"][0] == 2
assert result["traded_volume"][0] > 0
assert all(b.n_contracts > 0 for b in optimists) and all(b.n_contracts < 0 for b in pessimists)
assert sum(b.n_contracts for b in population) == 0
assert np.isclose(sum(b.budget for b in population), 400.0)
assert min(worst_case_wealth(b) for b in population) >= 0
print("\nAll checks passed: buyers and sellers interact as intended.")
