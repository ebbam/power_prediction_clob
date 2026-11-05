"""Construct experiment whales before handing them to the market simulator."""

import numpy as np

from bettor import bettor


def scheduled_whale(population, share, start, valuation, risk_av=1.0, agent_type=bettor):
    """Return an entry schedule and the whale's initial budget.

    ``share`` is the whale's fraction of total initial cash after entry.
    """
    if not np.isfinite(share) or not 0 <= share < 1:
        raise ValueError("whale share must be finite and in [0, 1)")
    if not np.isfinite(valuation) or not 0 <= valuation <= 1:
        raise ValueError("whale valuation must be finite and in [0, 1]")
    if share == 0:
        return {}, None
    budget = sum(agent.budget for agent in population) * share / (1 - share)
    entrant = agent_type(budget=budget, market_valuation=valuation,
                         risk_av=risk_av, whale=True)
    return {start: [entrant]}, budget
