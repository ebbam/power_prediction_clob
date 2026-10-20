import numpy as np
import random as random
import matplotlib.pyplot as plt
from orderbook import OrderBook
## Defining classes

class bettor:
    # Wordy but this allows us to more easily change the default values of bettors we initialise
    def __init__(self, budget=None, market_valuation=None, n_contracts=0, risk_av=None,
                 stubbornness=None, expertise=None, market_imitation=0, bias=0, whale=False):
        # Random defaults are drawn here, per bettor. (Defaults written in the signature are evaluated
        # once, when the class is defined, so every bettor would share the same draw.)
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
        self.budget = budget # Their personal contract valuation - this will ultimately depend on their beliefs and evolve in relation to market activity
        self.market_valuation = market_valuation
                # The number of contracts currently held
        self.n_contracts = n_contracts
                # Risk aversion score (if we indeed try to vary the risk aversion)
        self.risk_av = risk_av
                # Definition (loose): your willingness to change your opinion in the face of contradicting evidence
                # Will ideally be used to process "fact" or "news" - again, loosely defined for now
        self.stubbornness = stubbornness
                # This is the "clarity" with which a bettor views the true probability
                # Perhaps a different distribution...although this is something we can play with
        self.expertise = 1- expertise
                # Market imitation. This should be a value between -1 and 1 which incorporates market movement into an agent's internal valuation
        self.market_imitation = market_imitation
                # Bias in the bettor's beliefs - this is the expected difference between the true probability and the bettor's belief
        self.bias = bias
        self.whale = whale
                
    def exp_utility(self, mkt_price, new_c):
        ''' 
        Expected utility function that takes into account the market price, budget, and market valuation of a bettor to determine the value of any trade (new_c)
        mkt_price: current market price
        new_c: possible number of contracts to trade: negative (positive) new_c = sell (buy)
        Return: Expected utility of a particular trade volume offered to sell or buy
        '''

        return self.market_valuation * self.utility(self.budget - mkt_price*new_c + self.n_contracts + new_c) + ((1-self.market_valuation)*self.utility(self.budget - mkt_price*new_c))
    
    def utility(self, w):
        ''' 
        Function defining utility of wealth according to risk aversion factor
        W: wealth 
        Return: risk-adjusted value of wealth
        '''
        if self.risk_av ==1:
            return np.log(w)
        else:
            return (w**(1-self.risk_av))/(1-self.risk_av)

    def trade(self, m):
        ''' 
        Function maximising the expected utility above to decide how many contracts to buy or sell constrained by budget
        Return: The number of contracts offerd to either sell (negative) or buy (positive)
        '''
        c_range = np.arange((-1*(int(self.budget) + self.n_contracts)), int(self.budget)+1)
        # vectorised: same result as looping exp_utility over c_range, ~100x faster
        offered_contracts = c_range[np.argmax(self.exp_utility(m, c_range))]
        return offered_contracts

    def update_belief(self, true_value, mkt_price, price_history, k=1): # stubbornness, risk aversion, information
        ''' 
        Individual bettor updates their belief as a function of their stubbornnes, expertise-adjusted signal of the true election outcome probability, and their current market valuation. 
        Updates internal belief 

        If an agent has a propensity to imitate the market (market_imitation is -1 or 1), we will also add a factor of the change in market price over the most recent $k$ steps.
        '''
        # if (self.market_imitation != 0) and (len(price_history)>k):
        #     self.market_valuation += 2*self.market_imitation * (mkt_price - price_history[-(k)])
        if not self.whale:
            self.market_valuation += (1-self.stubbornness)*(np.random.normal(true_value, self.expertise) - self.bias - self.market_valuation)

            if self.market_imitation > 0:
                self.market_valuation = (1-self.market_imitation)*self.market_valuation + self.market_imitation*mkt_price

            # ensure value is within range [0,1]
            self.market_valuation = np.clip(self.market_valuation, 0, 1)

    def reservation_price(self):
        '''
        Price at which a marginal trade leaves expected utility unchanged: the solution of
        d exp_utility / d new_c = 0 at new_c = 0, i.e.
            r = v u'(W_yes) / (v u'(W_yes) + (1 - v) u'(W_no)),  with u'(w) = w^(-risk_av),
        where W_yes = budget + n_contracts and W_no = budget. With no holdings r = market_valuation.
        A long YES position lowers r and a short position raises it, because risk aversion makes
        the agent value an extra dollar less in the state where it is already richer.
        The optimiser trade(p) buys when p < r and sells when p > r (up to integer rounding).
        '''
        v = self.market_valuation
        w_yes, w_no = self.budget + self.n_contracts, self.budget
        with np.errstate(divide="ignore", invalid="ignore"):
            return v / (v + (1 - v) * (w_yes / w_no) ** self.risk_av)

    def limit_order(self, ref_price, ob, rng, max_markup=0.05):
        '''
        CLOB order decision (Chiarella & Iori 2002 style), re-using the existing CRRA demand.
        Side: buy if the reservation price exceeds the reference price, sell if below. This is the
        sign of the utility-optimal trade at the reference price, so it accounts for current holdings.
        Limit price: reservation price shaded by a random markup k ~ U(0, max_markup), so buyers bid
        below and sellers ask above the price at which they are indifferent.
        Quantity: the existing utility-maximising trade() evaluated AT the limit price, so any fill
        (always at the limit price or better) is one the bettor is happy with.
        Return: (side, limit_price, qty) or None
        '''
        r = self.reservation_price()
        if np.isnan(r) or r == ref_price:
            return None
        side = int(np.sign(r - ref_price))
        p = ob.round_price(r - side * rng.uniform(0, max_markup), side)
        q = self.trade(p)
        if np.sign(q) != side:
            return None
        return side, p, int(abs(q))


# Original code for managing order book
def fulfil_orders(buy_orders, sell_orders):
    sum_buy = np.sum(buy_orders)
    sum_sell = -np.sum(sell_orders)

    fulfilled_buy = np.zeros(len(buy_orders))
    fulfilled_sell = np.zeros(len(sell_orders))


    if sum_buy == sum_sell:
        fulfilled_buy = buy_orders
        fulfilled_sell = sell_orders
    
    elif sum_buy > sum_sell:
        fulfilled_sell = sell_orders

        # fulfil buy orders
        shuffled_indices = np.random.permutation(len(buy_orders))
        running_total = 0

        i=-1
        for i in range(np.sum(np.cumsum(buy_orders[shuffled_indices])<sum_sell)):
            j= shuffled_indices[i]
            running_total += buy_orders[j]
            fulfilled_buy[j] = buy_orders[j]

        i+=1
        j = shuffled_indices[i]
        fulfilled_buy[j] = sum_sell - running_total


    elif sum_buy < sum_sell:
        fulfilled_buy = buy_orders

        # fulfil sell orders
        shuffled_indices = np.random.permutation(len(sell_orders))
        running_total = 0
        i=-1
        for i in range(np.sum(np.cumsum(-sell_orders[shuffled_indices])<sum_buy)):
            j= shuffled_indices[i]
            running_total -= sell_orders[j]
            fulfilled_sell[j] = sell_orders[j]
        

        i+=1
        j = shuffled_indices[i]
        fulfilled_sell[j] = -(sum_buy - running_total)

    return fulfilled_buy, fulfilled_sell


def split_orders(all_orders):
    # Define a function which takes in a numpy array of orders and
    # returns two numpy arrays of buy and sell orders alongside a list of indices
    # to reconstruct the original array of orders from the fulfilled buy and sell orders.

    buy_orders = all_orders[all_orders > 0]
    sell_orders = all_orders[all_orders < 0]

    buy_indices = np.where(all_orders > 0)[0]
    sell_indices = np.where(all_orders < 0)[0]

    return buy_orders, sell_orders, buy_indices, sell_indices


def reconstruct_orders(buy_orders, sell_orders, buy_indices, sell_indices, N):
    # Define a function which takes in two numpy arrays of buy and sell orders
    # alongside two lists of indices and returns a single numpy array of orders
    # with the buy and sell orders in the correct place.

    all_orders = np.zeros(N)

    all_orders[buy_indices] = buy_orders
    all_orders[sell_indices] = sell_orders

    return all_orders


def manage_orders(order_book):
    # Define a function which takes in a list of orders and returns a list of fulfilled orders
    # alongside a list of unfilled orders.

    all_orders = np.array(order_book)
    buy_orders, sell_orders, buy_indices, sell_indices = split_orders(all_orders)

    fulfilled_buy, fulfilled_sell = fulfil_orders(buy_orders, sell_orders)

    return reconstruct_orders(fulfilled_buy, fulfilled_sell, buy_indices, sell_indices, len(order_book))


def set_market_price(m, net_supply_demand, total_orders,market_lambda=1/10):
    delta = market_lambda*net_supply_demand/(total_orders) if total_orders > 0 else 0
    m += delta
    return np.clip(m, 0, 1) # constrain market price to be within [0,1]

# Generate election outcome as a random walk
# Not sure whether this should be a series of probabilities or if it should be a series of 1 and 0
def gen_election(init_price, t_el, sd):
    el = [init_price]
    for k in range(t_el):
        el.append(np.clip(el[k] + np.random.normal(0, sd), 0, 1))
    return el


def run_market(n_bettors, t_election, initial_price, outcome_uncertainty, bettors, k=1, market_lambda=1/10, market_error=0):

    # Create true election probability
    gen_el = gen_election(initial_price, t_election, outcome_uncertainty)
    # Initialise betting population
    #bettors = [bettor() for _ in range(n_bettors)]
    # Set market price
    mkt_price = initial_price + market_error

    # Varioues records to store market price, volume on the market, average bettor beliefs, and net supply
    price_history = [mkt_price]
    vol_history = [sum(np.abs(k.n_contracts) for k in bettors)]
    beliefs = [np.mean([k.market_valuation for k in bettors])]
    beliefs_weighted = [np.average([k.market_valuation for k in bettors], weights = [k.budget for k in bettors])]
    market_pressure = [0]
    net_supply = [0]
    fulfilled_orders_list = []
    fulfilled_orders_sums = []

    for t in range(t_election):
        order_book = [] # Initialise order book for time step t
        for b in bettors:
            order_book.append(b.trade(mkt_price)) # add bettor's order volume to the order_book, positive for a buy order and negative for a sell order
        
        # Resolve order book
        fulfilled_orders = manage_orders(order_book)  # Fulfill buy-sell orders once every bettor has placed their order

        
        # Traders update portfolios according to fulfilled orders or not
        """ Q: For now, I have separated the process for buyers and sellers here as we do not allow for negative contract holdings...if a seller for example enters into/sells NEW contracts rather than contracts they already hold
            Not sure if this is a problem...we do not currently match each existing contract with a negative contract elsewhere...maybe we need this?
            Additionally, the below only works because of the budget constraint implicit in our trade() function -- correct? Perhaps assert or unit test here or in trade function.  """
        
        """ I don't actually think this is something we need to include - the budget constraint means you can sell contracts you don't have, but you need to put up $1 for each one (max payout). """

        fulfilled_orders_sums.append(np.sum(np.abs(fulfilled_orders)))
        for i in range(n_bettors):
            bettors[i].n_contracts += fulfilled_orders[i]
            bettors[i].budget -= fulfilled_orders[i]*mkt_price
            # Buyers
            # if fulfilled_orders[i] > 0:
            #     bettors[i].n_contracts += fulfilled_orders[i]
            #     bettors[i].budget -= fulfilled_orders[i]*mkt_price

            # # # Sellers 
            # elif fulfilled_orders[i] < 0:
            #     bettors[i].n_contracts += max(fulfilled_orders[i], -bettors[i].n_contracts) # Ensures no negative contract holdings
            #     bettors[i].budget -= fulfilled_orders[i]*mkt_price

        
        net_supply_demand = np.sum(order_book) # Get net supply (buy contracts - sell contracts)
        order_volume = np.sum(np.abs(order_book)) # Get total order volume
        market_pressure.append(net_supply_demand/order_volume if order_volume != 0 else 0) # Record market pressure
        assert(abs(net_supply_demand) <= order_volume)

        # Update beliefs feeding in the true probability of the election outcome at time t
        for b in bettors:
            b.update_belief(gen_el[t], mkt_price, price_history, k=k) 

        """ Q: The market price updating is not working...See last chunk/plot for demonstration of the problem: price converges to one or zero."""
        mkt_price = set_market_price(mkt_price, net_supply_demand, order_volume, market_lambda=market_lambda) #Update market price



        # Update records
        beliefs.append(np.mean([k.market_valuation for k in bettors]))
        beliefs_weighted.append(np.average([k.market_valuation for k in bettors], weights = [k.budget for k in bettors]))
        price_history.append(mkt_price)
        net_supply.append(net_supply_demand)       
        vol_history.append(sum([np.abs(k.n_contracts) for k in bettors])) 

        # to del
        fulfilled_orders_list.append(np.sum(fulfilled_orders))

    record = {'price_history': price_history,
            'beliefs': beliefs,
            'weighted_beliefs': beliefs_weighted,
            'vol_history': vol_history,
            'gen_el': gen_el,
            'net_supply': net_supply[1:],
            'market_pressure': market_pressure,
            'final_budgets': [k.budget for k in bettors]}

    return record




def run_market_fixed_el(n_bettors, t_election, initial_price, outcome_uncertainty, bettors, gen_el, market_lambda=1/10):

    # Initialise betting population
    #bettors = [bettor() for _ in range(n_bettors)]
    # Set market price
    mkt_price = initial_price

    # Varioues records to store market price, volume on the market, average bettor beliefs, and net supply
    price_history = [mkt_price]
    vol_history = [sum(k.n_contracts for k in bettors)]
    beliefs = [np.mean([k.market_valuation for k in bettors])]
    beliefs_weighted = [np.average([k.market_valuation for k in bettors], weights = [k.budget for k in bettors])]
    market_pressure = [0]
    net_supply = [0]
    fulfilled_orders_list = []

    for t in range(t_election):

        order_book = [] # Initialise order book for time step t
        for b in bettors:
            order_book.append(b.trade(mkt_price)) # add bettor's order volume to the order_book, positive for a buy order and negative for a sell order
        
        # Resolve order book
        fulfilled_orders = manage_orders(order_book)  # Fulfill buy-sell orders once every bettor has placed their order
        
        # Traders update portfolios according to fulfilled orders or not
        """ Q: For now, I have separated the process for buyers and sellers here as we do not allow for negative contract holdings...if a seller for example enters into/sells NEW contracts rather than contracts they already hold
            Not sure if this is a problem...we do not currently match each existing contract with a negative contract elsewhere...maybe we need this?
            Additionally, the below only works because of the budget constraint implicit in our trade() function -- correct? Perhaps assert or unit test here or in trade function.  """
        
        """ I don't actually think this is something we need to include - the budget constraint means you can sell contracts you don't have, but you need to put up $1 for each one (max payout). """
        
        for i in range(n_bettors):
            bettors[i].n_contracts += fulfilled_orders[i]
            bettors[i].budget -= fulfilled_orders[i]*mkt_price
            # Buyers
            # if fulfilled_orders[i] > 0:
            #     bettors[i].n_contracts += fulfilled_orders[i]
            #     bettors[i].budget -= fulfilled_orders[i]*mkt_price

            # # # Sellers 
            # elif fulfilled_orders[i] < 0:
            #     bettors[i].n_contracts += max(fulfilled_orders[i], -bettors[i].n_contracts) # Ensures no negative contract holdings
            #     bettors[i].budget -= fulfilled_orders[i]*mkt_price

        
        net_supply_demand = np.sum(order_book) # Get net supply (buy contracts - sell contracts)
        order_volume = np.sum(np.abs(order_book)) # Get total order volume
        market_pressure.append(net_supply_demand/order_volume if order_volume != 0 else 0) # Record market pressure
        assert(abs(net_supply_demand) <= order_volume)


        """ Q: The market price updating is not working...See last chunk/plot for demonstration of the problem: price converges to one or zero."""
        mkt_price = set_market_price(mkt_price, net_supply_demand, order_volume, market_lambda=market_lambda) #Update market price

        # Update beliefs feeding in the true probability of the election outcome at time t
        for b in bettors:
            b.update_belief(gen_el[t], mkt_price, price_history, k=1) 

        # Update records
        beliefs.append(np.mean([k.market_valuation for k in bettors]))
        beliefs_weighted.append(np.average([k.market_valuation for k in bettors], weights = [k.budget for k in bettors]))
        price_history.append(mkt_price)
        net_supply.append(net_supply_demand)       
        vol_history.append(sum(np.abs(k.n_contracts) for k in bettors)) 

        # to del
        fulfilled_orders_list.append(np.sum(fulfilled_orders))

    record = {'price_history': price_history,
            'beliefs': beliefs,
            'weighted_beliefs': beliefs_weighted,
            'vol_history': vol_history,
            'gen_el': gen_el,
            'net_supply': net_supply[1:],
            'market_pressure': market_pressure,
            'final_budgets': [k.budget for k in bettors]}

    return record



def worst_case_wealth(agent, signed_qty=0, price=0.0):
    '''
    Terminal wealth in the worse of the two outcomes if the agent trades signed_qty contracts at price.
    YES pays 1: W_yes = C + Q.  NO pays 0: W_no = C.  A position is fully collateralised iff this is >= 0.
    '''
    cash = agent.budget - signed_qty * price
    contracts = agent.n_contracts + signed_qty
    return min(cash + contracts, cash)


def run_market_lob(n_bettors, t_election, initial_price, outcome_uncertainty, bettors,
                   k=1, market_error=0, gen_el=None, tick=0.01, ttl=5, max_markup=0.05,
                   price_rule="mid", rng=None, convergence_tolerance=0.02,
                   convergence_window=5, reference_timescale=None, whale_share=0.0,
                   whale_start=0, whale_valuation=0.7):
    '''
    Run the prediction market with a persistent central limit order book (CLOB).

    Each period: expire old orders, activate agents one at a time in random order, record the
    end-of-period book, then update beliefs. An activated agent cancels its resting order, observes
    the live book (including orders placed earlier in the same period), and may submit one new
    order. Because every agent is activated every period, an order rests for at most one period
    unless ttl = 1 removes it sooner.

    Period-level series (length t_election) are stored in the returned dict alongside the
    original outputs (length t_election + 1). The recorded midquote is NaN whenever one side of
    the book is empty; the reference price agents use is OrderBook.mid() (see its fallbacks).
    '''
    from collections import defaultdict
    from convergence import first_passage_convergence, dimensionless_time

    if rng is None:
        rng = np.random.default_rng()
    if price_rule not in ("mid", "last"):
        raise ValueError("price_rule must be 'mid' or 'last'")
    if ttl < 1:
        raise ValueError("ttl must be >= 1")
    if not np.isfinite(whale_share) or not 0 <= whale_share < 1:
        raise ValueError("whale_share must be finite and in [0, 1)")
    if whale_share > 0 and (
        not isinstance(whale_start, (int, np.integer))
        or not 0 <= whale_start < t_election
    ):
        raise ValueError("whale_start must be an integer within the simulation")
    if not np.isfinite(whale_valuation) or not 0 <= whale_valuation <= 1:
        raise ValueError("whale_valuation must be finite and in [0, 1]")
    if gen_el is None:
        gen_el = gen_election(initial_price, t_election, outcome_uncertainty)
    gen_el = np.asarray(gen_el, dtype=float)
    if len(gen_el) < t_election:
        raise ValueError("gen_el must contain at least t_election values")

    ob = OrderBook(tick=tick)
    mkt_price = float(initial_price + market_error)
    if not 0 <= mkt_price <= 1:
        raise ValueError("Initial reference price must be in [0, 1]")

    # Trades only move cash and contracts between agents, so both totals are conserved.
    total_cash = sum(b.budget for b in bettors)
    total_contracts = sum(b.n_contracts for b in bettors)

    price_history = [mkt_price]
    vol_history = [sum(abs(b.n_contracts) for b in bettors)]
    beliefs = [np.mean([b.market_valuation for b in bettors])]
    beliefs_weighted = [np.average([b.market_valuation for b in bettors], weights=[b.budget for b in bettors])]
    history = defaultdict(list)   # one entry per period for every diagnostic below
    whale_initial_budget = None
    whale_added_at = None
    market_cash_before_whale = None

    for t in range(t_election):
        if whale_share > 0 and t == whale_start:
            # Reserve the whale's share outside the market before intervention.
            market_cash_before_whale = total_cash
            whale_budget = total_cash * whale_share / (1 - whale_share)
            whale = bettor(
                budget=whale_budget,
                market_valuation=whale_valuation,
                risk_av=0.5,
                whale=True,
            )
            bettors.append(whale)
            total_cash += whale_budget
            whale_initial_budget = whale_budget
            whale_added_at = t

        expired = ob.expire(t)
        period = {
            "submitted_buy_volume": 0, "submitted_sell_volume": 0,
            "n_buy_orders": 0, "n_sell_orders": 0,
            "traded_volume": 0, "n_trades": 0,
            "n_cancelled": 0, "n_expired": len(expired),
            # What each agent's utility optimiser wants at the reference price it observes,
            # and how limit_order() translated that into an order.
            "n_want_buy": 0, "n_want_sell": 0,
            "n_desired_trade_dropped": 0,      # optimiser wants to trade, limit_order() returned None
            "n_side_flipped": 0,               # order side opposite to optimiser's desired trade
            "n_collateral_rejected": 0,        # order would leave worst-case wealth negative
            "n_one_sided_reference": 0,        # activations where no midquote existed
        }

        for i in rng.permutation(len(bettors)):
            agent = bettors[i]
            # On activation the agent withdraws its resting order and re-decides from scratch,
            # so its only live order is the one reflecting its current beliefs and holdings.
            period["n_cancelled"] += len(ob.cancel_trader(i))
            one_sided = ob.best_bid() is None or ob.best_ask() is None
            ref = ob.mid(mkt_price)
            period["n_one_sided_reference"] += one_sided

            desired = agent.trade(ref)
            period["n_want_buy"] += desired > 0
            period["n_want_sell"] += desired < 0

            order = agent.limit_order(ref, ob, rng, max_markup)
            if order is None:
                period["n_desired_trade_dropped"] += desired != 0
                continue
            side, price, qty = order
            period["n_side_flipped"] += desired != 0 and side != np.sign(desired)
            if worst_case_wealth(agent, side * qty, price) < 0:
                period["n_collateral_rejected"] += 1
                continue

            period["n_buy_orders" if side > 0 else "n_sell_orders"] += 1
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
        assert min(worst_case_wealth(b) for b in bettors) >= -1e-9, "uncollateralised position"

        for key, value in {**period, **snapshot}.items():
            history[key].append(value)

        # Reference price agents see between periods (and the price used in belief updates).
        if price_rule == "mid":
            mkt_price = ob.mid(mkt_price)
        elif np.isfinite(snapshot["last_trade"]):
            mkt_price = snapshot["last_trade"]

        for b in bettors:
            b.update_belief(gen_el[t], mkt_price, price_history, k=k)

        price_history.append(mkt_price)
        beliefs.append(np.mean([b.market_valuation for b in bettors]))
        beliefs_weighted.append(np.average([b.market_valuation for b in bettors], weights=[b.budget for b in bettors]))
        vol_history.append(sum(abs(b.n_contracts) for b in bettors))

    history = {key: np.asarray(values) for key, values in history.items()}
    history["submitted_volume"] = history["submitted_buy_volume"] + history["submitted_sell_volume"]
    net_supply = history["submitted_buy_volume"] - history["submitted_sell_volume"]
    with np.errstate(invalid="ignore", divide="ignore"):
        pressure = np.where(history["submitted_volume"] > 0, net_supply / history["submitted_volume"], 0.0)

    # Midquote after trading in period t is compared with the fundamental gen_el[t]. Note the timing:
    # agents trading in period t have only observed gen_el[0..t-1] (beliefs update after trading), so
    # this error includes the unforecastable innovation gen_el[t] - gen_el[t-1] (mean |step| ~0.04
    # when outcome_uncertainty = 0.05), which exceeds the default 0.02 tolerance.
    convergence = first_passage_convergence(history["midquote"], gen_el[:t_election],
                                            tolerance=convergence_tolerance, window=convergence_window)
    time = np.arange(t_election)
    scaled_time = (dimensionless_time(time, reference_timescale) if reference_timescale is not None
                   else np.full(t_election, np.nan))

    return {
        "price_history": price_history,
        "reference_price": price_history,
        "beliefs": beliefs,
        "weighted_beliefs": beliefs_weighted,
        "vol_history": vol_history,
        "gen_el": gen_el,
        "net_supply": net_supply,
        "market_pressure": np.concatenate([[0.0], pressure]),
        "final_budgets": [b.budget for b in bettors],
        **history,
        "convergence": convergence,
        "time": time,
        "dimensionless_time": scaled_time,
        "convergence_tolerance": convergence_tolerance,
        "convergence_window": convergence_window,
        "reference_timescale": reference_timescale,
        "whale_initial_budget": whale_initial_budget,
        "whale_added_at": whale_added_at,
        "market_cash_before_whale": market_cash_before_whale,
        "market_cash_after_whale": total_cash,
    }


def plot_returns(rec, scale_fact = 10, step=False): ## Election Result

    # process market pressure
    mp =  np.array(rec['market_pressure'])[1:]
    upward_pressure = rec['price_history'][:-1] +  ([mp>0] * mp)[0]/scale_fact # positive market pressure
    downward_pressure = rec['price_history'][:-1]  + ([mp<0] * mp)[0]/scale_fact # negative market pressure
    N = len(rec["price_history"])

    fig, ax = plt.subplots(2,figsize=[8,7])
    plt.subplots_adjust(hspace=-5)

    if step:
        ax[0].plot(rec['price_history'], color = "black", label = "Market price", drawstyle = "steps")
        ax[0].fill_between(range(N-1), upward_pressure, rec['price_history'][:-1], color = "darkorange", alpha = 0.5, step = 'pre')
        ax[0].fill_between(range(N-1), downward_pressure, rec['price_history'][:-1], color = "dodgerblue", alpha = 0.5, step = 'pre')


        ax[0].plot(rec['gen_el'], color = "red", label = "Election outcome", drawstyle = "steps")
        ax[0].plot(rec['beliefs'], color = "purple", linestyle = "dotted", label = "Average market valuation of bettors", drawstyle = "steps")


        # volume and net supply
        ax[1].plot(rec['vol_history'], color = "red", label = "No. of Contracts", drawstyle = "steps")
        ax[1].plot(rec['net_supply'], color = "blue", linestyle = "dotted", label = "Net Supply", drawstyle = "steps")

    else:
        ax[0].plot(rec['price_history'], color = "black", label = "Market price")
        ax[0].fill_between(range(N-1), upward_pressure, rec['price_history'][:-1], color = "darkorange", alpha = 0.5)
        ax[0].fill_between(range(N-1), downward_pressure, rec['price_history'][:-1], color = "dodgerblue", alpha = 0.5)


        ax[0].plot(rec['gen_el'], color = "red", label = "Election outcome")
        ax[0].plot(rec['beliefs'], color = "purple", linestyle = "dotted", label = "Average market valuation of bettors")


        # volume and net supply
        ax[1].plot(rec['vol_history'], color = "red", label = "No. of Contracts")
        ax[1].plot(rec['net_supply'], color = "blue", linestyle = "dotted", label = "Net Supply")

    ax[0].set_ylim(0,1.02)
    ax[0].legend(loc = "lower center", bbox_to_anchor=(0.5, -.3), ncols=3)
    ax[1].legend(loc = "lower center", bbox_to_anchor=(0.5, -0.5), ncols=2)

    ax[0].set_title('Market Price')
    ax[1].set_title('Contract Volume & Net Supply')

    plt.xlabel('Time')
    ax[0].set_ylabel('Result')
    ax[1].set_ylabel('Result')
    plt.xticks(rotation=45)
    fig.tight_layout()

    # Display the plot
    plt.show()
