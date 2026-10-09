new CLOB

Agents act one at a time, in a random order. Each activated agent:
    1. Withdraws its old order, if it has one.

    2. Looks at the book as it is right now, including orders placed earlier this period. Its reference price is:
        the midquote, if there are both bids and asks;
        otherwise the one best price still on the book;
        otherwise, if the book is empty, the last trade (or the starting price).

    3. Works out its reservation price. This is the price at which its existing utility function makes it indifferent to trading. With no position it equals the agent’s belief. Holding YES pulls it down, and holding a short pulls it up.

    4. Picks a side. It buys if the reservation price is above the reference price and sells if below. This always matches what its utility optimiser wants at that price.
    Sets a limit price a random 0–5 cents away from its reservation price: below it for a buy, above it for a sell.

    5. Sets a quantity of trade(limit price), the utility-optimal amount at that price. The range of trades trade() considers guarantees the agent can cover its position whether the event resolves YES or NO.

    6. Submits the order. It trades against the opposite side, best price first and oldest order first at equal prices, always at the resting order’s price. Any unfilled part rests in the book. Cash and positions update on every fill, and selling without holdings opens a short (NO) position.

Record the period. This captures the book’s state, the diagnostic counters, and checks that cash and contracts are conserved and no position is under-collateralised.

Update beliefs. All agents update from the fundamental’s new value. The new reference price is passed in but has no effect unless market_imitation is set.