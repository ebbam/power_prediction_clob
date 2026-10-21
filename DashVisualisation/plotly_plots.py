import numpy as np
import plotly.graph_objects as go

# Market record conventions (run_market_lob): per-period series have one value for each
# period t = 0, ..., T-1. 'price_history' and 'beliefs' have an extra initial value, so the
# market price after trading in period t is price_history[t + 1], the beliefs agents traded on
# in period t are beliefs[t], and the true outcome they learn about after period t is gen_el[t].


def _periods(rec):
    return np.arange(len(rec["best_bid"]))


def market_error(rec):
    """Market price after each period minus the true outcome in that period."""
    T = len(rec["best_bid"])
    return np.array(rec["price_history"][1:T + 1]) - np.array(rec["gen_el"][:T])


def bid_ask_band(best_bid, best_ask):
    """
    Outline of the bid-ask band as one polygon per run of periods in which both sides of the
    book exist, separated by None so that the fill never spans periods with a one-sided book.
    Each period's quotes are drawn as a step from t to t + 1, matching the price line.
    """
    bid, ask = np.asarray(best_bid, dtype=float), np.asarray(best_ask, dtype=float)
    two_sided = np.isfinite(bid) & np.isfinite(ask)
    xs, ys = [], []
    t = 0
    while t < len(bid):
        if not two_sided[t]:
            t += 1
            continue
        end = t
        while end < len(bid) and two_sided[end]:
            end += 1
        periods = np.arange(t, end)
        top_x = np.repeat(periods, 2) + np.tile([0, 1], len(periods))
        top_y = np.repeat(ask[t:end], 2)
        bottom_y = np.repeat(bid[t:end], 2)
        xs += top_x.tolist() + top_x[::-1].tolist() + [int(top_x[0]), None]
        ys += top_y.tolist() + bottom_y[::-1].tolist() + [float(top_y[0]), None]
        t = end
    return xs, ys


def plot_market_price(rec, whale_valuation=None, whale_start=None):
    t = _periods(rec)
    T = len(t)
    fig = go.Figure()
    band_x, band_y = bid_ask_band(rec["best_bid"], rec["best_ask"])
    fig.add_trace(go.Scatter(x=band_x, y=band_y, mode="lines", line=dict(width=0), fill="toself",
                             fillcolor="rgba(42,120,214,0.25)", hoverinfo="skip", name="Best bid to best ask"))
    fig.add_trace(go.Scatter(x=t, y=rec["price_history"][1:T + 1], mode="lines",
                             line=dict(color="black", width=2.5, shape="hv"), name="Market price"))
    fig.add_trace(go.Scatter(x=t, y=rec["gen_el"][:T], mode="lines",
                             line=dict(color="#e34948", width=1.5, shape="hv"), name="Election outcome"))
    fig.add_trace(go.Scatter(x=t, y=rec["beliefs"][:T], mode="lines",
                             line=dict(color="#6250d6", dash="dot", width=1.5, shape="hv"), name="Mean valuation"))
    fig.add_trace(go.Scatter(x=t, y=rec["weighted_beliefs"][:T], mode="lines",
                             line=dict(color="#1baf7a", dash="dash", width=1.5, shape="hv"), name="Budget-weighted mean valuation"))
    if whale_valuation is not None:
        fig.add_hline(y=whale_valuation, line=dict(color="#52514e", dash="dot", width=1),
                      annotation_text="Whale valuation", annotation_position="top left")
        fig.add_vline(x=whale_start, line=dict(color="#52514e", dash="dot", width=1),
                      annotation_text="Whale enters", annotation_position="bottom right")
    fig.update_layout(title="Market price and order book", xaxis_title="Period",
                      yaxis=dict(title="Price / probability", range=[0, 1.02]),
                      legend=dict(x=0, y=-0.2, orientation="h"), height=550,
                      margin=dict(t=40, b=20), template="plotly_white")
    return fig


def plot_market_error(rec):
    t = _periods(rec)
    error = market_error(rec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.cumsum(error), mode="lines", line=dict(color="rgba(82,81,78,0.6)", shape="hv"),
                             fill="tozeroy", fillcolor="rgba(82,81,78,0.15)", name="Cumulative error"))
    fig.add_trace(go.Scatter(x=t, y=error, mode="lines", line=dict(color="#e34948", shape="hv"),
                             fill="tozeroy", fillcolor="rgba(227,73,72,0.35)", name="Error"))
    fig.update_layout(title="Market price minus election outcome", xaxis_title="Period",
                      yaxis=dict(title="Error"), legend=dict(x=0, y=-0.3, orientation="h"), height=320,
                      margin=dict(t=40, b=20), template="plotly_white")
    return fig


def plot_trading_activity(rec):
    t = _periods(rec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=rec["submitted_buy_volume"], mode="lines",
                             line=dict(color="#2a78d6", dash="dot", shape="hv"), name="Submitted buy orders"))
    fig.add_trace(go.Scatter(x=t, y=rec["submitted_sell_volume"], mode="lines",
                             line=dict(color="#eb6834", dash="dot", shape="hv"), name="Submitted sell orders"))
    fig.add_trace(go.Scatter(x=t, y=rec["traded_volume"], mode="lines",
                             line=dict(color="black", shape="hv"), name="Executed volume"))
    fig.update_layout(title="Order flow (contracts per period)", xaxis_title="Period",
                      yaxis=dict(title="Contracts"), legend=dict(x=0, y=-0.2, orientation="h"), height=400,
                      margin=dict(t=40, b=20), template="plotly_white")
    return fig


def plot_hist(data_, c, label):
    """Histogram of one sampled bettor attribute."""
    hist_fig = go.Figure(data=[go.Histogram(x=data_, marker=dict(color=c))])
    hist_fig.update_layout(title=label, xaxis_title=label, yaxis_title="Number of bettors",
                           autosize=True, minreducedwidth=250, minreducedheight=250, height=350,
                           margin=dict(t=50, b=40, l=50, r=10), bargap=0.05, hovermode="closest",
                           showlegend=False, template="plotly_white")
    return hist_fig


def joint_plot(bettor_data, variables_of_interest, var_1_ind, var_2_ind, c1, c2, label_1, label_2, correlation):
    """Scatter of the two correlated attributes, with each attribute's histogram on the margins."""
    x, y = bettor_data[variables_of_interest[var_1_ind]], bettor_data[variables_of_interest[var_2_ind]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, xaxis="x", yaxis="y", mode="markers",
                             marker=dict(color="rgba(0,0,0,0.5)", size=4)))
    fig.add_trace(go.Histogram(y=y, xaxis="x2", marker=dict(color=c2)))
    fig.add_trace(go.Histogram(x=x, yaxis="y2", marker=dict(color=c1)))
    margin_axis = dict(zeroline=False, domain=[0.8, 1], showgrid=False, showticklabels=False)
    fig.update_layout(title=f"{label_1} and {label_2}",
                      autosize=True, minreducedwidth=250, minreducedheight=250, height=350,
                      margin=dict(t=50, b=40, l=50, r=10),
                      xaxis=dict(title=label_1, zeroline=False, domain=[0, 0.75], showgrid=False, color="rgba(0,0,0,0.8)"),
                      yaxis=dict(title=label_2, zeroline=False, domain=[0, 0.75], showgrid=False, color="rgba(0,0,0,0.8)"),
                      xaxis2=margin_axis, yaxis2=margin_axis,
                      bargap=0, hovermode="closest", showlegend=False, template="plotly_white")
    return fig
