"""
Dash app for exploring the order-book prediction market.

Choose the distribution of bettor attributes (and the correlation between two of them),
add herding or a whale, and the app simulates the market and plots the result.
Run with:  python market_visualisation.py   then open http://127.0.0.1:8050
"""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

from market_functions import bettor, run_market_lob
from plotly_plots import joint_plot, plot_hist, plot_market_error, plot_market_price, plot_trading_activity
from processing_data import sample_parameters

# Default simulation parameters (editable in the "Simulation parameters" panel)
DEFAULTS = {"n_bettors": 100,            # number of (non-whale) bettors
            "t_election": 100,           # periods until the election
            "initial_price": 0.5,        # starting price, true outcome and mean initial valuation
            "outcome_uncertainty": 0.1,  # standard deviation of each step of the true-outcome random walk
            "max_markup": 0.05,          # largest offset between a reservation price and its limit price
            "ttl": 5}                    # periods an unfilled order stays in the book

variables_of_interest = ["budget", "risk_av", "stubbornness", "expertise"]
nice_names = ["Initial budget", "Risk aversion", "Stubbornness", "Expertise"]
bettor_attribute_colours = dict(zip(variables_of_interest, px.colors.qualitative.Plotly))


def attribute_row(name, colour, slider, sd_id, sd_value, sd_max):
    return dbc.Row([
        dbc.Col(html.Div(name, style={"color": colour}), width=3),
        dbc.Col(slider, width=5),
        dbc.Col(width=1),
        dbc.Col(dcc.Input(id=sd_id, type="number", value=sd_value, min=0, max=sd_max, step="any"), width=3),
    ], className="mb-3")


def control(label, component):
    return dbc.Col([html.Div(label), component], width=3)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Market Simulation Dashboard"),
    html.H3("Simulation parameters"),
    dbc.Row([
        control("Number of bettors", dcc.Input(id="n_bettors", type="number", value=DEFAULTS["n_bettors"], min=2, max=1000, step=1)),
        control("Number of periods", dcc.Input(id="t_election", type="number", value=DEFAULTS["t_election"], min=2, max=1000, step=1)),
        control("Initial price", dcc.Input(id="initial_price", type="number", value=DEFAULTS["initial_price"], min=0.01, max=0.99, step=0.01)),
        control("Outcome step s.d.", dcc.Input(id="outcome_uncertainty", type="number", value=DEFAULTS["outcome_uncertainty"], min=0, max=0.5, step=0.01)),
    ], className="mb-2"),
    dbc.Row([
        control("Max price offset", dcc.Input(id="max_markup", type="number", value=DEFAULTS["max_markup"], min=0.01, max=0.5, step=0.01)),
        control("Order lifetime (periods)", dcc.Input(id="ttl", type="number", value=DEFAULTS["ttl"], min=1, max=100, step=1)),
        control("Random seed", dcc.Input(id="seed", type="number", value=0, min=0, step=1)),
    ], className="mb-4"),
    dbc.Row([dbc.Col(html.H3("Bettor attributes"), width=3), dbc.Col(html.H3("Expected value"), width=5),
             dbc.Col(width=1), dbc.Col(html.H3("Standard deviation"), width=3)], className="mb-3"),
    attribute_row(nice_names[0], bettor_attribute_colours["budget"],
                  dcc.Slider(500, 1500, 100, marks={i: str(i) for i in range(500, 1501, 200)}, value=1000, id="budget_mean"),
                  "budget_sd", 100, 1000),
    attribute_row(nice_names[1], bettor_attribute_colours["risk_av"],
                  dcc.Slider(0, 1, 0.1, value=0.5, id="risk_av_mean"), "risk_av_sd", 0.1, 0.4),
    attribute_row(nice_names[2], bettor_attribute_colours["stubbornness"],
                  dcc.Slider(0, 1, 0.1, value=0.3, id="stubbornness_mean"), "stubbornness_sd", 0.05, 0.4),
    attribute_row(nice_names[3], bettor_attribute_colours["expertise"],
                  dcc.Slider(0, 1, 0.05, marks={i / 10: str(i / 10) for i in range(0, 11, 2)}, value=0.9, id="expertise_mean"),
                  "expertise_sd", 0.04, 0.4),
    dbc.Row([
        dbc.Col(html.Div("Correlate"), width=1),
        dbc.Col(dcc.Dropdown(id="var_1", value="budget",
                             options=[{"label": n, "value": v} for n, v in zip(nice_names, variables_of_interest)]), width=3),
        dbc.Col(html.Div("and"), width=1),
        dbc.Col(dcc.Dropdown(id="var_2", value="expertise",
                             options=[{"label": n, "value": v} for n, v in zip(nice_names, variables_of_interest)]), width=3),
        dbc.Col(html.Div("with correlation coefficient"), width=3),
        dbc.Col(dcc.Input(id="correlation", type="number", value=0, min=-1, max=1, step="any"), width=1),
    ], className="mb-3"),
    html.H3("Herding and whale"),
    dbc.Row([
        control("Herding weight h", dcc.Slider(0, 1, 0.05, marks={i / 10: str(i / 10) for i in range(0, 11, 2)}, value=0, id="herding")),
        control("Whale share of market wealth", dcc.Slider(0, 0.9, 0.05, marks={i / 10: str(i / 10) for i in range(0, 10, 2)}, value=0, id="whale_share")),
        control("Whale valuation", dcc.Slider(0, 1, 0.05, marks={i / 10: str(i / 10) for i in range(0, 11, 2)}, value=0.7, id="whale_valuation")),
        control("Whale enters in period", dcc.Input(id="whale_start", type="number", value=20, min=0, step=1)),
    ], className="mb-3"),

    dcc.Loading([
        html.H3("Sampled bettor attributes"),
        dbc.Row([dbc.Col(dcc.Graph(id="hist_plot_1"), width=4),
                 dbc.Col(dcc.Graph(id="hist_plot_2"), width=4),
                 dbc.Col(dcc.Graph(id="joint_plot"), width=4)], className="g-0"),
        dbc.Row([dcc.Graph(id="market_price_plot")]),
        dbc.Row([dcc.Graph(id="market_error_plot")]),
        dbc.Row([dcc.Graph(id="trading_activity_plot")]),
    ]),
])


@app.callback(
    [Output("hist_plot_1", "figure"), Output("hist_plot_2", "figure"), Output("joint_plot", "figure"),
     Output("market_price_plot", "figure"), Output("market_error_plot", "figure"), Output("trading_activity_plot", "figure")],
    [Input("var_1", "value"), Input("var_2", "value"), Input("correlation", "value"),
     Input("budget_mean", "value"), Input("budget_sd", "value"),
     Input("risk_av_mean", "value"), Input("risk_av_sd", "value"),
     Input("stubbornness_mean", "value"), Input("stubbornness_sd", "value"),
     Input("expertise_mean", "value"), Input("expertise_sd", "value"),
     Input("herding", "value"), Input("whale_share", "value"), Input("whale_valuation", "value"),
     Input("whale_start", "value"), Input("seed", "value"),
     Input("n_bettors", "value"), Input("t_election", "value"), Input("initial_price", "value"),
     Input("outcome_uncertainty", "value"), Input("max_markup", "value"), Input("ttl", "value")])
def update_plots(var_1, var_2, correlation, budget_mean, budget_sd, risk_av_mean, risk_av_sd,
                 stubbornness_mean, stubbornness_sd, expertise_mean, expertise_sd,
                 herding, whale_share, whale_valuation, whale_start, seed,
                 n_bettors, t_election, initial_price, outcome_uncertainty, max_markup, ttl):
    if var_1 == var_2:
        raise dash.exceptions.PreventUpdate      # need two different attributes to correlate
    # Blank or out-of-range inputs fall back to the defaults or the nearest allowed value
    value = lambda x, default: default if x is None else x
    n_bettors = int(np.clip(value(n_bettors, DEFAULTS["n_bettors"]), 2, 1000))
    t_election = int(np.clip(value(t_election, DEFAULTS["t_election"]), 2, 1000))
    initial_price = float(np.clip(value(initial_price, DEFAULTS["initial_price"]), 0.01, 0.99))
    outcome_uncertainty = float(np.clip(value(outcome_uncertainty, DEFAULTS["outcome_uncertainty"]), 0, 0.5))
    max_markup = float(np.clip(value(max_markup, DEFAULTS["max_markup"]), 0.01, 0.5))
    ttl = int(np.clip(value(ttl, DEFAULTS["ttl"]), 1, 100))
    seed = int(seed or 0)
    np.random.seed(seed)                         # attribute draws and private signals
    rng = np.random.default_rng(seed)            # order placement

    means = [budget_mean, risk_av_mean, stubbornness_mean, expertise_mean]
    sds = [budget_sd or 0, risk_av_sd or 0, stubbornness_sd or 0, expertise_sd or 0]
    var_1_ind, var_2_ind = variables_of_interest.index(var_1), variables_of_interest.index(var_2)
    sampled = sample_parameters(var_1_ind, var_2_ind, means, sds, np.clip(correlation or 0, -1, 1), n_bettors)

    bettors = [bettor(budget=max(sampled[i, 0], 1.0),
                      market_valuation=np.clip(np.random.normal(initial_price, 0.05), 0, 1),
                      risk_av=np.clip(sampled[i, 1], 0, 1),
                      stubbornness=np.clip(sampled[i, 2], 0, 1),
                      expertise=np.clip(sampled[i, 3], 0, 1),
                      market_imitation=herding)
               for i in range(n_bettors)]

    bettor_data = {"budget": [b.budget for b in bettors],
                   "risk_av": [b.risk_av for b in bettors],
                   "stubbornness": [b.stubbornness for b in bettors],
                   "expertise": [1 - b.expertise for b in bettors]}   # bettor stores 1 - expertise (signal s.d.)

    names = dict(zip(variables_of_interest, nice_names))
    other_var = [v for v in variables_of_interest if v not in (var_1, var_2)]
    fig_hist_1 = plot_hist(bettor_data[other_var[0]], bettor_attribute_colours[other_var[0]], names[other_var[0]])
    fig_hist_2 = plot_hist(bettor_data[other_var[1]], bettor_attribute_colours[other_var[1]], names[other_var[1]])
    fig_joint = joint_plot(bettor_data, variables_of_interest, var_1_ind, var_2_ind,
                           bettor_attribute_colours[var_1], bettor_attribute_colours[var_2],
                           names[var_1], names[var_2], float(np.clip(correlation or 0, -1, 1)))

    whale_start = int(np.clip(whale_start or 0, 0, t_election - 1))
    market_record = run_market_lob(n_bettors, t_election, initial_price, outcome_uncertainty, bettors, rng=rng,
                                   max_markup=max_markup, ttl=ttl, whale_share=whale_share,
                                   whale_start=whale_start, whale_valuation=whale_valuation)

    has_whale = whale_share > 0
    fig_market_price = plot_market_price(market_record, whale_valuation if has_whale else None,
                                         whale_start if has_whale else None)
    return (fig_hist_1, fig_hist_2, fig_joint, fig_market_price,
            plot_market_error(market_record), plot_trading_activity(market_record))


if __name__ == "__main__":
    app.run(debug=True)
