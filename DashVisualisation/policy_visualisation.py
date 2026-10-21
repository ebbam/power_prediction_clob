"""
Dash app for the position-limit policy counterfactual.

A buy-and-hold whale enters and buys (or sells) until it holds its target position, paying at most
its valuation W per contract, then holds it. A position limit caps the contracts any trader,
including the whale, may hold. The app runs the same markets (same seeds) without the whale, with the
whale and no limit, and, when a limit is set, with the whale and the limit. Every bettor has log utility. See policy_simulation.py.
Run with:  python policy_visualisation.py
"""
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

import policy_simulation as sim

# Number inputs use step="any": the browser treats a value off the step grid (e.g. 2000 with min=1, step=100)
# as invalid, and Dash then passes None, which would silently mean "no target" or "no limit".

N_ITER = 12                      # runs per market: all three markets take about 8 s on 4 cores
EXPECTED_CAPITAL = 550.0 * sim.N_BETTORS    # ordinary bettors' expected total budget, U(100, 1000) each
COLOURS = {"No whale": "#52514e", "No limit": "#eb6834", "With limit": "#0d6efd"}
PLOT_ORDER = ["No whale", "With limit", "No limit"]
NOISE_GREY = "rgba(82,81,78,0.12)"
# Plotly's "plotly_white" look, set explicitly so the figures are white whatever template Plotly applies
AXIS = dict(gridcolor="#ebf0f8", zerolinecolor="#ebf0f8", linecolor="#ebf0f8", showline=False)
_pool = None


def pool():
    """One worker pool for the life of the app (starting processes is slow)."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=os.cpu_count())
    return _pool


@lru_cache(maxsize=64)
def simulate(target, whale_valuation, limit, n_iter, seed):
    """Run n_iter independent markets in parallel (cached, so revisiting a setting is instant)."""
    seeds = [seed * 1000 + i for i in range(n_iter)]
    n = len(seeds)
    return list(pool().map(sim.run_once, seeds, [target] * n, [whale_valuation] * n, [limit] * n))


def money(x):
    return f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"


def section(title):
    """Section heading: the same large gap above and small gap below for every section."""
    return html.H4(title, className="mt-5 mb-1")


def control(label, component, width=3):
    return dbc.Col([html.Div(label, className="fw-semibold"), component], width=width)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("Position Limits and Whale Manipulation", className="mt-3"),
    html.P([f"{sim.N_BETTORS} ordinary bettors trade on an event with true probability {sim.ETA}. "
            f"After {sim.BASELINE} periods a whale buys contracts at prices up to W until it holds its target, "
            "then holds them until the market closes.",
            html.Br(),
            "A position limit caps the contracts any trader, including the whale, may hold. "
            "Each run is repeated without the whale for comparison. "
            "Each contract pays $1 if the event happens."],
           className="lead", style={"fontSize": "1.05em"}),
    dbc.Row([
        control("Whale target (contracts)",
                dcc.Input(id="target", type="number", value=10000, min=0, max=200000, step="any")),
        control("Whale valuation W",
                dcc.Slider(0, 1, 0.05, value=0.7, id="whale_valuation", marks={v / 10: f"{v / 10:g}" for v in range(0, 11, 2)}),
                width=4),
    ], className="mb-3"),
    dbc.Row([
        control("Position limit (contracts; leave blank for none)",
                dcc.Input(id="limit", type="number", value=2000, min=0, step="any", placeholder="no limit")),
    ], className="mb-3"),
    dbc.Row([
        control("Runs", dcc.Input(id="n_iter", type="number", value=N_ITER, min=2, max=60, step=1), width=2),
        control("Seed", dcc.Input(id="seed", type="number", value=0, min=0, step=1), width=2),
        dbc.Col(dbc.Button("Run simulations", id="run", color="primary", className="mt-4"), width=2),
    ], className="mb-2"),
    html.P(id="capital_text", className="text-muted mb-0"),
    dcc.Loading([
        section("Summary (median across runs)"),
        html.Div(id="summary_table"),
        section("How the price responds"),
        dcc.Graph(id="error_path"),
        dbc.Row([dbc.Col(dcc.Graph(id="steady_plot"), width=4), dbc.Col(dcc.Graph(id="max_error_plot"), width=4),
                 dbc.Col(dcc.Graph(id="above_plot"), width=4)]),
        section("Whale size and the position limit"),
        dbc.Row([dbc.Col([dcc.Graph(id="capital_plot"),
                          dcc.Markdown(r"Theory (log utility): $\delta_S = p^{\ast}-\eta_S \approx \dfrac{C_{N+1,S}\,p^{\ast}(1-p^{\ast})}"
                                       r"{\sum_{i=1}^{N} A_{i,S}}$, and $|\delta_S| \le |W-\eta_S|$. Here $p^{\ast}$ is the "
                                       r"market price, $\eta_S$ the true probability, $C_{N+1,S}$ the whale's position and "
                                       r"$A_{i,S} = B_{i,S} + p^{\ast}C_{i,S}$ bettor $i$'s wealth marked to market.",
                                       mathjax=True, className="text-muted small ms-3")], width=6),
                 dbc.Col(dcc.Graph(id="position_plot"), width=6)]),
    ]),
], fluid=True, className="policy-app", style={"maxWidth": "1600px"})


def layout(fig, title, xaxis_title, yaxis_title, height=400, legend_width=None, legend_below=False):
    """legend_below puts the legend under the x-axis title (for the half-width plots, where a legend
    above the plot would wrap into the title)."""
    legend = dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=12))
    margin = dict(t=110, b=60, l=90, r=25)
    if legend_below:
        legend.update(y=-0.2, yanchor="top")
        margin.update(t=70, b=120)
    if legend_width:                       # entries per legend row = 1 / legend_width
        legend.update(entrywidthmode="fraction", entrywidth=legend_width)
    main, _, subtitle = title.partition("<br><sup>")
    fig.update_layout(title=dict(text=main, x=0.01, font=dict(size=16), yref="container", y=1, yanchor="top",
                                 pad=dict(t=12),                # title at the top of the figure
                                 subtitle=dict(text=subtitle.removesuffix("</sup>"), font=dict(size=12))), template="plotly_white", height=height,
                      paper_bgcolor="white", plot_bgcolor="white",
                      xaxis=dict(title=dict(text=xaxis_title, font=dict(size=12)), tickfont=dict(size=11), **AXIS),
                      yaxis=dict(title=dict(text=yaxis_title, font=dict(size=12), standoff=14), tickfont=dict(size=11), **AXIS,
                                 ticklabelstandoff=8),          # gap between tick labels and the plot
                      font=dict(size=12),
                      margin=margin, legend=legend, hoverlabel=dict(font_size=11))
    return fig


def error_path_figure(scenarios, band):
    fig = go.Figure()
    fig.add_hrect(y0=-band, y1=band, fillcolor=NOISE_GREY, line_width=0)
    # Legend: the median runs on the first row and the theory lines on the second, one column per market
    for rank, (name, result) in enumerate(scenarios.items()):
        t = np.arange(result["errors"].shape[1])
        low, median, high = np.percentile(result["errors"], [10, 50, 90], axis=0)
        colour = COLOURS[name]
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[high, low[::-1]], fill="toself", line=dict(width=0),
                                 fillcolor=colour, opacity=0.18, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=median, mode="lines", line=dict(color=colour, width=2.5),
                                 name=f"{name}: median run", legendrank=rank))
        fig.add_trace(go.Scatter(x=[sim.BASELINE, t[-1]], y=[result["benchmark"]] * 2, mode="lines",
                                 line=dict(color=colour, dash="dash", width=2), name=f"{name}: theory",
                                 legendrank=len(scenarios) + rank))
    fig.add_vline(x=sim.BASELINE, line=dict(color="#52514e", dash="dot", width=1.5),
                  annotation_text="Whale enters", annotation_position="top right")
    fig = layout(fig, "Market price error over time<br><sup>Shading: 10th–90th percentile of runs. "
                       "Grey band: ordinary noise before the whale enters.</sup>",
                  "Period", "Market price − true probability", height=520, legend_width=1 / 3.2)
    fig.update_layout(margin=dict(t=130))           # room for two legend rows
    return fig


def runs_figure(scenarios, key, title, subtitle, xaxis_title, theory=False, percent=False):
    """One row per market; each dot is a run and the box shows the quartiles."""
    fig = go.Figure()
    for name, result in scenarios.items():
        values = result[key] * (100 if percent else 1)
        fig.add_trace(go.Box(x=values, name=name, orientation="h", boxpoints="all", jitter=0.5, pointpos=0,
                             marker=dict(color=COLOURS[name], size=7, opacity=0.8), line=dict(color=COLOURS[name], width=2),
                             fillcolor="rgba(0,0,0,0)", showlegend=False, hovertemplate="%{x:.3f}<extra></extra>"))
        if theory:
            fig.add_trace(go.Scatter(x=[result["benchmark"]], y=[name], mode="markers", showlegend=False,
                                     marker=dict(symbol="line-ns", size=34, line=dict(width=3, color="#1a1a19")),
                                     hovertemplate="Theory: %{x:.3f}<extra></extra>"))
    fig = layout(fig, f"{title}<br><sup>{subtitle}</sup>", xaxis_title, "", height=320)
    fig.update_yaxes(categoryorder="array", categoryarray=list(scenarios)[::-1])     # first market on top
    fig.update_xaxes(nticks=6, rangemode="tozero" if percent else "normal")
    fig.update_layout(margin=dict(t=80, b=60, l=95, r=25))
    return fig


def capital_figure(scenarios, whale_valuation, limit):
    """Theory: the error depends on the whale's position only through Q / (ordinary bettors' capital)."""
    capital = next(iter(scenarios.values()))["others_wealth"]
    largest = max(r["held"].max() for r in scenarios.values()) / capital
    share = np.linspace(0, max(1.5 * largest, 0.1), 200)
    theory = np.abs([sim.log_utility_benchmark(x * capital, capital, whale_valuation) for x in share])
    fig = go.Figure(go.Scatter(x=100 * share, y=theory, mode="lines", line=dict(color="#1a1a19", dash="dash", width=2),
                               name="Theory"))
    for name, result in scenarios.items():
        x = 100 * np.median(result["held"]) / capital
        low, median, high = np.percentile(np.abs(result["steady"]), [10, 50, 90])
        fig.add_trace(go.Scatter(x=[x], y=[median], mode="markers", name=f"{name} (simulated)",
                                 marker=dict(color=COLOURS[name], size=13),
                                 error_y=dict(type="data", array=[high - median], arrayminus=[median - low], thickness=2)))
    if limit is not None and limit / capital <= share[-1]:
        fig.add_vline(x=100 * limit / capital, line=dict(color=COLOURS["With limit"], dash="dot", width=1.5),
                      annotation_text="Position limit", annotation_position="top right")
    fig = layout(fig, "Price distortion against whale size<br><sup>Bars: 10th–90th percentile of runs. "
                      f"Ordinary bettors' capital ≈ ${capital:,.0f}.</sup>",
                 "Whale position ÷ ordinary bettors' capital (%)", "Steady-state price error (absolute)",
                 height=460, legend_width=0.5, legend_below=True)
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(rangemode="tozero")
    return fig


def position_figure(scenarios, limit):
    fig = go.Figure()
    for name, result in scenarios.items():
        if name == "No whale":
            continue
        t = np.arange(result["positions"].shape[1])
        fig.add_trace(go.Scatter(x=t, y=np.median(result["positions"], axis=0), mode="lines", name=name,
                                 line=dict(color=COLOURS[name], width=2.5)))
    if limit is not None:
        fig.add_hline(y=limit, line=dict(color=COLOURS["With limit"], dash="dot", width=1.5),
                      annotation_text="Position limit", annotation_position="top right")
    fig = layout(fig, "Whale's position over time<br><sup>Median run.</sup>", "Period", "Contracts held",
                 height=460, legend_width=0.5, legend_below=True)
    fig.update_yaxes(rangemode="tozero")
    return fig


def summary_table(scenarios, target):
    """Rows about the whale are left blank in the market without one."""
    rows = [("Whale's position", lambda r: f"{np.median(r['held']):,.0f} of {target:,} contracts"),
            ("Whale's position ÷ ordinary bettors' capital", lambda r: f"{np.median(r['held']) / r['others_wealth']:.0%}"),
            ("Periods for the whale to build its position",
             lambda r: "not reached" if np.isnan(r["build"]).all() else f"{np.nanmedian(r['build']):.0f}"),
            ("Steady-state price error (simulated)", lambda r: f"{np.median(r['steady']):+.3f}"),
            ("Steady-state price error (theory)", lambda r: f"{r['benchmark']:+.3f}"),
            ("Largest price error after the whale enters (P1)", lambda r: f"{np.median(r['max_error']):.3f}"),
            ("Time outside the ordinary noise band (P3)", lambda r: f"{np.median(r['above']):.0%}"),
            ("Probability that m_T > 1/2 at the final period T (m_T = 1/2 counts as half)",
             lambda r: f"{np.mean(r['final_above']):.0%}"),
            ("Whale's profit if the event happens", lambda r: money(np.median(r["if_yes"]))),
            ("Whale's profit if it doesn't", lambda r: money(np.median(r["if_no"])))]
    header = html.Thead(html.Tr([html.Th("")] + [html.Th(name, style={"color": COLOURS[name], "whiteSpace": "nowrap", "width": "18%"}) for name in scenarios]))
    whale_rows = {"Whale's position", "Whale's position ÷ ordinary bettors' capital",
                  "Periods for the whale to build its position", "Whale's profit if the event happens",
                  "Whale's profit if it doesn't"}
    body = html.Tbody([html.Tr([html.Td(label)] + [html.Td("—" if name == "No whale" and label in whale_rows else cell(r))
                                                    for name, r in scenarios.items()])
                       for label, cell in rows])
    return dbc.Table([header, body], bordered=False, hover=True, striped=True, className="mb-0",
                     style={"maxWidth": "820px", "fontSize": "0.95em"})


@app.callback(Output("capital_text", "children"),
              [Input("target", "value"), Input("whale_valuation", "value"), Input("limit", "value")])
def describe_capital(target, whale_valuation, limit):
    target = target or 0
    price = whale_valuation if whale_valuation >= sim.ETA else 1 - whale_valuation
    text = (f"{target:,} contracts is {target / EXPECTED_CAPITAL:.0%} of the ordinary bettors' capital "
            f"(about ${EXPECTED_CAPITAL:,.0f}) and costs the whale at most ${target * price:,.0f}.")
    if limit:
        return [text, html.Br(), f"A limit of {limit:,} contracts is {limit / EXPECTED_CAPITAL:.0%} of that capital."]
    return text


@app.callback(
    [Output("summary_table", "children"), Output("error_path", "figure"), Output("steady_plot", "figure"),
     Output("max_error_plot", "figure"), Output("above_plot", "figure"), Output("capital_plot", "figure"),
     Output("position_plot", "figure")],
    Input("run", "n_clicks"),
    [State("target", "value"), State("whale_valuation", "value"), State("limit", "value"),
     State("n_iter", "value"), State("seed", "value")])
def update(_, target, whale_valuation, limit, n_iter, seed):
    target = int(np.clip(target or 0, 0, 200000))
    limit = None if limit is None or limit <= 0 else int(limit)
    n_iter = int(np.clip(n_iter or N_ITER, 2, 60))
    whale_valuation, seed = float(whale_valuation), int(seed or 0)

    no_whale = simulate(0, 0.5, None, n_iter, seed)      # same seeds, so the first 20 periods match the others
    band = sim.noise_band(no_whale)
    scenarios = {"No whale": sim.summarise(no_whale, band),
                 "No limit": sim.summarise(simulate(target, whale_valuation, None, n_iter, seed), band)}
    if limit is not None:
        scenarios["With limit"] = sim.summarise(simulate(target, whale_valuation, limit, n_iter, seed), band)

    plotted = {name: scenarios[name] for name in PLOT_ORDER if name in scenarios}
    return (summary_table(scenarios, target),
            error_path_figure(plotted, band),
            runs_figure(plotted, "steady", "Steady-state price error", "Each dot is a run; black line: theory.",
                        f"Mean error over the last {sim.STEADY_WINDOW} periods", theory=True),
            runs_figure(plotted, "max_error", "Largest price error (P1)", "After the whale enters.",
                        "Maximum |price − true probability|"),
            runs_figure(plotted, "above", "Time outside the noise band (P3)", "After the whale enters.",
                        "% of periods outside the grey band", percent=True),
            capital_figure(plotted, whale_valuation, limit),
            position_figure(plotted, limit))


if __name__ == "__main__":
    app.run(debug=True)
