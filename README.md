# power_prediction_clob

Agent-based simulations of a prediction market run as a central limit order book (CLOB), and how a large trader (a "whale") distorts its price.

## Reproducing the manuscript figures


```
python3 paper_figures.py          # Figures 1–3, drawn from the cached simulations in sim_output/
python3 paper_figures.py --rerun  # rerun the simulations first (~25 min on 4 processes)
python3 stylised_mechanisms.py    # stylised mechanisms figure (no simulation)
```

Figures are saved to `figures/` as PDF and PNG:

| Manuscript figure | Output file(s) |
|---|---|
| Figure 1 | `figure1_clob` |
| Figure 2 | `figure2`, `figure2e` |
| Figure 3 (panels A–C; panels D–E use `theory_checks/figdata_AB.json`) | `figure3b`, `figure3b_theory` |
| Stylised mechanisms | `stylised_mechanisms_clob` |

### Files needed

| File | Role |
|---|---|
| `paper_figures.py` | Runs the simulations for Figures 1–3 (cached in `sim_output/`) and draws them |
| `stylised_mechanisms.py` | Draws the stylised mechanisms figure |
| `bettor.py` | The model: agents and the market (`run_market_lob`) |
| `orderbook.py` | The order book used by `run_market_lob` |
| `metrics.py` | Distortion and convergence measures, and their plots |
| `figure_style.py` | Shared figure style and saving |
| `experiments/whale_convergence_experiment.py` | Whale settings, fundamental paths and populations used by Figure 2 |

Figure 3 D–E also needs the theory results in `theory_checks/` (see below).

## Other folders

- `experiments/` – further experiments not in the manuscript.
- `tests/` – tests for the order book and the metrics.
- `sim_output/` – cached simulation results.
- `figures/` – figure outputs.

## theory_checks/

<!-- To be completed. -->

## DashVisualisation/

<!-- To be completed. -->
