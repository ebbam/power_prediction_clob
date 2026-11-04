# power_prediction_clob

Agent-based simulations of a prediction market run as a central limit order book (CLOB), and how a large trader (a "whale") distorts its price.

## Setup

Create the `prediction-mkt-abm` virtual environment from the repository root:

```
python3 -m venv prediction-mkt-abm
source prediction-mkt-abm/bin/activate
pip install -r requirements.txt
python3 -m pytest tests            # check everything works
```

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
| `bettor.py` | The model: agents (`Bettor`, `add_whale`), fundamental paths and the market (`run_market_lob`) |
| `orderbook.py` | The order book used by `run_market_lob` |
| `walrasian.py` | The original market mechanism, kept for comparison with the CLOB |
| `metrics.py` | Distortion and convergence measures, and their plots |
| `figure_style.py` | Shared figure style and saving |
| `experiments/whale_convergence_experiment.py` | Whale settings, fundamental paths and populations used by Figure 2 |

Figure 3 D–E also needs the theory results in `theory_checks/` (see below).

## Other folders

- `experiments/` – further experiments not in the manuscript. Run them from the repository root, e.g. `python3 -m experiments.run`.
- `tests/` – tests for the order book and the metrics.
- `sim_output/` – cached simulation results.
- `figures/` – figure outputs.
- `ignore/` – old outputs no code produces or reads any more, kept until you decide to delete them.

## Running a market

A whale is an ordinary bettor that never updates its belief. Build the population, add the
whale (it can join part-way through with `entry_time`), then run the market:

```python
import numpy as np
from bettor import Bettor, add_whale, fundamental_path, run_market_lob

np.random.seed(0)                                   # traits and belief signals use np.random
population = [Bettor(budget=100) for _ in range(100)]
add_whale(population, share=0.3, valuation=0.7, entry_time=50)
path = fundamental_path(0.5, 100, 0.05)
record = run_market_lob(population, 100, path, initial_price=0.5,
                        rng=np.random.default_rng(0))   # activation order and price shading
```

## theory_checks/

<!-- To be completed. -->

## DashVisualisation/

<!-- To be completed. -->
