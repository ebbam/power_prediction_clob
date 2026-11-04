"""The Dash app is deployed from DashVisualisation/ alone, so it carries copies of the model.
Copy bettor.py and orderbook.py into DashVisualisation/ again whenever this fails."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dash_app_model_copies_match_the_repository():
    for name in ("bettor.py", "orderbook.py"):
        assert (ROOT / name).read_text() == (ROOT / "DashVisualisation" / name).read_text(), name
