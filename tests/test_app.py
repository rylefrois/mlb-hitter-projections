from datetime import date
from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "mlb.duckdb"
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="needs the built database")


@pytest.mark.parametrize("stat", ["woba", "k_rate", "ubb_rate"])
def test_app_renders_every_stat_without_errors(stat):
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.sidebar.radio[0].set_value(stat).run()
    assert not app.exception
    assert len(app.metric) >= 7  # player tab (4) + league tab (3)


def test_app_handles_a_late_season_date_and_short_range():
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    # Options display as labels ("Sep 1"); set_value takes the underlying date.
    app.select_slider[0].set_value(date(2024, 9, 1)).run()
    app.slider[0].set_value(25).run()
    assert not app.exception
