import json
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


def run_app():
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    return app


def hitter_box(app):
    return next(box for box in app.selectbox if box.label == "Hitter")


def test_team_filter_limits_hitters_to_that_team():
    app = run_app()
    everyone = len(hitter_box(app).options)
    next(box for box in app.selectbox if box.label == "Team").set_value("NYY").run()
    yankees = hitter_box(app).options
    assert not app.exception
    assert 15 < len(yankees) < everyone
    assert "Aaron Judge" in yankees


def test_traded_hitter_is_listed_under_both_teams_with_full_season():
    # Jazz Chisholm was traded from MIA to NYY at the 2024 deadline.
    for team in ["MIA", "NYY"]:
        app = run_app()
        next(box for box in app.selectbox if box.label == "Team").set_value(team).run()
        hitter_box(app).set_value(665862).run()
        assert not app.exception
        assert len(app.select_slider[0].options) == 5  # May 1 through Sep 1


def test_chart_labels_each_forecast_date_once():
    app = run_app()
    spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
    axes = [layer["encoding"]["x"]["axis"] for layer in spec["layer"]]
    for axis in axes:
        assert axis["format"] == "%b %-d"
        assert [tick["month"] for tick in axis["values"]] == [5, 6, 7, 8, 9]
