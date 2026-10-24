import json
from datetime import date
from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "mlb.duckdb"
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="needs the built database")


# Tests pin a season so they don't depend on which season is newest.
SEASON = 2024


def run_app():
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.sidebar.selectbox[0].set_value(SEASON).run()
    return app


@pytest.mark.parametrize("stat", ["woba", "k_rate", "ubb_rate"])
def test_app_renders_every_stat_without_errors(stat):
    app = run_app()
    app.sidebar.radio[0].set_value(stat).run()
    assert not app.exception
    assert len(app.metric) >= 8  # player tab (4) + hitters-to-watch track record (4)


def test_app_handles_a_late_season_date_and_short_range():
    app = run_app()
    # Options display as labels ("Sep 1"); set_value takes the underlying date.
    app.select_slider[0].set_value(date(2024, 9, 1)).run()
    app.slider[0].set_value(25).run()
    assert not app.exception


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


def test_trends_tab_lists_ten_hitters_each_way_and_can_reveal_outcomes():
    app = run_app()
    lists = [df for df in app.dataframe if "Why" in df.value.columns]
    assert len(lists) == 2
    assert all(len(df.value) == 10 for df in lists)
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert all("What happened next" in df.value.columns
               for df in app.dataframe if "Why" in df.value.columns)


@pytest.mark.parametrize("season", [2026, 2024, 2018])
def test_switching_seasons_after_moving_the_date_slider(season):
    # Regression test: a date picked in one season used to crash another season.
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.select_slider[0].set_value(app.select_slider[0].value).run()
    app.sidebar.selectbox[0].set_value(season).run()
    assert not app.exception


def test_hitters_to_watch_can_compare_with_the_preseason_projection():
    app = run_app()
    next(r for r in app.radio if r.label == "Compare the forecast to").set_value("marcel").run()
    assert not app.exception
    lists = [df for df in app.dataframe if "Why" in df.value.columns]
    assert len(lists) == 2
    assert "Chance to beat preseason" in lists[0].value.columns
    assert all(len(df.value) == 10 for df in lists)


def test_model_performance_shows_accuracy_by_date_and_sample_size():
    app = run_app()
    indexes = [list(df.value.index) for df in app.dataframe]
    assert ["May 1", "Jun 1", "Jul 1", "Aug 1", "Sep 1"] in indexes
    assert any(index[0] == "1-49 PA so far" for index in indexes if index)


def keys(app):
    # AppTest's session state has .keys() but can't be iterated directly.
    return list(app.session_state.keys())


def state(app, name):
    """A filter's value, whatever reset generation its key is on."""
    matches = [k for k in keys(app) if k.startswith(f"{name}__reset")]
    assert len(matches) == 1, (name, matches)
    return app.session_state[matches[0]]


def test_reset_button_restores_every_filter_to_its_default():
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    names = ["season", "stat", "trends_min_pa", "trends_next_pa", "trends_reveal", "trends_versus"]
    defaults = {name: state(app, name) for name in names}
    default_hitter = hitter_box(app).value

    app.sidebar.selectbox[0].set_value(2024).run()
    app.sidebar.radio[0].set_value("k_rate").run()
    next(s for s in app.slider if s.label == "Minimum PA so far").set_value(150).run()
    next(s for s in app.slider if s.label == "Next N plate appearances").set_value(300).run()
    app.toggle[0].set_value(True).run()
    next(r for r in app.radio if r.label == "Compare the forecast to").set_value("marcel").run()
    next(b for b in app.selectbox if b.label == "Team").set_value("NYY").run()
    assert state(app, "season") == 2024

    app.sidebar.button[0].click().run()

    assert not app.exception
    for name, value in defaults.items():
        assert state(app, name) == value, name
    assert next(b for b in app.selectbox if b.label == "Team").value == "All teams"
    assert hitter_box(app).value == default_hitter


def test_reset_gives_every_filter_a_new_key():
    # A browser keeps its own copy of each widget's value and sends it back on the
    # next interaction, so a reset that only cleared saved values let old choices
    # reappear. New keys make brand-new widgets the browser has no copy of.
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    before = {k for k in keys(app) if "__reset" in k}
    app.sidebar.button[0].click().run()
    after = {k for k in keys(app) if "__reset" in k}
    assert before and after
    assert before.isdisjoint(after)
