# Rebuild everything with `make all`, then explore with `make app`.
# The first `make data` downloads about 8 million pitches and takes a few hours;
# later runs skip seasons that are already saved.

PY := .venv/bin/python
START ?= 2015
END ?= 2026

.PHONY: all install data features reports app-data app test lint clean

all: data features app-data reports

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,app]"

# Raw data: Statcast pitches, player register, ballpark of every game.
data:
	$(PY) -m mlb_ros_projections.ingest --start $(START) --end $(END)
	$(PY) -m mlb_ros_projections.register
	$(PY) -m mlb_ros_projections.venues --start $(START) --end $(END)
	$(PY) -m mlb_ros_projections.build_db

# Plate appearances, snapshots, Marcel, park factors, model inputs (DuckDB).
features:
	$(PY) -m mlb_ros_projections.features

# The final model's forecasts, saved for the app.
app-data:
	$(PY) -m mlb_ros_projections.forecasts

# Backtest, calibration, and the analyses behind the decision memo.
reports:
	$(PY) -m mlb_ros_projections.backtest
	$(PY) -m mlb_ros_projections.calibration
	$(PY) -m mlb_ros_projections.insights

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

# Removes the database and reports, keeping the downloaded raw data.
clean:
	rm -f data/mlb.duckdb reports/*.csv
