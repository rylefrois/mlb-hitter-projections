"""Run the final model once and save every forecast for the app.

Creates the `app_forecasts` table and a compressed copy of it in the repo
(app/data), which is all the app reads. So the app never retrains the model,
and it runs without the full database, e.g. when hosted.

Usage:
    python -m mlb_ros_projections.forecasts
"""

from pathlib import Path

import duckdb

from mlb_ros_projections.build_db import DB_PATH
from mlb_ros_projections.models import FINAL_MODEL, MODEL_SPECS, forward_chain, load_model_inputs

APP_DATA_PATH = Path(__file__).resolve().parents[2] / "app" / "data" / "app_forecasts.parquet"


def build_app_forecasts(con: duckdb.DuckDBPyConnection) -> int:
    forecast, _ = forward_chain(load_model_inputs(con), MODEL_SPECS[FINAL_MODEL])
    con.register("forecast_df", forecast)
    con.execute("CREATE OR REPLACE TABLE final_forecasts AS SELECT * FROM forecast_df")
    con.unregister("forecast_df")

    # One row per hitter and forecast date: who they are, what was known,
    # the forecast with the parts behind its range, and what actually happened.
    con.execute("""
        CREATE OR REPLACE TABLE app_forecasts AS
        SELECT
            f.season,
            f.asof_date,
            f.batter,
            concat_ws(' ', p.name_first, p.name_last, p.name_suffix) AS name,
            t.team,
            a.age,

            i.pa_ytd,
            i.woba_denom_ytd,
            i.pa_ros,
            i.woba_denom_ros,
            i.pa_ytd_share,
            i.marcel_weighted_pa,
            i.pf_hist_k_rate, i.pf_hist_ubb_rate, i.pf_hist_woba,
            i.pf_ytd_k_rate, i.pf_ytd_ubb_rate, i.pf_ytd_woba,
            i.pf_ros_k_rate, i.pf_ros_ubb_rate, i.pf_ros_woba,

            b.ytd_k_rate, b.ytd_ubb_rate, b.ytd_woba, b.ytd_xwoba,
            b.marcel_k_rate, b.marcel_ubb_rate, b.marcel_woba,

            f.k_rate, f.ubb_rate, f.woba,
            f.k_rate_per_pa_var, f.ubb_rate_per_pa_var, f.woba_per_pa_var,
            f.k_rate_evidence, f.ubb_rate_evidence, f.woba_evidence,
            f.k_rate_interval_scale, f.ubb_rate_interval_scale, f.woba_interval_scale,

            i.actual_k_rate, i.actual_ubb_rate, i.actual_woba
        FROM final_forecasts AS f
        JOIN model_inputs AS i USING (season, asof_date, batter)
        JOIN batter_baselines AS b USING (season, asof_date, batter)
        LEFT JOIN people AS p ON p.key_mlbam = f.batter
        LEFT JOIN batter_team_at_date AS t USING (season, asof_date, batter)
        LEFT JOIN batter_age AS a USING (season, batter)
        WHERE i.pa_ytd > 0
    """)
    APP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY app_forecasts TO '{APP_DATA_PATH}' (FORMAT parquet, COMPRESSION zstd)"
    )
    return con.execute("SELECT count(*) FROM app_forecasts").fetchone()[0]


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    n = build_app_forecasts(con)
    con.close()
    print(f"app_forecasts: {n:,} rows ({FINAL_MODEL}), saved to {APP_DATA_PATH}")


if __name__ == "__main__":
    main()
