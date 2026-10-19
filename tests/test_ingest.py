from mlb_ros_projections.ingest import month_ranges


def test_month_ranges_cover_march_through_october():
    ranges = month_ranges(2024)
    assert ranges[0] == ("2024-03-01", "2024-03-31")
    assert ranges[-1] == ("2024-10-01", "2024-10-31")
    assert len(ranges) == 8


def test_month_ranges_end_on_last_day_of_month():
    # April has 30 days, so this checks the end-of-month math.
    assert ("2024-04-01", "2024-04-30") in month_ranges(2024)
