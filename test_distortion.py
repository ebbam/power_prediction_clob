import numpy as np

from distortion import (
    distortion_metrics, kaplan_meier_curve, normalised_pull, smoothed_absolute_error,
)


def test_undistorted_run_reverts_immediately():
    error = np.full(20, 0.01)
    metrics = distortion_metrics(error, shock_time=5, band=0.05, window=3)
    assert np.isclose(metrics["max_error"], 0.01)
    assert metrics["reversion_time"] == 0.0 and metrics["reverted"]
    assert metrics["share_outside_band"] == 0.0


def test_reversion_is_timed_from_the_maximum_error():
    error = np.full(30, 0.01)
    error[12:18] = 0.20
    error[15] = 0.30  # maximum at 15; back inside the band from period 18
    metrics = distortion_metrics(error, shock_time=10, band=0.05, window=3)
    assert np.isclose(metrics["max_error"], 0.30)
    assert metrics["peak_time_after_shock"] == 5
    assert metrics["reversion_time"] == 3.0 and metrics["reverted"]
    assert np.isclose(metrics["share_outside_band"], 6 / 20)


def test_gradual_shock_is_not_an_instant_reversion():
    error = np.full(30, 0.01)
    error[16:] = 0.30  # shock at 10, price leaves the band only at 16
    metrics = distortion_metrics(error, shock_time=10, band=0.05, window=3)
    assert not metrics["reverted"]
    assert metrics["reversion_time"] == 20 - metrics["peak_time_after_shock"]


def test_smoothing_uses_available_periods_at_the_start():
    smoothed = smoothed_absolute_error([0.5, 0.6, 0.7], [0.5, 0.5, 0.5], window=2)
    assert np.allclose(smoothed, [0.0, 0.05, 0.15])


def test_missing_price_carries_last_quote_forward():
    smoothed = smoothed_absolute_error([0.6, np.nan, 0.5], [0.5, 0.5, 0.5], window=1)
    assert np.allclose(smoothed, [0.1, 0.1, 0.0])


def test_km_curve_starts_at_recoveries_at_time_zero():
    times, probability, _, _ = kaplan_meier_curve([0, 0, 5, 10], [True, True, True, False], 10)
    assert times[0] == 0.0 and np.isclose(probability[0], 0.5)


def test_km_curve_stops_when_too_few_runs_remain():
    durations = [1, 2, 3, 50, 100]
    times, probability, _, _ = kaplan_meier_curve(durations, [True] * 5, 300, min_at_risk=2)
    assert times[-1] == 50.0 and probability[-1] < 1.0


def test_normalised_pull_measures_share_of_gap_closed():
    fundamentals = np.array([0.5, 0.2, 0.4, 0.1, 0.3])
    assert np.isclose(normalised_pull(fundamentals, fundamentals, 0.7, 1), 0.0)
    assert np.isclose(normalised_pull(np.full(5, 0.7), fundamentals, 0.7, 1), 1.0)
    halfway = fundamentals + 0.5 * (0.7 - fundamentals)
    assert np.isclose(normalised_pull(halfway, fundamentals, 0.7, 1), 0.5)
