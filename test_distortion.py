import numpy as np

from distortion import distortion_metrics, kaplan_meier_curve, smoothed_absolute_error


def test_undistorted_run_has_zero_magnitude_and_duration():
    error = np.full(20, 0.01)
    metrics = distortion_metrics(error, shock_time=5, band=0.05, window=3)
    assert metrics["peak_excess_error"] == 0.0
    assert metrics["distortion_duration"] == 0.0
    assert metrics["recovered"]


def test_distortion_magnitude_and_duration_are_measured_from_shock():
    error = np.full(30, 0.01)
    error[10:18] = 0.30  # shock at 10, back inside the band from period 18
    metrics = distortion_metrics(error, shock_time=10, band=0.05, window=3)
    assert np.isclose(metrics["peak_excess_error"], 0.25)
    assert metrics["distortion_duration"] == 8.0
    assert metrics["recovered"]


def test_unrecovered_run_is_censored_at_horizon():
    error = np.full(30, 0.01)
    error[10:] = 0.30
    metrics = distortion_metrics(error, shock_time=10, band=0.05, window=3)
    assert not metrics["recovered"]
    assert metrics["distortion_duration"] == metrics["horizon_after_shock"] == 20


def test_smoothing_uses_available_periods_at_the_start():
    smoothed = smoothed_absolute_error([0.5, 0.6, 0.7], [0.5, 0.5, 0.5], window=2)
    assert np.allclose(smoothed, [0.0, 0.05, 0.15])


def test_km_curve_starts_at_recoveries_at_time_zero():
    times, probability, _, _ = kaplan_meier_curve([0, 0, 5, 10], [True, True, True, False], 10)
    assert times[0] == 0.0 and np.isclose(probability[0], 0.5)


def test_missing_price_carries_last_quote_forward():
    smoothed = smoothed_absolute_error([0.6, np.nan, 0.5], [0.5, 0.5, 0.5], window=1)
    assert np.allclose(smoothed, [0.1, 0.1, 0.0])
