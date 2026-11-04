import numpy as np


def first_passage_convergence(
    prices,
    fundamentals,
    tolerance=0.02,
    window=5,
):
    """
    Measure sustained convergence of market prices
    to the underlying fundamental probability.

    Parameters
    ----------
    prices : array-like
        Observed market prices, preferably midquotes.

    fundamentals : array-like
        True probabilities at matching simulation times.

    tolerance : float
        Maximum absolute pricing error.

    window : int
        Number of consecutive periods required
        inside the tolerance band.

    Returns
    -------
    dict
        first_passage_time:
            First period of the first qualifying window.

        confirmation_time:
            Last period of the first qualifying window.

        converged:
            Whether sustained convergence occurred.

        censored:
            Whether convergence was not observed.

        absolute_error:
            Absolute pricing-error trajectory.

        within_tolerance:
            Boolean trajectory indicating whether
            each valid observation is within tolerance.
    """

    prices = np.asarray(prices, dtype=float)

    fundamentals = np.asarray(
        fundamentals,
        dtype=float,
    )

    if prices.ndim != 1 or fundamentals.ndim != 1:
        raise ValueError(
            "Prices and fundamentals must be 1D arrays"
        )

    if len(prices) != len(fundamentals):
        raise ValueError(
            "Prices and fundamentals must have equal length"
        )

    if tolerance < 0:
        raise ValueError(
            "Tolerance must be nonnegative"
        )

    if not isinstance(window, (int, np.integer)):
        raise ValueError(
            "Window must be an integer"
        )

    if window < 1:
        raise ValueError(
            "Window must be at least 1"
        )

    # Missing prices must never count as convergence.
    valid = (
        np.isfinite(prices)
        & np.isfinite(fundamentals)
    )

    absolute_error = np.full(
        len(prices),
        np.nan,
        dtype=float,
    )

    absolute_error[valid] = np.abs(
        prices[valid] - fundamentals[valid]
    )

    within_tolerance = (
        valid
        & (absolute_error <= tolerance)
    )

    # Search for the first sustained convergence window.
    first_passage_time = np.nan
    confirmation_time = np.nan

    for start in range(
        len(prices) - window + 1
    ):

        end = start + window

        if np.all(within_tolerance[start:end]):

            first_passage_time = start
            confirmation_time = end - 1

            break

    converged = bool(
        np.isfinite(first_passage_time)
    )

    return {
        "first_passage_time": first_passage_time,
        "confirmation_time": confirmation_time,
        "converged": converged,
        "censored": not converged,
        "absolute_error": absolute_error,
        "within_tolerance": within_tolerance,
        "tolerance": tolerance,
        "window": window,
    }


def dimensionless_time(
    time,
    reference_timescale,
):
    """
    Convert simulation time into dimensionless time.

    t_hat = t / tau_ref

    The reference timescale should preferably be
    estimated from independent baseline simulations.
    """

    if (
        not np.isfinite(reference_timescale)
        or reference_timescale <= 0
    ):
        raise ValueError(
            "reference_timescale must be finite and positive"
        )

    time = np.asarray(time, dtype=float)

    return time / reference_timescale

def kaplan_meier_median(event_times, observed):
    """
    Median time to convergence across runs, counting runs that never converged as
    right-censored (they had not converged by the end of the simulation) rather than dropping them.

    Kaplan-Meier: S(t) = P(not yet converged by t) = prod over event times s <= t of
    (1 - n_converged_at_s / n_still_unconverged_at_s). The median is the first event time with
    S(t) <= 0.5. Returns NaN when fewer than half of the runs can be shown to converge, because
    the median is then not identified by the data.

    event_times : first-passage time for converged runs, censoring time (e.g. t_election) otherwise
    observed    : True where the run converged
    """
    event_times = np.asarray(event_times, dtype=float)
    observed = np.asarray(observed, dtype=bool)

    survival = 1.0
    for t in np.unique(event_times[observed]):
        at_risk = np.sum(event_times >= t)
        converged_now = np.sum(observed & (event_times == t))
        survival *= 1 - converged_now / at_risk
        if survival <= 0.5:
            return t
    return np.nan
