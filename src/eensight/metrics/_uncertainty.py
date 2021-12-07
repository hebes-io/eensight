# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.validation import column_or_1d


def picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray):
    """Compute the fraction of samples for which the ground truth lies within predicted interval.

    This is the Prediction Interval Coverage Probability (PICP) metric.
    From https://github.com/IBM/UQ360/blob/main/uq360/metrics/regression_metrics.py

    Args:
        y_true (numpy.ndarray): Ground truth.
        y_lower (numpy.ndarray): Predicted lower bound.
        y_upper (numpy.ndarray): Predicted upper bound.

    Returns:
        float: The fraction of samples for which the grounds truth lies within predicted interval.
    """
    y_true, y_lower, y_upper = (
        column_or_1d(y_true),
        column_or_1d(y_lower),
        column_or_1d(y_upper),
    )

    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


def mpiw(y_lower: np.ndarray, y_upper: np.ndarray):
    """Compute the average width of the prediction intervals.

    This is the Mean Prediction Interval Width (MPIW) metric. Measures the sharpness of intervals.
    From https://github.com/IBM/UQ360/blob/main/uq360/metrics/regression_metrics.py

    Args:
        y_lower (numpy.ndarray): Predicted lower bound.
        y_upper (numpy.ndarray): Predicted upper bound.

    Returns:
        float: The average width the prediction interval across samples.
    """
    y_lower, y_upper = column_or_1d(y_lower), column_or_1d(y_upper)
    return np.mean(np.abs(y_lower - y_upper))
