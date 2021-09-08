# -*- coding: utf-8 -*-

# From https://github.com/IBM/UQ360/blob/main/uq360/metrics/regression_metrics.py

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import column_or_1d


def picp(y_true: ArrayLike, y_lower: ArrayLike, y_upper: ArrayLike):
    """
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for
    which the ground truth lies within predicted interval.

    Args:
        y_true: Ground truth
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval.
    """
    y_true, y_lower, y_upper = (
        column_or_1d(y_true),
        column_or_1d(y_lower),
        column_or_1d(y_upper),
    )

    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


def mpiw(y_lower, y_upper):
    """
    Mean Prediction Interval Width (MPIW). Computes the average width of the the prediction
    intervals. Measures the sharpness of intervals.

    Args:
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: the average width the prediction interval across samples.
    """
    y_lower, y_upper = column_or_1d(y_lower), column_or_1d(y_upper)
    return np.mean(np.abs(y_lower - y_upper))
