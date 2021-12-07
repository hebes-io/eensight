# -*- coding: utf-8 -*-

import numpy as np


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute the Coefficient of Variation of the Root Mean Squared Error.

    Args:
        y_true (np.ndarray): Ground truth (correct) target values.
        y_pred (np.ndarray): Estimated target values.

    Returns:
        float: The metric's value.
    """
    resid = y_true - y_pred
    return float(np.sqrt((resid ** 2).sum() / len(resid)) / np.mean(y_true))


def nmbe(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute the Normalized Mean Bias Error.

    Args:
        y_true (np.ndarray): Ground truth (correct) target values.
        y_pred (np.ndarray): Estimated target values.

    Returns:
        float: The metric's value.
    """
    resid = y_true - y_pred
    return float(np.mean(resid) / np.mean(y_true))
