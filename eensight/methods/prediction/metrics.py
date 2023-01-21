# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn.utils.validation import column_or_1d


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute the Coefficient of Variation of the Root Mean Squared Error.

    Args:
        y_true (np.ndarray): Ground truth (correct) target values.
        y_pred (np.ndarray): Estimated target values.

    Returns:
        float: The metric's value.
    """

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)

    resid = y_true - y_pred
    return float(np.sqrt((resid**2).sum() / len(resid)) / np.mean(y_true))


def nmbe(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute the Normalized Mean Bias Error.

    Args:
        y_true (np.ndarray): Ground truth (correct) target values.
        y_pred (np.ndarray): Estimated target values.

    Returns:
        float: The metric's value.
    """

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)

    resid = y_true - y_pred
    return float(np.mean(resid) / np.mean(y_true))
