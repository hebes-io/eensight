# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np
import pandas as pd
from feature_encoders.utils import as_series
from scipy.interpolate import PchipInterpolator


def generate_samples(
    n_samples: int,
    *,
    prediction: Union[pd.Series, pd.DataFrame],
    quantiles: pd.DataFrame
):
    """Generate samples given a prediction and the quantiles of the non-conformity scores.

    Args:
        n_samples (int): The number of samples to generate.
        prediction (pandas.DataFrame or pandas.Series): The prediction time series.
        quantiles (pandas.DataFrame): A dataframe of shape `(len(X), len(significance))`
            with columns containing the significance levels, and data containing the
            corresponding quantiles of the non-conformity scores.

    Returns:
        pandas.DataFrame: Dataframe of the generated samples.
    """
    if isinstance(prediction, pd.DataFrame):
        prediction = as_series(prediction)

    significance = quantiles.columns.tolist()
    n_obs, n_significance = quantiles.shape

    cdf = pd.DataFrame(
        data=np.zeros((n_obs, 2 * n_significance)), index=prediction.index
    )
    columns = []

    for i, s in enumerate(significance):
        columns.append((1 - s) / 2)
        cdf.iloc[:, i] = prediction - quantiles[s]

    for i, s in enumerate(significance):
        columns.append((1 + s) / 2)
        cdf.iloc[:, i + n_significance] = prediction + quantiles[s]

    cdf.columns = columns
    seen = []
    for col in cdf.columns:
        if col in seen:
            cdf.pop(col)
        else:
            seen.append(col)

    cdf = cdf.reindex(sorted(cdf.columns), axis=1)
    samples = np.random.uniform(size=(n_obs, n_samples))

    for i, (_, row) in enumerate(cdf.iterrows()):
        f = PchipInterpolator(row.index.to_numpy(), row.values, extrapolate=False)
        samples[i, :] = f(samples[i, :])

    return (
        pd.DataFrame(data=samples, index=prediction.index)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
