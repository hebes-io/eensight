# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

from eensight.utils import as_list


def generate_samples(
    n_samples, *, prediction: pd.Series, significance: List[float], quantiles: ArrayLike
):
    if not isinstance(prediction, pd.Series):
        raise ValueError("`prediction` msut be a pandas Series")

    significance = as_list(significance)
    n_obs = len(prediction)
    n_significance = len(significance)

    cdf = np.zeros((n_obs, 2 * n_significance))
    columns = []

    for i, s in enumerate(significance):
        columns.append((1 - s) / 2)
        cdf[:, i] = prediction - pd.Series(data=quantiles[:, i], index=prediction.index)

    for i, s in enumerate(significance):
        columns.append((1 + s) / 2)
        cdf[:, i + n_significance] = prediction + pd.Series(
            data=quantiles[:, i], index=prediction.index
        )

    cdf = pd.DataFrame(data=cdf, index=prediction.index, columns=columns)
    cdf = cdf.reindex(sorted(cdf.columns), axis=1)
    samples = np.random.uniform(size=(n_obs, n_samples))

    for i, (_, row) in enumerate(cdf.iterrows()):
        f = PchipInterpolator(row.index.to_numpy(), row.values, extrapolate=True)
        samples[i, :] = f(samples[i, :])

    return pd.DataFrame(data=samples, index=prediction.index)
