# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd


def linear_impute(X: pd.Series, window=6, copy=True) -> pd.Series:
    if copy:
        X = X.copy()

    dt = X.index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    limit = int(window * pd.Timedelta("1H") / time_step)

    return X.interpolate(
        method="slinear",
        limit_area="inside",
        limit_direction="both",
        axis=0,
        limit=limit,
    )
