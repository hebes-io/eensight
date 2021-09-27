# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from eensight.pipelines.model_selection.nodes import split_training_data


def apply_ensemble(data, model, include_components):
    X_test, y_test, outliers = split_training_data(data, drop_outliers=False)

    other_outliers = X_test.filter(like="outlier", axis=1)
    if not other_outliers.empty:
        X_test = X_test.drop(other_outliers.columns, axis=1)
        outliers = np.logical_or(outliers, other_outliers.apply(any, axis=1))

    pred = model.predict(X_test, include_components=include_components)
    savings = pred["consumption"] - y_test["consumption"]
    return pred, savings
