# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def cvrmse(y_true, y_pred):
    resid = y_true - y_pred
    return float(np.sqrt((resid ** 2).sum() / len(resid)) / np.mean(y_true))


def nmbe(y_true, y_pred):
    resid = y_true - y_pred
    return float(np.mean(resid) / np.mean(y_true))
