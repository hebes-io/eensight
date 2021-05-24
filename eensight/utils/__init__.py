# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ._stats import fit_pdf
from ._configuration import load_configuration
from ._prediction import validate_input_data, validate_target_data


def maybe_reshape_2d(arr):
    # Reshape output so it's always 2-d and long
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)
    return arr


def as_list(val):
    """
    Helper function, always returns a list of the input value.
    """
    if isinstance(val, str):
        return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    if val is None:
        return []
    return [val]