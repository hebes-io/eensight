# -*- coding: utf-8 -*-
# From https://github.com/scikit-learn/scikit-learn/tree/main/sklearn
# since it is not yet part of the stable version
# License: BSD 3 clause

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._split import _RepeatedSplits


class RepeatedStratifiedGroupKFold(_RepeatedSplits):
    """Repeated Stratified K-Folds iterator variant with non-overlapping groups.

    Repeats StratifiedGroupKFold n times with different randomization in each repetition.

    Args:
        n_splits (int, optional): Number of folds. Must be at least 2. Defaults to 5.
        n_repeats (int, optional): Number of times the cross-validator needs to be
            repeated. Defaults to 5.
        random_state (int, RandomState instance or None, optional): Controls the
            randomness of each repeated cross-validation instance. Pass an int for
            reproducible output across multiple function calls. Defaults to None.
    """

    def __init__(self, *, n_splits=5, n_repeats=5, random_state=None):
        super().__init__(
            StratifiedGroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )
