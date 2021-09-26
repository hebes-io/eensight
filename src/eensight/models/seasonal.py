# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from eensight.features.compose import LinearModelFeatures
from eensight.utils import as_list, check_X


class SeasonalDecomposer(TransformerMixin, BaseEstimator):
    """Seasonal decomposition model for time series data.

    Args:
        feature: str
            The name of the time series feature to decompose.
        dt : str, default=None
            The name of the input dataframe's column that contains datetime information.
            If None, it is assumed that the datetime information is provided by the
            input dataframe's index.
        add_trend : bool, default=False
            If True, a linear time trend will be added.
        yearly_seasonality: Fit yearly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
            Default: 'auto'.
        weekly_seasonality: Fit weekly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
            Default: 'auto'.
        daily_seasonality: Fit daily seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
            Default: 'auto'.
        alpha : float, default=1
            Parameter for the underlying ridge estimator. It must be a positive float.
            Regularization improves the conditioning of the problem and reduces the
            variance of the estimates. Larger values specify stronger regularization.
    """

    def __init__(
        self,
        feature: str,
        dt: str = None,
        add_trend: bool = False,
        yearly_seasonality: Union[str, bool, int] = "auto",
        weekly_seasonality: Union[str, bool, int] = "auto",
        daily_seasonality: Union[str, bool, int] = "auto",
        alpha=1,
    ):
        self.feature = feature
        self.dt = dt
        self.add_trend = add_trend
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.alpha = alpha

        # Set during fitting
        self.seasonalities_ = OrderedDict({})
        self.base_estimator_ = Ridge(fit_intercept=True, alpha=alpha)

    def add_seasonality(
        self,
        name: str,
        period: float = None,
        fourier_order: int = None,
        condition_name: str = None,
    ):
        """Add a seasonal component with specified period and number of Fourier components.

        If condition_name is provided, the input dataframe passed to `fit` and `predict` should
        have a column with the specified condition_name containing booleans that indicate
        when to apply seasonality.

        Args:
            name : string name of the seasonality component.
            period : float number of days in one period.
            fourier_order : int number of Fourier components to use.
            condition_name : string name of the seasonality condition.

        Returns:
            The estimator object.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception("Seasonality must be added prior to model fitting.")

        if name not in ["daily", "weekly", "yearly"]:
            if (period is None) or (fourier_order is None):
                raise ValueError(
                    "When adding custom seasonalities, values for "
                    '"period" and "fourier_order" must be specified.'
                )

        if (period is not None) and (period <= 0):
            raise ValueError("Period must be > 0")
        if (fourier_order is not None) and (fourier_order <= 0):
            raise ValueError("Fourier order must be > 0")

        self.seasonalities_[name] = {
            "period": float(period) if period is not None else None,
            "fourier_order": int(fourier_order) if fourier_order is not None else None,
            "condition_name": condition_name,
        }
        return self

    def _set_seasonalities(self, X):
        dates = X.index.to_series() if self.dt is None else X[self.dt]
        first = dates.min()
        last = dates.max()
        dt = dates.diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()

        default_params = {"period": None, "fourier_order": None, "condition_name": None}

        # Set yearly seasonality
        if (self.yearly_seasonality is False) or ("yearly" in self.seasonalities_):
            pass
        elif self.yearly_seasonality is True:
            self.seasonalities_["yearly"] = default_params
        elif self.yearly_seasonality == "auto":
            # Turn on yearly seasonality if there is >=1 years of history
            if last - first >= pd.Timedelta(days=365):
                self.seasonalities_["yearly"] = default_params
        elif self.yearly_seasonality <= 0:
            raise ValueError("Fourier order must be > 0")
        else:
            self.seasonalities_["yearly"] = dict(
                default_params, fourier_order=self.yearly_seasonality
            )

        # Set weekly seasonality
        if (self.weekly_seasonality is False) or ("weekly" in self.seasonalities_):
            pass
        elif self.weekly_seasonality is True:
            self.seasonalities_["weekly"] = default_params
        elif self.weekly_seasonality == "auto":
            # Turn on yearly seasonality if there is >=1 years of history
            if (last - first >= pd.Timedelta(weeks=1)) and (
                time_step < pd.Timedelta(weeks=1)
            ):
                self.seasonalities_["weekly"] = default_params
        elif self.weekly_seasonality <= 0:
            raise ValueError("Fourier order must be > 0")
        else:
            self.seasonalities_["weekly"] = dict(
                default_params, fourier_order=self.weekly_seasonality
            )

        # Set daily seasonality
        if (self.daily_seasonality is False) or ("daily" in self.seasonalities_):
            pass
        elif self.daily_seasonality is True:
            self.seasonalities_["daily"] = default_params
        elif self.daily_seasonality == "auto":
            # Turn on yearly seasonality if there is >=1 years of history
            if (last - first >= pd.Timedelta(days=1)) and (
                time_step < pd.Timedelta(days=1)
            ):
                self.seasonalities_["daily"] = default_params
        elif self.daily_seasonality <= 0:
            raise ValueError("Fourier order must be > 0")
        else:
            self.seasonalities_["daily"] = dict(
                default_params, fourier_order=self.daily_seasonality
            )
        return self

    def _create_composer(self):
        composer = LinearModelFeatures()

        if self.add_trend:
            composer = composer.add_new_feature(
                name="add_trend",
                enc_type="trend",
                feature=self.dt,
                include_bias=False,
                remainder="passthrough",
                replace=False,
            )
            composer = composer.add_main_effect(
                name="trend",
                enc_type="linear",
                feature="growth",
                as_filter=False,
                include_bias=False,
            )

        for seasonality, props in self.seasonalities_.items():
            condition_name = props["condition_name"]

            composer = composer.add_new_feature(
                name=seasonality,
                enc_type="cyclical",
                seasonality=seasonality,
                feature=self.dt,
                period=props.get("period"),
                fourier_order=props.get("fourier_order"),
                remainder="passthrough",
                replace=False,
            )

            if condition_name is None:
                composer = composer.add_main_effect(
                    name=seasonality,
                    enc_type="linear",
                    feature=seasonality,
                    as_filter=True,
                    include_bias=False,
                )
            else:
                composer = composer.add_interaction(
                    left_name=condition_name,
                    right_name=seasonality,
                    left_enc_type="categorical",
                    right_enc_type="linear",
                    left_feature=condition_name,
                    right_feature=seasonality,
                    **{
                        condition_name: {"encode_as": "onehot"},
                        seasonality: {"as_filter": True, "include_bias": False},
                    },
                )
        return composer

    def _check_input(self, X):
        conditions = [
            props["condition_name"]
            for props in self.seasonalities_.values()
            if props["condition_name"] is not None
        ]

        regressors = as_list(self.feature) + as_list(self.dt) + conditions
        X = check_X(X, exists=regressors)

        for condition_name in conditions:
            if not X[condition_name].isin([True, False]).all():
                raise ValueError(f"Found non-boolean in column {condition_name!r}")
        return X

    def fit(self, X: pd.DataFrame, y=None):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. " "Instantiate a new object."
            )

        X = self._check_input(X)
        self._set_seasonalities(X)
        self.composer_ = self._create_composer()

        design_matrix = self.composer_.fit_transform(X)
        self.base_estimator_.fit(design_matrix, X[self.feature])
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        X = self._check_input(X)
        design_matrix = self.composer_.transform(X)

        component_names = self.composer_.component_names_
        components = pd.DataFrame(
            index=X.index, columns=component_names + ["yhat", "resid"]
        )
        component_matrix = self.composer_.component_matrix

        for name in component_names:
            subset = component_matrix[component_matrix[name] == 1].index.to_list()
            coef = self.base_estimator_.coef_.squeeze()
            pred = np.matmul(design_matrix[:, subset], coef[subset])
            components[name] = pred

        components["offset"] = self.base_estimator_.intercept_
        y_hat = pd.Series(
            data=self.base_estimator_.predict(design_matrix), index=X.index
        )
        components["yhat"] = y_hat
        components["resid"] = X[self.feature] - y_hat
        return components
