# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import holidays as hdays
import pandas as pd
from feature_encoders.encode import SplineEncoder
from feature_encoders.generate import DatetimeFeatures
from feature_encoders.utils import as_series, check_X
from pydantic import BaseModel, validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from unidecode import unidecode

from eensight.utils import merge_hours_days

#####################################################################################
# Schemas
#####################################################################################


class HolidaySchema(BaseModel):
    type: str
    country: str
    province: Optional[str] = None
    state: Optional[str] = None
    ds: Optional[str] = None
    remainder: str = "passthrough"
    replace: bool = False

    @validator("remainder")
    def check_remainder(cls, data):
        if data not in ("drop", "passthrough"):
            raise ValueError("can be either 'drop' or 'passthrough'")
        return data


class OccupancySchema(BaseModel):
    type: str
    ds: Optional[str] = None
    consumption: str = "consumption"
    temperature: str = "temperature"
    remainder: str = "passthrough"
    replace: bool = False

    @validator("remainder")
    def check_remainder(cls, data):
        if data not in ("drop", "passthrough"):
            raise ValueError("can be either 'drop' or 'passthrough'")
        return data


#####################################################################################
# Add new features
# All feature generators generate pandas DataFrames
#####################################################################################


class HolidayFeatures(TransformerMixin, BaseEstimator):
    """Add holiday features to a dataset.

    The feature generator uses the `holidays` library to generate holiday
    information. Visit https://github.com/dr-prodigy/python-holidays to
    see the supported countries.

    Args:
        country (str): The name of the country.
        province (str, optional): The name of the province. Defaults to
            None.
        state (str, optional): The name of the state. Defaults to None.
        ds (str, optional): The name of the input dataframe's column that contains
            datetime information. If None, it is assumed that the datetime information
            is provided by the input dataframe's index. Defaults to None.
        remainder ({'drop', 'passthrough'}, optional): By specifying
            ``remainder='passthrough'``, all the remaining columns of the input dataset
            will be automatically passed through (concatenated with the output of the
            transformer). Defaults to "passthrough".
        replace (bool, optional): Specifies whether replacing an existing column with
            the same name is allowed (when `remainder=passthrough`). Defaults to False.

    Raises:
        ValueError: If automatic holiday generation for the given `country` is not
            supported.
        ValueError: If ``remainder`` is neither 'drop' nor 'passthrough'.
    """

    def __init__(
        self,
        country: str,
        province: str = None,
        state: str = None,
        ds=None,
        remainder="passthrough",
        replace=False,
    ):
        try:
            getattr(hdays, country.capitalize())
        except AttributeError as e:
            raise ValueError(
                f"Holidays in {country} are not currently supported!"
            ) from e

        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.country = country
        self.province = province
        self.state = state
        self.ds = ds
        self.remainder = remainder
        self.replace = replace
        self.country_ = country.capitalize() if country is not None else None
        self.province_ = province.capitalize() if province is not None else None
        self.state_ = state.capitalize() if state is not None else None

    def _make_holidays_df(self, year_list: list):
        """Make dataframe of holidays for given years and countries.

        Args:
            year_list (list of int): List of years to generate holidays for.

        Returns:
            pandas.DataFrame: A dataframe with a column named "holiday".
        """
        holidays = getattr(hdays, self.country_)(
            prov=self.province_,
            state=self.state_,
            years=year_list,
            expand=False,
        )
        holidays_df = pd.DataFrame(
            data=[(date, holidays.get_list(date)) for date in holidays],
            columns=["timestamp", "holiday"],
        )
        holidays_df = holidays_df.explode("holiday")
        holidays_df["holiday"] = holidays_df["holiday"].map(unidecode)
        holidays_df = holidays_df.set_index("timestamp")
        return holidays_df

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature generator on the available data.

        Args:
            X (pandas.DataFrame, shape (n_samples, n_features)): The input dataframe.
            y (None, optional): There is no need of a target in the transformer, but the
                pipeline API requires this parameter. Defaults to None.

        Returns:
            HolidayFeatures: Returns the instance itself.
        """
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the feature generator.

        Args:
            X (pandas.DataFrame, shape (n_samples, n_features)): The input dataframe.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
            ValueError: If common columns are found and ``replace=False``.

        Returns:
            pandas.DataFrame: The transformed dataframe.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)

        if (
            (self.remainder == "passthrough")
            and ("holiday" in X)
            and (not self.replace)
        ):
            raise ValueError("Found common column name: `holiday`")

        if self.ds is not None:
            X = X.set_index(self.ds)

        year_list = X.groupby(lambda x: x.year).first().index.tolist()
        holidays = self._make_holidays_df(year_list)

        if (self.remainder == "passthrough") and ("holiday" in X):
            X = X.drop("holiday", axis=1)
            X = merge_hours_days(X, holidays)
        elif self.remainder == "passthrough":
            X = merge_hours_days(X, holidays)
        else:
            X = merge_hours_days(X.drop(X.columns, axis=1), holidays)

        X["holiday"] = X["holiday"].fillna(value="_novalue_")
        return X


class OccupancyFeatures(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        ds=None,
        consumption="consumption",
        temperature="temperature",
        remainder="passthrough",
        replace=False,
    ):
        """Generate date and time features.

        Args:
            ds (str, optional): The name of the input dataframe's column that contains
                datetime information. If None, it is assumed that the datetime information
                is provided by the input dataframe's index. Defaults to None.
            consumption (str, optional): The name of the input dataframe's column that contains
                the energy consumption data. Defaults to 'consumption.
            temperature (str, optional): The name of the input dataframe's column that contains
                the outdoor air temperature data. Defaults to 'temperature.
            remainder (str, :type: {'drop', 'passthrough'}, optional): By specifying
                ``remainder='passthrough'``, all the remaining columns of the input dataset
                will be automatically passed through (concatenated with the output of the
                transformer). Defaults to "passthrough".
            replace (bool, optional): Specifies whether replacing an existing column with
                the same name is allowed (when `remainder=passthrough`). Defaults to False.

        Raises:
            ValueError: If ``remainder`` is neither 'drop' nor 'passthrough'.
        """
        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.ds = ds
        self.consumption = consumption
        self.temperature = temperature
        self.remainder = remainder
        self.replace = replace

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature generator on the available data.

        Args:
            X (pandas.DataFrame): The input dataframe.
            y (None, optional): There is no need of a target in the transformer, but the
                pipeline API requires this parameter. Defaults to None.

        Returns:
            OccupancyFeatures: Returns the instance itself.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X = check_X(X, exists=[self.consumption, self.temperature])
        dmatrix = SplineEncoder(
            feature=self.temperature, degree=1, strategy="uniform"
        ).fit_transform(X)
        model = LinearRegression(fit_intercept=False).fit(dmatrix, X[self.consumption])
        pred = pd.DataFrame(
            data=model.predict(dmatrix), index=X.index, columns=[self.consumption]
        )

        resid = X[[self.consumption]] - pred
        mask = resid > 0
        mask = DatetimeFeatures(ds=self.ds, subset="hourofweek").fit_transform(mask)
        occupied = mask.groupby("hourofweek")["consumption"].mean() > 0.65
        self.mapping_ = occupied.to_dict()
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the feature generator.

        Args:
            X (pandas.DataFrame): The input dataframe.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
            ValueError: If common columns are found and ``replace=False``.

        Returns:
            pandas.DataFrame: The transformed dataframe.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)

        if (
            (self.remainder == "passthrough")
            and ("occupied" in X)
            and (not self.replace)
        ):
            raise ValueError("Found common column name: `occupied`")

        hourofweek = DatetimeFeatures(
            ds=self.ds, subset="hourofweek", remainder="drop"
        ).fit_transform(X)

        pred = (
            as_series(hourofweek).map(lambda x: self.mapping_[x]).to_frame("occupied")
        )

        if self.remainder == "passthrough":
            return X.assign(**{"occupied": pred["occupied"]})
        else:
            return pred


class MMCFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, month="month", dayofweek="dayofweek"):
        """Generate `month` and `dayofweek` one-hot encoded features for training an MMC
        (Mahalanobis Metric for Clustering) metric learning algorithm.

        Args:
            month (str, optional): The name of the input dataframe's column that contains
                the `month` feature values. Defaults to "month".
            dayofweek (str, optional): The name of the input dataframe's column that contains
            the `dayofweek` feature values. Defaults to "dayofweek".
        """
        self.month = month
        self.dayofweek = dayofweek

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature generator on the available data.

        Args:
            X (pandas.DataFrame, shape (n_samples, n_features)): The input dataframe.
            y (None, optional): There is no need of a target in the transformer, but the
                pipeline API requires this parameter. Defaults to None.

        Returns:
            MMCFeatures: Returns the instance itself.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X = check_X(X, exists=[self.month, self.dayofweek])
        self.fitted_ = True
        return self

    def transform(self, X):
        """Apply the feature generator.

        Args:
            X (pandas.DataFrame, shape (n_samples, n_features)): The input dataframe.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
            ValueError: If common columns are found and ``replace=False``.

        Returns:
            pandas.DataFrame: The transformed dataframe.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=[self.month, self.dayofweek])
        features = pd.DataFrame(0, index=X.index, columns=range(1, 20))

        temp = pd.get_dummies(X[self.month].astype(int))
        for col in temp.columns:
            features[col] = temp[col]

        temp = pd.get_dummies(X[self.dayofweek].astype(int))
        for col in temp.columns:
            features[col + 13] = temp[col]

        return features
