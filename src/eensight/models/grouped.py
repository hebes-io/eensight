# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import OrderedDict, defaultdict
from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from eensight.features.compose import encoder_by_type
from eensight.models.linear import LinearPredictor
from eensight.utils import as_list, check_X, check_y


class GroupedPredictor(RegressorMixin, BaseEstimator):
    """Construct one estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.

    Args:
        model_structure : dict
            A configuration dictionary that includes information about the base model's
            structure.
        group_feature : str
            The name of the column of the input dataframe to use as the grouping set.
        estimator_params : dict or tuple of tuples, default=tuple()
            The parameters to use when instantiating a new base estimator. If none are given,
            default parameters are used.
        fallback : bool (default=False)
            Whether or not to fall back to a global model in case a group parameter is not
            found during `.predict()`.
    """

    def __init__(
        self,
        *,
        model_structure,
        group_feature,
        estimator_params=tuple(),
        fallback=False,
    ):
        self.model_structure = model_structure
        self.group_feature = group_feature
        self.estimator_params = estimator_params
        self.fallback = fallback
        self.model_structure_ = copy.deepcopy(model_structure)
        self.estimators_ = OrderedDict({})
        self.transformers_ = []
        self.encoders_ = {
            "main_effects": defaultdict(dict),
            "interactions": defaultdict(dict),
        }

    def _fit_single_group(self, group_name, model_structure, X, y):
        try:
            params = (
                dict(self.estimator_params) if self.estimator_params is not None else {}
            )
            estimator = LinearPredictor(model_structure=model_structure, **params)
            estimator = estimator.fit(X, y)
        except Exception as e:
            raise type(e)(f"Exception for group {group_name}: {e}")
        else:
            return estimator

    def _update_local_conf(self, conf, X, y=None, fitting=True):
        if "main_effects" in conf:
            for name, regressor in conf["main_effects"].items():
                if regressor["type"] == "categorical":
                    if fitting:
                        stratify_by = (
                            None
                            if not regressor["stratify_by"]
                            else ",".join(
                                (self.group_feature, regressor["stratify_by"])
                            )
                        )
                        enc = encoder_by_type(
                            "categorical",
                            dict(
                                regressor, encode_as="ordinal", stratify_by=stratify_by
                            ),
                        )
                        encoded = enc.fit_transform(X, y).squeeze()
                        self.encoders_["main_effects"][name] = enc
                    else:
                        enc = self.encoders_["main_effects"][name]
                        encoded = enc.transform(X).squeeze()

                    new_name = "__for__".join((regressor.get("feature"), name))
                    X[new_name] = encoded
                    regressor.update(
                        {
                            "feature": new_name,
                            "max_n_categories": None,
                            "stratify_by": None,
                        }
                    )
        if "interactions" in conf:
            for pair_name, pair_props in conf["interactions"].items():
                for name in pair_name:
                    regressor = pair_props[name]
                    if regressor["type"] == "categorical":
                        if fitting:
                            stratify_by = (
                                None
                                if not regressor["stratify_by"]
                                else ",".join(
                                    (self.group_feature, regressor["stratify_by"])
                                )
                            )
                            enc = encoder_by_type(
                                "categorical",
                                dict(
                                    regressor,
                                    encode_as="ordinal",
                                    stratify_by=stratify_by,
                                ),
                            )
                            encoded = enc.fit_transform(X, y).squeeze()
                            self.encoders_["interactions"][pair_name].update(
                                {name: enc}
                            )
                        else:
                            enc = self.encoders_["interactions"][pair_name][name]
                            encoded = enc.transform(X).squeeze()

                        new_name = "__for__".join(
                            (regressor.get("feature"), ":".join(pair_name))
                        )
                        X[new_name] = encoded
                        regressor.update(
                            {
                                "feature": new_name,
                                "max_n_categories": None,
                                "stratify_by": None,
                            }
                        )
        return (conf, X) if fitting else X

    @property
    def n_parameters(self):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError as exc:
            raise ValueError(
                "The number of parameters is acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            n_parameters = 0
            for name, est in self.estimators_.items():
                if name != "_global_":
                    n_parameters += est.n_parameters
            return n_parameters

    @property
    def dof(self):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError as exc:
            raise ValueError(
                "The degrees of freedom are acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            dof = 0
            for name, est in self.estimators_.items():
                if name != "_global_":
                    dof += est.dof
            return dof

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        if "add_features" in self.model_structure_:
            added_features = self.model_structure_.pop("add_features")
            for _, props in added_features.items():
                enc_type = props.get("type")
                self.transformers_.append(encoder_by_type(enc_type, props))

            self.transformers_ = make_pipeline(*self.transformers_)
            X = self.transformers_.fit_transform(X)

        X = check_X(X, exists=self.group_feature)
        if self.fallback and ("_global_" in X[self.group_feature]):
            raise ValueError(
                "Name `_global_` is reserved and cannot be used as a group name"
            )
        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        local_model_structure = copy.deepcopy(self.model_structure_)
        local_model_structure, X = self._update_local_conf(
            local_model_structure, X, y=y, fitting=True
        )

        for group_name, group_data in X.groupby(self.group_feature):
            self.estimators_[group_name] = self._fit_single_group(
                group_name=group_name,
                model_structure=local_model_structure,
                X=group_data.drop(self.group_feature, axis=1),
                y=y.loc[group_data.index],
            )

        if self.fallback:
            global_model_structure = self.model_structure_
            self.estimators_["_global_"] = self._fit_single_group(
                group_name="_global_",
                model_structure=global_model_structure,
                X=X.drop(self.group_feature, axis=1),
                y=y,
            )

        self.groups_ = as_list(self.estimators_.keys())
        self.fitted_ = True
        return self

    def _predict_single_group(self, group_name, X, include_components):
        """Predict a single group by getting its estimator"""
        try:
            estimator = self.estimators_[group_name]
        except KeyError:
            if self.fallback:
                estimator = self.estimators_["_global_"]
            else:
                raise ValueError(f"Found new group {group_name} during predict")
        finally:
            pred = estimator.predict(X, include_components=include_components)
            if not isinstance(pred, (pd.Series, pd.DataFrame)):
                pred = pd.DataFrame(
                    data=pred, index=X.index, columns=[self.target_name_]
                )

            return pred

    def predict(
        self, X: pd.DataFrame, include_clusters=False, include_components=False
    ):
        check_is_fitted(self, "fitted_")
        if self.transformers_:
            X = self.transformers_.transform(X)
        X = check_X(X, exists=self.group_feature)

        local_model_structure = copy.deepcopy(self.model_structure_)
        X = self._update_local_conf(local_model_structure, X, fitting=False)

        pred = None
        for group_name, group_data in X.groupby(self.group_feature):
            group_pred = self._predict_single_group(
                group_name=group_name,
                X=group_data.drop(self.group_feature, axis=1),
                include_components=include_components,
            )
            if include_clusters:
                group_pred = pd.concat(
                    (group_pred, group_data[[self.group_feature]]),
                    axis=1,
                    ignore_index=False,
                )
            pred = pd.concat((pred, group_pred), axis=0, ignore_index=False)

        pred = pred.reindex(X.index).dropna()
        return pred
