# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict, defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from .encode import (
    CategoricalEncoder,
    ICatEncoder,
    ICatLinearEncoder,
    ICatSplineEncoder,
    IdentityEncoder,
    ISplineEncoder,
    ProductEncoder,
    SplineEncoder,
)
from .generate import CyclicalFeatures, DatetimeFeatures, TrendFeatures


def encoder_by_type(enc_type, props):
    if enc_type == "trend":
        return TrendFeatures(
            feature=props.get("feature"),
            include_bias=props.get("include_bias", False),
            remainder=props.get("remainder", "passthrough"),
            replace=props.get("replace", False),
        )
    elif enc_type == "datetime":
        return DatetimeFeatures(
            feature=props.get("feature"),
            remainder=props.get("remainder", "passthrough"),
            replace=props.get("replace", False),
            subset=None
            if not props.get("subset")
            else list(map(str.strip, props.get("subset").split(",")))
            if isinstance(props.get("subset"), str)
            else props.get("subset"),
        )
    elif enc_type == "cyclical":
        return CyclicalFeatures(
            seasonality=props.get("seasonality"),
            feature=props.get("feature"),
            period=props.get("period"),
            fourier_order=props.get("fourier_order"),
            remainder=props.get("remainder", "passthrough"),
            replace=props.get("replace", False),
        )
    elif enc_type == "categorical":
        return CategoricalEncoder(
            feature=props.get("feature"),
            max_n_categories=props.get("max_n_categories"),
            stratify_by=[]
            if not props.get("stratify_by")
            else list(map(str.strip, props.get("stratify_by").split(",")))
            if isinstance(props.get("stratify_by"), str)
            else props.get("stratify_by"),
            excluded_categories=[]
            if not props.get("excluded_categories")
            else list(map(str.strip, props.get("excluded_categories").split(",")))
            if isinstance(props.get("excluded_categories"), str)
            else props.get("excluded_categories"),
            unknown_value=props.get("unknown_value"),
            min_samples_leaf=props.get("min_samples_leaf", 1),
            max_features=props.get("max_features", "auto"),
            random_state=props.get("random_state"),
            encode_as=props.get("encode_as", "onehot"),
        )
    elif enc_type == "spline":
        return SplineEncoder(
            feature=props.get("feature"),
            n_knots=props.get("n_knots", 5),
            degree=props.get("degree", 3),
            strategy=props.get("strategy", "quantile"),
            extrapolation=props.get("extrapolation", "constant"),
            include_bias=props.get("include_bias", True),
        )
    elif enc_type == "linear":
        return IdentityEncoder(
            feature=props.get("feature"),
            as_filter=props.get("as_filter", False),
            include_bias=props.get("include_bias", False),
        )
    else:
        raise ValueError(f"Encoder type {enc_type} not supported")


def interaction_by_types(left_enc, right_enc, left_enc_type, right_enc_type):
    if (left_enc_type, right_enc_type) == ("categorical", "categorical"):
        return ICatEncoder(left_enc, right_enc)
    elif (left_enc_type, right_enc_type) == ("categorical", "linear"):
        return ICatLinearEncoder(encoder_cat=left_enc, encoder_num=right_enc)
    elif (left_enc_type, right_enc_type) == ("categorical", "spline"):
        return ICatSplineEncoder(encoder_cat=left_enc, encoder_num=right_enc)
    elif (left_enc_type, right_enc_type) == ("linear", "linear"):
        return ProductEncoder(left_enc, right_enc)
    elif (left_enc_type, right_enc_type) == ("linear", "categorical"):
        return ICatLinearEncoder(encoder_cat=right_enc, encoder_num=left_enc)
    elif (left_enc_type, right_enc_type) == ("spline", "spline"):
        return ISplineEncoder(left_enc, right_enc)
    elif (left_enc_type, right_enc_type) == ("spline", "categorical"):
        return ICatSplineEncoder(encoder_cat=right_enc, encoder_num=left_enc)
    else:
        raise NotImplementedError(
            f"Interactions between encoder type `{left_enc_type}` "
            f"and encoder type `{right_enc_type}` are not supported"
        )


class LinearModelFeatures(TransformerMixin, BaseEstimator):
    """A transformer that generates linear features and pairwise interactions.

    Args:
        model_structure : dict, default=None
            The model configuration
    """

    def __init__(self, model_structure: Dict[str, Dict] = None):
        self.model_structure = model_structure
        self.model_structure_ = {
            "add_features": defaultdict(dict),
            "main_effects": defaultdict(dict),
            "interactions": defaultdict(dict),
        }
        self.transformers_ = []
        self.encoders_ = {
            "main_effects": OrderedDict({}),
            "interactions": OrderedDict({}),
        }
        self.train_feature_cols_ = []
        self.component_names_ = []

    def add_new_feature(self, *, name, enc_type, feature=None, **kwargs):
        if name in self.model_structure_["add_features"]:
            raise ValueError(f"Feature generator named {name} has already been added")
        if enc_type not in ("trend", "datetime", "cyclical"):
            raise ValueError(f"Feature generator type {enc_type} is not supported")

        self.model_structure_["add_features"][name].update(
            dict(
                type=enc_type,
                feature=feature,
                **kwargs,
            )
        )
        return self

    def add_main_effect(self, *, name, enc_type, feature, **kwargs):
        if name in self.model_structure_["main_effects"]:
            raise ValueError(f"Encoder named {name} has already been added")
        if enc_type not in ("linear", "spline", "categorical"):
            raise ValueError(f"Encoder type enc_type {enc_type} is not supported")

        self.model_structure_["main_effects"][name].update(
            dict(
                type=enc_type,
                feature=feature,
                **kwargs,
            )
        )
        return self

    def add_interaction(
        self,
        *,
        left_name,
        right_name,
        left_enc_type,
        right_enc_type,
        left_feature,
        right_feature,
        **kwargs,
    ):
        for enc_type in (left_enc_type, right_enc_type):
            if enc_type not in ("linear", "spline", "categorical"):
                raise ValueError(f"enc_type {enc_type} is not supported")

        self.model_structure_["interactions"][(left_name, right_name)][
            left_name
        ] = dict(
            type=left_enc_type,
            feature=left_feature,
            **kwargs[left_name],
        )
        self.model_structure_["interactions"][(left_name, right_name)][
            right_name
        ] = dict(
            type=right_enc_type,
            feature=right_feature,
            **kwargs[right_name],
        )
        return self

    def _create_transformers(self):
        if (self.model_structure is not None) and (
            "add_features" in self.model_structure
        ):
            self.model_structure_["add_features"] = dict(
                self.model_structure["add_features"],
                **self.model_structure_["add_features"],
            )
        for _, props in self.model_structure_["add_features"].items():
            enc_type = props.get("type")
            self.transformers_.append(encoder_by_type(enc_type, props))

    def _create_encoders(self):
        if (self.model_structure is not None) and (
            "main_effects" in self.model_structure
        ):
            self.model_structure_["main_effects"] = dict(
                self.model_structure["main_effects"],
                **self.model_structure_["main_effects"],
            )

        if (self.model_structure is not None) and (
            "interactions" in self.model_structure
        ):
            self.model_structure_["interactions"] = dict(
                self.model_structure["interactions"],
                **self.model_structure_["interactions"],
            )

        for name, props in self.model_structure_["main_effects"].items():
            enc_type = props.get("type")
            self.encoders_["main_effects"][name] = encoder_by_type(enc_type, props)

        for name, props in self.model_structure_["interactions"].items():
            left, right = name
            left_enc_type = props[left].get("type")
            right_enc_type = props[right].get("type")
            left_enc = encoder_by_type(left_enc_type, props[left])
            right_enc = encoder_by_type(right_enc_type, props[right])
            interaction = interaction_by_types(
                left_enc, right_enc, left_enc_type, right_enc_type
            )
            self.encoders_["interactions"][name] = interaction

    def _main_effects(self, X, y=None, fitting=True):
        for name, encoder in self.encoders_["main_effects"].items():
            if fitting:
                encoder.fit(X, y)
                yield name, encoder.n_features_out_
            else:
                yield name, encoder.transform(X)

    def _interaction_effects(self, X, y=None, fitting=True):
        for name, encoder in self.encoders_["interactions"].items():
            if fitting:
                encoder.fit(X, y)
                yield name, encoder.n_features_out_
            else:
                yield name, encoder.transform(X)

    @property
    def component_matrix(self):
        """Dataframe indicating which columns of the feature matrix correspond
        to which components.

        Returns
        -------
        feature_cols: A binary indicator dataframe. Entry is 1 if that column is used
            in that component.
        """
        if self.train_feature_cols_ is None:
            raise ValueError(
                "The estimator must be fitted before the `component_matrix` can be accessed."
            )

        components = pd.DataFrame(
            {
                "col": np.arange(len(self.train_feature_cols_)),
                "component": [x.split("_delim_")[0] for x in self.train_feature_cols_],
            }
        )
        # Convert to a binary matrix
        feature_cols = pd.crosstab(
            components["col"],
            components["component"],
        ).sort_index(level="col")

        return feature_cols

    def fit(self, X, y=None):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        feature_cols = []
        self._create_transformers()
        # Apply the transformers
        if self.transformers_:
            self.transformers_ = make_pipeline(*self.transformers_)
            X = self.transformers_.fit_transform(X)

        self._create_encoders()
        # Fit the main effect encoders
        for name, n_features_out_ in self._main_effects(X, y, fitting=True):
            if "_delim_" in name:
                raise ValueError('The name of the regressor cannot include "_delim_"')
            feature_cols.extend([f"{name}_delim_{i}" for i in range(n_features_out_)])

        # Fit the interaction encoders
        for (left, right), n_features_out_ in self._interaction_effects(
            X, y, fitting=True
        ):
            if ("_delim_" in left) or ("_delim_" in right):
                raise ValueError('The name of the regressor cannot include "_delim_"')
            feature_cols.extend(
                [f"{left}:{right}_delim_{i}" for i in range(n_features_out_)]
            )

        self.train_feature_cols_ = feature_cols
        self.component_names_ = self.component_matrix.columns.tolist()
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")

        if self.transformers_:
            X = self.transformers_.transform(X)

        design_matrix = np.zeros((len(X), len(self.train_feature_cols_)))

        # Add the main effects
        for name, features in self._main_effects(X, fitting=False):
            relevant_cols = self.component_matrix.loc[
                self.component_matrix[name] == 1
            ].index
            design_matrix[:, relevant_cols] = features

        # Add the interactions
        for name, features in self._interaction_effects(X, fitting=False):
            left, right = name
            relevant_cols = self.component_matrix.loc[
                self.component_matrix[f"{left}:{right}"] == 1
            ].index
            design_matrix[:, relevant_cols] = features

        return design_matrix
