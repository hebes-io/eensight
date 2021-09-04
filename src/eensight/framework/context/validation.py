# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numbers
from collections import defaultdict

from marshmallow import INCLUDE, Schema, ValidationError, fields

##########################################################################
# Field validation functions
##########################################################################


def validate_max_features(x):
    if (x is None) or (x in ("auto", "sqrt", "log2")):
        return
    if x.replace(".", "", 1).isdigit():
        if isinstance(x, numbers.Integral):
            return
        elif (x > 0) and (x <= 1):
            return
    raise ValidationError(
        "`max_features` can be int, float between 0 and 1, or one of 'auto', 'sqrt', 'log2'"
    )


def validate_encode_as(x):
    if (x is not None) and (x not in ("onehot", "ordinal")):
        raise ValidationError("`encode_as` can be either 'onehot' or 'ordinal'")


def validate_strategy(x):
    if (x is not None) and (x not in ("uniform", "quantile")):
        raise ValidationError(
            "`strategy` can be one of 'uniform' or 'quantile', or array-like of numbers"
        )


def validate_extrapolation(x):
    if (x is not None) and (x not in ("error", "constant", "linear", "continue")):
        raise ValidationError(
            "`extrapolation` can be one of 'error', 'constant', 'linear', 'continue'"
        )


##########################################################################
# Validation schemas
##########################################################################

LinearSchema = Schema.from_dict(  # data is passed to IdentityEncoder
    {
        "feature": fields.String(required=False, allow_none=True, load_default=None),
        "as_filter": fields.Boolean(
            required=False, allow_none=True, load_default=False
        ),
        "include_bias": fields.Boolean(
            required=False, allow_none=True, load_default=False
        ),
        "interaction_only": fields.Boolean(
            required=False, allow_none=True, load_default=False
        ),
    }
)

CategoricalSchema = Schema.from_dict(
    {
        "feature": fields.String(required=True),
        "max_n_categories": fields.Integer(
            required=False, allow_none=True, load_default=None
        ),
        "stratify_by": fields.String(  # string of comma-separated values
            required=False, allow_none=True, load_default=None
        ),
        "excluded_categories": fields.String(  # string of comma-separated values
            required=False, allow_none=True, load_default=None
        ),
        "unknown_value": fields.Integer(
            required=False, allow_none=True, load_default=None
        ),
        "min_samples_leaf": fields.Integer(
            required=False, allow_none=True, load_default=None
        ),
        "max_features": fields.String(
            required=False,
            allow_none=True,
            load_default="auto",
            validate=validate_max_features,
        ),
        "random_state": fields.Integer(
            required=False, allow_none=True, load_default=None
        ),
        "encode_as": fields.String(
            required=False,
            allow_none=True,
            load_default="onehot",
            validate=validate_encode_as,
        ),
        "interaction_only": fields.Boolean(
            required=False, allow_none=True, load_default=False
        ),
    }
)

SplineSchema = Schema.from_dict(
    {
        "feature": fields.String(required=True),
        "n_knots": fields.Integer(required=False, allow_none=True, load_default=5),
        "degree": fields.Integer(required=False, allow_none=True, load_default=3),
        "strategy": fields.String(
            required=False,
            allow_none=True,
            load_default="quantile",
            validate=validate_strategy,
        ),
        "extrapolation": fields.String(
            required=False,
            allow_none=True,
            load_default="constant",
            validate=validate_extrapolation,
        ),
        "include_bias": fields.Boolean(
            required=False, allow_none=True, load_default=True
        ),
        "interaction_only": fields.Boolean(
            required=False, allow_none=True, load_default=False
        ),
    }
)


def _get_validated_props(props):
    validated_props = (
        CategoricalSchema(unknown=INCLUDE).load(props)
        if props["type"] == "categorical"
        else LinearSchema(unknown=INCLUDE).load(props)
        if props["type"] == "linear"
        else SplineSchema(unknown=INCLUDE).load(props)
        if props["type"] == "spline"
        else None
    )
    if validated_props is None:
        raise ValueError(f"Type {props['type']} not recognized")
    return validated_props


def parse_model_config(config):
    model_structure = {
        "main_effects": defaultdict(dict),
        "interactions": defaultdict(dict),
    }

    if "regressors" in config:
        for name, props in config["regressors"].items():
            model_structure["main_effects"][name] = _get_validated_props(props)

    if "interactions" in config:
        # example of pair_name: temperature, hour
        for pair_name, pair_props in config["interactions"].items():
            pair_name = tuple([x.strip() for x in pair_name.split(",")])
            if len(pair_name) != 2:
                raise ValueError("Only pairwise interactions are supported.")

            for name in pair_name:
                if name in model_structure["main_effects"]:
                    props = dict(
                        model_structure["main_effects"][name],
                        **pair_props.get(name, dict()),
                    )
                    model_structure["interactions"][pair_name].update(
                        {f"{name}": _get_validated_props(props)}
                    )
                    if model_structure["main_effects"][name]["interaction_only"]:
                        del model_structure["main_effects"][name]
                elif name in pair_props:
                    model_structure["interactions"][pair_name].update(
                        {f"{name}": _get_validated_props(pair_props[name])}
                    )
                else:
                    raise ValueError(
                        f"The regressor `{name}` has not been added yet and not "
                        "enough information has been provided so that to add it"
                    )
    return model_structure
