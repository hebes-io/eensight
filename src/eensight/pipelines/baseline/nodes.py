# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
from itertools import chain

import pandas as pd
from feature_encoders.generate import DatetimeFeatures
from feature_encoders.models import GroupedPredictor
from sklearn.pipeline import Pipeline

from eensight.features import ClusterFeatures, MMCFeatures
from eensight.metrics import cvrmse, nmbe
from eensight.models._composite import AggregatePredictor
from eensight.utils import split_training_data

logger = logging.getLogger("model-selection")


def _needs_holiday(model_config):
    if "holiday" in [
        v.get("feature")
        for v in model_config["regressors"].values()
        if isinstance(v, dict)
    ]:
        return True
    elif "holiday" in list(
        chain.from_iterable(
            [
                [k.get("feature") for k in v.values()]
                for v in model_config["interactions"].values()
                if isinstance(v, dict)
            ]
        )
    ):
        return True
    else:
        return False


def optimize_model(
    data: pd.DataFrame,
    location: dict,
    model_config: dict,
    feature_map: dict,
    distance_metrics: dict,
    of_optimize_model: dict,
):
    """Carry out the model optimization step.

    Args:
        data (pandas.DataFrame): The input data.
        location (dict): Information about the building's location (country,
            province, state)
        model_config (dict): A dictionary that includes information about the
            base model's structure.
        feature_map (dict): A dictionary that maps feature generator names to
            the classes for the generators' validation and creation.
        distance_metrics (dict): A dictionary that contains time interval information
            of the form: key: interval number, values: interval start time, interval
            end time, and components of the corresponding distance metric.
        of_optimize_model (dict): The parameters of the optimization step (typically
            read from `conf/base/parameters/baseline.yaml`).

    Raises:
        ValueError: If the selected base model requires a `holiday` feature but location
            data is not provided.

    Returns:
        tuple containing

        - **opt_params** (*dict*): The optimal parameters for the model.
        - **model** (*AggregatePredictor*): The model with the optimal parameters applied.
    """
    X_train, y_train = split_training_data(data)

    if _needs_holiday(model_config) and ("holiday" not in X_train.columns):
        if location and any(location.values()):
            logger.info("Adding holiday features to model configuration")
            model_config["add_features"].update(
                {"holidays": {"type": "holidays", **location}}
            )
        else:
            raise ValueError(
                "Cannot add holiday information because location is missing"
            )

    mcc_features = Pipeline(
        [
            ("dates", DatetimeFeatures(subset=["month", "dayofweek"])),
            ("features", MMCFeatures()),
        ]
    )
    model = AggregatePredictor(
        distance_metrics=distance_metrics,
        base_clusterer=ClusterFeatures(
            transformer=mcc_features,
            n_jobs=of_optimize_model["clusterer"].get("n_jobs", 1),
            n_neighbors=of_optimize_model["clusterer"].get("n_neighbors", 1),
            weights=of_optimize_model["clusterer"].get("weights", "uniform"),
            output_name=of_optimize_model["clusterer"].get("group_feature", "cluster"),
        ),
        base_regressor=GroupedPredictor(
            model_conf=model_config,
            feature_conf=feature_map,
            group_feature=of_optimize_model["clusterer"].get(
                "group_feature", "cluster"
            ),
            estimator_params=(
                ("alpha", of_optimize_model.get("alpha", 0.01)),
                ("fit_intercept", False),
            ),
        ),
        group_feature=of_optimize_model["clusterer"].get("group_feature", "cluster"),
    )
    opt_params = model.optimize(
        X_train,
        y_train,
        n_estimators=of_optimize_model.get("n_estimators", 3),
        test_size=of_optimize_model.get("test_size", 0.33),
        group_by=of_optimize_model.get("group_by", "week"),
        budget=of_optimize_model.get("budget", 50),
        timeout=of_optimize_model.get("timeout"),
        n_jobs=of_optimize_model.get("n_jobs"),
        verbose=of_optimize_model.get("verbose", True),
        **of_optimize_model.get("opt_space_kwards", {}),
    )
    model = model.set_params(cluster_params=opt_params)
    return opt_params, model


def fit_model(data: pd.DataFrame, model: AggregatePredictor):
    """Fit the optimized model.

    Args:
        data (pandas.DataFrame): The input data.
        model (AggregatePredictor): The optimized model.

    Returns:
        AggregatePredictor: The optimized model fitted on the input data.
    """
    X_train, y_train = split_training_data(data)
    model = model.fit(X_train, y_train)
    pred = model.predict(X_train)

    logger.info(
        "In-sample CV(RMSE) (%): "
        f"{cvrmse(y_train['consumption'], pred['consumption'])*100}"
    )
    logger.info(
        "In-sample NMBE (%): "
        f"{nmbe(y_train['consumption'], pred['consumption'])*100}"
    )
    logger.info(f"Number of parameters: {model.n_parameters}")
    logger.info(f"Degrees of freedom: {model.dof}")
    return model
