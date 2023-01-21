# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Generator, List, MutableMapping, Optional

import mlflow
from kedro.extras.datasets.json import JSONDataSet
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, DataSetError
from kedro.pipeline.node import Node

logger = logging.getLogger("eensight")


def _flatten_dict(params: Dict[Any, Any], delimiter: str = "/") -> Dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params (dict): Dictionary containing the hyperparameters.
        delimiter (str): Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(
        input_dict: Any, prefixes: List[Optional[str]] = None
    ) -> Generator[Any, Optional[List[str]], List[Any]]:
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, MutableMapping):
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


class ActivityHooks:
    @property
    def _logger(self):
        return logging.getLogger("eensight")

    @hook_impl
    def before_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ):

        if catalog.load("app").get("namespace") == "apply":
            if "apply.activity" in inputs:
                try:
                    activity = catalog.load("apply.activity-adjusted")
                except DataSetError:
                    return
                else:
                    self._logger.warning(
                        "Adjusted activity was found in "
                        "catalog and selected as activity feature"
                    )
                    return {**inputs, **{"apply.activity": activity}}

            if node.name == "apply.create_activity_feature":
                try:
                    activity = catalog.load("apply.activity-adjusted")
                except DataSetError:
                    self._logger.warning("bad")
                    return
                else:
                    return {
                        **inputs,
                        **{"params:activity.adjusted_activity": activity},
                    }


class MLflowHooks:
    @hook_impl
    def before_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None:
        """Hook to be invoked before a node runs.
        This hook logs in mlflow all the input parameters of the nodes.

        Args:
            node (Node): The ``Node`` to run.
            catalog (DataCatalog): A ``DataCatalog`` containing the node's inputs
                and outputs.
            inputs (dict): The dictionary of node's input dataset. The keys are
                dataset names and the values are the actual loaded input data, not
                the dataset instance.
            is_async (bool): Whether the node was run in ``async`` mode.
            session_id: The id of the session.
        """

        run_id = catalog.load("app").get("run_id")
        if run_id is not None:
            namespace = catalog.load("app").get("namespace")

            with mlflow.start_run(run_id=run_id, nested=True):
                # log parameters
                params = {}
                for k, v in inputs.items():
                    # detect parameters automatically based on kedro reserved names
                    if k.startswith("params:"):
                        params[k.split(":")[-1]] = v

                params = _flatten_dict(params)
                for k, v in params.items():
                    mlflow.log_param("_".join([namespace, k]), v)

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None:
        """Hook to be invoked after a node runs.
        This hook logs in mlflow all generated metrcis.

        Args:
            node (Node): The ``Node`` that ran.
            catalog (DataCatalog): A ``DataCatalog`` containing the node's inputs
                and outputs.
            inputs (dict): The dictionary of inputs dataset. The keys are dataset
                names and the values are the actual loaded input data, not the dataset
                instance.
            outputs (dict): The dictionary of outputs dataset. The keys are dataset
                names and the values are the actual computed output data, not the dataset
                instance.
            is_async: Whether the node was run in ``async`` mode.
            session_id: The id of the session.
        """

        run_id = catalog.load("app").get("run_id")
        if run_id is not None:
            namespace = catalog.load("app").get("namespace")

            with mlflow.start_run(run_id=run_id, nested=True):
                # Log metrics
                for name, content in outputs.items():
                    dataset = catalog._get_dataset(name)
                    if isinstance(dataset, JSONDataSet):
                        for k, v in content.items():
                            mlflow.log_metric("_".join([namespace, k]), v)
