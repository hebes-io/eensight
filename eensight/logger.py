# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Code adapted from https://github.com/PyTorchLightning/pytorch-lightning

import os 
import re 
import sys 
import logging
import mlflow
import mlflow.sklearn

from time import time
from typing import Any, Dict
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO, 
                    format='%(asctime)s:%(levelname)-8s: %(message)s'
)
logger = logging.getLogger(__file__)



class MLFlowLogger:
    """ Log using `MLflow <https://mlflow.org>`.
    """
    def __init__(
        self,
        tracking_uri    : str,
        experiment_name : str = 'Default',
        run_id          : str = None,
        tags            : Dict[str, Any] = None,
        prefix          : str = '',
    ):
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id = run_id
        self._experiment_id = None
        self.tags = {} if tags is None else tags
        self._prefix = prefix
        self._mlflow_client = MlflowClient(tracking_uri)


    @property
    def experiment(self) -> MlflowClient:
        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            
            if expt is not None:
                self._experiment_id = expt.experiment_id
            else:
                if self._run_id is not None:
                    raise ValueError(f'Experiment with name {self._experiment_name} not found')
                
                logger.warning(f'Experiment with name {self._experiment_name} not found. Creating it.')
                self._experiment_id = self._mlflow_client.create_experiment(name=self._experiment_name)

        if self._run_id is None:
            run = self._mlflow_client.create_run(experiment_id=self._experiment_id, 
                                                 tags=resolve_tags(self.tags)
            )
            self._run_id = run.info.run_id

        return self._mlflow_client


    @property
    def run_id(self):
        # create the experiment if it does not exist to get the run id
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self):
        # create the experiment if it does not exist to get the experiment id
        _ = self.experiment
        return self._experiment_id


    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.experiment.log_param(self.run_id, k, v)


    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        if self._prefix:
            metrics = {f'{self._prefix}/{k}': v for k, v in metrics.items()}

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(f'Discarding metric with string value {k}={v}.')
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                logger.warning(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {k} with {new_k}.")
                k = new_k

            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step)


    def log_artifact(self, local_path: str, artifact_path: str=None) -> None:
        local_path = os.path.abspath(local_path)
        self.experiment.log_artifact(self.run_id, local_path, artifact_path=artifact_path)


    def log_model(self, model, artifact_path: str):
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        
        with mlflow.start_run(run_id=self.run_id, tags=resolve_tags(self.tags)):
            mlflow.sklearn.log_model(model, artifact_path)