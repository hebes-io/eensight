# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Any
from sklearn.utils import Bunch
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, MetaEstimatorMixin, is_classifier, is_regressor



class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for models composed of named estimators.
    Copied verbatim from sklearn.utils.metaestimators._BaseComposition  v0.24.0
    """
    steps: List[Any]

    @abstractmethod
    def __init__(self):
        pass

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
        return out

    def _set_params(self, attr, **params):
        if attr in params:
            setattr(self, attr, params.pop(attr))
        
        items = getattr(self, attr)
        names = []
        if items:
            names, _ = zip(*items)
        for name in list(params.keys()):
            if '__' not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))


class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta):
    """Base class for heterogeneous ensemble of learners.
    Copied verbatim from sklearn.ensemble._base._BaseHeterogeneousEnsemble  v0.24.0
    
    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.
    
    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    _required_parameters = ['estimators']

    @property
    def named_estimators(self):
        return Bunch(**dict(self.estimators))

    @abstractmethod
    def __init__(self, estimators):
        self.estimators = estimators

    def _validate_estimators(self):
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        # defined by MetaEstimatorMixin
        self._validate_names(names)

        has_estimator = any(est != 'drop' for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        is_estimator_type = (is_classifier if is_classifier(self) else is_regressor)

        for est in estimators:
            if est != 'drop' and not is_estimator_type(est):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        est.__class__.__name__, is_estimator_type.__name__[3:]
                    )
                )

        return names, estimators

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.
        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.
        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.
        """
        super()._set_params('estimators', **params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.
        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.
        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.
        """
        return super()._get_params('estimators', deep=deep)






    
    
    
    