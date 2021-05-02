# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args

from eensight.base import _BaseHeterogeneousEnsemble
from eensight.preprocessing.utils import as_list
from eensight.prediction._gradient_boosting import Regressor
from eensight.preprocessing._clustering import Clusterer, NOISE
from eensight.preprocessing._day_typing import DateFeatureTransformer, MMCFeatureTransformer


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)



class ClusterPredictor(_BaseHeterogeneousEnsemble):
    """
    A composite model that uses that uses a eensight.preprocessing.Clusterer to cluster the 
    input and a classifier to predict the cluster for unseen inputs.

    Parameters
    ----------
    clusterer : A clustering estimator (default=eensight.preprocessing.Clusterer)
    classifier : A scikit-learn classifier (default=sklearn.neighbors.KNeighborsClassifier)
        A classifier that answers the question "To which cluster should I allocate a given observation?".
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    """
    @_deprecate_positional_args
    def __init__(self, clusterer=None, classifier=None, n_neighbors=5, weights='distance'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.clusterer = clusterer or Clusterer()
        self.classifier = classifier or KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self.estimators = [('assign_clusters', self.clusterer), ('predict_clusters', self.classifier)] 


    def fit(self, X, y=None):
        X = pd.DataFrame(data=check_array(X), index=X.index, columns=X.columns)
        
        labels = self.named_estimators['assign_clusters'].fit_predict(X)
        labels = labels[labels['label'] != NOISE] #filter the noise
        X = X[X.index.isin(labels.index)]
        
        self.named_estimators['predict_clusters'].fit(np.array(X), np.array(labels).ravel())
        self.fitted_ = True
        return self 


    def predict(self, X):
        check_is_fitted(self, 'fitted_')
        
        X = pd.DataFrame(data=check_array(X), index=X.index, columns=X.columns)

        if ((self.named_estimators['assign_clusters'].year_coverage_ > 0.9) and 
            (self.named_estimators['predict_clusters'].get_params()['n_neighbors'] > 1)
        ):
            self.named_estimators['predict_clusters'].set_params(**{'n_neighbors': 1})
            logger.info(f'Coverage is large enough to use n_neighbors=1')
        
        return pd.DataFrame(
                data=self.named_estimators['predict_clusters'].predict(X), 
                index=X.index,
                columns=['label']
        )


    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.named_estimators['assign_clusters'].labels_
        
        


class ConsumptionPredictor(RegressorMixin, _BaseHeterogeneousEnsemble):
    """Regression model for predicting energy consumption.
    
    Parameters
    ----------
    classifier : A classifier (default=eensight.prediction.ClusterPredictor)
        A classifier that answers the question "To which cluster should I allocate a given observation?".
    regressor : A regressor (default=eensight.prediction.Regressor)
        A regressor for predicting the target.
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe passed to `fit` and 
        `predict` should have a column with the specified names.
    """

    @_deprecate_positional_args
    def __init__(self, classifier=None, 
                       regressor=None, 
                       extra_regressors='temperature'):
        
        self.classifier = classifier or ClusterPredictor()
        self.regressor = regressor or Regressor()
        self.extra_regressors = None if extra_regressors is None else as_list(extra_regressors)
        self.estimators = [('classifier', self.classifier), ('regressor', self.regressor)] 


    def _validate_input_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input values are expected as pandas DataFrames (ndim=2).') 
        
        X = X.dropna(axis=0, how='any')
        X = pd.DataFrame(data=check_array(X), index=X.index, columns=X.columns)
        
        if self.extra_regressors is not None:
            for name in self.extra_regressors:
                if name not in X:
                    raise ValueError(f'Regressor {name} missing from dataframe')
                X[name] = pd.to_numeric(X[name])    
        return X


    def _validate_target_data(self, y, index):
        if not isinstance(y, pd.DataFrame):
            raise ValueError('Target values are expected as pandas DataFrame (ndim=2).')
        
        if y.shape[1] > 1:
            raise ValueError('This estimator expects a univariate target')
        else:
            self.target_name_ = y.columns[0]
        
        y = y[y.index.isin(index)]
        y = pd.DataFrame(data=check_array(y), index=y.index, columns=y.columns)
        return y
    
    
    def fit(self, X, y):
        try:
            check_is_fitted(self, 'fitted_')
        except NotFittedError:
            pass
        else:
            raise Exception('Estimator object can only be fit once. '
                            'Instantiate a new object.')

        X = self._validate_input_data(X)
        y = self._validate_target_data(y, X.index)
        check_consistent_length(X, y)

        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        mmc_features = MMCFeatureTransformer().fit_transform(X_with_dates)

        classifier = self.named_estimators['classifier']
        labels = classifier.fit_predict(mmc_features)
        
        probabilities = classifier.named_estimators['assign_clusters'].probabilities_
        labels = pd.concat((labels, probabilities), axis=1)
        exemplars = classifier.named_estimators['assign_clusters'].exemplars_

        target_name = y.columns[0]
        y['date'] = y.index.date
        y['time'] = y.index.time
        
        self.codebook_ = {}
        for cat in exemplars.keys():
            subset = (y[np.isin(y.index.date, exemplars[cat].date)]
                       .pivot(index='time', columns='date', values=target_name)
            )
        
            self.codebook_[cat] = subset.median(axis=1)
        
        
        self.time_step_ = X_with_dates.index.to_series().diff().min()
        labels = labels.resample(self.time_step_).first().groupby(lambda x: x.date).pad()
        labels = labels[labels['label'] != NOISE]
        
        common_index = labels.index.intersection(X_with_dates.index)
        labels = labels.loc[common_index]
        X_with_dates = X_with_dates.loc[common_index]
        
        features = ['month', 'dayofweek', 'hour']
        if self.extra_regressors is not None:
            features.extend(self.extra_regressors)
        
        features = X_with_dates[features]
        features.insert(0, 'label', labels['label'].astype(np.int16).astype('category'))
    
        y = y.loc[common_index, target_name]
        y = pd.concat((y, labels[['label']]), axis=1)

        def _apply_codebook(x):
            x.index = x.index.map(lambda x: x.time)
            return x[target_name] - self.codebook_[x['label'][0]]

        y = y.groupby(['label', lambda x: x.date]).apply(lambda x: _apply_codebook(x)).stack()
        y.index = y.index.droplevel().map(lambda x: datetime.combine(x[0], x[1]))
        y = y.sort_index()
        y.name = target_name
        
        regressor = self.named_estimators['regressor']
        regressor.cat_features = regressor.cat_features + ['label']
        regressor.fit(features, y, sample_weight=labels['probability'])
        self.fitted_ = True
        return self 


    def predict(self, X):
        check_is_fitted(self, 'fitted_')
        X = self._validate_input_data(X)
        
        X_with_dates = DateFeatureTransformer(remainder='passthrough').transform(X)
        mmc_features = MMCFeatureTransformer().transform(X_with_dates)
        
        classifier = self.named_estimators['classifier']
        labels = classifier.predict(mmc_features)
        labels = labels.resample(self.time_step_).first().groupby(lambda x: x.date).pad()
        
        common_index = labels.index.intersection(X_with_dates.index)
        labels = labels.loc[common_index]
        X_with_dates = X_with_dates.loc[common_index]
        
        features = ['month', 'dayofweek', 'hour']
        if self.extra_regressors is not None:
            features.extend(self.extra_regressors)
        
        features = X_with_dates[features]
        features.insert(0, 'label', labels['label'].astype(np.int16).astype('category'))

        regressor = self.named_estimators['regressor']
        pred = regressor.predict(features)
        pred = pd.concat((pred, labels[['label']]), axis=1)

        def _inverse_apply_codebook(x):
            x.index = x.index.map(lambda x: x.time)
            return x[regressor.target_name_] + self.codebook_[x['label'][0]]

        pred = pred.groupby(['label', lambda x: x.date]).apply(lambda x: _inverse_apply_codebook(x)).stack()
        pred.index = pred.index.droplevel().map(lambda x: datetime.combine(x[0], x[1]))
        pred = pred.sort_index()
        return pred 





        
        
        








