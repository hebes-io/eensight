# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.base import clone 
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklego.preprocessing import ColumnSelector, IntervalEncoder, PatsyTransformer
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder

from eensight.preprocessing.utils import as_list
from eensight.base import _BaseHeterogeneousEnsemble
from eensight.prediction._gradient_boosting import GBRegressor
from eensight.preprocessing._clustering import Clusterer, NOISE
from eensight.preprocessing._day_typing import DateFeatureTransformer, MMCFeatureTransformer



logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)



fillna          = lambda data, method=None: data.fillna(method=method)
match_dates     = lambda data, dates: pd.DataFrame(index=dates).join(data, how='left')
match_and_fill  = lambda data, dates, method=None: data.pipe(match_dates, dates).pipe(fillna, method=method)


def _validate_input_data(X, extra_regressors, missing, fitting=True):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('Input values are expected as pandas DataFrames (ndim=2).') 
    
    for name in extra_regressors:
        if name not in X:
            raise ValueError(f'Regressor {name} missing from dataframe')

    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    categorical_extra = [col for col in categorical_cols if col in extra_regressors]

    numeric_cols = X.select_dtypes(include='number').columns
    numeric_extra = [col for col in numeric_cols if col in extra_regressors]
    
    if X.isnull().values.any():
        if missing == 'impute':
            if len(categorical_cols) > 0:
                X[categorical_cols] = X[categorical_cols].fillna(value='')
            if len(numeric_cols) > 0:
                X[numeric_cols] = X[numeric_cols].fillna(value=X[numeric_cols].median())
        elif missing == 'drop':
            X = X.dropna(axis=0, how='any')
        else:
            raise ValueError('Found missing values in input data')

    X[numeric_cols] = check_array(X[numeric_cols])
    
    if fitting:
        return X, categorical_extra, numeric_extra
    else:
        return X 


def _validate_target_data(y, index=None):
    if not isinstance(y, pd.DataFrame):
        raise ValueError('Target values are expected as pandas DataFrame (ndim=2).')
    if y.shape[1] > 1:
        raise ValueError('This estimator expects a univariate target')
    
    if index is not None:
        y = y[y.index.isin(index)]
    
    y = pd.DataFrame(data=check_array(y), index=y.index, columns=y.columns)
    return y


def _create_codebook(y, exemplars, target_name):
    codebook = {}
    
    y['date'] = y.index.date
    y['time'] = y.index.time

    for cat in exemplars.keys():
        subset = (y[np.isin(y.index.date, exemplars[cat].date)]
                        .pivot(index='time', columns='date', values=target_name)
        )
        codebook[cat] = subset.median(axis=1)
    return codebook 


def _apply_codebook(y, labels, codebook, target_name):
    y = pd.concat((y, labels), axis=1)

    def _forward_apply(x):
        x.index = x.index.map(lambda x: x.time)
        return x[target_name] - codebook[x['label'][0]]

    y = y.groupby(['label', lambda x: x.date]).apply(lambda x: _forward_apply(x)).stack()
    y.index = y.index.droplevel().map(lambda x: datetime.combine(x[0], x[1]))
    y = y.sort_index()
    y.name = target_name
    return y 

def inverse_apply_codebook(y_hat, labels, codebook, target_name):
    y_hat = pd.concat((y_hat, labels), axis=1)

    def _inverse_apply(x):
        x.index = x.index.map(lambda x: x.time)
        return x[target_name] + codebook[x['label'][0]]

    y_hat = y_hat.groupby(['label', lambda x: x.date]).apply(lambda x: _inverse_apply(x)).stack()
    y_hat.index = y_hat.index.droplevel().map(lambda x: datetime.combine(x[0], x[1]))
    y_hat = y_hat.sort_index()
    return y_hat  



#############################################################################################
#################### ClusterPredictor #######################################################
#############################################################################################


class ClusterPredictor(_BaseHeterogeneousEnsemble):
    """
    A composite model that uses an eensight.preprocessing.Clusterer to cluster the 
    input and a classifier to predict the clusters for unseen inputs.

    Parameters
    ----------
    clusterer : An eensight.preprocessing.Clusterer
    classifier : A scikit-learn classifier (default=sklearn.neighbors.KNeighborsClassifier)
        A classifier that answers the question "To which cluster should I allocate a given observation?".
    metric : string, or callable, optional (default=None)
        The metric to use if a clusterer is not provided.
    ignored_index : array-like of datetime.date objects, optional (default=None)
        Includes dates the profiles of which should not be included in the exemplars.
        Needed if a clusterer is not provided.
    exemplar_size: int, optional (default=4)
        The number of exemplars to store. Needed if a clusterer is not provided.
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        Needed if a classifier is not provided.
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
        Needed if a classifier is not provided.
    """
    @_deprecate_positional_args
    def __init__(self, clusterer=None, 
                       classifier=None, 
                       metric=None, 
                       ignored_index=None, 
                       exemplar_size=4, 
                       n_neighbors=5, 
                       weights='distance'):
        self.clusterer = clusterer
        self.classifier = classifier
        self.metric = metric
        self.ignored_index = ignored_index
        self.exemplar_size = exemplar_size
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.estimators = [
            ('assign_clusters', clusterer or Clusterer(metric=metric, 
                                                       ignored_index=ignored_index, 
                                                       exemplar_size=exemplar_size 
                                            )
            ), 
            ('predict_clusters', classifier or KNeighborsClassifier(n_neighbors=n_neighbors, 
                                                                    weights=weights
                                               )
            )
        ] 


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



#############################################################################################
#################### LinearPredictor ########################################################
#############################################################################################


class LinearPredictor(RegressorMixin, _BaseHeterogeneousEnsemble):
    """Linear regression model for predicting energy consumption.
    
    Parameters
    ----------
    classifier : An eensight.prediction.ClusterPredictor
        A classifier that answers the question "To which cluster should I allocate a given observation?".
    regressor : A regressor (default=sklearn.linear_model.LinearRegression)
        A regressor for predicting the target.
    metric : string, or callable, optional (default=None)
        The metric to use if a classifier is not provided.
    ignored_index : array-like of datetime.date objects, optional (default=None)
        Includes dates the profiles of which should not be included in the exemplars.
        Needed if a classifier is not provided.
    exemplar_size: int, optional (default=4)
        The number of exemplars to store. Needed if a classifier is not provided.
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        Needed if a classifier is not provided.
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
        Needed if a classifier is not provided.
    temperature_col : str (default='temperature')
        The name of the column containing the temperature data. The dataframe passed to `fit` and 
        `predict` should have a column with the specified name.
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe passed to `fit` and 
        `predict` should have a column with the specified names.
    missing : str (default='impute')
        Defines who missing values in input data are treated. It can be 'impute' or 'drop'
    """

    @_deprecate_positional_args
    def __init__(self, classifier=None, 
                       regressor=None, 
                       metric=None, 
                       ignored_index=None, 
                       exemplar_size=4, 
                       n_neighbors=5, 
                       weights='distance',
                       temperature_col='temperature',
                       extra_regressors=None,
                       missing='impute'):
        
        self.classifier = classifier
        self.regressor = regressor
        self.metric = metric
        self.ignored_index = ignored_index
        self.exemplar_size = exemplar_size
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.temperature_col = temperature_col
        self.extra_regressors = extra_regressors
        self.missing = missing
        self.extra_regressors_ = as_list(extra_regressors)

        self.estimators = [
            ('classifier', classifier or ClusterPredictor(metric=metric, 
                                                          ignored_index=ignored_index, 
                                                          exemplar_size=exemplar_size, 
                                                          n_neighbors=n_neighbors, 
                                                          weights=weights
                                        )
            ), 
            ('regressor',  regressor or LinearRegression(fit_intercept=False))
        ]    

    
    def _get_feature_pipeline(self, X, y=None):
        n_months = X['month'].nunique()
        n_days = X['dayofweek'].nunique()
        
        pipeline = Pipeline([
            ('features', FeatureUnion([
                    ('months', 'drop' if n_months == 1 else 
                        Pipeline([
                            ('grab_months',   ColumnSelector(['month'])),
                            ('encode_months', IntervalEncoder(n_chunks=n_months, 
                                                              span=0.1, 
                                                              method='normal'
                                              )
                            )
                        ])
                    ),

                    ('days', 'drop' if n_days == 1 else 
                        Pipeline([
                            ('grab_days',    ColumnSelector(['dayofweek'])),
                            ('encode_days',  IntervalEncoder(n_chunks=n_days, 
                                                             span=0.1, 
                                                             method='normal'
                                             )
                            )
                        ])
                    ),

                    ('hours', 
                        Pipeline([
                            ('grab_hours',   ColumnSelector(['hour'])),
                            ('encode_hours', OneHotEncoder()
                            )
                        ])
                    ),

                    ('temperature', 
                        Pipeline([
                            ('grab_temperature',   ColumnSelector([self.temperature_col])),
                            ('encode_temperature', IntervalEncoder(n_chunks=15, 
                                                                   span=0.1*X[self.temperature_col].std(), 
                                                                   method='normal'
                                                   )
                            ),
                        ])
                    ),

                    ('interactions', 
                        Pipeline([
                            ('split', FeatureUnion([
                                    ('temperature_part', Pipeline([
                                                ('grab_temperature', ColumnSelector([self.temperature_col])),
                                                ('bins', KBinsDiscretizer(n_bins=5, 
                                                                            strategy='kmeans',
                                                                            encode='ordinal')
                                                )
                                            ])
                                    ),
                                    ('hour_part', Pipeline([
                                                ('grab_hours', ColumnSelector(['hour']))
                                        ])
                                    )
                                ])
                            ),
                            ('pandarize', FunctionTransformer(lambda x: 
                                            pd.DataFrame(x, columns=[self.temperature_col, 'hour'])
                                        )
                            ),
                            ('term', PatsyTransformer(f'-1 + C({self.temperature_col}):C(hour)'))
                        ])
                    ),
                    
                    ('extra_numeric_regressors', 'drop' if not self.numeric_extra_ else 
                                                 ColumnSelector(self.numeric_extra_)
                    ),

                    ('extra_categoric_regressors', 'drop' if not self.categorical_extra_ else 
                            Pipeline([
                                ('grab_features', ColumnSelector(self.categorical_extra_)),
                                ('encode_features', OneHotEncoder(handle_unknown='ignore')
                                )
                            ])                  
                    ) 
                ])
            )
        ])
        return pipeline
        
    
    def fit(self, X, y):
        try:
            check_is_fitted(self.named_estimators['classifier'], 'fitted_')
            check_is_fitted(self.named_estimators['regressor'], 'fitted_')
            check_is_fitted(self, 'fitted_')
        except NotFittedError:
            pass
        else:
            raise Exception('Estimator object can only be fit once. '
                            'Instantiate a new object.')

        X, categorical_extra, numeric_extra = _validate_input_data(X, self.extra_regressors_, self.missing)
        self.categorical_extra_ = categorical_extra
        self.numeric_extra_ = numeric_extra
        
        if self.missing == 'drop':
            y = _validate_target_data(y, index=X.index)
        else:
            y = _validate_target_data(y)
        check_consistent_length(X, y)
        self.target_name_ = y.columns[0]

        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        mmc_features = MMCFeatureTransformer().fit_transform(X_with_dates)

        classifier = self.named_estimators['classifier']
        labels = classifier.fit_predict(mmc_features)
        
        to_match = X_with_dates.index.min().time()
        labels.index = labels.index.map(lambda t: t.replace(hour=to_match.hour, 
                                                            minute=to_match.minute, 
                                                            second=to_match.second)
        )
        
        labels = match_and_fill(labels, X_with_dates.index, method='ffill')
        labels = labels[labels['label'] != NOISE]
        X_with_dates = X_with_dates.loc[labels.index]
        X_with_dates.insert(0, 'label', labels['label'])
        y = y.loc[X_with_dates.index]

        classifier = self.named_estimators['classifier']
        exemplars = classifier.named_estimators['assign_clusters'].exemplars_
        self.codebook_ = _create_codebook(y, exemplars, self.target_name_)
        y = _apply_codebook(y, labels, self.codebook_, self.target_name_)
        
        self.estimators_ = {}
        for label, group in X_with_dates.groupby('label'):
            y_ = y.loc[group.index]
            pipe = self._get_feature_pipeline(group, y_)
            features = pipe.fit_transform(group, y_)
        
            regressor = clone(self.named_estimators['regressor'])
            regressor.fit(features, y_)
            self.estimators_[label] = (pipe, regressor)
        
        self.fitted_ = True
        return self


    def predict(self, X):
        check_is_fitted(self, 'fitted_')
        X = _validate_input_data(X, self.extra_regressors_, self.missing, fitting=False)
        
        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        mmc_features = MMCFeatureTransformer().fit_transform(X_with_dates)

        classifier = self.named_estimators['classifier']
        labels = classifier.predict(mmc_features)

        to_match = X_with_dates.index.min().time()
        labels.index = labels.index.map(lambda t: t.replace(hour=to_match.hour, 
                                                            minute=to_match.minute, 
                                                            second=to_match.second)
        )
        
        labels = match_and_fill(labels, X_with_dates.index, method='ffill')
        X_with_dates.insert(0, 'label', labels['label'])
        
        pred = None
        for label, group in X_with_dates.groupby('label'):
            pipe, regressor = self.estimators_[label]
            features = pipe.transform(group)
            res = regressor.predict(features)
            pred = pd.concat((pred, pd.DataFrame(data=res, index=group.index, columns=[self.target_name_])))
            
        pred = pred.reindex(labels.index)
        return inverse_apply_codebook(pred, labels, self.codebook_, self.target_name_)
        
        
        
#############################################################################################
#################### BoostPredictor #########################################################
#############################################################################################


class BoostPredictor(RegressorMixin, _BaseHeterogeneousEnsemble):
    """Gradient boosted tree regression model for predicting energy consumption.
    
    Parameters
    ----------
    classifier : An eensight.prediction.ClusterPredictor
        A classifier that answers the question "To which cluster should I allocate a given observation?".
    regressor : An eensight.prediction.GBPredictor
        A regressor for predicting the target.
    metric : string, or callable, optional (default=None)
        The metric to use if a classifier is not provided.
    ignored_index : array-like of datetime.date objects, optional (default=None)
        Includes dates the profiles of which should not be included in the exemplars.
        Needed if a classifier is not provided.
    exemplar_size: int, optional (default=4)
        The number of exemplars to store. Needed if a classifier is not provided.
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        Needed if a classifier is not provided.
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
        Needed if a classifier is not provided.
    temperature_col : str (default='temperature')
        The name of the column containing the temperature data. The dataframe passed to `fit` and 
        `predict` should have a column with the specified name.
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe passed to `fit` and 
        `predict` should have a column with the specified names.
    cat_features : list of str (default=None)
        The names of the categorical features
    missing : str (default='impute')
        Defines who missing values in input data are treated. It can be 'impute' or 'drop'
    """

    @_deprecate_positional_args
    def __init__(self, classifier=None, 
                       regressor=None, 
                       metric=None, 
                       ignored_index=None, 
                       exemplar_size=4, 
                       n_neighbors=5, 
                       weights='distance',
                       temperature_col='temperature',
                       extra_regressors=None,
                       cat_features=None,
                       missing='impute'):
        
        self.classifier = classifier
        self.regressor = regressor
        self.metric = metric
        self.ignored_index = ignored_index
        self.exemplar_size = exemplar_size
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.temperature_col = temperature_col
        self.extra_regressors = extra_regressors
        self.cat_features = cat_features
        self.missing = missing
        self.extra_regressors_ = as_list(extra_regressors)

        self.estimators = [
            ('classifier', classifier or ClusterPredictor(metric=metric, 
                                                          ignored_index=ignored_index, 
                                                          exemplar_size=exemplar_size, 
                                                          n_neighbors=n_neighbors, 
                                                          weights=weights
                                        )
            ), 
            ('regressor', regressor or GBRegressor(cat_features=cat_features))
        ] 


    def fit(self, X, y):
        try:
            check_is_fitted(self.named_estimators['classifier'], 'fitted_')
            check_is_fitted(self.named_estimators['regressor'], 'fitted_')
            check_is_fitted(self, 'fitted_')
        except NotFittedError:
            pass
        else:
            raise Exception('Estimator object can only be fit once. '
                            'Instantiate a new object.')

        X, categorical_extra, _ = _validate_input_data(X, self.extra_regressors_, self.missing)
        
        if categorical_extra != as_list(self.cat_features):
            left_out = [i for i in categorical_extra if i not in self.cat_features]
            raise ValueError(f'Features {left_out} are categorical but were not included in `cat_features`')
        
        if self.missing == 'drop':
            y = _validate_target_data(y, index=X.index)
        else:
            y = _validate_target_data(y)
        
        check_consistent_length(X, y)
        self.target_name_ = y.columns[0]

        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        mmc_features = MMCFeatureTransformer().fit_transform(X_with_dates)

        classifier = self.named_estimators['classifier']
        labels = classifier.fit_predict(mmc_features)
        probabilities = classifier.named_estimators['assign_clusters'].probabilities_
        labels = pd.concat((labels, probabilities), axis=1)
        
        to_match = X_with_dates.index.min().time()
        labels.index = labels.index.map(lambda t: t.replace(hour=to_match.hour, 
                                                            minute=to_match.minute, 
                                                            second=to_match.second)
        )
        
        labels = match_and_fill(labels, X_with_dates.index, method='ffill')
        labels = labels[labels['label'] != NOISE]
        X_with_dates = X_with_dates.loc[labels.index]
        X_with_dates.insert(0, 'label', labels['label'].astype(np.int16).astype('category'))
        y = y.loc[X_with_dates.index]

        #### Possible but no really useful for gradient boosting
        #classifier = self.named_estimators['classifier']
        #exemplars = classifier.named_estimators['assign_clusters'].exemplars_
        #self.codebook_ = _create_codebook(y, exemplars, self.target_name_)
        #y = _apply_codebook(y, labels, self.codebook_, self.target_name_)
        
        features = ['label', 'month', 'dayofweek', 'hour', self.temperature_col]
        features = features + self.extra_regressors_
        features = X_with_dates[features]
        
        regressor = self.named_estimators['regressor']
        regressor.cat_features = as_list(self.cat_features) + ['label']
        regressor.fit(features, y, sample_weight=labels['probability'])
        
        self.fitted_ = True
        return self
    
    
    def predict(self, X):
        check_is_fitted(self, 'fitted_')
        X = _validate_input_data(X, self.extra_regressors_, self.missing, fitting=False)
        
        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        mmc_features = MMCFeatureTransformer().fit_transform(X_with_dates)

        classifier = self.named_estimators['classifier']
        labels = classifier.predict(mmc_features)

        to_match = X_with_dates.index.min().time()
        labels.index = labels.index.map(lambda t: t.replace(hour=to_match.hour, 
                                                            minute=to_match.minute, 
                                                            second=to_match.second)
        )
        
        labels = match_and_fill(labels, X_with_dates.index, method='ffill')
        X_with_dates.insert(0, 'label', labels['label'].astype(np.int16).astype('category'))

        features = ['label', 'month', 'dayofweek', 'hour', self.temperature_col]
        features = features + self.extra_regressors_
        features = X_with_dates[features]
        
        regressor = self.named_estimators['regressor']
        regressor.cat_features = as_list(self.cat_features) + ['label']
        pred = regressor.predict(features)
        
        #### If codebook is used
        #return inverse_apply_codebook(pred, labels, self.codebook_, self.target_name_)
        return pred
    
    
    
    
    

    





        
        
        








