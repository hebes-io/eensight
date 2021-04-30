# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd 

from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklego.preprocessing import ColumnSelector, RepeatingBasisFunction, PatsyTransformer

from eensight.base import _BaseHeterogeneousEnsemble
from eensight.preprocessing.utils import DateFeatureTransformer
from eensight.preprocessing._day_typing import  MMCFeatureTransformer
from eensight.preprocessing._clustering import ClusterPredictor, NOISE



class ConsumptionPredictorLin(RegressorMixin, _BaseHeterogeneousEnsemble):
    """Linear regression model for predicting energy consumption.
    
    Parameters
    ----------
    classifier : A scikit-learn classifier (default=eensight.preprocessing.ClusterPredictor)
        A classifier that answers the question "To which cluster should I allocate a given observation?".
    regressor : A scikit-learn regressor (default=sklearn.linear_model.LinearRegression)
        A regressor for predicting the target.
    n_basis_temperature : int (default=5)
        The number of degrees of freedom to use for the spline that approximates the `temperature` feature
    n_basis_hours : int (default=8)
         The number of degrees of freedom to use for the spline that approximates the `hour_of_day` feature
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe passed to `fit` and 
        `predict` should have a column with the specified names.
    """

    @_deprecate_positional_args
    def __init__(self, classifier=None, 
                       regressor=None, 
                       n_basis_temperature=5, 
                       n_basis_hours=8, 
                       extra_regressors=None):
        
        self.classifier = classifier or ClusterPredictor()
        self.regressor = regressor or LinearRegression(fit_intercept=False)
        self.n_basis_temperature = n_basis_temperature
        self.n_basis_hours = n_basis_hours
        self.extra_regressors = None if extra_regressors is None else list(extra_regressors)
        self.estimators = [('classifier', self.classifier), ('regressor', self.regressor)] 


    def _validate_input_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input values are expected as pandas DataFrames (ndim=2).') 
        
        X = pd.DataFrame(data=check_array(X), index=X.index, columns=X.columns)
        
        if self.extra_regressors is not None:
            for name in self.extra_regressors:
                if name not in X:
                    raise ValueError(f'Regressor {name} missing from dataframe')
                if name == 'temperature':
                    raise ValueError('Temperature is already treated as an interaction term. '
                                     'It cannot be added as an extra linear regressor too')
                X[name] = pd.to_numeric(X[name])    
        return X


    def _validate_target_data(self, y):
        if not isinstance(y, pd.DataFrame):
            raise ValueError('Target values are expected as pandas DataFrame (ndim=2).')
        
        if y.shape[1] > 1:
            raise ValueError('This estimator expects a univariate target')
        else:
            self.target_name_ = y.columns[0]
        
        y = pd.DataFrame(data=check_array(y, copy=True), index=y.index, columns=y.columns)
        return y
    
    
    def _design_matrix_blocks(self, X, clusters):
        try:
            self.feature_pipeline_
        
        except AttributeError:
            self.feature_pipeline_ = Pipeline([
                ('design_matrix', FeatureUnion([
                        ('month',       RepeatingBasisFunction(
                                                n_periods=12, 
                                                remainder='drop', 
                                                column='month', 
                                                input_range=(1,12)
                                        )
                        ),
                        ('day_of_week', RepeatingBasisFunction(
                                                n_periods=7, 
                                                remainder='drop',
                                                column='dayofweek', 
                                                input_range=(0, 6)
                                        )
                        ),
                        ('temperature',  PatsyTransformer(
                                            f'te(cr(temperature, {self.n_basis_temperature}),'
                                               f'cc(hour, {self.n_basis_hours})) - 1')
                        ),
                        ('grab_extra_regressors', 'drop' if self.extra_regressors is None 
                                                  else ColumnSelector(self.extra_regressors)
                        ),
                    ])
                )
            ])
            
            self.feature_pipeline_.fit(X)
        
        finally:
            features = pd.DataFrame(data=self.feature_pipeline_.transform(X), 
                                    index=X.index
            )
    
            for cat in sorted(clusters.unique()):
                yield features.mask(
                    cond=pd.Series(
                        data=features.index.isin(clusters[clusters != cat].index), 
                        index=features.index
                    ), 
                    other=0
                )
            
    
    def fit(self, X, y):
        try:
            check_is_fitted(self, 'fitted_')
        except NotFittedError:
            pass
        else:
            raise Exception('Estimator object can only be fit once. '
                            'Instantiate a new object.')

        X = self._validate_input_data(X)
        y = self._validate_target_data(y)
        check_consistent_length(X, y)

        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        mmc_features = MMCFeatureTransformer().fit_transform(X_with_dates)
        
        classifier = self.named_estimators['classifier']
        clusters = classifier.fit_predict(mmc_features)
        clusters = pd.concat((clusters, classifier.probabilities_), axis=1)
        clusters.index = clusters.index.map(lambda x: x.date())
        clusters = (
            pd.DataFrame(index=X_with_dates.index).join(clusters, how='left')
                                                  .fillna(method='ffill')
        )
        clusters = clusters[clusters['cluster'] != NOISE]
        X_with_dates = X_with_dates.loc[clusters.index]
        
        features = []
        for block in self._design_matrix_blocks(X_with_dates, clusters['cluster']):
            features.append(block)
        
        features  = pd.concat(features, axis=1)
        y         = y.loc[features.index].values.ravel()
        regressor = self.named_estimators['regressor']
        
        regressor.fit(features, y, sample_weight=clusters['probability'])
        self.fitted_ = True
        return self 


    def predict(self, X):
        check_is_fitted(self, 'fitted_')
        X = self._validate_input_data(X)
        
        X_with_dates = DateFeatureTransformer(remainder='passthrough').transform(X)
        mmc_features = MMCFeatureTransformer().transform(X_with_dates)
        
        classifier = self.named_estimators['classifier']
        clusters = classifier.predict(mmc_features)
        clusters.index = clusters.index.map(lambda x: x.date())
        clusters = (
            pd.DataFrame(index=X_with_dates.index).join(clusters, how='left')
                                                  .fillna(method='ffill')
        )

        features = []
        for block in self._design_matrix_blocks(X_with_dates, clusters['cluster']):
            features.append(block)
        features = pd.concat(features, axis=1)
        
        regressor = self.named_estimators['regressor']
        return pd.DataFrame(data=regressor.predict(features).squeeze(), 
                            columns=[f'{self.target_name_}_pred'], 
                            index=X.index
        )




        
        
        








