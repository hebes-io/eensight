# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#import logging
import calendar

import numpy as np 
import pandas as pd
import statsmodels.api as sm

from typing import Union
from pampy import match
from collections import OrderedDict
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted


#logger = logging.getLogger(__file__)
#logger.setLevel(logging.INFO)


class SeasonalModel(BaseEstimator):
    """ Predictor for time series data using seasonal decomposition.
    Includes adapted code from https://github.com/facebook/prophet 

    Parameters
    ----------
    name: str. The name of the target. 
    trend : The trend to include in the model {'n', 'c', 't', 'ct'}
        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.
        Default: 'c'.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate. 
        Default: 'auto'.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
        Default: 'auto'.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
        Default: 'auto'.
    """
    def __init__(self, name: str, 
                       trend: str = 'c',
                       yearly_seasonality: Union[str, bool, int] = 'auto', 
                       weekly_seasonality: Union[str, bool, int] = 'auto', 
                       daily_seasonality: Union[str, bool, int] = 'auto'
    ):
        if trend not in ('n', 'c', 't', 'ct'):
            raise ValueError('Parameter "trend" should be "n", "c", "t", "ct".')
        
        self.name = name 
        self.trend = trend 
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality 
        # Set during fitting
        self.t_scaler = None
        self.model = None
        self.seasonalities = OrderedDict({})
        
    
    def add_seasonality(self, name: str, period: float = None, 
                                         fourier_order: int = None, 
                                         condition_name: str = None):
        """Add a seasonal component with specified period and number of Fourier components.
        
        If condition_name is provided, the dataframe passed to `fit` and `predict` should 
        have a column with the specified condition_name containing booleans that indicate 
        when to apply seasonality.
        
        Parameters
        ----------
        name: string name of the seasonality component.
        period: float number of days in one period.
        fourier_order: int number of Fourier components to use.
        condition_name: string name of the seasonality condition.
        
        Returns
        -------
        The estimator object.
        """
        if self.model is not None:
            raise Exception('Seasonality must be added prior to model fitting.')
        
        if name not in ['daily', 'weekly', 'yearly']:
            if (period is None) or (fourier_order is None):
                raise ValueError('When adding custom seasonalities, values for '
                                 '"period" and "fourier_order" must be specified.')

        if (period is not None) and (period <= 0):
            raise ValueError('Period must be > 0')
        if (fourier_order is not None) and (fourier_order <= 0):
            raise ValueError('Fourier Order must be > 0')
        
        self.seasonalities[name] = {
            'period': float(period) if period is not None else None,
            'fourier_order': int(fourier_order) if fourier_order is not None else None,
            'condition_name': condition_name,
        }
        return self


    def _set_seasonalities(self, dates):
        first = dates.min()
        last = dates.max()
        dt = dates.diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()

        default_params = {'period': None,
                          'fourier_order': None,
                          'condition_name': None}

        # Set yearly seasonality
        if (self.yearly_seasonality is False) or ('yearly' in self.seasonalities):
            pass 
        elif self.yearly_seasonality is True:
            self.seasonalities['yearly'] = default_params
        elif self.yearly_seasonality == 'auto':
            #Turn on yearly seasonality if there is >=1 years of history
            if last - first >= pd.Timedelta(days=365):
                self.seasonalities['yearly'] = default_params
        elif self.yearly_seasonality <= 0:
                raise ValueError('Fourier order must be > 0')
        else:
            self.seasonalities['yearly'] = dict(default_params, fourier_order=self.yearly_seasonality)
            
        # Set weekly seasonality
        if (self.weekly_seasonality is False) or ('weekly' in self.seasonalities):
            pass 
        elif self.weekly_seasonality is True:
            self.seasonalities['weekly'] = default_params
        elif self.weekly_seasonality == 'auto':
            #Turn on yearly seasonality if there is >=1 years of history
            if ((last - first >= pd.Timedelta(weeks=1)) and
                                    (time_step < pd.Timedelta(weeks=1))):
                self.seasonalities['weekly'] = default_params
        elif self.weekly_seasonality <= 0:
                raise ValueError('Fourier order must be > 0')
        else:
            self.seasonalities['weekly'] = dict(default_params, fourier_order=self.weekly_seasonality)
           
        # Set daily seasonality
        if (self.daily_seasonality is False) or ('daily' in self.seasonalities):
            pass 
        elif self.daily_seasonality is True:
            self.seasonalities['daily'] = default_params
        elif self.daily_seasonality == 'auto':
            #Turn on yearly seasonality if there is >=1 years of history
            if ((last - first >= pd.Timedelta(days=1)) and
                                    (time_step < pd.Timedelta(days=1))):
                self.seasonalities['daily'] = default_params
        elif self.daily_seasonality <= 0:
                raise ValueError('Fourier order must be > 0')
        else:
            self.seasonalities['daily'] = dict(default_params, fourier_order=self.daily_seasonality)
            
        return self  


    @staticmethod
    def _fourier_series(dates, period, order):
        """Provides Fourier series components with the specified frequency
        and order.
        
        Parameters
        ----------
        dates: pd.Series containing timestamps.
        period: Number of days of the period.
        order: Number of components.
        
        Returns
        -------
        Matrix with seasonality features.
        """
        # convert to days since epoch
        t = np.array(
            (dates - dates[0].to_pydatetime())
                .dt.total_seconds()
                .astype(np.float64)
        ) / (3600 * 24.)
        
        return np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / period))
            for i in range(order)
            for fun in (np.sin, np.cos)
        ])


    def _seasonality_features(self, dates, X):
        """ Generate seasonality features. """
        for name, props in self.seasonalities.items():
            condition_name = props['condition_name']

            period = props['period'] or match(name,  'daily',    1,
                                                     'weekly',   7,
                                                     'yearly',   365.25)
            
            fourier_order = props['fourier_order'] or match(name, 'daily',    4,
                                                                  'weekly',   3,
                                                                  'yearly',   10)

            features = pd.DataFrame(
                            data=self._fourier_series(dates, period, fourier_order), 
                            columns=[f'{name}_{order}' for order in range(1, 2*fourier_order+1)],
                            index=dates)
        
            if condition_name is not None:
                features[~X[condition_name]] = 0

            yield features  


    def _gather_all_features(self, X):
        """Creates a consolidated Dataframe with all features."""
        all_features = []
        dates = X.index.to_series()
    
        if self.trend in ['c', 'ct']:
            all_features.append(
                pd.DataFrame(data=np.ones([len(dates), 1]),
                             index=dates,
                             columns=['offset']
                )
            )
                    
        if self.trend in ['t', 'ct']:
            if self.t_scaler is None:
                self.t_scaler = MinMaxScaler().fit(dates.to_frame())
            
            all_features.append(
                pd.DataFrame(data=self.t_scaler.transform(dates.to_frame()).squeeze(),
                             index=dates,
                             columns=['growth']
                )
            )
        
        for feature in self._seasonality_features(dates, X):
            all_features.append(feature)

        all_features = pd.concat(all_features, axis=1)
        return all_features


    def _validate_input_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input values are expected as pandas DataFrames (ndim=2).') 

        if self.name not in X:
            raise ValueError(f'Dataframe must have column {self.name}') 
    
        if np.isinf(X[self.name].values).any():
            raise ValueError(f'Found infinity in column {X[self.name]}.')
        
        if np.any(X.index.duplicated()):
            raise ValueError("Index includes duplicate entries.")
        # check NaN
        if X.index.isna().any():
            raise ValueError('Found NaN in index.')
        # check sorted
        if not X.index.is_monotonic_increasing:
            raise ValueError("Time series must have an ascending time index. ")
        
        for props in self.seasonalities.values():
            condition_name = props['condition_name']
            
            if condition_name is not None:
                if condition_name not in X.columns:
                    raise ValueError(
                        'Condition {condition_name!r} missing from input'
                        .format(condition_name=condition_name)
                    )

                if not X[condition_name].isin([True, False]).all():
                    raise ValueError(
                        'Found non-boolean in column {condition_name!r}'
                        .format(condition_name=condition_name)
                    )
                X.loc[:, condition_name] = X[condition_name].astype('bool')
                
        return X

    
    def fit(self, X: pd.DataFrame, y=None):
        if self.model is not None:
            raise Exception('Estimator object can only be fit once. '
                            'Instantiate a new object.')
        
        X = self._validate_input_data(X)
        self._set_seasonalities(X.index.to_series())
        features = self._gather_all_features(X)
        
        target = X[self.name].dropna()
        features = features.loc[target.index]
        
        self.model = sm.RLM(target, features, M=sm.robust.norms.HuberT()).fit()
        self.is_fitted_ = True
        return self


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input_data(X)
        features = self._gather_all_features(X)
        pred = self.model.predict(features)
        return pd.DataFrame(data=pred.squeeze(), 
                            columns=[f'{self.name}_seasonal_pred'], 
                            index=X.index
        )


    def fit_predict(self, X: pd.DataFrame, y=0) -> pd.DataFrame:
        return self.fit(X, y).predict(X)



def seasonal_predict(data: pd.DataFrame, 
                     target_name: str='consumption',
                     trend: str = 'c',
                     yearly_seasonality: Union[str, bool, int] = 'auto', 
                     weekly_seasonality: Union[str, bool, int] = 'auto', 
                     daily_seasonality: Union[str, bool, int] = 'auto',
                     return_model=True) -> pd.DataFrame:
    data = data[[target_name]].copy()

    if target_name == 'temperature':
        model = SeasonalModel(name=target_name, trend=trend, yearly_seasonality=True, 
                              weekly_seasonality=False, daily_seasonality=True) 

    elif target_name == 'consumption':
        model = SeasonalModel(name=target_name, trend=trend, yearly_seasonality=True, 
                              weekly_seasonality=False, daily_seasonality=False)
        
        data['DayOfWeek'] = data.index.dayofweek.map(lambda x: calendar.day_abbr[x])
        data = (data.merge(pd.get_dummies(data['DayOfWeek']), left_index=True, right_index=True)
                    .drop('DayOfWeek', axis=1))

        for i in range(7):
            day = calendar.day_abbr[i]
            model.add_seasonality(f'daily_{day}', period=1, fourier_order=4, condition_name=day)

    else:
        model = SeasonalModel(name=target_name, trend=trend, 
                              yearly_seasonality=yearly_seasonality, 
                              weekly_seasonality=weekly_seasonality, 
                              daily_seasonality=daily_seasonality)

    pred = model.fit_predict(data)
    
    if return_model:
        return model, pred
    else:
        return pred

