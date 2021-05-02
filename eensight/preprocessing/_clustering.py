# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd

from joblib import Memory
from hdbscan import HDBSCAN
from collections import OrderedDict
from sklearn.utils import check_array
from scipy.spatial.distance import cdist
from sklearn.utils.validation import _deprecate_positional_args

from eensight.base import _BaseHeterogeneousEnsemble


NOISE = -1

DEFAULT_HDBSCAN_PARAMS = {
        'min_cluster_size' : 5, 
        'min_samples' : None, 
        'cluster_selection_epsilon' : 0.0, 
        'metric' : 'euclidean', 
        'alpha' : 1.0, 
        'p' : None, 
        'algorithm' : 'best', 
        'leaf_size' : 40, 
        'memory' : Memory(location=None), 
        'approx_min_span_tree' : True, 
        'gen_min_span_tree' : False, 
        'core_dist_n_jobs' : 4, 
        'cluster_selection_method' : 'eom', 
        'allow_single_cluster' : False,
}


class RankedPoints:
    def __init__(self, points, clusterer, metric='euclidean', 
                                          selection_method='centroid', 
                                          ignored_index=None):
        """ Rank points in a cluster based on their distance to the cluster centroid/medoid
        
        Parameters
        ----------
        points : array of shape (n_samples, n_features), and must be the same data passed into
                 HDBSCAN
        clusterer : Instance of HDBSCAN that has been fit to data
        metric: string or callable, optional (default='euclidean')
            The metric to use when calculating distance between points in a cluster and 
            the cluster centroid/medoid. If metric is a string or callable, it must be one of
            the options allowed by scipy.spatial.distance.cdist for its metric parameter.
        
        selection_method: string, optional (default='centroid')
            Method to use to find the weighted cluster center. Allowed options are 'centroid' 
            and 'medoid'.
        
        """
        self.clusterer = clusterer
        self.metric = metric
        
        allowed_methods = ['centroid', 'medoid']
        if selection_method not in allowed_methods:
            raise ValueError(f'Selection method must be one of {allowed_methods}')
        
        if selection_method == 'centroid' and metric != 'euclidean':
            raise ValueError(f'Metric must be euclidian when using selection_method centroid. '
                             f'Current metric is {metric}')
        
        self.selection_method = selection_method
        
        self._embedding_cols = [str(i) for i in range(points.shape[1])]
        self.embedding_df = pd.DataFrame(data=np.array(points), 
                                         index=points.index, 
                                         columns=self._embedding_cols)
        self.embedding_df['cluster'] = clusterer.labels_

        if ignored_index is not None:
            self.embedding_df = self.embedding_df[~np.isin(self.embedding_df.index, ignored_index)]


    def calculate_all_distances_to_center(self):
        """For each cluster calculate the distance from each point to the centroid/medoid"""
        all_distances = pd.DataFrame()
        for label in np.unique(self.embedding_df['cluster']):           
            distance_df = self.calculate_distances_for_cluster(label)
            all_distances = pd.concat([all_distances, distance_df])
        
        self.embedding_df = self.embedding_df.merge(all_distances, left_index=True, right_index=True)
    

    def calculate_distances_for_cluster(self, cluster_id):
        """For a given cluster_id calculate the distance from each point to the centroid/medoid.
        
        Parameters
        ----------
        cluster_id : int
            The id of the cluster to compute the distances for. If the cluster id is -1 which
            corresponds to the noise point cluster, then this will return a distance of NaN.

        Returns
        -------
        df : A pandas DataFrame containing the distances from each point to the cluster centroid/medoid.
             The index of the dataframe corresponds to the index in the original data. 

        """
        cluster_of_interest = self.embedding_df[self.embedding_df['cluster'] == cluster_id].copy()
        
        if cluster_of_interest.empty:
            raise ValueError(f'Cluster id {cluster_id} not found')
        
        # Don't calculate distances for the noise cluster
        if cluster_id == -1:
            return pd.DataFrame(np.nan, columns=['dist_to_rep_point'], index=cluster_of_interest.index)
        
        if self.selection_method == 'centroid':
            rep_point = self.clusterer.weighted_cluster_centroid(cluster_id)
        if self.selection_method == 'medoid':
            rep_point = self.clusterer.weighted_cluster_medoid(cluster_id)
        
        dists = cdist(rep_point.reshape((1,len(self._embedding_cols))), 
                      cluster_of_interest[self._embedding_cols].values, metric=self.metric
        )
        return pd.DataFrame(dists[0], columns=['dist_to_rep_point'], index=cluster_of_interest.index)
    

    def rank_cluster_points_by_distance(self, cluster_id):
        """For a given cluster return a pandas dataframe of points ranked 
           by distance to the cluster centroid/medoid
        """
        cluster_of_interest = self.embedding_df[self.embedding_df['cluster'] == cluster_id].copy()
        
        if cluster_of_interest.empty:
            raise ValueError(f'Cluster id {cluster_id} not found')
            
        if 'dist_to_rep_point' not in self.embedding_df.columns:
            distance_df = self.calculate_distances_for_cluster(cluster_id)
            cluster_of_interest = cluster_of_interest.merge(distance_df, 
                                        left_index=True, right_index=True)
        
        cluster_of_interest.sort_values('dist_to_rep_point', inplace=True)
        return cluster_of_interest
    

    def get_closest_samples_for_cluster(self, cluster_id, n_samples=5):
        """Get the N closest points to the cluster centroid/medoid"""
        return self.rank_cluster_points_by_distance(cluster_id).head(n_samples)
    
    
    def get_furthest_samples_for_cluster(self, cluster_id, n_samples=5):
        """Get the N points furthest away from the cluster centroid/medoid"""
        return self.rank_cluster_points_by_distance(cluster_id).tail(n_samples)



class Clusterer(_BaseHeterogeneousEnsemble):
    """ A wrapper around HDBSCAN models.
        
    Parameters
    ----------
    min_cluster_size : int, optional (default=5)
        The minimum size of clusters; single linkage splits that contain
        fewer points than this will be considered points "falling out" of a
        cluster rather than a cluster splitting into two new clusters.
    min_samples : int, optional (default=None)
        The number of samples in a neighbourhood for a point to be
        considered a core point.
    metric : string, or callable, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
    p : int, optional (default=None)
        p value to use if using the minkowski metric.
    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
    cluster_selection_epsilon: float, optional (default=0.0)
		A distance threshold. Clusters below this value will be merged.
    algorithm : string, optional (default='best')
        Exactly which algorithm to use; hdbscan has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``best``
            * ``generic``
            * ``prims_kdtree``
            * ``prims_balltree``
            * ``boruvka_kdtree``
            * ``boruvka_balltree``
    leaf_size: int, optional (default=40)
        If using a space tree algorithm (kdtree, or balltree) the number
        of points ina leaf node of the tree. This does not alter the
        resulting clustering, but may have an effect on the runtime
        of the algorithm.
    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.
    gen_min_span_tree: bool, optional (default=False)
        Whether to generate the minimum spanning tree with regard
        to mutual reachability distance for later analysis.
    core_dist_n_jobs : int, optional (default=4)
        Number of parallel jobs to run in core distance computations (if
        supported by the specific algorithm). For ``core_dist_n_jobs``
        below -1, (n_cpus + 1 + core_dist_n_jobs) are used.
    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:
            * ``eom``
            * ``leaf``
    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.
    ignored_index : array-like of datetime.date objects, optional (default=None)
        Includes dates the profiles of which should not be included in the exemplars.
        Exemplars are members of the input set that are representative of their clusters.
    exemplar_size: int, optional (default=4)
        The number of exemplars to store 
    """
    @_deprecate_positional_args
    def __init__(self, ignored_index=None, exemplar_size=4, **params):
        self.ignored_index = ignored_index
        self.exemplar_size = exemplar_size
        # Duplicates are resolved in favor of the value in params
        self._estimator_params = dict(DEFAULT_HDBSCAN_PARAMS, **params)
        self.estimators = [('estimator', HDBSCAN(**self._estimator_params))] 
        
        
    def fit(self, X, y=None):
        X = pd.DataFrame(data=check_array(X), index=X.index, columns=X.columns)

        estimator = self.named_estimators['estimator']
        estimator.fit(X)
        
        self.labels_ = pd.DataFrame(data=estimator.labels_, 
                                    index=X.index, 
                                    columns=['label']
        )
        self.probabilities_ = pd.DataFrame(data=estimator.probabilities_, 
                                           index=X.index, 
                                           columns=['probability']
        )
        self.year_coverage_ = len(np.unique(X.index.dayofyear)) / 365
        
        ranked_points = RankedPoints(X, estimator, metric=estimator.metric, 
                                                   selection_method='medoid',
                                                   ignored_index=self.ignored_index)
        self.exemplars_ = OrderedDict()
        for cat in sorted(np.unique(estimator.labels_)):
            if cat == NOISE:
                continue 
            else:
                self.exemplars_[cat] = ranked_points.get_closest_samples_for_cluster(cat, 
                                                        n_samples=self.exemplar_size
                                        ).index
        self.fitted_ = True
        return self 


    def fit_predict(self, X, y=None):
        """
        Perform clustering on `X` and returns cluster labels.
        """
        self.fit(X)
        return self.labels_
        
        
        


        

