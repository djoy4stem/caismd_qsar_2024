from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree, KDTree
import numpy as np
import pandas as pd
from typing import Union
import warnings


def get_knn_distances(target, nn_model, k=5, return_mean=True):
    if isinstance(nn_model, (KDTree, BallTree)):
        dist = nn_model.query(target, k=k)[0]
        if not return_mean:
            return dist
        else:
            return [d.mean() for d in dist]

    else:
        raise NotImplementedError(
            "This function is implemented only for KDTree and BallTree models."
        )


class ApplicationDomainEstimator:
    def __init__(
        self,
        nn_model=None,
        max_dist: float = None,
        n_neighbors: int = 5,
        metric: str = "jaccard",
        leaf_size: int = 40,
    ):
        self.nn_model = None
        self.max_dist = None
        self.n_neighbors = None
        self.metric = metric
        self.leaf_size = leaf_size

    @classmethod
    def from_dict(cls, params_: dict):
        nn_model = params_.get("nn_model", None)
        max_dist = params_.get("max_dist", None)
        n_neighbors = params_.get("n_neighbors", 5)
        metric = params_.get("metric", "jaccard")
        leaf_size = params_.get("leaf_size", 40)

        valid = all(
            [
                not nn_model is None,
                not max_dist is None,
                not n_neighbors is None,
                not metric is None,
                not leaf_size is None,
            ]
        )
        if valid:
            return cls(
                nn_model=nn_model,
                max_dist=max_dist,
                n_neighbors=n_neighbors,
                metric=metric,
            )
        else:
            raise ValueError(
                "All of the following arguments must be non-null: 'nn_model', 'max_dist', 'n_neighbors', 'metric'."
            )

    def to_dict(self):
        return {
            "nn_model": {"data": self.nn_model.data.shape},
            "max_dist": self.max_dist,
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "leaf_size": self.leaf_size,
        }

    def train_nn_model(self, training_data, leaf_size=None):
        """
        # We personally select the Jaccard metric, because
        # the value occupy a range between 0 and 1.
        # It makes it easier to select a threshold
        """
        if leaf_size is None:
            leaf_size = self.leaf_size
        elif leaf_size != self.leaf_size:
            print(f"Changing the leaf size of the model to {leaf_size}.")
            self.leaf_size = leaf_size
        ball_tree_model = BallTree(
            training_data, leaf_size=leaf_size, metric=self.metric
        )
        self.nn_model = ball_tree_model

    def find_max_distance_treshold(
        self, test_data, step=0.05, perc=90, n_neighbors=5, max_dist=None
    ):
        """
        The function implements the following method:
        1. For each point in the training data, calculate the average distance to its 'self.n_neighbors' (default=5) nearest neighbors
        2. Determine pdval, the 'perc'-th perticile of the average distances; i.e: For 'perc'% of the data points in the training set,
            the avg. distance to their 'self.n_neighbors' neighbors is lower than this value. This will be the initial value for the max. threshold.
        3. Unless a non-null value is specified, determine max_dist, the maximum distance that any training data point has to its 'self.n_neighbors' neighbors
        3. Iterate of over the list of test points, and for each point, calculate the average distance to its 'self.n_neighbors' in the training (not test) dataset.
        4. while threshold<max_dist:
            - if less than 90% of the test set have an avg. value to theit training neighbors of less than the thresold, then increase the threshold (to accept more distant points)
            - else: break
        5. Return/set the max_threshold value
        """
        assert not (
            self.metric == "jaccard" and (not max_dist is None) and max_dist > 1
        ), "The max jaccard distance cannot be greater that 1."
        assert not (
            self.n_neighbors is None and n_neighbors is None
        ), "The default n_neighbors is None. Make sure to provide a non-null value"

        # if max_dist is None:
        #     if not self.metric in ['jaccard']:
        #         max_dist=1000
        #     else:
        #         max_dist=1

        if self.n_neighbors is None:
            self.n_neighbors = n_neighbors
        elif self.n_neighbors != n_neighbors:
            warnings.warn(
                f"You provided a n_neighbors params that is differnet from the default value. This will impact the domain of applicability estimation. The default value has been modified to {n_neighbors}."
            )
            self.n_neighbors = n_neighbors

        avg_k_nn_distances_train = []

        trc = self.nn_model.data.copy()
        tec = test_data.copy()
        if isinstance(trc, pd.DataFrame):
            trc = trc.values
        if isinstance(tec, pd.DataFrame):
            tec = tec.values

        for i in trc:
            nn_k_distances = self.nn_model.query(
                [i], k=self.n_neighbors, return_distance=True
            )[0]
            avg_k_nn_distances_train.append(nn_k_distances.mean())
        # print(avg_k_nn_distances_train)

        avg_dist = round(
            sum(avg_k_nn_distances_train) / len(avg_k_nn_distances_train), 3
        )
        min_dist = round(min(avg_k_nn_distances_train), 3)

        if max_dist is None:
            max_dist = round(max(avg_k_nn_distances_train), 3)

        pdval = round(np.percentile(np.array(avg_k_nn_distances_train), 75), 3)
        print(
            f"Min: {min_dist} - Max = {max_dist} - Avg: {avg_dist} - {perc}th-percentile: {pdval}"
        )

        nn_k_distances_test_to_train = [
            self.nn_model.query([j], k=self.n_neighbors, return_distance=True)[0].mean()
            for j in tec
        ]

        threshold = pdval
        # while threshold<starting_max_distance:
        while threshold < max_dist:
            valid_test_points = [
                int(y <= threshold) for y in nn_k_distances_test_to_train
            ]
            valid_perc_test = sum(valid_test_points) / len(valid_test_points)
            print(f"valid_perc_test = {round(valid_perc_test,3)}")

            if valid_perc_test < 0.90:
                threshold += step
            else:
                break
            if self.metric == "jaccard" and threshold > 1.0:
                raise ValueError(
                    "The threshold reached passed the max value of 1.0 imposed for the jaccard metric. Consider reducing the value of 'perc', and/or max_dist."
                )

        valid = [int(x <= threshold) for x in avg_k_nn_distances_train]
        valid_perc = round(sum(valid) / len(valid), 3)
        print(
            f"Percentage of training data points with avg. distance to {self.n_neighbors}-nearest training data neighbors < {round(threshold, 3)}: {valid_perc}"
        )

        valid_test_points = [int(y <= threshold) for y in nn_k_distances_test_to_train]
        valid_perc_test = round(sum(valid_test_points) / len(valid_test_points), 3)
        print(
            f"Percentage of test data points with avg. distance to {self.n_neighbors}-nearest training data neighbors  < {round(threshold, 3)}: {valid_perc_test}"
        )

        self.max_dist = threshold

        return self.max_dist

    def get_avg_knn_distance(self, X):
        return get_knn_distances(X, self.nn_model, k=5, return_mean=True)

    def is_within_ad(self, X: Union[np.ndarray, pd.DataFrame]):
        # if isinstance(X, pd.DataFrame):
        #     return [c<=self.max_dist for c in get_knn_distances]
        # elif isinstance(X, np.ndarray):
        return [c <= self.max_dist for c in self.get_avg_knn_distance(X)]
