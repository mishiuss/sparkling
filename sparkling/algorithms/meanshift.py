from typing import List
from pyspark import Row

from .base import *


class MeanShiftModel(ClusteringModel):
    def __init__(self, jvm_model):
        super().__init__(jvm_model)

    @property
    def means(self) -> List[Row]:
        return self._jvm_model.means()


class MeanShift(ClusteringAlgo):
    """
    Mean Shift is a clustering algorithm commonly used in machine learning and computer vision. It works by
    iteratively shifting the centroids of the clusters towards the mean of the points in their local neighborhood
    until convergence.

    The algorithm starts by initializing each data point as a cluster centroid. Then, for each centroid, it computes
    the mean of the points in its neighborhood within a specified bandwidth. The centroid is then shifted towards
    this mean, and the process repeats until convergence.

    The resulting clusters are defined by their mode and can have arbitrary shapes. Mean Shift is often used for
    datasets with non-uniform density or complex shapes, and can handle noise and outliers effectively. It requires
    one main parameter: the bandwidth, which determines the size of the neighborhood used to compute the mean.
    """

    def __init__(
            self, *,
            radius: float,
            max_clusters: int,
            max_iterations: int,
            initial: int,
            convergence: float = 1e-7,
            seed: Optional[int] = None
    ):
        """
        :param radius: mean-object's neighbourhood radius
        :param max_clusters: upper bound for number of clusters
        :param max_iterations: maximum number of steps after which the stop condition is met
        :param initial: number of initial candidates. Should fit into driver memory
        :param convergence: the minimum distance considered significant.
        If means shifted less than param, then stop condition is met
        :param seed: random seed for picking initial candidates
        """
        super().__init__(
            radius=radius,
            max_clusters=max_clusters,
            max_iterations=max_iterations,
            initial=initial,
            convergence=convergence,
            seed=ClusteringAlgo.make_seed(seed)
        )

    def fit(self, sparkling_df: SparklingDF) -> MeanShiftModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        return MeanShiftModel(jvm_model)

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[MeanShiftModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.predict(sparkling_df)

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.meanshift.MeanShift(
            kwargs['radius'],
            kwargs['max_clusters'],
            kwargs['max_iterations'],
            kwargs['initial'],
            kwargs['convergence'],
            kwargs['seed']
        )


class MeanShiftDefaults:
    @staticmethod
    def initial(sparkling_df: SparklingDF) -> int:
        return 2 * Defaults.max_clusters(sparkling_df)


class MeanShiftConf(AlgoConf):
    def __init__(
            self, *,
            radius=(1e-5, 0.61),
            max_clusters=Defaults.max_clusters,
            max_iterations=Defaults.max_iterations,
            initial=MeanShiftDefaults.initial,
            convergence=1e-7
    ):
        super().__init__(
            MeanShift,
            radius=radius,
            max_clusters=max_clusters,
            initial=initial,
            max_iterations=max_iterations,
            convergence=convergence
        )
