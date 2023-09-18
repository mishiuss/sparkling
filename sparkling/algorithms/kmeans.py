from typing import List
from pyspark import Row

from .base import *


class KMeansModel(ClusteringModel):
    def __init__(self, jvm_model):
        super().__init__(jvm_model)

    @property
    def centroids(self) -> List[Row]:
        return self._jvm_model.centroids()


class KMeans(ClusteringAlgo):
    """
    K-means is a popular clustering algorithm used in machine learning and data mining. It aims to partition
    a dataset into k clusters by iteratively assigning each point to the nearest centroid and then updating
    the centroids to the mean of the points in each cluster.

    The algorithm works by randomly initializing k centroids, then repeating two steps until convergence:
    (1) assigning each point to the closest centroid based on a distance metric, and (2) updating the centroids
    to the mean of the points assigned to each cluster.

    The resulting clusters are defined by their centroid and are relatively spherical in shape. K-means is often
    used for datasets with high dimensionality or a large number of points. It requires only one parameter: k, the
    number of desired clusters.
    """

    def __init__(
            self, *,
            k: int,
            max_iterations: int,
            init_steps: int = 2,
            convergence: float = 1e-7,
            seed: Optional[int] = None
    ):
        """
        :param: k: target number of clusters
        :param max_iterations: maximum number of steps after which the stop condition is met
        :convergence: the minimum distance considered significant.
        If centroids moved less than param, then stop condition is met
        :param seed: random seed for picking initial candidates
        """
        super().__init__(
            k=k,
            max_iterations=max_iterations,
            init_steps=init_steps,
            convergence=convergence,
            seed=ClusteringAlgo.make_seed(seed)
        )

    def fit(self, sparkling_df: SparklingDF) -> KMeansModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        return KMeansModel(jvm_model)

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[KMeansModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.predict(sparkling_df)

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.kmeans.KMeans(
            kwargs['k'],
            kwargs['max_iterations'],
            kwargs['init_steps'],
            kwargs['convergence'],
            kwargs['seed']
        )


class KMeansConf(AlgoConf):
    def __init__(
            self, *,
            k=(2, Defaults.max_clusters),
            max_iterations=Defaults.max_iterations,
            init_steps=2,
            convergence=1e-7,
    ):
        super().__init__(
            KMeans,
            k=k,
            max_iterations=max_iterations,
            init_steps=init_steps,
            convergence=convergence,
        )
