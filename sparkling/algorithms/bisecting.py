from typing import List
from pyspark import Row

from .base import *


class BisectingKMeansModel(ClusteringModel):
    """
    Bisecting K-means is a variation of the traditional K-means clustering algorithm. It follows a divisive hierarchical
    clustering approach, where clusters are split recursively into smaller clusters until a desired number of clusters
    is achieved.
    """
    def __init__(self, jvm_model):
        super().__init__(jvm_model)

    @property
    def centroids(self) -> List[Row]:
        return self._jvm_model.centroids()


class BisectingKMeans(ClusteringAlgo):
    def __init__(
            self, *,
            k: int,
            max_iterations: int,
            min_cluster_size: float = 1.0,
            convergence: float = 1e-7,
            seed: Optional[int] = None
    ):
        """
        :param k: number of clusters
        :param max_iterations: maximum number of steps after which the stop condition is met
        :param: min_cluster_size: minimal number of objects (percentage) in cluster.
        :convergence: the minimum distance considered significant.
        If centroids moved less than param, then stop condition is met
        :seed: random seed
        """
        super().__init__(
            k=k,
            max_iterations=max_iterations,
            min_cluster_size=min_cluster_size,
            convergence=convergence,
            seed=ClusteringAlgo.make_seed(seed)
        )

    def fit(self, sparkling_df: SparklingDF) -> BisectingKMeansModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        return BisectingKMeansModel(jvm_model)

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[BisectingKMeansModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.predict(sparkling_df)

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.bisecting.BisectingKMeans(
            kwargs['k'],
            kwargs['max_iterations'],
            kwargs['min_cluster_size'],
            kwargs['convergence'],
            kwargs['seed']
        )


class BisectingKMeansConf(AlgoConf):
    def __init__(
            self, *,
            k=(2, Defaults.max_clusters),
            max_iterations=Defaults.max_iterations,
            min_cluster_size=(0.01, 1.0),
            convergence=1e-7
    ):
        super().__init__(
            BisectingKMeans,
            k=k,
            max_iterations=max_iterations,
            min_cluster_size=min_cluster_size,
            convergence=convergence
        )
