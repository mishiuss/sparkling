from .base import *
from .kmeans import KMeansModel


class BirchModel(KMeansModel):
    def __init__(self, jvm_model):
        super().__init__(jvm_model)


class Birch(ClusteringAlgo):
    """
    BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is a hierarchical clustering algorithm that
    is designed to handle large datasets efficiently. It was proposed by Tian Zhang, Raghu Ramakrishnan,
    and Miron Livny in 1996.

    The BIRCH algorithm builds a hierarchical structure by incrementally clustering data points into a tree-like
    structure called the CF (Clustering Feature) tree. The CF tree consists of internal nodes and leaf nodes. Internal
    nodes represent clusters, while leaf nodes represent subclusters or individual data points.

    The algorithm starts with an empty CF tree and iteratively processes the input data points. It uses a combination
    of clustering features, including the cluster center, radius, and number of data points, to represent each internal
    node. The algorithm dynamically adjusts the CF tree as new data points arrive.
    """

    def __init__(
            self, *,
            k: int,
            max_branches: int,
            threshold: float,
            max_iterations: int
    ):
        """
        :param: k: target number of clusters
        :param max_branches: maximum number of subclusters in each node.
        :param threshold: radius of the subcluster after merging a new sample
        and the closest subcluster must be lesser than this value
        :param max_iterations: maximum number of steps after which the stop condition is met
        """
        super().__init__(
            k=k,
            max_branches=max_branches,
            threshold=threshold,
            max_iterations=max_iterations
        )

    def fit(self, sparkling_df: SparklingDF) -> BirchModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        return BirchModel(jvm_model)

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[BirchModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.predict(sparkling_df)

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.birch.Birch(
            kwargs['k'],
            kwargs['max_branches'],
            kwargs['threshold'],
            kwargs['max_iterations']
        )


class BirchConf(AlgoConf):
    def __init__(
            self, *,
            k=(2, Defaults.max_clusters),
            max_branches=(2, Defaults.sqrt_n),
            threshold=(0.0, 1.0),
            max_iterations=Defaults.max_iterations
    ):
        super().__init__(
            Birch,
            k=k,
            max_branches=max_branches,
            threshold=threshold,
            max_iterations=max_iterations
        )
