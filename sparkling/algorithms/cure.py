from .base import *


class CUREModel(ClusteringModel):
    def __init__(self, jvm_model):
        super().__init__(jvm_model)


class CURE(ClusteringAlgo):
    """
    CURE (Clustering Using Representatives) is a hierarchical clustering algorithm that is designed to handle large
    datasets efficiently.
    The CURE algorithm works in several main steps:
    1. Representative selection: Randomly select a subset of data points as initial representatives.
    2. Cluster assignment: Assign each data point to the nearest representative based on a distance measure.
    3. Local clustering: Apply a local clustering algorithm (such as K-means) to each cluster formed by the
    representatives.
    4. Cluster merging: Merge similar clusters by comparing their representatives.
    5. Shrinking: Adjust the positions of representatives towards the centroid of their respective clusters.
    6. Repeat steps 2-5 until the desired number of clusters is obtained or a stopping criterion is met.
    """
    def __init__(
            self, *,
            k: int,
            representatives: int,
            shrink_factor: float,
            remove_outliers: bool
    ):
        """
        :param k: number of clusters
        :param representatives: number of initial data points
        :param shrink_factor: factor that impacts moving speed of the representatives towards centroid
        :param remove_outliers: true, if algorithm should label outliers with noise label
        """
        super().__init__(
            k=k,
            representatives=representatives,
            shrink_factor=shrink_factor,
            remove_outliers=remove_outliers
        )

    def fit(self, sparkling_df: SparklingDF) -> CUREModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        return CUREModel(jvm_model)

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[ClusteringModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.predict(sparkling_df)

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.cure.CURE(
            kwargs['k'],
            kwargs['representatives'],
            kwargs['shrink_factor'],
            kwargs['remove_outliers']
        )


class CUREConf(AlgoConf):
    def __init__(
            self, *,
            k=(2, Defaults.max_clusters),
            representatives=(2, Defaults.sqrt_n),
            shrink_factor=(1e-5, 1.0),
            remove_outliers=frozenset([True, False])
    ):
        super().__init__(
            CURE,
            k=k,
            representatives=representatives,
            shrink_factor=shrink_factor,
            remove_outliers=remove_outliers
        )
