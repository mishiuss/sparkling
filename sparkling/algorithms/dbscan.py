from .base import *


class DBSCANModel(ClusteringModel):
    def __init__(self, jvm_model, dataframe):
        super().__init__(jvm_model)
        self._dataframe = dataframe

    @property
    def dataframe(self) -> SparklingDF:
        return self._dataframe


class DBSCAN(ClusteringAlgo):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm commonly used in
    machine learning and data mining. It groups together points that are close to each other based on a distance
    metric and a specified minimum number of points in their neighborhood.

    The algorithm works by iteratively expanding clusters from points that have a sufficient number of nearby
    points, and marking points that do not belong to any cluster as noise. The resulting clusters can be
    of any shape and size and are defined by their density.

    DBSCAN is useful for identifying clusters in datasets with non-uniform density or irregular shapes,
    and can handle noise and outliers effectively. It requires two main parameters: epsilon (the maximum distance
    between two points for them to be considered in the same neighborhood) and min_points (the minimum number of
    points required to form a dense region).
    """

    def __init__(
            self, *,
            epsilon: float,
            min_points: int,
            border_noise: bool,
            max_clusters: int,
            points_in_box: int,
            levels: int,
            axis_splits: int = 2
    ):
        """
        :param epsilon: neighborhood radius
        :param min_points: the number of objects in a point neighborhood for it to be considered as a core point
        :param border_noise: true, if border points should be considered as noise and not be assigned to clusters
        :param max_clusters: maximum number of clusters
        :param points_in_box: minimum number of points in a single subdivision of dimensions.
         Does not affect the result of a clustering, but affects performance only
        :param levels: number of procedure that split space on some dimension
         Does not affect the result of a clustering, but affects performance only
        :param axis_splits: number of splits per one dimension
         Does not affect the result of a clustering, but affects performance only
        """
        super().__init__(
            epsilon=epsilon,
            min_points=min_points,
            border_noise=border_noise,
            max_clusters=max_clusters,
            points_in_box=points_in_box,
            axis_splits=axis_splits,
            levels=levels
        )

    def fit(self, sparkling_df: SparklingDF) -> DBSCANModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        labeled_df = DataFrame(jvm_model.dataframe(), sparkling_df.sql_ctx)
        return DBSCANModel(jvm_model, sparkling_df.like(labeled_df))

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[DBSCANModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.dataframe

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.dbscan.DBSCAN(
            kwargs['epsilon'],
            kwargs['min_points'],
            kwargs['border_noise'],
            kwargs['max_clusters'],
            kwargs['points_in_box'],
            kwargs['axis_splits'],
            kwargs['levels']
        )


class DBSCANDefaults:
    @staticmethod
    def default_points_in_box(sparkling_df: SparklingDF) -> int:
        return int(5 * np.sqrt(sparkling_df.amount))

    @staticmethod
    def default_levels(sparkling_df: SparklingDF) -> int:
        return int(2 * np.log10(sparkling_df.amount))


class DBSCANConf(AlgoConf):
    def __init__(
            self, *,
            epsilon=(1e-5, 0.5),
            min_points=(2, Defaults.sqrt_n),
            border_noise=frozenset([True, False]),
            max_clusters=Defaults.max_clusters,
            points_in_box=DBSCANDefaults.default_points_in_box,
            levels=DBSCANDefaults.default_levels,
            axis_splits=2,
    ):
        super().__init__(
            DBSCAN,
            epsilon=epsilon,
            min_points=min_points,
            border_noise=border_noise,
            max_clusters=max_clusters,
            points_in_box=points_in_box,
            axis_splits=axis_splits,
            levels=levels
        )
