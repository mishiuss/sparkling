from .base import *


class CLIQUEModel(ClusteringModel):
    def __init__(self, jvm_model):
        super().__init__(jvm_model)


class CLIQUE(ClusteringAlgo):
    """
    CLIQUE is a clustering algorithm that is specifically designed for clustering categorical data.
    The Clique algorithm works by identifying cohesive groups of categorical data items that share common attribute
    values. It operates in two main steps:
    1. Formation of cliques: Initially, each data item is considered as a separate clique. Then, the algorithm
    iteratively merges cliques that have a high degree of similarity based on the shared attribute values.
    The merging process continues until no more cliques can be merged.
    2. Pruning of cliques: After the formation of cliques, the algorithm prunes them to remove any outliers or noise.
    It evaluates the cohesion and density of each clique and removes those that do not meet certain criteria.
    The CLIQUE algorithm takes into account both the intra-clique similarity (similarity among data items within
    a clique) and the inter-clique dissimilarity (dissimilarity among different cliques) to identify
    meaningful clusters.
    """

    def __init__(
            self, *,
            threshold: float,
            splits: int,
            levels: int
    ):
        """
        :param threshold: ratio of total number of points
        :param splits: number of subdivisions for a single dimension
        :param levels: number of attempts to cartesian grids
        """
        super().__init__(
            threshold=threshold,
            splits=splits,
            levels=levels
        )

    def fit(self, sparkling_df: SparklingDF) -> CLIQUEModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        return CLIQUEModel(jvm_model)

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[ClusteringModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.predict(sparkling_df)

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.clique.CLIQUE(
            kwargs['threshold'],
            kwargs['splits'],
            kwargs['levels']
        )


class CLIQUEConf(AlgoConf):
    def __init__(
            self, *,
            threshold=(1e-5, 0.8),
            splits=(2, Defaults.log10_n),
            levels=(1, Defaults.global_dim)
    ):
        super().__init__(
            CLIQUE,
            threshold=threshold,
            splits=splits,
            levels=levels
        )
