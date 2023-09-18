from .base import *


class SpectralClusteringModel(ClusteringModel):
    def __init__(self, jvm_model, dataframe):
        super().__init__(jvm_model)
        self._dataframe = dataframe

    @property
    def dataframe(self) -> SparklingDF:
        return self._dataframe


class SpectralClustering(ClusteringAlgo, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, sparkling_df: SparklingDF) -> SpectralClusteringModel:
        jvm_model = self._jvm_algo.fit(sparkling_df.jdf, sparkling_df.dist)
        labeled_df = DataFrame(jvm_model.dataframe(), sparkling_df.sql_ctx)
        return SpectralClusteringModel(jvm_model, sparkling_df.like(labeled_df))

    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[SpectralClusteringModel, SparklingDF]:
        model = self.fit(sparkling_df)
        return model, model.dataframe


class SpectralDefaults:
    @staticmethod
    def default_eigen(sparkling_df: SparklingDF) -> int:
        return int(np.log(sparkling_df.amount)) + 1


class SpectralAdjacency(SpectralClustering):
    def __init__(
            self, *,
            neighbours: int,
            eigen: int,
            k: int,
            max_iterations: int,
            seed: Optional[int] = None
    ):
        super().__init__(
            neighbours=neighbours,
            eigen=eigen,
            k=k,
            max_iterations=max_iterations,
            seed=ClusteringAlgo.make_seed(seed)
        )

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.spectral.SpectralAdjacency(
            kwargs['neighbours'],
            kwargs['eigen'],
            kwargs['k'],
            kwargs['max_iterations'],
            kwargs['seed']
        )


class SpectralAdjacencyConf(AlgoConf):
    def __init__(
            self, *,
            neighbours=(2, Defaults.sqrt_n),
            eigen=(2, SpectralDefaults.default_eigen),
            k=(2, Defaults.max_clusters),
            max_iterations=Defaults.max_iterations
    ):
        super().__init__(
            SpectralAdjacency,
            neighbours=neighbours,
            eigen=eigen,
            k=k,
            max_iterations=max_iterations
        )


class SpectralSimilarity(SpectralClustering):
    def __init__(
            self, *,
            gamma: float,
            eigen: int,
            k: int,
            max_iterations: int,
            seed: Optional[int] = None
    ):
        super().__init__(
            gamma=gamma,
            eigen=eigen,
            k=k,
            max_iterations=max_iterations,
            seed=ClusteringAlgo.make_seed(seed)
        )

    def _jvm_builder(self, jvm, **kwargs):
        return jvm.ru.ifmo.rain.algorithms.spectral.SpectralSimilarity(
            kwargs['gamma'],
            kwargs['eigen'],
            kwargs['k'],
            kwargs['max_iterations'],
            kwargs['seed']
        )


class SpectralSimilarityConf(AlgoConf):
    def __init__(
            self, *,
            gamma=(0.01, 1.0),
            eigen=(2, SpectralDefaults.default_eigen),
            k=(2, Defaults.max_clusters),
            max_iterations=Defaults.max_iterations
    ):
        super().__init__(
            SpectralSimilarity,
            gamma=gamma,
            eigen=eigen,
            k=k,
            max_iterations=max_iterations
        )
