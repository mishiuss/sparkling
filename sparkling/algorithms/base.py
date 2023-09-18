import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
from pyspark import SparkContext
from pyspark.sql import DataFrame

from sparkling.data.dataframe import SparklingDF


class Defaults:
    """ Some functions that provide default parameters for algorithms depending on :class`SparklingDF` """

    @staticmethod
    def max_clusters(sparkling_df: SparklingDF) -> int:
        return int(np.cbrt(2 * sparkling_df.amount)) + 1
    
    @staticmethod
    def max_iterations(sparkling_df: SparklingDF) -> int:
        return int(np.sqrt(sparkling_df.amount) / np.log(sparkling_df.amount)) + 1

    @staticmethod
    def sqrt_n(sparkling_df: SparklingDF) -> int:
        return int(np.sqrt(sparkling_df.amount))

    @staticmethod
    def log10_n(sparkling_df: SparklingDF) -> int:
        return int(np.log10(sparkling_df.amount))

    @staticmethod
    def global_dim(sparkling_df: SparklingDF) -> int:
        return sparkling_df.global_dim


class ClusteringModel(ABC):
    """ Basic wrapper for fitted model, holds reference to specified jvm object """

    def __init__(self, jvm_model):
        self._jvm_model = jvm_model

    def predict(self, sparkling_df: SparklingDF) -> SparklingDF:
        """ Returns new :class`SparklingDF` with predictions in new column. Invokes evaluations on jvm """
        label_jdf = self._jvm_model.predict(sparkling_df.jdf)
        label_df = DataFrame(label_jdf, sparkling_df.sql_ctx)
        return sparkling_df.like(label_df)


class ClusteringAlgo(ABC):
    """
    Basic wrapper for ready-to-fit algorithm with defined hyperparameters,
    holds reference to specified jvm object
    """

    def __init__(self, **kwargs):
        """
        Creates wrapper and jvm object with specified arguments

        :param kwargs: clustering algorithm hyperparameters
        """

        self.algo_name, self.params = self.__class__.__name__, kwargs
        self._jvm_algo = self._jvm_builder(SparkContext.getOrCreate()._jvm, **kwargs)

    @abstractmethod
    def fit(self, sparkling_df: SparklingDF) -> ClusteringModel:
        """ Invoke according method on jvm side and return fitted model """
        pass

    @abstractmethod
    def fit_predict_with_model(self, sparkling_df: SparklingDF) -> Tuple[ClusteringModel, SparklingDF]:
        """ Invoke according method on jvm side and return fitted model with labeled dataframe """
        pass

    def fit_predict(self, sparkling_df: SparklingDF) -> SparklingDF:
        """ Invoke according method on jvm side and return labeled dataframe """
        _, label_sdf = self.fit_predict_with_model(sparkling_df)
        return label_sdf

    @abstractmethod
    def _jvm_builder(self, jvm, **kwargs):
        """ Base method to create jvm instance of clustering algorithm """
        pass

    @staticmethod
    def make_seed(seed: Optional[int]) -> int:
        return seed if seed is not None else random.randint(0, 2 ** 32)


class AlgoConf(ABC):
    """
    Basic class for hyperparameters search space for particular clustering algorithm.
    Configuration is defined by the following rules:
    - primitive value (int, float, bool) is treated as constant and does not change during the whole optimisation
    - tuple (int, int) is treated as integer range (both borders are inclusive)
    - tuple (float, float) is treated as continuous span (both borders are inclusive)
    - set of float/int/bool is treated as a set of categorical values
    """

    def __init__(self, builder, **kwargs):
        self.config_space, self.builder = kwargs, builder
        self.algo_name = builder.__name__

    def build(self, **kwargs) -> ClusteringAlgo:
        return self.builder(**kwargs)
