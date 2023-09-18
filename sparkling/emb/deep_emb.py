from abc import abstractmethod
from typing import Dict, Any, Tuple

import pyspark.sql.functions as F
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import DataFrame

from sparkling.data.monad import DataMonad, DataStage
from sparkling.emb.stages import Normaliser, ForceEval, DimReduction, PretrainedDim


class DeepRunner(DataStage):
    """ Launches deep learning model inference on dataframe by using pandas_udf """

    def __init__(self, col_name, udf_builder, **kwargs):
        self.col_name = col_name
        self.udf_builder, self.kwargs = udf_builder, kwargs

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        udf_wrapper = self.udf_builder.__call__(**self.kwargs)
        deep_expr = udf_wrapper(F.col(self.col_name))
        return df.withColumn(self.col_name, deep_expr), {}


class ListConverter(DataStage):
    """
    Converts bare array of floats to mllib vector.
    TODO: should be replaced by native function when upgrading to pyspark 3.1+
    """

    def __init__(self, col_name):
        self.col_name = col_name

    @staticmethod
    @F.udf(VectorUDT())
    def _vectorise(as_list):
        return Vectors.dense(as_list)

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        vector_expr = self._vectorise(F.col(self.col_name))
        return df.withColumn(self.col_name, vector_expr), {}


class DeepEmb(DataMonad):
    """ Base pipeline for processing image/text modalities by deep learning models """

    def __init__(self, name: str, model_name: str, orig_dim: int, reduce_dim: bool, **kwargs):
        pipeline = [
            DeepRunner(name, self.udf_builder, **kwargs),
            ListConverter(name),
            ForceEval(),
            PretrainedDim(orig_dim),
            Normaliser('_deep_norm', name)
        ]
        if reduce_dim:
            pipeline.extend([
                DimReduction(name),
                Normaliser('_reduced_norm', name)
            ])
        super().__init__(name, pipeline)
        self.model_name = model_name

    @staticmethod
    @abstractmethod
    def udf_builder(**kwargs):
        pass
