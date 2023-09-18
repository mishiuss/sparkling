from typing import Tuple, Dict, Any, Optional, Set

import numpy as np

from pyspark.ml.feature import MinMaxScaler, VectorAssembler, PCA, VectorSlicer
from pyspark.sql import DataFrame

from sparkling.util.logger import SparklingLogger
from sparkling.data.monad import DataStage


class InplaceAssembler(DataStage):
    """ Gathers multiple column into single vector column with cleaning input and intermediate columns """

    def __init__(self, name, out: str, from_state: Optional[str] = None, from_set: Optional[Set[str]] = None):
        if from_state is None and from_set is None:
            raise ValueError("Expected one argument to be defined")
        if from_state is not None and from_set is not None:
            raise ValueError("Expected one argument to be defined")
        self.name, self.out, self.from_state, self.from_set = name, out, from_state, from_set

    def __str__(self):
        return 'InplaceAssembler'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        inputs = self.from_set if self.from_set is not None else state[self.from_state]
        inputs = inputs & set(df.columns)
        if len(inputs) == 0:
            return df, {}
        assembler = VectorAssembler(inputCols=list(inputs), outputCol=self.out)
        return assembler.transform(df).drop(*inputs), {self.name: assembler}


class Normaliser(DataStage):
    """ Normalises modality values with cleaning intermediate columns """

    def __init__(self, name: str, input_col: str):
        self.name, self.input_col = name, input_col
        self._out = f'_{self.input_col}_norm'

    def __str__(self):
        return 'Normaliser'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        if self.input_col not in df.columns:
            return df, {}
        normaliser = MinMaxScaler(inputCol=self.input_col, outputCol=self._out, min=-1.0, max=1.0).fit(df)
        df = normaliser.transform(df).drop(self.input_col).withColumnRenamed(self._out, self.input_col)
        return df, {self.name: normaliser}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        if self.input_col not in df.columns:
            return df
        return state[self.name].transform(df).drop(self.input_col).withColumnRenamed(self._out, self.input_col)


class VectorDim(DataStage):
    """
    Extracts dimension of a vector column.
    TODO: Should be replaced with method, that will not use terminal dataframe operation
    """

    def __init__(self, col_name):
        self.col_name = col_name

    def __str__(self):
        return 'VectorDim'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        dim = df.first()[self.col_name].size
        return df, {'_dim': dim}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        return df


class PretrainedDim(DataStage):
    def __init__(self, dim):
        self.dim = dim

    def __str__(self):
        return 'PretrainedDim'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        return df, {'_dim': self.dim}


# TODO: Check if can delete this stage
class ForceEval(DataStage):
    def __str__(self):
        return 'ForceEval'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        return df.persist(), {}


class DimReduction(DataStage):
    """
    Utilises PCA feature extraction technique to obtain number of dimensions
    that will hit at least 90 percent of explained variance.
    However, achieved explained variance can be less in case PCA requires to
    calculate too many eigenvalues to satisfy the threshold
    """

    VARIANCE_THRESHOLD = 0.9

    def __init__(self, name):
        self.name, self._out = name, f'_{name}_reduce'

    def __str__(self):
        return 'DimReduction'

    def _extract(self, var_array):
        variance_acc = 0.0
        for component_idx, variance in enumerate(var_array):
            variance_acc += variance
            if variance_acc >= self.VARIANCE_THRESHOLD:
                return component_idx + 1, variance_acc
        return len(var_array), variance_acc

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        approx_dim = int(np.sqrt(2 * state['_dim']))
        model = PCA(k=approx_dim, inputCol=self.name, outputCol=self._out).fit(df)
        df = model.transform(df).drop(self.name)
        var_array = model.explainedVariance.values
        num_components, explained = self._extract(var_array)
        new_states = {'_var_array': var_array, '_pca': model, '_dim': num_components}
        if num_components == len(var_array):
            SparklingLogger.pca_not_enough_dims(self.name, explained, var_array)
            return df.withColumnRenamed(self._out, self.name), new_states
        slicer = VectorSlicer(inputCol=self._out, outputCol=self.name, indices=list(range(num_components)))
        return slicer.transform(df).drop(self._out), new_states
