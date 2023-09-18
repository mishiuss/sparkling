from itertools import chain
from typing import Optional, Dict, Any, Tuple, Iterable, Union

import numpy as np
from pyspark import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from .dataframe import SparklingDF, ModalInfo
from .monad import DataMonad, DataStage
from .modals import Distance


class ClassColumn(DataStage):
    """
    Handles dataframe with provided external labels.
    Converts labels to the range [0, <num_labels>] and renames column for inner purposes
    """
    def __init__(self, class_col: Optional[str]):
        self.class_col = class_col

    def __str__(self):
        return f"ClassColumn ['{self.class_col}']"

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        if self.class_col is None:
            return df, {}
        src, target = self.class_col, SparklingDF.CLASS_COL
        uniques = df.select(src).distinct().collect()
        classes = [clazz_row[src] for clazz_row in uniques]
        class_map = {clazz: index for index, clazz in enumerate(sorted(classes))}
        mapping_fun = F.create_map([F.lit(x) for x in chain(*class_map.items())])
        mapping_expr = mapping_fun.getItem(F.col(src)).cast(IntegerType())
        monad_updates = {'_class_map': class_map, '_class_map_expr': mapping_expr}
        return df.withColumn(src, mapping_expr).withColumnRenamed(src, target), monad_updates

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        mapping_expr = state.get('_class_map_expr')
        if mapping_expr is None:
            return df
        src, target = self.class_col, SparklingDF.CLASS_COL
        return df.withColumn(src, mapping_expr).withColumnRenamed(src, target)


class FilterValues(DataStage):
    """
    Removes not suitable for work objects from dataframe
    """
    def __init__(self, columns):
        self.columns = sorted(columns)

    def __str__(self):
        return 'FilterValues'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        useful_cols = self.columns if '_class_map' not in state else self.columns + [SparklingDF.CLASS_COL]
        return df.select(useful_cols).dropna(), {}


class IdColumn(DataStage):
    """
    Make identification for each object for inner purposes
    """
    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        return df.withColumn(SparklingDF.ID_COL, F.monotonically_increasing_id()), {}

    def __str__(self):
        return 'IdColumn'


class Partition(DataStage):
    """ Repartition dataframe if needed """

    def __init__(self, partitions: Optional[Union[int, str]]):
        self.partitions = partitions

    def __str__(self):
        return f'Partition [{self.partitions}]'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        if self.partitions is None:
            return df, {}
        if isinstance(self.partitions, int):
            partitions = self.partitions
        elif self.partitions == 'auto':
            amount = state.get('_amount', df.rdd.countApprox(3000, 0.75))
            partitions = int(np.sqrt(amount) / np.log10(amount)) + 1
        elif self.partitions == 'default':
            partitions = df.rdd.context.defaultParallelism
        else:
            raise ValueError(f'Unexpected partitions value: {self.partitions}')
        return df.repartition(partitions), {}


class EmbWrapper(DataStage):
    """
    Wraps an embedding monad, e.g. the one, that will process single modality
    """

    def __init__(self, emb: DataMonad):
        self.emb = emb

    def __str__(self):
        return self.emb.__str__()

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        df = self.emb.fit_transform(df)
        w_dim = self.emb._state.get('_w_dim')
        modality = ModalInfo(self.emb.name, dim=self.emb['_dim'], w_dim=w_dim)
        return df, {f'_modality.{modality.name}': modality}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        return self.emb.transform(df)


class CountAmountAndDimensions(DataStage):
    """
    Gathers intermediate statistics about dataframe and puts it into monad state
    """

    def __init__(self, modalities: Iterable[str]):
        self.modalities = modalities

    def __str__(self):
        return 'CountAmountAndDimensions'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        amount = df.persist().count()
        dims = sum(state[f'_modality.{modality}'].dim for modality in self.modalities)
        w_dims = sum(state[f'_modality.{modality}'].w_dim for modality in self.modalities)
        return df, {'_amount': amount, '_global_dim': dims, '_global_w_dim': w_dims}


class ModalWeights(DataStage):
    """
    Calculate modalities' importance. Normalises user defined weights or evaluate automatically
    """

    def __init__(self, weights: Dict[str, Optional[float]]):
        self.weights = weights

    def __str__(self):
        return 'ModalWeights'

    def _auto_weights(self, state):
        norm = state['_global_w_dim']
        for modality in self.weights.keys():
            modal_info = state[f'_modality.{modality}']
            modal_info.weight = modal_info.w_dim / norm

    def _norm_weights(self, state):
        norm = sum(self.weights.values())
        for modality, weight in self.weights.items():
            modal_info = state[f'_modality.{modality}']
            modal_info.weight = weight / norm

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        if len([w for w in self.weights.values() if w is None]) == 0:
            self._norm_weights(state)
        else:
            self._auto_weights(state)
        return df, {}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        return df


class JvmModalities(DataStage):
    """ Creates jvm multimodal distance and puts meta information into monad state """

    def __init__(self, jvm, metrics: Dict[str, Distance]):
        self.jvm, self.metrics = jvm, metrics

    def __str__(self):
        return 'JvmModalities'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        builder = self.jvm.ru.ifmo.rain.distances.MultiBuilder(df._jdf)
        modalities = [self._fill(info, builder) for key, info in state.items() if key.startswith('_modality.')]
        return df, {'_modalities': modalities, '_jvm_dist': builder.create()}

    def _fill(self, info: ModalInfo, builder):
        info.metric = self.metrics[info.name].name
        info.norm = builder.newModality(info.name, info.metric, info.dim, info.weight)
        return info


class Checkpoint(DataStage):
    """ Note that checkpoint directory should be set for :class`SparkContext` """

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        return df.checkpoint(eager=True), {}

    def __init__(self, sc: SparkContext):
        if sc._jsc.sc().getCheckpointDir().isEmpty():
            raise ValueError('Checkpoint directory should be set [SparkContext.setCheckpointDir("path/dir")]')

    def __str__(self):
        return 'Checkpoint'
