from typing import Set, Dict, Any, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator as OneHot
from pyspark.sql import DataFrame
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import VectorUDT as OldVectorType
from pyspark.sql.types import StringType, NumericType

from sparkling.emb.stages import VectorDim, InplaceAssembler, Normaliser
from sparkling.data.monad import DataMonad, DataStage


class SplitColumns(DataStage):
    """ Given a set of column names, splits them into numeric subset and categorical subset """

    def __init__(self, features, out_num, out_cat):
        self.features, self.out_num, self.out_cat = features, out_num, out_cat

    def __str__(self):
        return f'SplitColumns'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        dtypes = {str(f.name): f.dataType for f in df.select(*self.features).schema.fields}
        numerics = {name for name, col_type in dtypes.items() if isinstance(col_type, NumericType)}
        categories = {name for name, col_type in dtypes.items() if isinstance(col_type, StringType)}
        return df, {self.out_num: numerics, self.out_cat: categories}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        return df


class CategoriesEncoder(DataStage):
    """ Transforms set of categories column into one hot encoded vectors """

    def __init__(self, name, input_cats, out_dim):
        self.name, self.input_cats, self.out_dim = name, input_cats, out_dim

    def __str__(self):
        return 'CategoriesEncoder'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        categories = list(state[self.input_cats])
        if len(categories) == 0:
            return df, {}
        indexers, index_names = list(), list()
        for cat in categories:
            df = df.replace('', 'N/A', cat)
            index_names.append(f'_{cat}_idx')
            indexers.append(StringIndexer(inputCol=cat, outputCol=index_names[-1]))
        indexer_pipe = Pipeline(stages=indexers).fit(df)
        df = indexer_pipe.transform(df).drop(*categories)

        one_hot = OneHot(inputCols=index_names, outputCols=categories, dropLast=False).fit(df)
        df = one_hot.transform(df).drop(*index_names)
        cat_dim = sum(map(lambda stage: len(stage.labels), indexer_pipe.stages))
        return df, {self.name: (indexer_pipe, index_names, one_hot), self.out_dim: cat_dim}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        categories = state[self.input_cats]
        if len(categories) == 0:
            return df
        indexer_pipe, index_names, one_hot = state[self.name]
        df = indexer_pipe.transform(df).drop(*categories)
        return one_hot.transform(df).drop(*index_names)


class CalcMultiColDimension(DataStage):
    """ Puts dimensions information into monad's state """

    def __init__(self, numerics, categories, cats_dim):
        self.numerics, self.categories, self.cats_dim = numerics, categories, cats_dim

    def __str__(self):
        return 'TabularDimension'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        num_cols, cat_cols = state.get(self.numerics), state.get(self.categories)
        num_amount = 0 if num_cols is None else len(num_cols)
        cat_amount = 0 if cat_cols is None else len(cat_cols)
        cat_dim = state.get(self.cats_dim, 0)
        return df, {'_dim': num_amount + cat_dim, '_w_dim': num_amount + cat_amount}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        return df


class MultiColTabularEmb(DataMonad):
    """ Processes tabular data in form of multiple columns (both numeric and categorical) into single vector """

    def __init__(self, name, features: Set[str]):
        num_cols, num_vec_col = '_num_cols', '_num_vec_col'
        cat_cols, cat_vec_col = '_cat_cols', '_cat_vec_col'
        cats_dim_key = '_cat_dim'
        pipeline = [
            SplitColumns(features, out_num=num_cols, out_cat=cat_cols),
            InplaceAssembler('_num_assembler', out=num_vec_col, from_state=num_cols),
            Normaliser('_num_norm', input_col=num_vec_col),
            CategoriesEncoder('_cat_encoder', input_cats=cat_cols, out_dim=cats_dim_key),
            InplaceAssembler('_cat_assembler', out=cat_vec_col, from_state=cat_cols),
            InplaceAssembler('_tab_assembler', out=name, from_set={num_vec_col, cat_vec_col}),
            CalcMultiColDimension(numerics=num_cols, categories=cat_cols, cats_dim=cats_dim_key)
        ]
        super().__init__(name, pipeline)

    def __str__(self):
        return f"MultiColTabular ['{self.name}']"


class VectorConverter(DataStage):
    """ Converts old mllib vector to ml vector if detected """

    def __init__(self, col_name):
        self.col_name = col_name
        self._state_key = '_old_type'

    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        col_type = df.schema[self.col_name].dataType
        if isinstance(col_type, OldVectorType):
            return MLUtils.convertVectorColumnsToML(df, self.col_name), {self._state_key: True}
        else:
            return df, {self._state_key: False}

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        return MLUtils.convertVectorColumnsToML(df, self.col_name) if state[self._state_key] else df


class VectorColTabularEmb(DataMonad):
    """ Processes tabular data in form of vector column """

    def __init__(self, name):
        pipeline = [
            VectorConverter(name),
            Normaliser(f'_vec_norm', name),
            VectorDim(name),
        ]
        super().__init__(name, pipeline)

    def __str__(self):
        return f"VectorTabular ['{self.name}']"
