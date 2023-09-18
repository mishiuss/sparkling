import pyspark
import pandas as pd
from pyspark import SparkContext, SQLContext
from pyspark.sql.types import StructField, StringType, LongType, IntegerType, FloatType, StructType


class PandasConverter:
    PANDAS_TYPES = {'int64': LongType, 'int32': IntegerType, 'float64': FloatType}

    @staticmethod
    def convert(sc: SparkContext, df: pd.DataFrame) -> pyspark.sql.DataFrame:
        """
        Converts Pandas Dataframe to PySpark DataFrame
        """
        columns, types = list(df.columns), list(df.dtypes)
        structs = [StructField(
            column, StringType() if t not in PandasConverter.PANDAS_TYPES else PandasConverter.PANDAS_TYPES[t]()
        ) for column, t in zip(columns, types)]
        return SQLContext(sc).createDataFrame(df, StructType(structs))
