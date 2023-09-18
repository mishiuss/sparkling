from typing import List, Dict, Any, Union
from pyspark.sql import DataFrame

from sparkling.data.modals import ModalInfo


class SparklingDF:
    """ :class`Dataframe` with custom multimodal distance metric """

    ID_COL, LABEL_COL, CLASS_COL = '_id', '_label', '_class'

    def __init__(self, df: DataFrame, dist, modalities: List[ModalInfo], amount: int):
        """
        :param df: preprocessed :class`Dataframe`
        :param dist: py4j object, referencing jvm instance of MultiDistance
        :param modalities: list of modalities' meta information
        :param amount: amount of objects in dataframe
        NOTE: You should not call this method directly,
        use either :class`SparklingBuilder` to preprocess raw dataset
        or :method`SparklingDF.create()` to load serialized dataframe
        """
        self.df, self.dist = df, dist
        self.modalities, self.amount = modalities, amount

    @property
    def modal_dict(self):
        """
        Serialized version of modalities' meta information, that can be saved, for example, in json format

        >>> import json
        >>>
        >>> sparkling_df = ...
        >>> modals_info = sparkling_df.modal_dict
        >>> json.dumps(modals_info)
        """
        return [modal.__dict__ for modal in self.modalities]

    @property
    def global_dim(self):
        """
        Sum of dimensions of each modality
        """
        return sum([modality.dim for modality in self.modalities])

    @property
    def jdf(self):
        return self.df._jdf

    @property
    def jvm(self):
        return self.df._sc._jvm

    @property
    def sql_ctx(self):
        return self.df.sql_ctx

    def estimate_mem(self) -> int:
        """
        Approximate amount of memory consumed by dataframe
        """
        return self.amount * self.global_dim * 10

    def like(self, df: DataFrame):
        """
        Creates dataframe with the same objects and meta information,
        potentially with extra columns (e.g., labels).
        NOTE: For internal purposes only you should not use it
        """
        return SparklingDF(df, self.dist, self.modalities, self.amount)

    @staticmethod
    def create(df: DataFrame, modalities: List[Union[ModalInfo, Dict[str, Any]]]):
        """
        Creates instance for preprocessed by :class`SparklingBuilder` dataframe.
        You can use this method to restore serialized dataframe

        >>> import json
        >>>
        >>> sparkling_df = ...
        >>> # Save meta information on driver
        >>> with open('modals-info.json', 'w') as modals_fp:
        >>>     json.dump(sparkling_df.modal_dict, modals_fp)
        >>> # Format should support :class`Vector` serialization
        >>> sparkling_df.df.write.parquet('serialized/dataframe')
        >>>
        >>> serialized_df = sparkSession.read.parquet('serialized/dataframe')
        >>> with open('modals-info.json') as modals_fp:
        >>>     modals_info = json.load(modals_fp)
        >>> sparkling_df_restored = SparklingDF.create(serialized_df, modals)

        :param df: deserialized :class`DataFrame`
        :param modalities: (de)serialized modalities meta information
        """
        modals = [m if isinstance(m, ModalInfo) else ModalInfo(**m) for m in modalities]
        serialized = "\n".join([f'{m.metric};{m.name};{m.dim};{m.weight};{m.norm}' for m in modals])
        jvm_dist = df._sc._jvm.ru.ifmo.rain.distances.MultiBuilder.fromPy4J(serialized)
        return SparklingDF(df, jvm_dist, modals, df.count())
