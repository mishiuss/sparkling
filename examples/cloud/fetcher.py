import importlib
import json

from abc import ABC, abstractmethod
from pyspark.sql import SparkSession

from sparkling import *


class MultiModalFetcher(ABC):
    @abstractmethod
    def preprocessed_path(self) -> str:
        """ Where to store preprocessed dataframe (hdfs path) """
        pass

    @abstractmethod
    def modals_path(self) -> str:
        """ Where to store modalities meta information (master's local path) """
        pass

    @abstractmethod
    def from_raw(self, session: SparkSession) -> SparklingDF:
        """ Transforms raw original dataset to preprocessed SparklingDF """
        pass

    def raw(self, session: SparkSession) -> SparklingDF:
        """ Also serialize transformed dataframe into parquet and meta information into json """
        preprocessed_df = self.from_raw(session)

        with open(self.modals_path(), 'w') as modals_fp:
            json.dump(preprocessed_df.modal_dict, modals_fp)
        preprocessed_df.df.write.parquet(self.preprocessed_path())

        return preprocessed_df

    def preprocessed(self, session: SparkSession) -> SparklingDF:
        """ Restore dataframe from cached parquet files and meta information from json """
        with open(self.modals_path()) as modals_fp:
            modals = json.load(modals_fp)
        serialized_df = session.read.parquet(self.preprocessed_path())
        return SparklingDF.create(serialized_df, modals)

    @staticmethod
    def by_name(name: str):
        data_module = importlib.import_module(name)
        data_runner = getattr(data_module, name[0].upper() + name[1:])
        return data_runner()
