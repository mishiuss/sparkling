from abc import abstractmethod, ABC
from typing import Tuple, Dict, Any, List

from pyspark.sql import DataFrame

from sparkling.util.logger import SparklingLogger


class DataStage(ABC):
    """
    Base class for a single logical action applied to :class`DataFrame`.
    """

    @abstractmethod
    def bind(self, df: DataFrame, state: Dict[str, Any]) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Execute action on dataframe, returns updated dataframe and new values in monad state
        """
        pass

    def apply(self, df: DataFrame, state: Dict[str, Any]) -> DataFrame:
        """
        Perform action on dataframe, potentially using bind values from previous runs monad state
        """
        return self.bind(df, state)[0]


class DataMonad:
    """
    Represents sequence of :class`DataStage` and holds monad state
    """

    def __init__(self, name, pipeline: List[DataStage]):
        """
        Create monad with empty state and ready-to-run pipeline

        :param name: unique name for monad
        :param pipeline: sequence of :class`DataStage`, that will be consecutively executed
        """

        self.name, self._pipeline, self._state = name, pipeline, {}

    def __getitem__(self, item):
        return self._state[item]

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Execute pipeline and cache monad state. Returns updated dataframe
        """
        for stage in self._pipeline:
            t_start = SparklingLogger.start_stage(stage)
            df, new_items = stage.bind(df, self._state)
            self._state = {**self._state, **new_items}
            SparklingLogger.finish_stage(t_start, stage, new_items)
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Execute pipeline with cached state. Returns updated dataframe
        """
        for stage in self._pipeline:
            df = stage.apply(df, self._state)
        return df
