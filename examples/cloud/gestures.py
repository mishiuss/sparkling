from itertools import chain
from pyspark.sql import SparkSession

from fetcher import MultiModalFetcher
from sparkling import SparklingDF, SparklingBuilder, Distance


class Gestures(MultiModalFetcher):
    """
    https://www.kaggle.com/datasets/arthurfindelair/openhand-hand-gesture-recognition
    """

    DATA_PATH = '/user/unimodal/OpenHand_dataset.csv'

    def preprocessed_path(self) -> str:
        return '/user/unimodal/gestures-preprocessed'

    def modals_path(self) -> str:
        return 'examples/data/gestures-modals.json'

    def from_raw(self, session: SparkSession) -> SparklingDF:
        df = session.read.csv(self.DATA_PATH, header=True, inferSchema=True)
        coordinates = chain.from_iterable([(f'x{idx}', f'y{idx}') for idx in range(21)])

        return SparklingBuilder(df, class_col='label', partitions='default') \
            .tabular('tabular', {'accuracy', *coordinates}, Distance.EUCLIDEAN) \
            .create()
