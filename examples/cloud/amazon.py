from pyspark.sql import SparkSession

from fetcher import MultiModalFetcher
from sparkling import Distance, SparklingBuilder, SparklingDF
from sparkling.emb.torch import TorchTexts


class Amazon(MultiModalFetcher):
    """
    https://huggingface.co/datasets/amazon_reviews_multi/viewer/en/train
    """

    DATA_PATH = '/user/multimodal/amazon.csv'
    TEXT_MODEL = TorchTexts.ALBERT

    def preprocessed_path(self) -> str:
        return '/user/multimodal/text/preprocessed'

    def modals_path(self) -> str:
        return 'examples/data/amazon-modals.json'

    def from_raw(self, session: SparkSession) -> SparklingDF:
        df = session.read.csv(self.DATA_PATH, header=True, sep='|')

        return SparklingBuilder(df, partitions='default', class_col='stars') \
            .text('review_body', self.TEXT_MODEL, Distance.MANHATTAN) \
            .text('review_title', self.TEXT_MODEL, Distance.MANHATTAN) \
            .text('product_category', self.TEXT_MODEL, Distance.MANHATTAN).create()
