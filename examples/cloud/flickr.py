from pyspark import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, expr, split

from fetcher import MultiModalFetcher
from sparkling import SparklingDF, SparklingBuilder, Distance
from sparkling.emb.torch import TorchImages, TorchTexts


class Flickr(MultiModalFetcher):
    """
    https://www.kaggle.com/datasets/adityajn105/flickr30k
    """

    IMAGE_DIR = '/user/multimodal/multimodal_flickr30k/Images/flickr30k_images'
    TEXT_PATH = '/user/multimodal/multimodal_flickr30k/Images/results.csv'

    IMAGE_MODEL = TorchImages.SWIN_TRANSFORMER
    TEXT_MODEL = TorchTexts.BERT

    def preprocessed_path(self) -> str:
        return '/user/multimodal/multimodal_flickr30k/preprocessed'

    def modals_path(self) -> str:
        return 'examples/data/flickr-modals.json'

    def from_raw(self, session: SparkSession) -> SparklingDF:
        text_df = session.read.format("csv") \
            .option("delimiter", "|") \
            .option("header", "true") \
            .load(self.TEXT_PATH)\
            .withColumnRenamed(' comment', 'comment')\
            .withColumnRenamed(' comment_number', 'comment_number')\
            .groupBy("image_name")\
            .agg(concat_ws(" ", expr("collect_list(comment)")).alias("text_data"))

        image_paths = session.sparkContext.wholeTextFiles(self.IMAGE_DIR)\
            .keys().map(lambda x: Row(img_path=x))
        image_df = session.createDataFrame(image_paths)\
            .withColumn("image_name", split("img_path", "/")[8])

        df = text_df.join(image_df, on=["image_name"])

        return SparklingBuilder(df, partitions='default') \
            .image('img_path', self.IMAGE_MODEL, '', Distance.MANHATTAN) \
            .text('text_data', self.TEXT_MODEL, Distance.MANHATTAN).create()
