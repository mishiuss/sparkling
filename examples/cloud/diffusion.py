from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.sql.types import Row

from fetcher import MultiModalFetcher
from sparkling import *
from sparkling.emb.torch import TorchImages, TorchTexts


class Diffusion(MultiModalFetcher):
    """
    https://huggingface.co/datasets/poloclub/diffusiondb
    """

    IMAGE_DIR = '/user/multimodal/diffusion/images'
    DATA_PATH = '/user/multimodal/diffusion/diffusiondb_v2.csv'

    IMAGE_MODEL = TorchImages.SWIN_TRANSFORMER
    TEXT_MODEL = TorchTexts.ALBERT

    def preprocessed_path(self) -> str:
        return '/user/multimodal/diffusion/preprocessed'

    def modals_path(self) -> str:
        return 'examples/data/diffusion-modals.json'

    def from_raw(self, session: SparkSession) -> SparklingDF:
        text_df = session.read.format('csv') \
            .option('delimiter', '|') \
            .option('header', 'true') \
            .load(self.DATA_PATH) \
            .withColumnRenamed('prompt', 'text_data')

        images_paths = session.sparkContext.wholeTextFiles(self.IMAGE_DIR) \
            .keys().map(lambda x: Row(img_path=x))

        image_df = session.createDataFrame(images_paths) \
            .withColumn('image_filename', split('img_path', '/')[7])

        df = text_df.join(image_df, on=['image_filename'])

        return SparklingBuilder(df, partitions='default') \
            .image('img_path', self.IMAGE_MODEL, '', Distance.MANHATTAN) \
            .text('text_data', self.TEXT_MODEL, Distance.MANHATTAN).create()
