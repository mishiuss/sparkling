from pyspark.sql import SparkSession
from pyspark.sql.functions import split, concat
from pyspark.sql.types import Row

from fetcher import MultiModalFetcher
from sparkling import SparklingBuilder, Distance, SparklingDF
from sparkling.emb.torch import TorchImages


class Birds(MultiModalFetcher):
    """
    https://www.kaggle.com/datasets/gpiosenka/100-bird-species
    """

    IMAGE_DIR = '/user/unimodal/images/birds/train'
    DATA_PATH = '/user/unimodal/images/birds/birds.csv'

    MODEL = TorchImages.SWIN_TRANSFORMER

    def preprocessed_path(self):
        return '/user/unimodal/images/birds/preprocessed'

    def modals_path(self):
        return 'examples/data/birds-modals.json'

    def from_raw(self, session: SparkSession) -> SparklingDF:
        label_df = session.read.csv(self.DATA_PATH, header=True)\
            .withColumnRenamed('class id', 'label')\
            .withColumn('filename', concat(split('filepaths', '/')[1], split('filepaths', '/')[2]))

        images = session.sparkContext.wholeTextFiles(self.IMAGE_DIR + '/*/*')\
            .keys().map(lambda x: Row(img_path=x))

        image_df = session.createDataFrame(images)\
            .withColumn('filename', concat(split('img_path', '/')[8], split('img_path', '/')[9]))

        df = label_df.join(image_df, on=['filename'])

        return SparklingBuilder(df, class_col='label', partitions='default') \
            .image('img_path', self.MODEL, '', Distance.MANHATTAN).create()
