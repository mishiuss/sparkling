import pyspark.sql.functions as F
from pyspark import Row
from pyspark.sql import SparkSession

from fetcher import MultiModalFetcher
from sparkling import SparklingDF, SparklingBuilder, Distance
from sparkling.emb.torch import TorchImages


class Houses(MultiModalFetcher):
    """
    Ahmed E., Moustafa M. House price estimation from visual and textual features
    """

    IMAGE_DIR = '/user/multimodal/Houses-dataset/Houses Dataset/'
    DATA_PATH = '/user/multimodal/Houses-dataset/HousesInfo.txt'

    IMAGE_MODEL = TorchImages.EFFICIENT_NET

    def preprocessed_path(self) -> str:
        return '/user/multimodal/Houses-dataset/houses/preprocessed'

    def modals_path(self) -> str:
        return 'examples/data/houses-modals.json'

    def from_raw(self, session: SparkSession) -> SparklingDF:
        df = session.read.format("csv")\
            .option("delimiter", " ") \
            .option("header", "false")\
            .load(self.DATA_PATH)

        img_paths = session.sparkContext.wholeTextFiles(self.IMAGE_DIR)\
            .keys().map(lambda x: Row(img_path=x))

        fp_df = session.createDataFrame(img_paths)
        fp_df = fp_df.filter(~(fp_df["img_path"].contains('HousesInfo')))\
                     .withColumn("idx", F.split(F.split("img_path", "/")[7], "_")[0])\
                     .groupBy("idx")\
                     .agg(F.concat_ws("\t", F.expr("sort_array(collect_list(img_path))")).alias("paths"))\
                     .withColumn("bathroom", F.split("paths", "\t")[0])\
                     .withColumn("bedroom", F.split("paths", "\t")[1])\
                     .withColumn("frontal", F.split("paths", "\t")[2])\
                     .withColumn("kitchen", F.split("paths", "\t")[3])\
                     .drop("paths").drop("idx")
        df = df.coalesce(1).rdd.zip(fp_df.coalesce(1).rdd).toDF()
        for i in ["_c0", "_c1", "_c2", "_c3", "_c4"]:
            df = df.withColumn(f"{i}", F.col("_1").getItem(i))
        for i in ["bathroom", "bedroom", "frontal", "kitchen"]:
            df = df.withColumn(f"{i}", F.col("_2").getItem(i))
        df = df.drop("_1").drop("_2")

        return SparklingBuilder(df, partitions='default') \
            .image('bathroom', self.IMAGE_MODEL, '', Distance.MANHATTAN) \
            .image('bedroom', self.IMAGE_MODEL, '', Distance.MANHATTAN) \
            .image('frontal', self.IMAGE_MODEL, '', Distance.MANHATTAN) \
            .image('kitchen', self.IMAGE_MODEL, '', Distance.MANHATTAN) \
            .tabular('tabular', {'cola', 'colb', 'colc', 'cold', 'cole'}, Distance.EUCLIDEAN) \
            .create()
