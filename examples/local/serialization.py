import json

import pyspark
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F

if __name__ == '__main__':
    from sparkling import SparklingBuilder, SparklingDF
    # Using predefined pytorch NLP models from huggingface
    # transformers (https://huggingface.co/docs/transformers/index)
    from sparkling.emb.torch import TorchTexts

    # Note, that you need to enable pyarrow for text processing and include heaven.jar
    conf = pyspark.SparkConf()\
        .setAppName('sparkling-serialization')\
        .setMaster('local-cluster[2, 1, 4096]')\
        .set("spark.sql.execution.arrow.enabled", "true")\
        .set("spark.jars", "bin/heaven.jar")\
        .setExecutorEnv('ARROW_PRE_0_15_IPC_FORMAT', '1')
    ss = pyspark.sql.SparkSession.Builder().config(conf=conf).getOrCreate()

    # Do not forget to set Spark checkpoint directory
    ss.sparkContext.setCheckpointDir('examples/checkpoint')

    # Dataset Popular Quotes (https://www.kaggle.com/datasets/faseehurrehman/popular-quotes)
    # has been already downloaded into repository
    df = ss.read.csv('examples/data/popular-quotes/raw.csv', inferSchema=True, header=True) \
        .withColumn('Likes', F.col('Likes').cast(IntegerType())) \
        .withColumn('quotes', F.trim(F.col('Popular Quotes'))) \
        .withColumn('Author Names', F.trim(F.col('Author Names')))

    # Define two modalities. Tabular contains one categorical and one integer feature.
    # Text column `quotes` will be transformed via pytorch model 'lordtt13/emo-mobilebert'
    builder = SparklingBuilder(df, partitions='default') \
        .tabular('popularity', {'Author Names', 'Likes'}) \
        .text('quotes', TorchTexts.MOBILE)

    # Can take a while, because computing text embeddings is a heavy operation
    sparkling_df = builder.create()

    # Save modalities info on driver and dataframe on cluster
    with open('examples/data/popular-quotes/modals.json', 'w') as modals_fp:
        json.dump(sparkling_df.modal_dict, modals_fp)
    sparkling_df.df.write.parquet('examples/data/popular-quotes/preprocessed')

    # Restore dataframe from distributed storage and modals from driver
    serialized_df = ss.read.parquet('examples/data/popular-quotes/preprocessed')
    with open('examples/data/popular-quotes/modals.json') as modals_fp:
        modals = json.load(modals_fp)
    sparkling_df_restored = SparklingDF.create(serialized_df, modals)
    sparkling_df_restored.df.show()
