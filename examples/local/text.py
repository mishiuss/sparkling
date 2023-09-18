import pyspark

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


if __name__ == '__main__':
    from sparkling import *
    # Using predefined pytorch NLP models from huggingface
    # transformers (https://huggingface.co/docs/transformers/index)
    from sparkling.emb.torch import TorchTexts

    # Note, that you need to enable pyarrow for text processing and include heaven.jar
    conf = pyspark.SparkConf()\
        .setAppName('sparkling_pytorch_text')\
        .setMaster('local-cluster[2, 1, 4096]')\
        .set('spark.sql.execution.arrow.enabled', 'true')\
        .setExecutorEnv('ARROW_PRE_0_15_IPC_FORMAT', '1')\
        .set('spark.jars', 'bin/heaven.jar')
    ss = pyspark.sql.SparkSession.Builder().config(conf=conf).getOrCreate()

    # Do not forget to set Spark checkpoint directory
    ss.sparkContext.setCheckpointDir('examples/checkpoint')

    # Enable extra logging (default is SparklingLogLevel.EXCERPT)
    SparklingLogger.level = SparklingLogLevel.INFO

    # Dataset Popular Quotes (https://www.kaggle.com/datasets/faseehurrehman/popular-quotes)
    # has been already downloaded into repository
    df = ss.read.csv('examples/data/popular-quotes/raw.csv', inferSchema=True, header=True)\
        .withColumn('Likes', F.col('Likes').cast(IntegerType()))\
        .withColumn('quotes', F.trim(F.col('Popular Quotes')))\
        .withColumn('Author Names', F.trim(F.col('Author Names')))

    # Define two modalities. Tabular contains one categorical and one integer feature.
    # Text column `quotes` will be transformed via pytorch model 'lordtt13/emo-mobilebert'
    builder = SparklingBuilder(df, partitions='default')\
        .tabular('popularity', {'Author Names', 'Likes'})\
        .text('quotes', TorchTexts.MOBILE)

    # Can take a while, because computing text embeddings is a heavy operation
    sparkling_df = builder.create()

    # Note, that column 'quotes; now contains vector representations of text data
    sparkling_df.df.show()

    # Configure optimiser to use approximate silhouette function as target measure
    optimiser = Sparkling(sparkling_df, measure=Internal.SILHOUETTE_APPROX)
    print(optimiser.run(time_limit=80))

    # You can save optimisation history (invoked clustering algorithms with hyperparameters) into json format:
    with open(f'examples/logs/popular-quotes-local.json', 'w') as fp:
        fp.write(optimiser.history_json())

    ss.stop()
