import os
import pyspark

from pathlib import Path


if __name__ == '__main__':
    from sparkling import *
    # Using predefined pytorch image models from huggingface
    # transformers (https://huggingface.co/docs/transformers/index)
    from sparkling.emb.torch import TorchImages

    # Note, that you need to enable pyarrow for text processing and include heaven.jar
    conf = pyspark.SparkConf()\
        .setAppName('sparkling-pytorch-image')\
        .setMaster('local-cluster[2, 2, 2048]')\
        .set("spark.sql.execution.arrow.enabled", "true")\
        .set("spark.jars", "bin/heaven.jar")\
        .setExecutorEnv('ARROW_PRE_0_15_IPC_FORMAT', '1')
    ss = pyspark.sql.SparkSession.Builder().config(conf=conf).getOrCreate()

    # Do not forget to set Spark checkpoint directory
    ss.sparkContext.setCheckpointDir(str(Path('examples', 'checkpoint').absolute()))

    # Enable extra logging (default is SparklingLogLevel.EXCERPT)
    SparklingLogger.level = SparklingLogLevel.INFO

    # Dataset 'sports celebrity' (https://www.kaggle.com/datasets/yaswanthgali/sport-celebrity-image-classification)
    # has been already downloaded into repository
    data, root = list(), Path('examples', 'data', 'sports-celebrity').absolute()

    # Note, that you only need to define images paths, and not to read content yourself
    for clazz in os.listdir(root):
        # This file is not present in original dataset
        if clazz == 'celebrity.csv':
            continue
        for image_file in os.listdir(root / clazz):
            data.append((f'/{clazz}/{image_file}', clazz))

    # Create Spark dataframe from in-memory list with following column names
    df = ss.createDataFrame(data, ['celebrity', 'class'])

    # Configure single image modality to be processed by 'microsoft/swin-tiny-patch4-window7-224' model.
    # Note, that in line 35 we put into dataframe only relative paths, so you also need to provide root path
    builder = SparklingBuilder(df, class_col='class', partitions='default')\
        .image('celebrity', TorchImages.SWIN_TRANSFORMER, str(root), Distance.MANHATTAN)

    # Can take a while, because reading images and computing their embeddings are heavy operations
    sparkling_df = builder.create()

    # Note, that column 'celebrity' now contains vector representations of images
    sparkling_df.df.show()

    # You can choose another strategy to distribute time budgets between clustering algorithms
    optimiser = Sparkling(sparkling_df, mab_solver=MabSolver.UCB, measure=Internal.DAVIES_BOULDIN)
    print(optimiser.run(time_limit=90))

    # You can save optimisation history (invoked clustering algorithms with hyperparameters) into json format:
    with open(f'examples/logs/sports-celebrity-local.json', 'w') as fp:
        fp.write(optimiser.history_json())

    ss.stop()
