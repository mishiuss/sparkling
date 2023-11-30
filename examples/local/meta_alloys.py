import os
import pyspark

if __name__ == '__main__':
    from sparkling import SparklingBuilder, Distance, Sparkling

    conf = pyspark.SparkConf() \
        .setAppName(f'sparkling-alloys') \
        .setMaster('local[*]') \
        .set('spark.jars', 'bin/heaven.jar')

    sc = pyspark.SparkContext.getOrCreate(conf)
    sc.setCheckpointDir('examples/checkpoint')
    sql = pyspark.SQLContext(sc)

    DATA_ROOT, DATASET = 'examples/data', 'alloys_ml_coordinates'

    df = sql.read.csv(f'{DATA_ROOT}/{DATASET}.csv', inferSchema=True, header=False)

    sparkling_df = SparklingBuilder(df, partitions='default') \
        .tabular('features', set(df.columns), Distance.EUCLIDEAN) \
        .create()

    optimiser = Sparkling(sparkling_df, measure=None)