import sys
import pyspark
from pyspark.sql import SparkSession


if __name__ == '__main__':
    from sparkling import SparklingBuilder, Distance, Sparkling, Internal
    data_path, out_path, checkpoint_dir, budget = sys.argv[1:5]

    spark = SparkSession.Builder().getOrCreate()
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    sql = pyspark.SQLContext(spark.sparkContext)
    df = sql.read.csv(data_path, inferSchema=True, header=True)

    feature_cols = set(df.columns) - {'class'}
    sparkling_df = SparklingBuilder(df, class_col='class') \
        .tabular('features', feature_cols, Distance.EUCLIDEAN) \
        .create()

    optimiser = Sparkling(sparkling_df, measure=Internal.CALINSKI_HARABASZ)
    optimiser.run(time_limit=int(budget))

    with open(out_path, 'w') as fp:
        fp.write(optimiser.history_json())
