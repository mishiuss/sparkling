import os
import pyspark

if __name__ == '__main__':
    from sparkling import SparklingBuilder, Distance, Sparkling

    conf = pyspark.SparkConf()\
        .setAppName('sparkling-meta-cvi') \
        .setMaster('local-cluster[4, 2, 1024]') \
        .set('spark.default.parallelism', '4') \
        .set("spark.jars", "bin/heaven.jar")

    sc = pyspark.SparkContext.getOrCreate(conf)
    sc.setCheckpointDir('examples/checkpoint')
    sql = pyspark.SQLContext(sc)

    # Run metaclassifier on some unimodal tabular datasets
    DATA_ROOT = 'heaven/src/test/data'

    for data_file in os.listdir(DATA_ROOT):
        print(f'=== {data_file} ===')
        df = sql.read.csv(f'{DATA_ROOT}/{data_file}', inferSchema=True, header=True)

        feature_cols = set(df.columns) - {'class'}
        sparkling_df = SparklingBuilder(df, class_col='class', partitions='default') \
            .tabular('features', feature_cols, Distance.EUCLIDEAN) \
            .create()

        optimiser = Sparkling(sparkling_df, measure=None)

    sc.stop()

    # === visualizing_soil.csv ===
    # |SPARKLING|>  Finished preprocessing in 7.300827503204346s
    # |SPARKLING|>  CVI predictor recommended CALINSKI_HARABASZ in 90.71584367752075s
    # === wall-robot-navigation.csv ===
    # |SPARKLING|>  Finished preprocessing in 1.6549997329711914s
    # |SPARKLING|>  CVI predictor recommended GD41_APPROX in 43.234046936035156s
    # === character.csv ===
    # |SPARKLING|>  Finished preprocessing in 1.6277883052825928s
    # |SPARKLING|>  CVI predictor recommended CALINSKI_HARABASZ in 119.70400381088257s
    # === iris.csv ===
    # |SPARKLING|>  Finished preprocessing in 1.0443212985992432s
    # |SPARKLING|>  CVI predictor recommended CALINSKI_HARABASZ in 0.6992194652557373s
    # === blocks.csv ===
    # |SPARKLING|>  Finished preprocessing in 1.0706491470336914s
    # |SPARKLING|>  CVI predictor recommended GD41_APPROX in 44.36152911186218s
    # === mfeature.csv ===
    # |SPARKLING|>  Finished preprocessing in 0.9157025814056396s
    # |SPARKLING|>  CVI predictor recommended SF in 9.798141956329346s
    # === volcanoes-d4.csv ===
    # |SPARKLING|>  Finished preprocessing in 1.122236728668213s
    # |SPARKLING|>  CVI predictor recommended SF in 91.46459031105042s
    # === wine-quality-white.csv ===
    # |SPARKLING|>  Finished preprocessing in 0.9923214912414551s
    # |SPARKLING|>  CVI predictor recommended GD41_APPROX in 37.30893635749817s
    # === abalone.csv ===
    # |SPARKLING|>  Finished preprocessing in 0.8053402900695801s
    # |SPARKLING|>  CVI predictor recommended SF in 28.83964252471924s
