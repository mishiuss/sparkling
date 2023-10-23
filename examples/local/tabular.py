import pyspark

if __name__ == '__main__':
    from sparkling import *

    # There are some downloaded datasets in repository. You can choose one of the following:
    # 'abalone', 'blocks', 'character', 'iris', 'mfeature', 'visualizing_soil',
    # 'volcanoes-d4', 'wall-robot-navigation', 'wine-quality-white'.
    DATA_ROOT, DATASET = 'heaven/src/test/data', 'abalone'

    # Enable extra logging (default is SparklingLogLevel.EXCERPT)
    SparklingLogger.level = SparklingLogLevel.INFO

    # Create local Spark context. Including heaven.jar into Spark context is obligatory
    conf = pyspark.SparkConf() \
        .setAppName('sparkling-unimodal') \
        .setMaster('local-cluster[4, 2, 1024]') \
        .set('spark.default.parallelism', '4') \
        .set('spark.jars', 'bin/heaven.jar')
    sc = pyspark.SparkContext.getOrCreate(conf)

    # Checkpoint dir is also obligatory for sparkling app
    sc.setCheckpointDir('examples/checkpoint')

    # Load dataframe into Spark context
    sql = pyspark.SQLContext(sc)
    df = sql.read.csv(f'{DATA_ROOT}/{DATASET}.csv', inferSchema=True, header=True)

    # All columns except 'class' are features, which represents single tabular modality
    feature_cols = set(df.columns) - {'class'}

    # Make sure you pass class column name into SparklingBuilder, otherwise this column
    # will be lost, and you will not be able to evaluate external measures.
    # Define tabular modality, so that all feature_cols will be assembled into single column 'features'
    builder = SparklingBuilder(df, class_col='class', partitions='default') \
        .tabular('features', feature_cols, Distance.EUCLIDEAN)

    # Execute preprocessing pipeline to obtain ready-to-optimise dataframe (e.g., SparklingDF)
    sparkling_df = builder.create()

    # You can access underlying Spark dataframe, but modifying it leads to undefined behaviour
    sparkling_df.df.show()

    # Configure optimisation pipeline for preprocessed dataframe.
    # By default, using KMeans, Birch, MeanShift and BisectingKMeans clustering algorithms with default search space.
    # Note, that you can also include CLIQUE, CURE, DBSCAN, SpectralAdjacency and SpectralSimilarity, but spectral
    # algorithms implies huge memory consumption and others showed unstable performance on real YARN cluster.
    # Using None as a target measure invokes meta-classifier, which recommends for optimisation
    # one of the following CVIs: CALINSKI_HARABASZ, SILHOUETTE_APPROX, SF, GD41_APPROX
    optimiser = Sparkling(sparkling_df, measure=None)

    # Run optimiser for nearly 100 seconds and obtain labeled dataframe,
    # which was evaluated as the best according to target measure
    optimal = optimiser.run(time_limit=100)

    # You can obtain underlying labeled Spark dataframe
    optimal.label_sdf.df.show()

    # You can check achieved target measure value:
    print(f'{optimiser.measure.name}: {optimal.value}')

    # You can evaluate any external measure (provided dataframe possesses external classes)
    print(f'F1: {External.F1.evaluate(optimal.label_sdf)}')
    print(f'RAND: {External.RAND.evaluate(optimal.label_sdf)}')

    # You can also evaluate any internal measure for SparklingDF
    print(f'SILHOUETTE_APPROX: {Internal.SILHOUETTE_APPROX.evaluate(optimal.label_sdf)}')
    print(f'GD41_APPROX: {Internal.GD41_APPROX.evaluate(optimal.label_sdf)}')

    # You can save optimisation history (invoked clustering algorithms with hyperparameters) into json format:
    with open(f'examples/logs/{DATASET}-local.json', 'w') as fp:
        fp.write(optimiser.history_json())

    sc.stop()
