import sys
import pyspark

if __name__ == '__main__':
    from sparkling import Distance, SparklingBuilder, Sparkling, Internal
    celebrity_path, out_path, checkpoint_dir, framework, budget = sys.argv[1:6]

    if framework == 'pytorch':
        from sparkling.emb.torch import TorchImages
        model = TorchImages.SWIN_TRANSFORMER
    elif framework == 'tensorflow':
        from sparkling.emb.tf import TFImages
        model = TFImages.SWIN_TRANSFORMER
    else:
        raise ValueError(f'Unknown framework {framework}')

    ss = pyspark.sql.SparkSession.Builder().getOrCreate()
    ss.sparkContext.setCheckpointDir(checkpoint_dir)

    df = ss.read.csv(f'{celebrity_path}/celebrity.csv', header=True)

    sparkling_df = SparklingBuilder(df, class_col='class', partitions='default') \
        .image('celebrity', model, celebrity_path, Distance.MANHATTAN) \
        .create()

    sparkling_df.df.show()

    optimiser = Sparkling(sparkling_df, measure=Internal.DAVIES_BOULDIN)
    print(optimiser.run(time_limit=int(budget)))

    with open(out_path, 'w') as fp:
        fp.write(optimiser.history_json())
