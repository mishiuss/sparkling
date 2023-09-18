import numpy as np
import pandas as pd
import pyspark


# Обработка данных о научных сотрудниках университета и их научных интересах.
# Из этических соображений имена были убраны из набора данных

DATASET_NAME = 'itmo-authors'


def get_cumulative(history):
    values = [{'val': run['value'], 'algo': run['algo']} for run in history]
    history_df = pd.DataFrame(values)
    history_df.replace(np.inf, np.nan, inplace=True)
    history_df.replace(-np.inf, np.nan, inplace=True)

    data = history_df.pivot(columns="algo", values='val')
    algos = {col: data[col].fillna(method='ffill') for col in data.columns}
    return {key: algos[key].cummin() for key in algos.keys()}, algos


def plot_history(history):
    try:
        from matplotlib import pyplot as plt

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(DATASET_NAME, fontsize=20)

        algos_cumulative, algos = get_cumulative(history)
        for key in algos.keys():
            algo_runs = list(range(len(algos_cumulative[key])))
            plt.step(algo_runs, algos_cumulative[key], label=key)

        plt.legend()
        plt.savefig(f'examples/logs/{DATASET_NAME}.png')
    except ImportError:
        print('matplotlib is not installed')


if __name__ == '__main__':
    from sparkling import *

    conf = pyspark.SparkConf()\
        .setAppName(f'sparkling-{DATASET_NAME}') \
        .setMaster('local[*]') \
        .set('spark.jars', 'bin/heaven.jar')

    sc = pyspark.SparkContext.getOrCreate(conf)
    sc.setCheckpointDir('examples/checkpoint')
    sql = pyspark.SQLContext(sc)

    df = sql.read.csv(f'examples/data/{DATASET_NAME}.csv', inferSchema=True, header=True)

    sparkling_df = SparklingBuilder(df, partitions='default') \
        .tabular('features', set(df.columns), Distance.EUCLIDEAN) \
        .create()
    sparkling_df.df.show()

    optimiser = Sparkling(sparkling_df, measure=Internal.SILHOUETTE_APPROX)
    result = optimiser.run(time_limit=90)

    result.label_sdf.df.show()
    opt_history = optimiser.history_json()

    with open(f'examples/logs/itmo-authors-local.json', 'w') as fp:
        fp.write(opt_history)
    plot_history(opt_history)
