import numpy as np
import pandas as pd
import pyspark
import json

def get_cumulative(history):
    values = [{'val': -run['value'], 'algo': run['algo']} for run in history]
    history_df = pd.DataFrame(values)
    history_df.replace(np.inf, np.nan, inplace=True)
    history_df.replace(-np.inf, np.nan, inplace=True)

    data = history_df.pivot(columns="algo", values='val')
    algos = {col: data[col].fillna(method='ffill') for col in data.columns}
    return {key: algos[key].cummax() for key in algos.keys()}, algos


def plot_history(history, dataset_name, measure_name):
    try:
        from matplotlib import pyplot as plt

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(dataset_name, fontsize=20)

        algos_cumulative, algos = get_cumulative(history)
        for key in algos.keys():
            algo_runs = list(range(len(algos_cumulative[key])))
            plt.step(algo_runs, algos_cumulative[key], label=key)

        plt.legend()
        plt.savefig(f'examples/logs/archeology/images/{dataset_name}_{measure_name}.png')
    except ImportError:
        print('matplotlib is not installed')


if __name__ == '__main__':
    from sparkling import *

    measure = ('sf', Internal.SF)

    DATA_ROOT, DATASET = 'examples/data', 'alloys_ml_coordinates'

    bests = []
    for iteration in range(10):
        conf = pyspark.SparkConf() \
            .setAppName(f'sparkling-{DATASET}') \
            .setMaster('local[*]') \
            .set('spark.jars', 'bin/heaven.jar')

        sc = pyspark.SparkContext.getOrCreate(conf)
        sc.setCheckpointDir('examples/checkpoint')
        sql = pyspark.SQLContext(sc)

        df = sql.read.csv(f'{DATA_ROOT}/{DATASET}.csv', inferSchema=True, header=False)

        sparkling_df = SparklingBuilder(df, partitions='default') \
            .tabular('features', set(df.columns), Distance.EUCLIDEAN) \
            .create()
        sparkling_df.df.show()

        optimiser = Sparkling(sparkling_df, measure=measure[1])
        result = optimiser.run(time_limit=90)

        result.label_sdf.df.show()
        opt_history = optimiser.history_json()
        with open(f'examples/logs/archeology/{DATASET}_{measure[0]}.json', 'w') as fp:
            fp.write(opt_history)

        plot_history(json.loads(opt_history), DATASET, measure[0])

        bests.append(min(json.loads(opt_history), key=lambda x: x['value']))

    print("Final results:")
    for best in bests:
        print(f"Final results:\n{best}")
