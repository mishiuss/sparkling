import sys
import pyspark

from fetcher import MultiModalFetcher
from sparkling import *

# This script is the common facade for processing cloud dataframes

dataframe = sys.argv[1]  # dataframe's short name
fetch_mode = sys.argv[2]  # 'raw' to preprocess from scratch or 'preprocessed' to use cached transformed version
checkpoint_dir = sys.argv[3]  # should be path in hdfs
logs_path = sys.argv[4]  # local path on master node

mab_solver = MabSolver[sys.argv[5].upper()]  # 'softmax' or 'ucb'
hyper_opt = HyperOpt[sys.argv[6].upper()]  # 'optuna' or 'smac'
measure = Internal[sys.argv[7].upper()]  # members of Internal enum
time_budget = int(sys.argv[8])  # number of seconds for optimisation

data_fetcher = MultiModalFetcher.by_name(dataframe)

ss = pyspark.sql.SparkSession.Builder().getOrCreate()
ss.sparkContext.setCheckpointDir(checkpoint_dir)

sparkling_df = getattr(data_fetcher, fetch_mode)(ss)  # read and prepare dataframe

optimiser = Sparkling(sparkling_df, mab_solver=mab_solver, hyper_opt=hyper_opt, measure=measure)
optimal = optimiser.run(time_limit=time_budget)

with open(logs_path, 'w') as fp:
    fp.write(optimiser.history_json())
