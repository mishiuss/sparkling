dataframe=$1
mab_solver=$2
hyper_opt=$3
measure=$4
time_budget=$5

checkpoint_dir=hdfs:///user/checkpoint
logs_path=examples/logs/$dataframe/$measure-$hyper_opt-$mab_solver.json

# If dataset was once preprocessed, you can use cached transformed
# version instead of executing preprocessing pipeline from scratch
fetch_mode=preprocessed

case "$fetch_mode" in
  "preprocessed" ) py_env=minimal;;
  "raw"          ) py_env=pytorch;;
  *              ) exit 1;;
esac

env_file=$py_env.tar.gz

zip -r sparkling.zip sparkling

PYSPARK_DRIVER_PYTHON=/opt/conda/envs/$py_env/bin/python \
PYSPARK_PYTHON=./environment/bin/python \
/usr/bin/spark-submit \
--archives $env_file#environment \
--py-files sparkling.zip \
--master yarn \
--jars bin/heaven.jar \
--deploy-mode client \
--conf spark.sql.execution.arrow.enabled=true \
--conf spark.sql.execution.arrow.maxRecordsPerBatch=512 \
--conf spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT=1 \
--conf spark.broadcast.blockSize=4m \
--conf spark.default.parallelism=10 \
examples/cloud/multimodal.py \
$dataframe $fetch_mode $checkpoint_dir $logs_path $mab_solver $hyper_opt $measure $time_budget
