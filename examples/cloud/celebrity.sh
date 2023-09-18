celebrity_path=$1
framework=$2
time_limit=$3

out_path=examples/logs/celebrity-cloud.json
checkpoint_dir=hdfs:///user/checkpoint
env_file=$framework.tar.gz

zip -r sparkling.zip sparkling

PYSPARK_DRIVER_PYTHON=/opt/conda/envs/$framework/bin/python \
PYSPARK_PYTHON=./environment/bin/python \
/usr/bin/spark-submit \
--archives $env_file#environment \
--py-files sparkling.zip \
--master yarn \
--jars bin/heaven.jar \
--deploy-mode client \
--conf spark.sql.execution.arrow.enabled=true \
--conf spark.sql.execution.arrow.maxRecordsPerBatch=128 \
--conf spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT=1 \
--conf spark.broadcast.blockSize=4m \
--conf spark.default.parallelism=10 \
examples/cloud/celebrity.py \
$celebrity_path $out_path $checkpoint_dir $framework $time_limit
