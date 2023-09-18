data_name=$1
time_limit=$2

# Expect dataset file is in hadoop's /user/unimodal
datafile=/user/unimodal/$data_name.csv

# Master node's local path to store history json
outfile=examples/logs/$data_name-cloud.json

# Checkpoint should be in hadoop
checkpoint=hdfs:///user/checkpoint

# In guide sparkling-env.tar.gz was built on top of requirements/pytorch.txt,
# but this environment contains only requirements/minimal.txt,
# which is much lighter and enough to process tabular data
envfile=minimal.tar.gz

zip -r sparkling.zip sparkling

PYSPARK_DRIVER_PYTHON=python \
PYSPARK_PYTHON=./environment/bin/python \
/usr/bin/spark-submit \
--archives $envfile#environment \
--py-files sparkling.zip \
--master yarn \
--jars bin/heaven.jar \
--deploy-mode client \
--conf spark.default.parallelism=10 \
examples/cloud/unimodal.py \
$datafile $outfile $checkpoint $time_limit
