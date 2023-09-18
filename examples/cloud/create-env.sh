# NOTE: You should manually print this commands in shell,
# because 'conda activate' does not work as expected inside script

env_name=$1

conda create -y --name $env_name python=3.7
conda activate $env_name
pip install -r requirements/$env_name.txt
conda install -y conda-pack
conda pack -o $env_name.tar.gz
