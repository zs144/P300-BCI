#!/bin/bash
#SBATCH --output miscg_out.err # output log file
#SBATCH -e miscg_err.err # error log file

#SBATCH --gres=gpu:1
#SBATCH --mem=8G # 8 GBs RAM 
#SBATCH -p gpu-common
#SBATCH --exclude=dcc-core-gpu-27

# # add cuda and cudnn path
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
# add my library path
export PYTHONPATH=$PYTHONPATH:/hpc/home/xc130/Code_Python

# execute my file
python -u misc.py