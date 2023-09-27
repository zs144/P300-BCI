#!/bin/bash
#SBATCH --output misc_out.err # output log file
#SBATCH -e misc_err.err # error log file

# #SBATCH --gres=gpu:RTXA5000:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G # 8 GBs RAM
#SBATCH -p mainsahlab-gpu --account=mainsahlab
# #SBATCH -p gpu-common

# add cuda and cudnn path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
# add my library path
export PYTHONPATH=$PYTHONPATH:/hpc/home/xc130/Code_Python

# execute my file
python -u misc.py