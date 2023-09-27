#!/bin/bash
#SBATCH --output misc_out.err # output log file
#SBATCH -e misc_err.err # error log file

#SBATCH -c 4
#SBATCH --mem-per-cpu=4G # 4 GBs RAM

# add cuda and cudnn path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
# add my library path
export PYTHONPATH=$PYTHONPATH:/hpc/home/xc130/Code_Python

# execute my file
python -u misc.py