#!/usr/bin/env bash

## These Double # signs are interpreted by SLURM as comments
## while single # signs before SBATCH are not

# Job name:
#SBATCH --job-name=example_matlab
#
# Account: railabs or bashirlab
#SBATCH --account=railabs
#
# Partition: # It is default queue for now.
#Will be modified as nodes are added
#
#SBATCH --partition=defq 
#
# Number of nodes: (1 should be sufficient)
#SBATCH --nodes=1 
#
# Number of tasks (If your is parallel feel free to increase this
# Pro trip make sure total number of workers for your job will be
# ntasks*cpus-per-task. cpus-per-task is by default 1. If you order 1
# task with 12 cpus-per-task yo will have 12 cores utilized.
# Decide carefully. Some nodes have 98 cores (numworkers) and some have
# 128 cores. Consider that other will be using the system too. If you 
# request 64 total, slurm may put your job  pending untill 
# 64 cpu cores are available.

#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=12
#
# Output and Error Files
##  %x -> Job name
##  %N -> which node it ran on
##  %J -> Your job id
##  %u -> Your username 
#SBATCH -o %x.%N.%J.%u.out # STDOUT
#SBATCH -e matlab.%N.%J.%u.err # STDERR


# load a MatLab module
module load matlab/R2022b 

# create a variable for the basename of the script to be run
# without .m extension
BASE_MFILE_NAME=helloworld_par

# execute code without a GUI
cd $SLURM_SUBMIT_DIR
# -r Run Immediately
#
# -nodisplay Display disabled (must)
# Also starts the JVM and does not start the
# desktop in addition it also ignores Commands
# and the DISPLAY environment variable in LINUX.
#
# -nojvm JVM disabled
# Does not start the JVM software and uses current window.
# Graphics will not work without the JVM.
#
matlab -nodisplay  -r "$BASE_MFILE_NAME"
