#!/bin/bash
#

#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=100000  # Requested Memory
#SBATCH -p gypsum-2080ti # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 07-00:00:00  # Job time limit
#SBATCH -o script_output/weight_link_inference.out  # %j = job ID

python3 model_evaluation.py -lo $1 -m $2 -nt $3 -bs $4 -tl $5