#!/bin/bash
#

#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=160000  # Requested Memory
#SBATCH -p gypsum-2080ti # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 07-00:00:00  # Job time limit
#SBATCH -o script_output/weight_link_inference.out  # %j = job ID

python3 load_model_inference.py -i $1 -s $2 -e $3 -lo $4 -nt $5 -bs $6 -tl $7