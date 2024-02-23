#!/bin/bash
#
#SBATCH --job-name=create_stat_index
#SBATCH --output=create_stat_index.txt  # output file
#SBATCH -e create_stat_index.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=100000


python3 create_stat_index.py -i $1 -o $2