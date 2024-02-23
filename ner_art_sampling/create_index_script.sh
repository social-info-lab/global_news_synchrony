#!/bin/bash
#

#SBATCH --job-name=create_index
#SBATCH --output="script_output/create_index.txt"  # output file
#SBATCH -e "script_output/create_index.err"        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=100000

python3 create_index.py -i "$1" -o $2 