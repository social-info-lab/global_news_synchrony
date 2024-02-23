#!/bin/bash
#

#SBATCH --job-name=pair_candidate
#SBATCH --output="script_output/pair_candidate.txt"  # output file
#SBATCH -e "script_output/pair_candidate.err"        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=100000

python3 pair_candidate.py -i $1 -s $2 -e $3