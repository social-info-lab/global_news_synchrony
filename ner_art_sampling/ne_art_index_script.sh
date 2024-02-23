#!/bin/bash
#

#SBATCH --job-name=ne_art_index
#SBATCH --output="script_output/ne_art_index.txt"  # output file
#SBATCH -e "script_output/ne_art_index.err"        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=100000

python3 ne_art_index.py -s $1 -e $2