#!/bin/bash
#
#SBATCH --job-name=pair_deduplicate
#SBATCH --output=pair_deduplicate.txt  # output file
#SBATCH -e pair_deduplicate.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=150000

python3 pair_deduplicate.py