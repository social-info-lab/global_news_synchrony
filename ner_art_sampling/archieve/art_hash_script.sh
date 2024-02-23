#!/bin/bash
#
#SBATCH --job-name=art_hash
#SBATCH --output=art_hash.txt  # output file
#SBATCH -e art_hash.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=60000

python3 art_hash.py