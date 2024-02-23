#!/bin/bash
#

#SBATCH --job-name=integrate_matched_inference
#SBATCH --output="script_output/integrate_matched_inference/integrate_matched_inference.txt"  # output file
#SBATCH -e "script_output/integrate_matched_inference/integrate_matched_inference.err"        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=150000

python3 integrate_matched_inference.py -s $1 -e $2