#!/bin/bash
#

#SBATCH --job-name=match_inference_country_bias
#SBATCH --output="script_output/visualize_matched_inference/visualize_matched_inference.txt"  # output file
#SBATCH -e "script_output/visualize_matched_inference/visualize_matched_inference.err"        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=150000

python3 visualize_matched_inference.py -i $1 -opt $2