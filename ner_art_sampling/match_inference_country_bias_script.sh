#!/bin/bash
#

#SBATCH --job-name=match_inference_country_bias
#SBATCH --output="script_output/match_inference_country_bias/match_inference_country_bias.txt"  # output file
#SBATCH -e "script_output/match_inference_country_bias/match_inference_country_bias.err"        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=150000

python3 match_inference_country_bias.py