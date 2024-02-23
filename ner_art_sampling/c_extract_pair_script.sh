#!/bin/bash
#

#SBATCH --job-name=c_extract_pair
#SBATCH --output="script_output/c_extract_pair.txt"  # output file
#SBATCH -e "script_output/c_extract_pair.err"        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=190000

g++ -std=c++11 -o c_extract_pair extract_pair.cpp
./c_extract_pair $1 $2