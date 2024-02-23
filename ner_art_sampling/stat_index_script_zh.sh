#!/bin/bash
#
#SBATCH --job-name=stat_index_zh
#SBATCH --output=stat_index_zh.txt  # output file
#SBATCH -e stat_index_zh.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=100000


python3 stat_index.py -i indexes/zh-wiki-v2.index -o indexes/zh-wiki-v2-stats.index