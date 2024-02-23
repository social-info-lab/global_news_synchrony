#!/bin/bash
#
#SBATCH --job-name=stat_index
#SBATCH --output=stat_index.txt  # output file
#SBATCH -e stat_index.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem=100000

python3 stat_index.py -i indexes/en-wiki-v2.index -o indexes/en-wiki-v2-stats.index
python3 stat_index.py -i indexes/es-wiki-v2.index -o indexes/es-wiki-v2-stats.index
python3 stat_index.py -i indexes/de-wiki-v2.index -o indexes/de-wiki-v2-stats.index
python3 stat_index.py -i indexes/fr-wiki-v2.index -o indexes/fr-wiki-v2-stats.index
python3 stat_index.py -i indexes/ar-wiki-v2.index -o indexes/ar-wiki-v2-stats.index
python3 stat_index.py -i indexes/tr-wiki-v2.index -o indexes/tr-wiki-v2-stats.index
python3 stat_index.py -i indexes/pl-wiki-v2.index -o indexes/pl-wiki-v2-stats.index
python3 stat_index.py -i indexes/zh-wiki-v2.index -o indexes/zh-wiki-v2-stats.index
python3 stat_index.py -i indexes/ru-wiki-v2.index -o indexes/ru-wiki-v2-stats.index
python3 stat_index.py -i indexes/it-wiki-v2.index -o indexes/it-wiki-v2-stats.index