# integrated all the matched inference pairs for removing duplicates across time bins
# this is last step before visualization

''''''
''' sbatch --output=script_output/integrate_matched_inference/integrate_matched_inference_0_180.txt -e script_output/integrate_matched_inference/integrate_matched_inference_0_180.err integrate_matched_inference_script.sh 0 180'''

import json
from datetime import datetime
import collections
import random
import os
import gc
import socket
from argparse import ArgumentParser
import numpy as np
import time
import sys
import gzip
from utils import LANG_FAMILY, unify_url
import pandas as pd
import glob
from collections import defaultdict
import geopy.distance




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-glob", dest="input_glob",
                            default="network_pairs/matched_prediction/matched*/*", type=str,
                            help="Input globstring.")
    parser.add_argument("-o", "--output-prefix", dest="output_prefix",
                            default="network_pairs/matched_prediction/integrated/integrated_matched_prediction", type=str,
                            help="output file.")
    parser.add_argument("-s", "--start-date", dest="start_date",
                            default=0, type=int,
                            help="The start date for this ne-art index.")
    parser.add_argument("-e", "--end-date", dest="end_date",
                            default=180, type=int,
                            help="The end date for this ne-art index.")

    args = parser.parse_args()

    n_processedpairs = 0

    data_path = args.input_glob
    output_file = args.output_prefix + f"_{args.start_date}_{args.end_date}.csv"
    pair_dict = {}


    files = list(glob.glob(data_path))
    for file in files:
        cur_dir = file.split("/")[-2].split("_")
        cur_start_date = int(cur_dir[-2])
        cur_end_date = int(cur_dir[-1])
        if (cur_start_date < args.start_date) or (cur_end_date > args.end_date):
            continue

        with open(file, "r") as fh:
            for line in fh:
                n_processedpairs += 1
                if n_processedpairs % 1000 == 0:
                    print(
                        f"processed {n_processedpairs} pairs... ",
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), flush=True)

                pair = json.loads(line)
                if pair["stories_id1"] < pair["stories_id2"]:
                    pair_id = str(pair["stories_id1"]) + "_" +  str(pair["stories_id2"])
                else:
                    pair_id = str(pair["stories_id2"]) + "_" + str(pair["stories_id1"])

                if pair_id in pair_dict:
                    continue
                pair_dict[pair_id] = pair

    print("total pair number: ",len(pair_dict.keys()))

    # with open(output_file, "w") as outfile:
    #     for key in pair_dict.keys():
    #         outfile.write(json.dumps(pair_dict[key]))
    #         outfile.write("\n")


    df = pd.DataFrame.from_dict(pair_dict, orient='index')
    df.to_csv(output_file)

    print("finished integrating the matched inference pairs....")