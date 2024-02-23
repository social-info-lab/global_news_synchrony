
# for trial, not used

# get key info of articles to estimate total article pair numbers of the global network before fitering out

import itertools
import ast
# from scipy.spatial import distance
import json
import datetime
import multiprocessing
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
import pandas as pd



def load_article(file,lineno):
    try:
        lineno=int(lineno)
    except ValueError:
        print("DEBUG", lineno, "Note: known issue, needs debugging.")
        # probably some issue about saving/loading namedtuples
        sys.exit()
    file = file.replace("home","mnt/nfs/work1/grabowicz/xchen4/mediacloud_temp")
    # print(file, flush=True)
    # symbolic link doesn't look good with current inter-lang data storage format since the current wiki data is in scott's directory while the offsets are in my directory. Even use symbolic links it will always change the code since we use the json files and offsets at different time.
    with open(file.replace(".json", ".offsets").replace(".gz", "").replace("scott/wikilinked","xichen/mediacloud/ner_art_sampling/wikilinked"), "r") as fh:
    # with open(file.replace(".json", ".offsets").replace(".gz", ""), "r") as fh:
        offsets = [int(x) for x in fh]

    if ".gz" not in file:
        with open(file,"r") as fh:
            fh.seek(offsets[lineno])
            line=fh.readline()
    else:
        with gzip.open(file,"rt") as fh:
            fh.seek(offsets[lineno])
            line=fh.readline()
    return json.loads(line)

def read_and_filter_data(index_filename):
    # print("start 1")
    # country_outlets_list = pd.read_csv("country_info/integrated_country.csv")
    # country_outlets_id = country_outlets_list["media_id"].to_list()
    # country_outlets_pub_country = country_outlets_list["pub_country"].to_list()
    # print("start 2")
    # country_outlets_id_pub_country = {}
    # for i in range(len(country_outlets_id)):
    #     country_outlets_id_pub_country[country_outlets_id[i]] = country_outlets_pub_country[i]

    data = []
    print(datetime.datetime.now(), f"Reading data from {index_filename}", flush=True)
    with open(args.output_filename, "w") as out:
        with open(index_filename, "r") as fh:
            n_lines = 0
            print("Loaded...", end=" ")
            for line in fh:
                file, lineno, reladate, url, vec = line.strip().split("\t")

                art = load_article(file, lineno)
                media_id = art["media_id"]

                # if art["media_id"] in country_outlets_id_pub_country:
                #     pub_country = country_outlets_id_pub_country[art["media_id"]]
                #     print("in ", end=" ", flush=True)
                # else:
                #     pub_country = ""
                #     print("out ", end=" ", flush=True)

                # # we don't consider the repeat times of words in the same document here
                # vec = tuple(k for k, v in vec)
                out.write(f"{reladate}\t{media_id}\n")
                n_lines += 1
                if n_lines%10000 == 0:
                    print(n_lines,"lines", end=" ", flush=True)

    print(datetime.datetime.now(), "Loaded", len(data), "news from",index_filename)
    # we want to get different pairs every time, some pairs may randomly repeat,
    # but chances of this are low and we can deal with them later
    # before sending them to annotators
    gc.collect()

    return 0

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--index-filename", dest="index_filename",
                        default="indexes/en-5k.index", type=str,
                        help="Index filename.")
    parser.add_argument("-o", "--output-filename", dest="output_filename",
                        default="indexes/en-5k.index", type=str,
                        help="Index filename.")

    args = parser.parse_args()
    print("parse.....", flush=True)
    read_and_filter_data(args.index_filename)

    print(f"successfully stats {args.index_filename}...")
