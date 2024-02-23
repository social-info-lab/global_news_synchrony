# filter articles in the indexes for further generating the ne_art_index

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

MIN_ART_NE_NUM = 10

def read_and_filter_data(index_filename):
    data = []
    print(datetime.datetime.now(), f"Reading data from {index_filename}")
    with open(args.output_filename, "w") as out:
        with open(index_filename, "r") as fh:
            n_lines = 0
            print("Loaded...", end=" ")
            for line in fh:
                file, lineno, reladate, url, vec = line.strip().split("\t")
                # we've heavily oversampled this day, so now we can undersample it
                # please move this step to create_index once index versioning and tracking is implemented (we need similar tracking for the remaining two scripts as well)
                # it's easy to forget it, especially on a local machine with alternative setup and small data samples
                # all article-specific filtering should happen at the index level anyway

                if "2020-01-01" in file:
                    if socket.gethostname()!="pms-mm.local":
                        continue

                vec = tuple(ast.literal_eval(vec))
                if len(vec) <= MIN_ART_NE_NUM:
                    continue

                # tuples have lower footprint than lists, lists have lower than sets
                # https://stackoverflow.com/questions/46664007/why-do-tuples-take-less-space-in-memory-than-lists/46664277
                # https://stackoverflow.com/questions/39914266/memory-consumption-of-a-list-and-set-in-python

                # # we don't consider the repeat times of words in the same document here
                # vec = tuple(k for k, v in vec)
                out.write(f"{file}\t{lineno}\t{reladate}\t{vec}\n")
                n_lines += 1
                if n_lines%500000 == 0:
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
    read_and_filter_data(args.index_filename)

    print(f"successfully filtered {args.index_filename}...")
