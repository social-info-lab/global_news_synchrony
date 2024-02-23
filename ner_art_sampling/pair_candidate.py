'''script usage example:'''
'''sbatch --output=script_output/pair_candidate/pair_candidate_0_31.txt -e script_output/pair_candidate/pair_candidate_0_31.err pair_candidate_script.sh network_pairs/pairs-top10-ne-art-wiki-filtered_0_31.txt 0 31'''


import itertools
import ast
# from scipy.spatial import distance
import json
import datetime
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

from utils import lang_list, News, Biased_News, strong_bias_class, GEN_RECORD_FILENAME, MIN_NE_SIM, CANDIDATE_NUM, DATE_WINDOW, date_diff, text2tokens, MAX_TXT_SIM

import cython_jaccard_sim

script_version = 8


def load_article(file,lineno):
	try:
		lineno=int(lineno)
	except ValueError:
		print("DEBUG", lineno, "Note: known issue, needs debugging.")
		# probably some issue about saving/loading namedtuples
		sys.exit()
	# symbolic link doesn't look good with current inter-lang data storage format since the current wiki data is in scott's directory while the offsets are in my directory. Even use symbolic links it will always change the code since we use the json files and offsets at different time.
	with open(file.replace(".json", ".offsets").replace(".gz", "").replace("home/scott/wikilinked","home/xichen/mediacloud/ner_art_sampling/wikilinked"), "r") as fh:
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


def dense2sparse(dense):
	sparse = [0] * 26 ** 2
	for k, v in dense:
		sparse[k] = v
	return sparse


def read_data(index_filename):
	data = []
	print(datetime.datetime.now(), f"Reading data from {index_filename}")
	with open(index_filename, "r") as fh:
		n_lines = 0
		print("Loaded...", end=" ")
		for line in fh:
			file, lineno, reladate, vec = line.strip().split("\t")
			# we've heavily oversampled this day, so now we can undersample it
			# please move this step to create_index once index versioning and tracking is implemented (we need similar tracking for the remaining two scripts as well)
			# it's easy to forget it, especially on a local machine with alternative setup and small data samples
			# all article-specific filtering should happen at the index level anyway

			if "2020-01-01" in file:
				if socket.gethostname()!="pms-mm.local":
					continue

			vec = tuple(ast.literal_eval(vec))
			# tuples have lower footprint than lists, lists have lower than sets
			# https://stackoverflow.com/questions/46664007/why-do-tuples-take-less-space-in-memory-than-lists/46664277
			# https://stackoverflow.com/questions/39914266/memory-consumption-of-a-list-and-set-in-python

			# # we don't consider the repeat times of words in the same document here
			# vec = tuple(k for k, v in vec)
			tup = News(file, int(lineno), int(reladate), vec)
			data.append( tup )
			n_lines += 1
			if n_lines%500000 == 0:
				print(n_lines,"lines", end=" ", flush=True)
				gc.collect()
	print(datetime.datetime.now(), "Loaded", len(data), "news from",index_filename)

	gc.collect()
	return data


def set_default(obj):
	if isinstance(obj, set):
		return list(obj)
	raise TypeError


# rank starts at 0
def find_pair_candidates(input_file, start_date, end_date):
	print("Finding candidate pairs...")

	lang_dict = {0:"en", 1:"de", 2:"es", 3:"pl", 4:"zh", 5:"fr", 6:"ar", 7:"tr", 8:"it", 9:"ru"}
	# lang_dict = {1:"de"}
	data_list = []
	for lang in lang_dict.values():
		data_list.append(read_data(f"indexes/{lang}-{args.index_suffix}"))


	outdir = f"network_pairs/candidates/candidates-top10-ne-filtered_{start_date}_{end_date}"
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	print("Will save results to directory:", outdir, flush=True)
	filename = f"{outdir}/{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.jsonl"
	print("Saving to", filename)

	total_pair_count = 0
	unique_pair_count = 0
	intra_lang_unique_pair_count = 0
	inter_lang_unique_pair_count = 0
	dup_count = 0
	with open(filename, "w") as output:
		with open(input_file, "r") as fh:
			for line in fh:
				total_pair_count += 1
				try:
					if total_pair_count%100000==0:
						print(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S} - processing {total_pair_count} pairs, {unique_pair_count} unique pairs", flush=True)

					cur_content = line.strip().split(" ")
					type = cur_content[0]
					line_num_composition = cur_content[-1]

					if type[2] == type[3]:
						input_type = "intra-lang"
					else:
						input_type = "inter-lang"

					# if pair_count % 100000 == 0:
					# 	print(f"type: {type};  line_num_composition:{line_num_composition}")

					lang1_index = int(type[2])
					lang2_index = int(type[3])
					line_num_split_pos = int(type[0])
					line_num_sec1 = int(line_num_composition[:line_num_split_pos])
					line_num_sec2 = int(line_num_composition[line_num_split_pos:])
					'''read article from the indexes'''

					art1 = data_list[lang1_index][line_num_sec1-1]
					art2 = data_list[lang2_index][line_num_sec2-1]

					jaccard_sim_struct = cython_jaccard_sim.cython_jaccard_similarity(art1.vec, art2.vec)
					jaccard, union = jaccard_sim_struct["similarity"], jaccard_sim_struct["size_union"]
					intersect = jaccard * union

					# filter those without shared name entities
					if union < 1:
						continue


					if jaccard >= MIN_NE_SIM[input_type]:

						# filtering duplicates
						if input_type == 'intra-lang':
							a1 = load_article(art1.file.replace(".gz", "").replace("home/scott/wikilinked","mnt/nfs/work1/grabowicz/xchen4/mediacloud_temp/scott/ner"), art1.lineno)
							a2 = load_article(art2.file.replace(".gz", "").replace("home/scott/wikilinked","mnt/nfs/work1/grabowicz/xchen4/mediacloud_temp/scott/ner"), art2.lineno)
							text1 = a1["story_text"].strip()
							text2 = a2["story_text"].strip()
							words1 = text2tokens(text1, lang_dict[lang1_index])
							words2 = text2tokens(text2, lang_dict[lang2_index])

							# text similarity threshold to reduce the duplicates in samples (only for intra-lang)
							text_jaccard_sim_struct_non_repeat = cython_jaccard_sim.cython_jaccard_similarity3(words1,words2)
							text_jaccard_non_repeat, text_union_non_repeat = text_jaccard_sim_struct_non_repeat["similarity"], text_jaccard_sim_struct_non_repeat["size_union"]
							# print("text_jaccard_non_repeat: ", text_jaccard_non_repeat)
							if text_jaccard_non_repeat > MAX_TXT_SIM:
								dup_count += 1
								continue

						pair = {
							# "cosine":cos,
							"script_version": script_version,
							"jaccard": jaccard,
							"union_ne_count": union,
							"intersect_ne_count": intersect,
							"a1_ne_count": len(art1.vec),
							"a2_ne_count": len(art2.vec),
							"article1": art1,
							"article2": art2,
						}
						output.write(json.dumps(pair, default=set_default))
						output.write("\n")

						if input_type == "intra-lang":
							intra_lang_unique_pair_count += 1
						else:
							inter_lang_unique_pair_count += 1
						unique_pair_count += 1
				except:
					print("this line didn't work..")

	print(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S} - : DONE")
	print(f"processed {total_pair_count} pairs, {unique_pair_count} unique pairs, and {dup_count} duplicates successfully...", flush=True)
	print(f"there are {intra_lang_unique_pair_count} intra-lang unique pairs and {inter_lang_unique_pair_count} inter-lang unique pairs...", flush=True)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-file", dest="input_file",
						default="network_pairs/pairs-top10-ne-art-wiki-filtered_0_31.txt", type=str,
						help="Input file where the pairs are stored.")
	parser.add_argument("-ind", "--index-suffix", dest="index_suffix",
						default="wiki-v2-filtered.index", type=str,
						help="Index suffix.")
	parser.add_argument("-s", "--start-date", dest="start_date",
						default=0, type=int,
						help="Start date.")
	parser.add_argument("-e", "--end-date", dest="end_date",
						default=0, type=int,
						help="End date.")
	args = parser.parse_args()

	'''loading the pairs, continue here'''
	find_pair_candidates(args.input_file, args.start_date, args.end_date)
