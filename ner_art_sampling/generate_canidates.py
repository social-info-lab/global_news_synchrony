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

from utils import News, Biased_News, strong_bias_class, GEN_RECORD_FILENAME, MIN_NE_SIM, CANDIDATE_NUM, DATE_WINDOW, date_diff

import cython_jaccard_sim

script_version = 8

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
			story_id, file, lineno, reladate, url, vec = line.strip().split("\t")
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
	# we want to get different pairs every time, some pairs may randomly repeat,
	# but chances of this are low and we can deal with them later
	# before sending them to annotators
	random.seed()
	random.shuffle(data)
	print(datetime.datetime.now(), "Shuffled the index", index_filename)

	gc.collect()
	return data

def read_biased_data(index_filename):
	data = []
	print(datetime.datetime.now(), f"Reading data from {index_filename}")
	with open(index_filename, "r") as fh:
		n_lines = 0
		print("Loaded...", end=" ")
		for line in fh:
			outlet_flag, match_country, mbfc_match_bias, abyz_match_url, mbfc_match_url, story_id, file, lineno, reladate, url, vec = line.strip().split("\t")
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
			tup = Biased_News(outlet_flag, match_country, mbfc_match_bias, file, int(lineno), int(reladate), vec)
			data.append( tup )
			n_lines += 1
			if n_lines%500000 == 0:
				print(n_lines,"lines", end=" ", flush=True)
				gc.collect()
	print(datetime.datetime.now(), "Loaded", len(data), "news from",index_filename)
	# we want to get different pairs every time, some pairs may randomly repeat,
	# but chances of this are low and we can deal with them later
	# before sending them to annotators
	random.seed()
	random.shuffle(data)
	print(datetime.datetime.now(), "Shuffled the index", index_filename)

	gc.collect()
	return data


def set_default(obj):
	if isinstance(obj, set):
		return list(obj)
	raise TypeError


# rank starts at 0
def find_candidates(script_version, input_type, bias_type, lang, data, index_filename, output_suffix, n_articles, rank, debug):
	if n_articles == 0:
		n_arts = len(data)
	else:
		n_arts = n_articles
	per_process = min(n_arts,len(data)) // CPUS
	start_index = per_process * rank
	end_index = start_index + per_process
	if n_articles == 0 and rank == CPUS - 1:
		end_index = len(data)
	if debug: print("DEBUG1", n_arts, per_process, start_index, end_index)

	fnm = os.path.basename(index_filename)
	index_name = os.path.splitext(fnm)[0]
	if output_suffix == "":
		outdir = "candidates/candidates-v" + str(script_version) + "_lang-" + index_name
	else:
		outdir = "candidates/candidates-v" + str(script_version) + "_lang-" + index_name + "_" + output_suffix
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	print("Will save results to directory:", outdir, flush=True)
	filename = f"{outdir}/n{n_articles}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}_rank{rank:02}.jsonl"
	print("Saving to", filename)
	with open(filename, "w") as output:
		print(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S} - rank {rank} processing {start_index} to {end_index}", flush=True)

		articles_to_process = end_index - start_index
		count = 0
		if debug: print("DEBUG2", start_index, end_index, len(data))
		for i in range(start_index, end_index):
			if (i - start_index) % 10 == 0:
				print(
					f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - rank {rank} : {i - start_index}/{articles_to_process} ({100 * (i - start_index) / articles_to_process:.0f}%)", flush=True)
			if debug and (i - start_index) > 100:
				# print("Stopped early for debugging.")
				break

			sim_pairs = list()
			art1 = data[i]
			# art1_date = art1.file.split("/")[-1].replace(".json","").replace(".gz","")

			# for English articles prioritize the MBFC articles with bias comparison
			# for non English artciels prioritize the ABYZ articles
			if bias_type:
				if lang == 'en':
					if art1.mbfc_match_bias != "left" and art1.mbfc_match_bias != "right":
						continue
				if lang != 'en' and 'ABYZ' not in art1.outlet_flag:
					continue

			for j in range(start_index + 1, len(data)):
				if i == j:
					continue

				art2 = data[j]

				# for english articles, we prioritize the articles with opposite biases "left" or "left_center" and "right" or "right-center"
				if bias_type and lang == 'en':
					cur_bias = [art1.mbfc_match_bias, art2.mbfc_match_bias]
					if ("left" not in cur_bias) or ("right" not in cur_bias):
						continue
					print("cur_bias: ", cur_bias)

				# art2_date = art2.file.split("/")[-1].replace(".json", "").replace(".gz", "")

				# temporal window, we don't consider the articles farther than a certain days
				# art_date_diff = int(date_diff(art1_date, art2_date).days)
				if abs(art1.reladate - art2.reladate) > DATE_WINDOW:
					continue

				jaccard_sim_struct = cython_jaccard_sim.cython_jaccard_similarity(art1.vec, art2.vec)
				jaccard, union = jaccard_sim_struct["similarity"], jaccard_sim_struct["size_union"]
				intersect = jaccard * union

				# filter those without shared name entities
				if union < 1:
					continue

				# this threshold value is based on jurgens figure
				# which shows that many nearest neighbors have NE similarity of 0.05
				# if jaccard >= 0.01:
				if jaccard >= MIN_NE_SIM[input_type]:
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
					sim_pairs.append(pair)

			# order pairs by similarity and save the most similar ones
			sim_pairs = sorted(sim_pairs, reverse=True, key=lambda x: x["jaccard"])
			max_saved = min(len(sim_pairs), CANDIDATE_NUM)
			for j in range(max_saved):
				output.write(json.dumps(sim_pairs[j], default=set_default))
				output.write("\n")
				count += 1

	print(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S} - rank {rank} : DONE")

	with open(GEN_RECORD_FILENAME, "a+") as genfile:
		genfile.write(f"{filename}\n")
	print(f"generate {count} candidates successfully...", flush=True)


# current we only support the matching from a specific language to English
def find_wiki_candidates(script_version, input_type, bias_type, lang, cmp_lang, data, cmp_data, index_filename, cmp_index_filename, output_suffix, n_articles, rank, debug):

	if n_articles == 0:
		n_arts = len(data)
	else:
		n_arts = n_articles
	per_process = n_arts // CPUS
	start_index = per_process * rank
	end_index = start_index + per_process
	if n_articles == 0 and rank == CPUS - 1:
		end_index = len(data)

	fnm = os.path.basename(index_filename)
	index_name = os.path.splitext(fnm)[0]
	fnm2 = os.path.basename(cmp_index_filename)
	cmp_index_name = os.path.splitext(fnm2)[0]
	if output_suffix == "":
		outdir = "candidates/candidates-v" + str(script_version) + "_lang-" + index_name + "-" + cmp_index_name
	else:
		outdir = "candidates/candidates-v" + str(script_version) + "_lang-" + index_name + "-" + cmp_index_name + "_" + output_suffix
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	print("Will save results to directory:", outdir)
	filename = f"{outdir}/n{n_articles}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}_rank{rank:02}.jsonl"
	print("Saving to", filename)
	with open(filename, "w") as output:
		print(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S} - rank {rank} processing {start_index} to {end_index}")

		articles_to_process = end_index - start_index
		count = 0
		for i in range(start_index, end_index):
			if (i - start_index) % 100 == 0:
				print(
					f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S} - rank {rank} : {i - start_index}/{articles_to_process} ({100 * (i - start_index) / articles_to_process:.0f}%)")
			if debug and (i - start_index) > 100:
				# print("Stopped early for debugging.")
				break

			sim_pairs = list()
			art1 = data[i]

			# for inter-lang articles prioritize the ABYZ articles
			if bias_type and 'ABYZ' not in art1.outlet_flag:
				continue

			# art1_date = art1.file.split("/")[-1].replace(".json","").replace(".gz","")
			for j in range(len(cmp_data)):
				art2 = cmp_data[j]
				if bias_type and 'ABYZ' not in art2.outlet_flag:
					continue

				# art2_date = art2.file.split("/")[-1].replace(".json", "").replace(".gz", "")

				# temporal window, we don't consider the articles farther than a certain days
				# art_date_diff = int(date_diff(art1_date, art2_date).days)
				if abs(art1.reladate - art2.reladate) > DATE_WINDOW:
					continue

				jaccard_sim_struct = cython_jaccard_sim.cython_jaccard_similarity(art1.vec, art2.vec)
				jaccard, union = jaccard_sim_struct["similarity"], jaccard_sim_struct["size_union"]
				intersect = jaccard * union

				# this threshold value is based on jurgens figure
				# which shows that many nearest neighbors have NE similarity of 0.05
				# if jaccard >= 0.01:
				if jaccard >= MIN_NE_SIM[input_type]:
					pair = {
						"script_version": script_version,
						"jaccard": jaccard,
						"union_ne_count": union,
						"intersect_ne_count": intersect,
						"a1_ne_count": len(art1.vec),
						"a2_ne_count": len(art2.vec),
						"article1": art1,
						"article2": art2,
					}
					sim_pairs.append(pair)
				# print(pair)
			# order pairs by similarity and save the most similar ones
			sim_pairs = sorted(sim_pairs, reverse=True, key=lambda x: x["jaccard"])
			# max_saved = min(len(sim_pairs), CANDIDATE_NUM)
			for j in range(len(sim_pairs)):
				output.write(json.dumps(sim_pairs[j], default=set_default))
				output.write("\n")
				count += 1

	print(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S} - rank {rank} : DONE")

	with open(GEN_RECORD_FILENAME, "a+") as genfile:
		genfile.write(f"{filename}\n")
	print(f"generate {count} candidates successfully...")


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-t", "--input-type", dest="input_type",
						default="intra-lang", type=str,
						help="the input type that we use for generate samples. It can be choose from 2 types: intra-lang or inter-lang(only support a specific language compared to English now).")
	parser.add_argument("-i", "--index-filename", dest="index_filename",
						default="indexes/en-5k.index", type=str,
						help="Index filename.")
	parser.add_argument("-cmp", "--compared-index-filename", dest="cmp_index_filename", default="indexes/en-wiki-v1.index", type=str,
						help="Compared Index filename.")
	parser.add_argument("-o", "--output-suffix", dest="output_suffix",
						default="", type=str,
						help="Output suffix.")
	parser.add_argument("-c", "--cpu-count", dest="cpu_count",
						default=1, type=int,
						help="Number of CPUs.")
	parser.add_argument("-n", "--number-of-articles", dest="n_articles",
						default=0, type=int,
						help="Number of articles.")
	parser.add_argument("-b", "--bias", dest="bias",
						default=False, action='store_true',
						help="Switch to bias dataset matching based index mode.")
	parser.add_argument("-d", "--debug", dest="debug",
						default=False, action='store_true',
						help="Debug mode running for only a small chunk of data.")
	args = parser.parse_args()

	print("Finding candidate pairs...")

	if args.cpu_count == 0:
		CPUS = multiprocessing.cpu_count()
	else:
		CPUS = args.cpu_count

	if not args.bias:
		if args.input_type == 'intra-lang':
			lang = args.index_filename.split("/")[-1][:2]
			data = read_data(args.index_filename)
			rank = 0
			find_candidates(script_version, args.input_type, args.bias, lang, data, args.index_filename, args.output_suffix, args.n_articles, rank, args.debug)
		elif args.input_type == 'inter-lang':
			lang = args.index_filename.split("/")[-1][:2]
			cmp_lang = args.index_filename.split("/")[-1][:2]
			data = read_data(args.index_filename)
			cmp_data = read_data(args.cmp_index_filename)
			gc.collect()
			rank = 0
			find_wiki_candidates(script_version, args.input_type, args.bias, lang, cmp_lang, data, cmp_data, args.index_filename, args.cmp_index_filename, args.output_suffix, args.n_articles, rank, args.debug)
	else:
		if args.input_type == 'intra-lang':
			lang = args.index_filename.split("/")[-1][:2]
			data = read_biased_data(args.index_filename)
			rank = 0
			find_candidates(script_version, args.input_type, args.bias, lang, data, args.index_filename, args.output_suffix, args.n_articles, rank, args.debug)
		elif args.input_type == 'inter-lang':
			lang = args.index_filename.split("/")[-1][:2]
			cmp_lang = args.index_filename.split("/")[-1][:2]
			data = read_biased_data(args.index_filename)
			cmp_data = read_biased_data(args.cmp_index_filename)
			gc.collect()
			rank = 0
			find_wiki_candidates(script_version, args.input_type, args.bias, lang, cmp_lang, data, cmp_data, args.index_filename, args.cmp_index_filename, args.output_suffix, args.n_articles, rank, args.debug)


	# if args.input_type == 'intra-lang':
	# 	def fun_for_pool(rank):
	# 		find_candidates(script_version, data, args.index_filename, args.output_suffix, args.n_articles, rank, args.debug)
	#
	# 	if CPUS == 1:
	# 		data = read_data(args.index_filename)
	# 	elif CPUS > 1:
	# 		data = multiprocessing.Manager().list(read_data(args.index_filename))
	# 	gc.collect()
	# 	print(f"finishing the garbage collection for {args.index_filename}...", flush=True)
	#
	# 	with multiprocessing.Pool(CPUS) as pool:
	# 		pool.map(fun_for_pool, list(range(CPUS)))
	#
	# elif args.input_type == 'inter-lang':
	# 	def fun_for_pool(rank):
	# 		find_wiki_candidates(script_version, data, cmp_data, args.index_filename, args.cmp_index_filename, args.output_suffix, args.n_articles, rank, args.debug)
	#
	# 	if CPUS == 1:
	# 		data = read_data(args.index_filename)
	# 		cmp_data = read_data(args.cmp_index_filename)
	# 		gc.collect()
	# 	elif CPUS > 1:
	# 		data = multiprocessing.Manager().list(read_data(args.index_filename))
	# 		gc.collect()
	# 		print(f"finishing the garbage collection for {args.index_filename}...", flush=True)
	#
	# 		cmp_data = multiprocessing.Manager().list(read_data(args.cmp_index_filename))
	# 		gc.collect()
	# 		print(f"finishing the garbage collection for {args.cmp_index_filename}...", flush=True)
	#
	# 	with multiprocessing.Pool(CPUS) as pool:
	# 		pool.map(fun_for_pool, list(range(CPUS)))


