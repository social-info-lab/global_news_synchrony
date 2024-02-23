import json
import sys
import re
import glob
import datetime
import collections
import os
import gzip
# from cachetools import LRUCache
from argparse import ArgumentParser
import numpy as np
import ast


''' compute the inversed document frequency of words in the articles with repeat counting, and save them as a dictionary into json file'''
''' the version number here should be consistent to the version number in create_index.py '''
script_version = 2

# based on repeat computation
def create_classic_idf_dict(name):
	name += "-v"+str(script_version)
	print("Save idf dictionary to file ", name+"-tf.json", flush=True)

	ne_count_dict = collections.defaultdict(int)
	idf_dict = collections.defaultdict(float)

	with open(name+".index","r") as file_in:
		for line in file_in:
			vec = line.strip().split("\t")[-1]
			vec = ast.literal_eval(vec)
			for ne in vec:
				ne_name = ne[0]
				ne_count = ne[1]
				ne_count_dict[ne_name] += ne_count

	doc_sum = sum(ne_count_dict.values())

	with open(name+".index","r") as file_in:
		for line in file_in:
			vec = line.strip().split("\t")[-1]
			vec = ast.literal_eval(vec)
			for ne in vec:
				ne_name = ne[0]
				ne_count = ne[1]
				# skip if the word has removed or the idf of this word has been computed earlier
				if ne_name in ne_count_dict and ne_name not in idf_dict:
					idf_dict[ne_name] = np.log10(doc_sum / (ne_count_dict[ne_name] + 1))
	with open(name+"-tf.json","w") as file_out:
		json.dump(idf_dict, file_out)

# based on repeat computation
def create_bm25_idf_dict(name):
	name += "-v"+str(script_version)
	print("Save BM25 idf dictionary to file ", name+"-bm25.json", flush=True)

	ne_count_dict = collections.defaultdict(int)
	idf_dict = collections.defaultdict(float)

	with open(name+".index","r") as file_in:
		for line in file_in:
			vec = line.strip().split("\t")[-1]
			vec = ast.literal_eval(vec)
			for ne in vec:
				ne_name = ne[0]
				ne_count = ne[1]
				ne_count_dict[ne_name] += ne_count

	doc_sum = sum(ne_count_dict.values())


	with open(name+".index","r") as file_in:
		for line in file_in:
			vec = line.strip().split("\t")[-1]
			vec = ast.literal_eval(vec)
			for ne in vec:
				ne_name = ne[0]
				ne_count = ne[1]

				# skip if the word has removed or the idf of this word has been computed earlier
				if ne_name in ne_count_dict and ne_name not in idf_dict:
					idf_dict[ne_name] = np.log10((doc_sum - ne_count_dict[ne_name] + 0.5) / (ne_count_dict[ne_name] + 0.5))
	with open(name+"-bm25.json","w") as file_out:
		json.dump(idf_dict, file_out)

	return idf_dict


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-o", "--output-name", dest="output_name", default="indexes/en", type=str, help="Input index name and Output dict name.")
	args = parser.parse_args()

	create_classic_idf_dict(args.output_name)
	# create_bm25_idf_dict(args.output_name)