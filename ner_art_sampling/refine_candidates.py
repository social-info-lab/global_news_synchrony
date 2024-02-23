import json
import glob
import re
import multiprocessing
import datetime
import os
import sys
import gzip
import collections
import pandas as pd
import numpy as np
import time

from argparse import ArgumentParser

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import auc, matthews_corrcoef, f1_score, precision_score, accuracy_score, recall_score

from utils import News, Biased_News, text2tokens, text2sentence, MIN_ARTICLE_LENGTH, MAX_SENTENCE_DUP_NUM, translation_ratio, MIN_SHARED_TF_IDF_SUM, MIN_NE_SIM, MAX_TXT_SIM, SHA, max_NE, MIN_NE_NUM
from utils import REFINE_RECORD_FILENAME, count_shared_sum, DATE_WINDOW, decay_factor, unify_url
from utils import compute_pairwise_tf_idf, compute_classic_tf_idf, cosine_similarity, compute_shared_ne, compute_bm25, compute_pairwise_bm25, list_to_counter_tuple_list


import cython_jaccard_sim

# TEXT_THRESHOLD=0.3 #Discard pairs with full text jaccard similarity *above* this value as potential duplicates
#NER_THRESHOLD=0.7 #Discard pairs with NER jaccard similarity *below* this value as too dissimilar

script_version = 3

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



def refine_candidates(globstring, input_type, bias_type, lang, cmp_lang, trans, min_ne_sim, min_ne_num, max_txt_sim, min_shared_tf_idf_sum, sha, stats, debug):
	print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} starting {globstring}")
	assert "candidates/" in globstring, "candidates/ not in file"
	new_fnm = globstring.replace("candidates/","refined/")
	out_classdir, out_langdir, output_filename = new_fnm.split("/")
	outdir = out_classdir + "/" + "refined-v" + str(script_version) + "_" + out_langdir + "/" + output_filename

	repeat = "repeat"
	df_name = 'refined_df'
	convert = 'very-vs-other'
	output_name = 'OVERALL(' + convert + ')'

	art_feature = ['art1_len', 'art2_len']
	repeat_feature = {'intra-lang':['shared_text_sum' + '('+repeat+')', 'shared_ne_sum' + '('+repeat+')', 'tf_idf_cosine_sim' + '('+repeat+')', 'text_jaccard_sim' + '('+repeat+')'], 'inter-lang':['shared_text_sum' + '('+repeat+')', 'shared_bm25_sum' + '('+repeat+')', 'bm25_cosine_sim' + '('+repeat+')', 'text_cosine_sim' + '('+repeat+')']}
	date_feature = {'intra-lang':['published_date_diff_decay6'], 'inter-lang':['published_date_diff_decay3']}
	feature_list = art_feature + repeat_feature[input_type] + date_feature[input_type] + [output_name]

	# loading idf dict
	if lang == cmp_lang:
		tf_json_suffix = "-v2-tf.json"
		bm25_json_suffix = "-v2-bm25.json"
		with open("indexes/" + lang + tf_json_suffix, 'r') as idf_json:
			tf_idf_dict = json.load(idf_json)
		with open("indexes/" + lang + bm25_json_suffix, 'r') as idf_json:
			bm25_idf_dict = json.load(idf_json)
	else:
		tf_json_suffix = "-wiki-v2-tf.json"
		bm25_json_suffix = "-wiki-v2-bm25.json"
		with open("indexes/" + lang + tf_json_suffix, 'r') as idf_json:
			tf_idf_dict1 = json.load(idf_json)
		with open("indexes/" + cmp_lang + tf_json_suffix, 'r') as idf_json:
			tf_idf_dict2 = json.load(idf_json)
		with open("indexes/" + lang + bm25_json_suffix, 'r') as idf_json:
			bm25_idf_dict1 = json.load(idf_json)
		with open("indexes/" + cmp_lang + bm25_json_suffix, 'r') as idf_json:
			bm25_idf_dict2 = json.load(idf_json)

		tf_idf_dict = dict(collections.Counter(tf_idf_dict1) + collections.Counter(tf_idf_dict2))
		bm25_idf_dict = dict(collections.Counter(bm25_idf_dict1) + collections.Counter(bm25_idf_dict2))

	for key in list(tf_idf_dict):
		tf_idf_dict[int(key)] = tf_idf_dict[key]
		tf_idf_dict.pop(key)
	for key in list(bm25_idf_dict):
		bm25_idf_dict[int(key)] = bm25_idf_dict[key]
		bm25_idf_dict.pop(key)

	'''building classifier'''
	feature_dir = "../annotation_analysis/feature/"
	origin_df = pd.read_csv(feature_dir + args.input_type + "_survey_items_feature.csv")
	# preprocessing data to feed model
	origin_df['art1_time'] = origin_df.apply(lambda x: str(x['art1_time'].replace(' days','')), axis = 1)
	origin_df['art2_time'] = origin_df.apply(lambda x: str(x['art2_time'].replace(' days','')), axis = 1)
	# describe an expotential decay factor to smooth the impact of time difference, use a set to test the best factor value
	for i in range(len(decay_factor)):
		origin_df['published_date_diff_decay' + str(i+1)] = origin_df.apply(lambda x: pow(decay_factor[i], int(x['published_date_diff'].replace(' days',''))), axis = 1)
	refined_df = origin_df[origin_df['art1_time'] != '2020-01-01']
	df_dict = {'origin_df': origin_df, 'refined_df':refined_df}

	x_train, x_test, y_train, y_test = train_test_split(df_dict[df_name][feature_list[:-1]], df_dict[df_name][feature_list[-1]], test_size=0.3)
	aver_art_len = df_dict[df_name]['aver_art_len'].mean()

	# normalization, y is class label so it doesn't need normalization
	std = StandardScaler()
	x_train = std.fit_transform(x_train)
	x_test = std.transform(x_test)

	# model_name = "SVC-linear"
	# print(model_name + '...')
	# model = SVC(kernel='linear', gamma='auto', class_weight='balanced')
	model_name = 'logistic_regression'
	print(model_name + '...')
	model = LogisticRegression(class_weight='balanced', C=1.0)  # default penalty is 'l2', C is the strength of regularization
	model.fit(x_train, y_train)
	y_predict = model.predict(x_test)
	# regression coefficient
	print(model.coef_)  # [[1.12796779  0.28741414  0.66944043 ...]]

	# compute metrics
	result = {}
	# result['confusion_matrix'] = confusion_matrix(y_test, y_predict)
	fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
	result['auc'] = auc(fpr, tpr)
	result['matthews_corrcoef'] = matthews_corrcoef(y_test, y_predict)
	result['f1_score'] = f1_score(y_test, y_predict)
	result['precision_score'] = precision_score(y_test, y_predict)
	result['accuracy_score'] = accuracy_score(y_test, y_predict)
	result['recall_score'] = recall_score(y_test, y_predict)
	print(result)

	cur_count = 0
	with open(outdir,"w") as outfile:
		prev_news=None
		best_match=None
		best_matches=list()
		best_score=0
		n_duplicates = 0
		n_notxt = 0
		n_classify = 0
		n_processedpairs = 0
		n_goodpairs = 0
		files = glob.glob(globstring)

		if trans:
			trans_ratio1 = translation_ratio(lang)
			trans_ratio2 = translation_ratio(cmp_lang)
		else:
			trans_ratio1 = 1
			trans_ratio2 = 1

		if debug:
			print("Will process files:", files)
		if not os.path.exists("dups/"):
			os.makedirs("dups/")
		dup_filename = "dups/" + "refined-v" + str(script_version) + "_" + out_langdir + ".dups"
		with open(dup_filename, "w") as dups:
			for file in files:
				if debug:
					print("Processing", file)
				with open(file,"r") as fh:
					for line in fh:
						cur_count += 1
						print(cur_count)

						# truncate for immediately getting some samples
						# if cur_count < 3000:
						# 	continue
						# if cur_count > 12000:
						# 	break

						if n_processedpairs % 1000 == 0:
							print(f"Already processed {n_processedpairs} pairs.", flush=True)

						canidate=json.loads(line)
						if not bias_type:
							a1 = News(*canidate["article1"])
							a2 = News(*canidate["article2"])
						else:
							a1 = Biased_News(*canidate["article1"])
							a2 = Biased_News(*canidate["article2"])
						this_news=a1
						# switch to refine the next pair
						if this_news!=prev_news and prev_news!=None:
							# save the best_match for the just refined pair and intialize for the next pair
							if best_match!=None:
								best_matches.append(best_match)
								n_goodpairs += 1
							best_match=None
							best_score=0
						prev_news=this_news

						if "jaccard" in canidate:
							ner_jaccard = canidate["jaccard"]
							# print("ner_jaccard:", ner_jaccard)
							if ner_jaccard<min_ne_sim:
								continue
						if "intersect_ne_count" in canidate:
							ner_intersect = canidate["intersect_ne_count"]
							# print("ner_intersect:", ner_intersect)
							if ner_intersect < min_ne_num:
								continue

						if ner_jaccard>best_score:
							if debug:
								print("pair", n_processedpairs, end=" ")
							n_processedpairs += 1

							pair_tf_idf_intersect_set, pair_tf_idf_dict = compute_pairwise_tf_idf(a1.vec, a2.vec, tf_idf_dict, repeat)
							shared_tf_idf_sum = count_shared_sum(pair_tf_idf_intersect_set, pair_tf_idf_dict)
							print("shared_tf_idf_sum: ", shared_tf_idf_sum)
							if shared_tf_idf_sum < min_shared_tf_idf_sum:
								continue

							art1=load_article(a1.file,a1.lineno)
							art2=load_article(a2.file,a2.lineno)

							art1['url'] = art1['url'].strip()
							art2['url'] = art2['url'].strip()

							# load text for inter-lang pairs
							if 'wikilinked' in a1.file:
								art1["story_text"] = load_article(a1.file.replace("home/scott/wikilinked", "home/scott/ner").replace(".gz",""), a1.lineno)["story_text"]
							if 'wikilinked' in a2.file:
								art2["story_text"] = load_article(a2.file.replace("home/scott/wikilinked", "home/scott/ner").replace(".gz",""), a2.lineno)["story_text"]

							text1 = art1["story_text"].strip()
							text2 = art2["story_text"].strip()
							words1 = text2tokens(text1, lang)
							words2 = text2tokens(text2, cmp_lang)
							art1_len = len(words1)
							art2_len = len(words2)

							# '''for test'''
							# ne_jaccard_sim_struct = cython_jaccard_sim.cython_jaccard_similarity2(tuple(a1.vec), tuple(a2.vec))
							# ne_jaccard, ne_union = ne_jaccard_sim_struct["similarity"], ne_jaccard_sim_struct["size_union"]
							# print(a1.vec)
							# print(a2.vec)
							# print("ne_jaccard: ",ne_jaccard, "ne_union: ", ne_union)


							# index filtering takes care of this
							# at this stage we don't see any articles meeting this criterion
							if text1 == None or len(words1) < MIN_ARTICLE_LENGTH * trans_ratio1 or text2 == None or len(words2) < MIN_ARTICLE_LENGTH * trans_ratio2:
								n_notxt += 1
								continue

							# filtering duplicates
							if input_type == 'intra-lang':
								sentence1_hashes = text2sentence(text1, lang)
								sentence2_hashes = text2sentence(text2, cmp_lang)
								len_sentence_intersection = len(sentence1_hashes & sentence2_hashes)
								if len_sentence_intersection > MAX_SENTENCE_DUP_NUM:
									# text similarity threshold to reduce the duplicates in samples (only for intra-lang)
									text_jaccard_sim_struct_non_repeat = cython_jaccard_sim.cython_jaccard_similarity3(words1, words2)
									text_jaccard_non_repeat, text_union_non_repeat = text_jaccard_sim_struct_non_repeat["similarity"], text_jaccard_sim_struct_non_repeat["size_union"]
									print("text_jaccard_non_repeat: ", text_jaccard_non_repeat)
									if text_jaccard_non_repeat > max_txt_sim:
										# Potential duplicate
										n_duplicates += 1
										art1_url = art1["url"]
										art2_url = art2["url"]
										art1_title = art1["title"]
										art2_title = art2["title"]

										sentence1_len = len(sentence1_hashes)
										sentence2_len = len(sentence2_hashes)
										dups.write(f"\t{text_jaccard_non_repeat}\t{len_sentence_intersection}\t{sentence1_len}\t{sentence2_len}\t{art1_len}\t{art2_len}\t{art1_title}\t{art2_title}\t{art1_url}\t{art2_url}\n")
										continue

							# filtering by the classifier
							# if input_type == 'intra-lang':
							# 	text_jaccard_sim_struct = cython_jaccard_sim.cython_jaccard_similarity2(tuple(collections.Counter(words1).most_common()), tuple(collections.Counter(words2).most_common()))
							# 	text_jaccard, text_union = text_jaccard_sim_struct["similarity"], text_jaccard_sim_struct["size_union"]
							# 	shared_text_num = text_jaccard * text_union
							#
							# 	ne_jaccard_sim_struct = cython_jaccard_sim.cython_jaccard_similarity2(tuple(a1.vec), tuple(a2.vec))
							# 	ne_jaccard, ne_union = ne_jaccard_sim_struct["similarity"], ne_jaccard_sim_struct["size_union"]
							# 	shared_ne_num = ne_jaccard * ne_union
							#
							# 	art1_bm25_list = compute_bm25(a1.vec, bm25_idf_dict, art1_len, aver_art_len)
							# 	art2_bm25_list = compute_bm25(a2.vec, bm25_idf_dict, art2_len, aver_art_len)
							# 	ne_intersect_set = compute_shared_ne(a1.vec, a2.vec, repeat)[0]
							# 	bm25_cosine_sim = cosine_similarity(art1_bm25_list, art2_bm25_list, ne_intersect_set)
							#
							# 	# art1_tf_idf_list = compute_classic_tf_idf(a1.vec, idf_dict)
							# 	# art2_tf_idf_list = compute_classic_tf_idf(a2.vec, idf_dict)
							# 	# ne_intersect_set = compute_shared_ne(a1.vec, a2.vec, repeat)[0]
							# 	# tf_idf_cosine_sim = cosine_similarity(art1_tf_idf_list, art2_tf_idf_list, ne_intersect_set)
							#
							# 	cur_decay_factor = decay_factor[6]
							# 	date_decay = pow(cur_decay_factor, abs(a1.reladate - a2.reladate))
							#
							# 	print("art1_len: ", art1_len, " art2_len: ", art2_len, " shared_text_num: ", shared_text_num, " shared_ne_num: ", shared_ne_num, " bm25_cosine_sim: ", bm25_cosine_sim, " text_jaccard: ", text_jaccard, " date_decay: ", date_decay)
							# 	this_test = np.array([art1_len, art2_len, shared_text_num, shared_ne_num, bm25_cosine_sim, text_jaccard, date_decay]).reshape(1,-1)
							# elif input_type == 'inter-lang':
							# 	text_jaccard_sim_struct = cython_jaccard_sim.cython_jaccard_similarity2(tuple(collections.Counter(words1).most_common()), tuple(collections.Counter(words2).most_common()))
							# 	text_jaccard, text_union = text_jaccard_sim_struct["similarity"], text_jaccard_sim_struct["size_union"]
							# 	shared_text_num = text_jaccard * text_union
							#
							# 	pair_bm25_intersect_set, pair_bm25_dict = compute_pairwise_bm25(a1.vec, a2.vec, bm25_idf_dict, art1_len, art2_len, aver_art_len, repeat)
							# 	shared_bm25_sum = count_shared_sum(pair_bm25_intersect_set, pair_bm25_dict)
							#
							# 	art1_bm25_list = compute_bm25(a1.vec, bm25_idf_dict, art1_len, aver_art_len)
							# 	art2_bm25_list = compute_bm25(a2.vec, bm25_idf_dict, art2_len, aver_art_len)
							# 	ne_intersect_set = compute_shared_ne(a1.vec, a2.vec, repeat)[0]
							# 	bm25_cosine_sim = cosine_similarity(art1_bm25_list, art2_bm25_list, ne_intersect_set)
							#
							# 	words1_tuple_list = list_to_counter_tuple_list(words1, repeat)
							# 	words2_tuple_list = list_to_counter_tuple_list(words2, repeat)
							# 	text_intersect_set = compute_shared_ne(words1_tuple_list, words2_tuple_list, repeat)[0]
							# 	text_cosine_sim = cosine_similarity(words1_tuple_list, words2_tuple_list, text_intersect_set)
							#
							# 	cur_decay_factor = decay_factor[3]
							# 	date_decay = pow(cur_decay_factor, abs(a1.reladate - a2.reladate))
							# 	print("art1_len: ", art1_len, " art2_len: ", art2_len, " shared_text_num: ",
							# 		  shared_text_num, " shared_bm25_sum: ", shared_bm25_sum, " bm25_cosine_sim: ",
							# 		  bm25_cosine_sim, " text_cosine_sim: ", text_cosine_sim, " date_decay: ", date_decay)
							# 	this_test = np.array(
							# 		[art1_len, art2_len, shared_text_num, shared_bm25_sum, bm25_cosine_sim, text_cosine_sim,
							# 		 date_decay]).reshape(1, -1)
							#
							# this_predict = model.predict(this_test)
							# if this_predict != 1:
							# 	n_classify += 1
							# 	continue

							best_score=ner_jaccard
							best_match={
								"similarity":ner_jaccard,
								"pair_id":f'{art1["stories_id"]}_{art2["stories_id"]}',
								"method":f'Indexes v4 ABYZ/MBFC dataset filtered namedtuples with contry info and bias info; Candidates v8, MBFC prioritized with left-right bias comparison for english articles and ABYZ prioritized for other intra-lang and inter-lang pairs, temporal window {DATE_WINDOW} days, min NE Jaccard {min_ne_sim}, min NE sim {min_ne_sim}, min NE num {min_ne_num}; Refining v2.2 txt Jaccard, max TXT sim {max_txt_sim}, min shared tf-idf sum {min_shared_tf_idf_sum}, translation ratio applied as option, language and compared language',
								"parameter":{"globstring": globstring, "input_type": input_type, "lang": lang, "cmp_lang": cmp_lang, "trans": trans, "temporal_window":DATE_WINDOW, "min_ne_sim": min_ne_sim, "min_ne_num": min_ne_num, "max_txt_sim": max_txt_sim, "min_shared_tf_idf_sum": min_shared_tf_idf_sum, "git_commit":sha, "trans":stats,} ,
								"url1":art1["url"],
								"url2":art2["url"],
								"article1":art1,
								"article2":art2,
							}
				# save the last refined pair of the file in the cache to output.
				if best_match!=None:
					best_matches.append(best_match)
					n_goodpairs += 1
				with open(REFINE_RECORD_FILENAME, "a+") as refinefile:
					refinefile.write(f"{file}\n")
			# print the most frequent urls in the samples
			if stats:
				l1 = [e["url1"] for e in best_matches]
				l2 = [e["url2"] for e in best_matches]
				common = collections.Counter(l1+l2).most_common()
				print("Most common URLs:")
				for e in common:
					if e[1]>1: print(e)
			for best_match in best_matches:
				outfile.write(json.dumps(best_match))
				outfile.write("\n")
				outfile.flush()
	print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {globstring} PROCESSSED: {n_processedpairs} checked pairs, {n_duplicates} duplicates, {n_notxt} without text, {n_classify} filtered by classifier, {n_goodpairs} good pairs")




if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument("-t", "--input-type", dest="input_type",
						default="intra-lang", type=str,
						help="the input type that we use for generate samples. It can be choose from 2 types: intra-lang or inter-lang(only support a specific language compared to English now).")
	parser.add_argument("-i", "--input-files", dest="input",
		default="candidates/en-full/*.jsonl", type=str,
		help="Path to input files.")
	parser.add_argument("-l", "--lang", dest="lang",
		default="en", type=str,
		help="Language (impacts tokenization).")
	parser.add_argument("-cl", "--lang2", dest="cmp_lang",
		default="en", type=str,
		help="compared Language (impacts tokenization).")
	parser.add_argument("-s", "--stats", dest="stats",
		default=False, action='store_true',
		help="Get the articles that appear multiple times in pairs and other stats")
	# parser.add_argument("-m", "--min-similarity", dest="min_ne_sim",
	# 	default=0.2, type=float,
	# 	help="Min NE similarity")
	# parser.add_argument("-x", "--max-similarity", dest="max_txt_sim",
	# 	default=0.3, type=float,
	# 	help="Max TXT similarity")
	parser.add_argument("-c", "--cpu-count", dest="cpu_count",
		default=1, type=int,
		help="Number of CPUs. Ignored if --stats is enabled.")
	parser.add_argument("-b", "--bias", dest="bias",
		default=False, action='store_true',
		help="Switch to bias dataset matching based index mode.")
	parser.add_argument("-trans", "--translation", dest="trans",
		default=True, type=str,
		help="Enable translation ratio.")
	parser.add_argument("-d", "--debug", dest="debug",
		default=False, action='store_true',
		help="Debug mode running for only a small chunk of data.")
	args = parser.parse_args()

	files=list(glob.glob(args.input))
	new_fnm = files[0].replace("candidates/","refined/")
	out_classdir, out_langdir, _ = new_fnm.split("/")
	outdir = out_classdir + "/" + "refined-v" + str(script_version) + "_" + out_langdir

	if not os.path.exists(outdir):
		os.makedirs(outdir)
	# pass the files that have been refined
	if not os.path.exists(REFINE_RECORD_FILENAME):
		refinefile = open(REFINE_RECORD_FILENAME, 'w')
		refinefile.close()
	with open(REFINE_RECORD_FILENAME, "r") as refinefile:
		line = refinefile.readline()
		# print(f"line: {line}")
		while line:
			genfile = line.replace("\n", "")
			print(f"genfile: {genfile}")
			if genfile in files:
				print("yes, this file have been refined...")
				files.remove(genfile)
			line = refinefile.readline()

	if args.cpu_count==0:
		CPUS=multiprocessing.cpu_count()
	else:
		CPUS=args.cpu_count

	def fun_for_pool(file):
		refine_candidates(file, args.input_type, args.bias, args.lang, args.cmp_lang, args.trans, MIN_NE_SIM[args.input_type], MIN_NE_NUM, MAX_TXT_SIM, MIN_SHARED_TF_IDF_SUM,
			SHA, args.stats, args.debug)

	# cpus=min(multiprocessing.cpu_count()//2,len(files))
	cpus=min(CPUS,len(files))
	print("Cpus, files:", cpus, files)
	if cpus <= 0:
		print("There is no file expected to be refined...")
		sys.exit(0)
	with multiprocessing.Pool(cpus) as pool:
		pool.map(fun_for_pool,files)

