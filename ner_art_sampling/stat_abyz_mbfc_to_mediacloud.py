from utils import lang_list
import collections
import datetime
import pandas as pd
import os

def stat_biased_data():
	abyz_counter = collections.Counter()
	mbfc_counter = collections.Counter()
	for lang in lang_list:
		index_filename = f"indexes/{lang}-bias-v3.index"
		print(datetime.datetime.now(), f"Stating data from {index_filename}")
		with open(index_filename, "r") as fh:
			n_lines = 0
			print("Loaded...", end=" ")
			for line in fh:
				outlet_flag, abyz_match_url, mbfc_match_url, story_id, file, lineno, reladate, url, vec = line.strip().split("\t")
				# we've heavily oversampled this day, so now we can undersample it
				# please move this step to create_index once index versioning and tracking is implemented (we need similar tracking for the remaining two scripts as well)
				# it's easy to forget it, especially on a local machine with alternative setup and small data samples
				# all article-specific filtering should happen at the index level anyway

				# tuples have lower footprint than lists, lists have lower than sets
				# https://stackoverflow.com/questions/46664007/why-do-tuples-take-less-space-in-memory-than-lists/46664277
				# https://stackoverflow.com/questions/39914266/memory-consumption-of-a-list-and-set-in-python

				# # we don't consider the repeat times of words in the same document here
				# vec = tuple(k for k, v in vec)
				abyz_counter[abyz_match_url] += 1
				mbfc_counter[mbfc_match_url] += 1
				n_lines += 1
				if n_lines%500000 == 0:
					print(n_lines,"lines", end=" ", flush=True)
					# gc.collect()
		# we want to get different pairs every time, some pairs may randomly repeat,
		# but chances of this are low and we can deal with them later
		# before sending them to annotators

		# gc.collect()


	del abyz_counter['None']
	del mbfc_counter['None']
	total_counter = abyz_counter + mbfc_counter

	# print("Top outlets for the ABYZ/MBFC are:")
	# for outlet in total_counter.most_common(50):
	# 	print(outlet)
	# print()
	# print("Top outlets for the ABYZ are:")
	# for outlet in abyz_counter.most_common(50):
	# 	print(outlet)
	# print()
	# print("Top outlets for the MBFC are:")
	# for outlet in mbfc_counter.most_common(50):
	# 	print(outlet)

	attributes_name = ['media_url','counts']
	most_common_total_counter = total_counter.most_common(50)
	most_common_abyz_counter = abyz_counter.most_common(50)
	most_common_mbfc_counter = mbfc_counter.most_common(50)

	most_common_total_counter_df = pd.DataFrame(data=most_common_total_counter, columns=attributes_name)
	most_common_abyz_counter_df = pd.DataFrame(data=most_common_abyz_counter, columns=attributes_name)
	most_common_mbfc_counter_df = pd.DataFrame(data=most_common_mbfc_counter, columns=attributes_name)

	output_folder = 'bias_stat/'
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	most_common_total_counter_df.to_csv(output_folder + 'most_common_outlets_for_ABYZ_and_MBFC.csv')
	most_common_abyz_counter_df.to_csv(output_folder + 'most_common_outlets_for_ABYZ.csv')
	most_common_mbfc_counter_df.to_csv(output_folder + 'most_common_outlets_for_MBFC.csv')

	return 0

stat_biased_data()