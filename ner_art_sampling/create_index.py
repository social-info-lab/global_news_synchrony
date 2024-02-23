import json
import sys
import re
import glob
import datetime
import collections
import os
import gzip
import pandas as pd
# from cachetools import LRUCache
from argparse import ArgumentParser

# version 5 added story_id and title
script_version = 5

letters="abcdefghijklmnopqrstuvwxyz"
assert len(letters)==26
idx=["".join([a,b]) for a in letters for b in letters]

from utils import blockwords,text2tokens, date_diff, MIN_ARTICLE_LENGTH, SHA, START_DATE, DATE_WINDOW,unify_url


def create_index(name, globstring, lang, debug):
	#Keep a list of the 1m most recent urls seen. LRU - least recently used - urls will be removed after hitting this limit
	# recent_url_cache=LRUCache(1000000)

	all_urls = set()
	all_titles = set()
	ne_index = dict()
	total_ent_count = 0

	# let's use "en_short.index" for the two-character NE representation
	name += "-v"+str(script_version)
	print("Will print to file", name+".index", flush=True)

	# with open(name+".index","w") as out, open(name+"_memes.index","w") as article_meme_file, open(name+".dups","w") as dups:
	with open(name+".index","w") as out:
		# we want in the reverse order, to keep the most recent article
		# we could reverse this ordering
		sorted_files = sorted(glob.glob(globstring), reverse=True)
		if debug:
			print(globstring, sorted_files, flush=True)
		for file in sorted_files:
			print(file,datetime.datetime.now(), flush=True)
			if lang=="auto":
				lang=os.path.dirname(file)[-2:]
				if lang not in [ "ar", "de", "es", "en", "fr", "it", "ko", "pl", "pt", "ru", "tr", "zh" ]:
					print("Last two letters of the directory don't specify a known language code. Please use the -l command line argument to specify the language. Exiting.", flush=True)
					sys.exit()
				if debug: print(lang, flush=True)
			if ".gz" not in file:
				with open(file,"r") as fh:
					for lineno,line in enumerate(fh):
						if debug and lineno>100:
							break

						art=json.loads(line)

						if not "stories_id" in art:
							continue
						if not "url" in art:
							continue

						stories_id = art["stories_id"]
						url=art["url"].strip()

						if any([w in art["url"] for w in blockwords]):
							continue

						# skip URLs that we have seen already
						if url not in all_urls:
							all_urls.add(url)
						else:
							continue

						# if recent_url_cache.get(url)!=None:
						# 	continue
						# recent_url_cache[url]=True

						if not "story_text" in art:
							continue

						words=text2tokens(art["story_text"], lang)
						if words==None or len(words)<MIN_ARTICLE_LENGTH:
							continue

						# each indexed article needs a title and we don't admit duplicates
						# if the same titles, then log them for inspection
						if "title" in art:
							title = art["title"].strip()
							# title_memeslist[title].append(memes)
							if title not in all_titles:
								all_titles.add(title)
								# title_url[title] = url
							else:
								continue
								# we looked at the resulting dup files, they contained:
								# i) many exact duplicates
								# ii) many articles not relevant socially/politically
						else:
							title = "null"
							# print("Article without title", url)
							continue

						# index of full NEs
						vec=[] #[0]*len(idx) #len(idx)==26**2
						ent_count=0

						if "spacy" in art and art["spacy"] != None:
							for ent in art["spacy"]:
								text=ent["text"]
								if len(text)<2 and lang not in ["zh", "ko"]:
									continue
								if text in ne_index:
									vec.append(ne_index[text])
								else:
									ne_index[text] = total_ent_count
									vec.append(ne_index[text])
									total_ent_count += 1
								ent_count+=1
							if ent_count>0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)

								out.write(f"{stories_id}\t{file}\t{lineno}\t{reladate}\t{url}\t{vec}\n")
						if "polyglot" in art and art["polyglot"]!=None:
							for ent in art["polyglot"]:
								# the implemention used by xi chen
								for word in ent["text"]:
									if len(word)<2 and lang not in ["zh", "ko"]:
										continue
									if word in ne_index:
										vec.append(ne_index[word])
									else:
										ne_index[word] = total_ent_count
										vec.append(ne_index[word])
										total_ent_count += 1
									ent_count+=1
								# implemention that joins tokenized named entities
								# which seems to correspond to what spacy does
								# however, xi chen's experience is that
								# we get better results without joining
								# text=" ".join(ent["text"])
								# if len(text)<2 and lang not in ["zh", "ko"]:
								# 	continue
								# if text in ne_index:
								# 	vec.append(ne_index[text])
								# else:
								# 	ne_index[text] = total_ent_count
								# 	total_ent_count += 1
								# ent_count+=1
							if ent_count>0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)
								out.write(f"{stories_id}\t{title}\t{file}\t{lineno}\t{reladate}\t{url}\t{vec}\n")
			else:
				with gzip.open(file,"rt") as fh:
					for lineno, line in enumerate(fh):
						if debug and lineno > 100:
							break

						art = json.loads(line)

						if not "stories_id" in art:
							continue
						if not "url" in art:
							continue

						stories_id = art["stories_id"]
						url = art["url"]

						if any([w in art["url"] for w in blockwords]):
							continue

						#there is no "story_text" attribute in wiki data now

						# if not "story_text" in art:
						# 	continue
						#
						# words = text2tokens(art["story_text"], lang)
						# if words == None or len(words) < MIN_ARTICLE_LENGTH:
						# 	continue

						# skip URLs that we have seen already
						if url not in all_urls:
							all_urls.add(url)
						else:
							continue

						if "title" in art:
							title = art["title"].strip()
							# title_memeslist[title].append(memes)
							if title not in all_titles:
								all_titles.add(title)
								# title_url[title] = url
							else:
								continue
								# we looked at the resulting dup files, they contained:
								# i) many exact duplicates
								# ii) many articles not relevant socially/politically
						else:
							title = "null"
							# print("Article without title", url)
							continue

						# index of full NEs
						vec = []  # [0]*len(idx) #len(idx)==26**2
						ent_count = 0

						if "wiki_concepts" in art and art["wiki_concepts"] != None:
							for ent in art["wiki_concepts"]:
								if len(ent['term']) < 2 and lang not in ["zh", "ko"]:
									continue
								if ent['term'] in ne_index:
									vec.append(int(ent['term_id']))
								else:
									ne_index[ent['term']] = ent['term_id']
									vec.append(int(ent['term_id']))
								ent_count += 1

							if ent_count > 0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)
								out.write(f"{stories_id}\t{title}\t{file}\t{lineno}\t{reladate}\t{url}\t{vec}\n")

	with open(name+"_ne.index", "w") as out:
		for k,v in ne_index.items():
			key = k.replace('\t', ' ')
			out.write(f"{key}\t{v}\n")

	with open(name+".info","w") as infofile:
		infodict = {
			"sha":SHA,
			"version":script_version,
			"description":"We moved to sorted by key NE counter/list.",
			"input_description":"Initial NE. Large model only for IT, small models for other languages.",
			"unique_named_entities":len(ne_index),
		}
		infofile.write(json.dumps(infodict, indent=4))

	print("created index successfully...", flush=True)



# a modified indexing with MBFC/ABYZ dataset
# in MBFC list,
# we have media name, media url, and bias trend
# some urls are invalid, each media has a bias , include "center", "left", "left_center", "right_center", "right", "fake_news", "pro_science", "conspiracy"(all without valid urls)
# in ABYZ list,
# we have greater region, sub-region(differ according to the region class of the media), region class(choose from "national", "foreign", "regional", "state", "local"),
# media name, media type("Broadcast", "Internet", "Newspaper", "Press Agency", "Magazine", etc), media focus("General Interest", "Business", "Sport", etc)
# language and media url (for national media they all have valid urls)

# in this index, each row would have dataset flag(ABYZ/MBFC), media url, stories_id, file name, lineno, reladate, url, vec
''' the biased index file name is like 'en-bias-v3.index' or 'en-wiki-bias-v3.index'  '''
def create_baised_index(name, globstring, lang, debug):
	abyz_data = pd.read_csv("abyz_outlets.csv")
	# mbfc_data = pd.read_csv("mbfc_sources.csv")
	mbfc_data = pd.read_csv("integrated_bias.csv")

	abyz_data['url'] = abyz_data.apply(lambda x: unify_url(x['url']), axis=1)
	mbfc_data['link'] = mbfc_data.apply(lambda x: unify_url(x['link']), axis=1)
	# old version of mbfc without country info
	# mbfc_data = pd.read_csv("mbfc_outlets.csv", header=None, names=['media_name','url','bias'])

	# we only consider national outlets for abyz here, 6000/26589 outlets
	new_abyz_data = abyz_data.loc[abyz_data["local, national or foreign"] == "national"]
	abyz_data = new_abyz_data
	# unify the format of country info
	mbfc_data = mbfc_data.dropna(subset = ["country"])
	mbfc_data["unified_country"] = mbfc_data.apply(lambda x: x['country'].split("(")[0].lower(), axis=1)




	abyz_outlet_list = abyz_data['url'].to_list()
	abyz_coutry_list = abyz_data['sub-region'].to_list()  # for national outlets the 'greater region' and 'sub-region' should be the same in the csv files, so doesn't matter which one we use
	mbfc_outlet_list = mbfc_data['link'].to_list()
	mbfc_bias_list = mbfc_data['bias'].to_list()
	mbfc_country_list = mbfc_data['unified_country'].to_list()


	#Keep a list of the 1m most recent urls seen. LRU - least recently used - urls will be removed after hitting this limit
	# recent_url_cache=LRUCache(1000000)

	all_urls = set()
	all_titles = set()
	ne_index = dict()
	total_ent_count = 0

	total_art_num = 0
	abyz_matches = 0
	mbfc_matches = 0

	# let's use "en_short.index" for the two-character NE representation
	name += "-bias-v"+str(script_version)
	print("Will print to file", name+".index", flush=True)

	# with open(name+".index","w") as out, open(name+"_memes.index","w") as article_meme_file, open(name+".dups","w") as dups:
	with open(name+".index","w") as out:
		# we want in the reverse order, to keep the most recent article
		# we could reverse this ordering
		sorted_files = sorted(glob.glob(globstring), reverse=True)
		print("sorted files: ", sorted_files)
		if debug:
			print(globstring, sorted_files, flush=True)
		for file in sorted_files:
			print(file,datetime.datetime.now(), flush=True)
			if lang=="auto":
				lang=os.path.dirname(file)[-2:]
				if lang not in [ "ar", "de", "es", "en", "fr", "it", "ko", "pl", "pt", "ru", "tr", "zh" ]:
					print("Last two letters of the directory don't specify a known language code. Please use the -l command line argument to specify the language. Exiting.", flush=True)
					sys.exit()
				if debug: print(lang, flush=True)
			if ".gz" not in file:
				with open(file,"r") as fh:
					for lineno,line in enumerate(fh):
						total_art_num += 1
						print(f"processed {total_art_num} articles, get {abyz_matches} ABYZ matches, get {mbfc_matches} MBFC matches.")

						#for debug
						if debug and lineno>100:
							break

						art=json.loads(line)

						if not "stories_id" in art:
							continue
						if not "url" in art:
							continue


						stories_id = art["stories_id"]
						url=art["url"].strip()
						media_url = unify_url(art["media_url"])

						outlet_flag = ''  # '' stands for not matching, otherwise stands for matching
						match_country = []
						abyz_match_url = 'None'
						mbfc_match_url = 'None'
						mbfc_match_bias = 'None'
						if any([w in art["url"] for w in blockwords]):
							continue
						for i in range(len(abyz_outlet_list)):
							if media_url in abyz_outlet_list[i] or abyz_outlet_list[i] in media_url:
								outlet_flag += 'ABYZ-' # '-' is to recognize the case an article in both ABYZ and MBFC
								abyz_match_url = abyz_outlet_list[i]
								match_country.append(abyz_coutry_list[i])
								abyz_matches += 1
								break
						for i in range(len(mbfc_outlet_list)):
							if media_url in mbfc_outlet_list[i] or mbfc_outlet_list[i] in media_url:
								outlet_flag += 'MBFC-' # '-' is to recognize the case an article in both ABYZ and MBFC
								mbfc_match_url = mbfc_outlet_list[i]
								mbfc_match_bias = mbfc_bias_list[i]
								match_country.append(mbfc_country_list[i])
								mbfc_matches += 1
								break
						if outlet_flag == '':
							continue

						# skip URLs that we have seen already
						if url not in all_urls:
							all_urls.add(url)
						else:
							continue

						# if recent_url_cache.get(url)!=None:
						# 	continue
						# recent_url_cache[url]=True

						if not "story_text" in art:
							continue

						words=text2tokens(art["story_text"], lang)
						if words==None or len(words)<MIN_ARTICLE_LENGTH:
							continue

						# each indexed article needs a title and we don't admit duplicates
						# if the same titles, then log them for inspection
						if "title" in art:
							title = art["title"].strip()
							# title_memeslist[title].append(memes)
							if title not in all_titles:
								all_titles.add(title)
								# title_url[title] = url
							else:
								continue
								# we looked at the resulting dup files, they contained:
								# i) many exact duplicates
								# ii) many articles not relevant socially/politically
						else:
							title = "null"
							# print("Article without title", url)
							continue

						# index of full NEs
						vec=[] #[0]*len(idx) #len(idx)==26**2
						ent_count=0

						if "spacy" in art and art["spacy"] != None:
							for ent in art["spacy"]:
								text=ent["text"]
								if len(text)<2 and lang not in ["zh", "ko"]:
									continue
								if text in ne_index:
									vec.append(ne_index[text])
								else:
									ne_index[text] = total_ent_count
									vec.append(ne_index[text])
									total_ent_count += 1
								ent_count+=1
							if ent_count>0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)

								out.write(f"{outlet_flag}\t{match_country}\t{mbfc_match_bias}\t{abyz_match_url}\t{mbfc_match_url}\t{stories_id}\t{file}\t{lineno}\t{reladate}\t{url}\t{vec}\n")
						if "polyglot" in art and art["polyglot"]!=None:
							for ent in art["polyglot"]:
								# the implemention used by xi chen
								for word in ent["text"]:
									if len(word)<2 and lang not in ["zh", "ko"]:
										continue
									if word in ne_index:
										vec.append(ne_index[word])
									else:
										ne_index[word] = total_ent_count
										vec.append(ne_index[word])
										total_ent_count += 1
									ent_count+=1
								# implemention that joins tokenized named entities
								# which seems to correspond to what spacy does
								# however, xi chen's experience is that
								# we get better results without joining
								# text=" ".join(ent["text"])
								# if len(text)<2 and lang not in ["zh", "ko"]:
								# 	continue
								# if text in ne_index:
								# 	vec.append(ne_index[text])
								# else:
								# 	ne_index[text] = total_ent_count
								# 	total_ent_count += 1
								# ent_count+=1
							if ent_count>0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)
								out.write(f"{outlet_flag}\t{match_country}\t{mbfc_match_bias}\t{abyz_match_url}\t{mbfc_match_url}\t{stories_id}\t{title}\t{file}\t{lineno}\t{reladate}\t{url}\t{vec}\n")
			else:
				with gzip.open(file,"rt") as fh:
					for lineno, line in enumerate(fh):
						if debug and lineno > 100:
							break

						art = json.loads(line)

						if not "stories_id" in art:
							continue
						if not "url" in art:
							continue

						stories_id = art["stories_id"]
						url = art["url"].strip()
						media_url = unify_url(art["media_url"])

						outlet_flag = ''  # '' stands for not matching, otherwise stands for matching
						match_country = []
						abyz_match_url = 'None'
						mbfc_match_url = 'None'
						mbfc_match_bias = 'None'
						if any([w in art["url"] for w in blockwords]):
							continue
						for i in range(len(abyz_outlet_list)):
							if media_url in abyz_outlet_list[i] or abyz_outlet_list[i] in media_url:
								outlet_flag += 'ABYZ-'  # '-' is to recognize the case an article in both ABYZ and MBFC
								abyz_match_url = abyz_outlet_list[i]
								match_country.append(abyz_coutry_list[i])
								abyz_matches += 1
								break
						for i in range(len(mbfc_outlet_list)):
							if media_url in mbfc_outlet_list[i] or mbfc_outlet_list[i] in media_url:
								outlet_flag += 'MBFC-'  # '-' is to recognize the case an article in both ABYZ and MBFC
								mbfc_match_url = mbfc_outlet_list[i]
								mbfc_match_bias = mbfc_bias_list[i]
								match_country.append(mbfc_country_list[i])
								mbfc_matches += 1
								break
						if outlet_flag == '':
							continue

						#there is no "story_text" attribute in wiki data now

						# if not "story_text" in art:
						# 	continue
						#
						# words = text2tokens(art["story_text"], lang)
						# if words == None or len(words) < MIN_ARTICLE_LENGTH:
						# 	continue

						# skip URLs that we have seen already
						if url not in all_urls:
							all_urls.add(url)
						else:
							continue

						if "title" in art:
							title = art["title"].strip()
							# title_memeslist[title].append(memes)
							if title not in all_titles:
								all_titles.add(title)
								# title_url[title] = url
							else:
								continue
								# we looked at the resulting dup files, they contained:
								# i) many exact duplicates
								# ii) many articles not relevant socially/politically
						else:
							title = "null"
							# print("Article without title", url)
							continue

						# index of full NEs
						vec = []  # [0]*len(idx) #len(idx)==26**2
						ent_count = 0

						if "wiki_concepts" in art and art["wiki_concepts"] != None:
							for ent in art["wiki_concepts"]:
								if len(ent['term']) < 2 and lang not in ["zh", "ko"]:
									continue
								if ent['term'] in ne_index:
									vec.append(int(ent['term_id']))
								else:
									ne_index[ent['term']] = ent['term_id']
									vec.append(int(ent['term_id']))
								ent_count += 1

							if ent_count > 0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)
								out.write(f"{outlet_flag}\t{match_country}\t{mbfc_match_bias}\t{abyz_match_url}\t{mbfc_match_url}\t{stories_id}\t{title}\t{file}\t{lineno}\t{reladate}\t{url}\t{vec}\n")

	with open(name+"_ne.index", "w") as out:
		for k,v in ne_index.items():
			key = k.replace('\t', ' ')
			out.write(f"{key}\t{v}\n")

	with open(name+".info","w") as infofile:
		infodict = {
			"sha":SHA,
			"version":script_version,
			"description":"We moved to sorted by key NE counter/list.",
			"input_description":"Initial NE. Large model only for IT, small models for other languages.",
			"unique_named_entities":len(ne_index),
		}
		infofile.write(json.dumps(infodict, indent=4))

	print("created index successfully...", flush=True)


# create temp index from story id to url
def create_storyid_url_index(name, globstring, lang, debug):
	#Keep a list of the 1m most recent urls seen. LRU - least recently used - urls will be removed after hitting this limit
	# recent_url_cache=LRUCache(1000000)

	globstring = globstring.replace("\\","")

	all_urls = set()
	all_titles = set()
	ne_index = dict()
	total_ent_count = 0

	# let's use "en_short.index" for the two-character NE representation
	name += "-temp-v"+str(script_version)
	print("Will print to file", name+".index", flush=True)

	# with open(name+".index","w") as out, open(name+"_memes.index","w") as article_meme_file, open(name+".dups","w") as dups:
	with open(name+".index","w") as out:
		# we want in the reverse order, to keep the most recent article
		# we could reverse this ordering
		sorted_files = sorted(glob.glob(globstring), reverse=True)
		if debug:
			print(globstring, sorted_files, flush=True)
		for file in sorted_files:
			print(file,datetime.datetime.now(), flush=True)
			if lang=="auto":
				lang=os.path.dirname(file)[-2:]
				if lang not in [ "ar", "de", "es", "en", "fr", "it", "ko", "pl", "pt", "ru", "tr", "zh" ]:
					print("Last two letters of the directory don't specify a known language code. Please use the -l command line argument to specify the language. Exiting.", flush=True)
					sys.exit()
				if debug: print(lang, flush=True)
			if ".gz" not in file:
				with open(file,"r") as fh:
					for lineno,line in enumerate(fh):
						if debug and lineno>100:
							break

						art=json.loads(line)

						if not "stories_id" in art:
							continue
						if not "url" in art:
							continue

						stories_id = art["stories_id"]
						url=art["url"].strip()

						if any([w in art["url"] for w in blockwords]):
							continue

						# skip URLs that we have seen already
						if url not in all_urls:
							all_urls.add(url)
						else:
							continue

						# if recent_url_cache.get(url)!=None:
						# 	continue
						# recent_url_cache[url]=True

						if not "story_text" in art:
							continue

						words=text2tokens(art["story_text"], lang)
						if words==None or len(words)<MIN_ARTICLE_LENGTH:
							continue

						# each indexed article needs a title and we don't admit duplicates
						# if the same titles, then log them for inspection
						if "title" in art:
							title = art["title"].strip()
							# title_memeslist[title].append(memes)
							if title not in all_titles:
								all_titles.add(title)
								# title_url[title] = url
							else:
								continue
								# we looked at the resulting dup files, they contained:
								# i) many exact duplicates
								# ii) many articles not relevant socially/politically
						else:
							title = "null"
							# print("Article without title", url)
							continue

						# index of full NEs
						vec=[] #[0]*len(idx) #len(idx)==26**2
						ent_count=0

						if "spacy" in art and art["spacy"] != None:
							for ent in art["spacy"]:
								text=ent["text"]
								if len(text)<2 and lang not in ["zh", "ko"]:
									continue
								if text in ne_index:
									vec.append(ne_index[text])
								else:
									ne_index[text] = total_ent_count
									vec.append(ne_index[text])
									total_ent_count += 1
								ent_count+=1
							if ent_count>0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)

								out.write(f"{stories_id}\t{file}\t{lineno}\t{url}\n")
						if "polyglot" in art and art["polyglot"]!=None:
							for ent in art["polyglot"]:
								# the implemention used by xi chen
								for word in ent["text"]:
									if len(word)<2 and lang not in ["zh", "ko"]:
										continue
									if word in ne_index:
										vec.append(ne_index[word])
									else:
										ne_index[word] = total_ent_count
										vec.append(ne_index[word])
										total_ent_count += 1
									ent_count+=1
								# implemention that joins tokenized named entities
								# which seems to correspond to what spacy does
								# however, xi chen's experience is that
								# we get better results without joining
								# text=" ".join(ent["text"])
								# if len(text)<2 and lang not in ["zh", "ko"]:
								# 	continue
								# if text in ne_index:
								# 	vec.append(ne_index[text])
								# else:
								# 	ne_index[text] = total_ent_count
								# 	total_ent_count += 1
								# ent_count+=1
							if ent_count>0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)
								out.write(f"{stories_id}\t{title}\t{file}\t{lineno}\t{url}\n")
			else:
				with gzip.open(file,"rt") as fh:
					for lineno, line in enumerate(fh):
						if debug and lineno > 100:
							break

						art = json.loads(line)

						if not "stories_id" in art:
							continue
						if not "url" in art:
							continue

						stories_id = art["stories_id"]
						url = art["url"]

						if any([w in art["url"] for w in blockwords]):
							continue

						#there is no "story_text" attribute in wiki data now

						# if not "story_text" in art:
						# 	continue
						#
						# words = text2tokens(art["story_text"], lang)
						# if words == None or len(words) < MIN_ARTICLE_LENGTH:
						# 	continue

						# skip URLs that we have seen already
						if url not in all_urls:
							all_urls.add(url)
						else:
							continue

						if "title" in art:
							title = art["title"].strip()
							# title_memeslist[title].append(memes)
							if title not in all_titles:
								all_titles.add(title)
								# title_url[title] = url
							else:
								continue
								# we looked at the resulting dup files, they contained:
								# i) many exact duplicates
								# ii) many articles not relevant socially/politically
						else:
							title = "null"
							# print("Article without title", url)
							continue

						# index of full NEs
						vec = []  # [0]*len(idx) #len(idx)==26**2
						ent_count = 0

						if "wiki_concepts" in art and art["wiki_concepts"] != None:
							for ent in art["wiki_concepts"]:
								if len(ent['term']) < 2 and lang not in ["zh", "ko"]:
									continue
								if ent['term'] in ne_index:
									vec.append(int(ent['term_id']))
								else:
									ne_index[ent['term']] = ent['term_id']
									vec.append(int(ent['term_id']))
								ent_count += 1

							if ent_count > 0:
								vec = sorted(collections.Counter(vec).items())
								# compute relative date
								cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
								reladate = int(date_diff(cur_date, START_DATE).days)
								out.write(f"{stories_id}\t{title}\t{file}\t{lineno}\t{url}\n")

	with open(name+"_ne.index", "w") as out:
		for k,v in ne_index.items():
			key = k.replace('\t', ' ')
			out.write(f"{key}\t{v}\n")

	with open(name+".info","w") as infofile:
		infodict = {
			"sha":SHA,
			"version":script_version,
			"description":"We moved to sorted by key NE counter/list.",
			"input_description":"Initial NE. Large model only for IT, small models for other languages.",
			"unique_named_entities":len(ne_index),
		}
		infofile.write(json.dumps(infodict, indent=4))

	print("created index successfully...", flush=True)




if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-o", "--output-name", dest="output_name",
		default="indexes/en", type=str,
		help="Output index name.")
	parser.add_argument("-i", "--input-files", dest="input",
		default="/home/scott/ner/en/*.json", type=str,
		help="Path to input files.")
	parser.add_argument("-l", "--lang", dest="lang",
		default="auto", type=str,
		help="Language (impacts tokenization).")
	parser.add_argument("-b", "--bias", dest="bias",
		default=False, action='store_true',
		help="Switch to bias dataset matching based index mode.")
	parser.add_argument("-d", "--debug", dest="debug",
		default=False, action='store_true',
		help="Debug mode running for only a small chunk of data.")
	args = parser.parse_args()

	if not args.bias:
		create_index(args.output_name, args.input, args.lang, args.debug)
		# create_storyid_url_index(args.output_name, args.input, args.lang, args.debug)
	else:
		create_baised_index(args.output_name, args.input, args.lang, args.debug)






