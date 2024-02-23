# (alternative of stat_index.py)
# create a stat index to get key info of articles to estimate total article pair numbers of the global network before fitering out

''''''
'''sbatch -o create_stat_index_en.txt create_stat_index_script.sh "/mnt/nfs/work1/grabowicz/xchen4/mediacloud_temp/scott/wikilinked/en" "indexes/en-wiki-stat" '''

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

script_version = 2

letters="abcdefghijklmnopqrstuvwxyz"
assert len(letters)==26
idx=["".join([a,b]) for a in letters for b in letters]

from utils import blockwords,text2tokens, date_diff, MIN_ARTICLE_LENGTH, SHA, START_DATE, DATE_WINDOW,unify_url


def create_index(name, globstring, lang, debug):
    #Keep a list of the 1m most recent urls seen. LRU - least recently used - urls will be removed after hitting this limit
    # recent_url_cache=LRUCache(1000000)

    # for temporal script usage on swarm
    globstring += "/2020*.json.gz"

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

                        if lineno % 10000 == 0:
                            print(lineno, "lines", end=" ", flush=True)

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
                            # print("Article without title", url)
                            continue

                        cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
                        reladate = int(date_diff(cur_date, START_DATE).days)
                        media_id = art["media_id"]
                        out.write(f"{reladate}\t{media_id}\n")

            else:
                with gzip.open(file,"rt") as fh:
                    for lineno, line in enumerate(fh):
                        if debug and lineno > 100:
                            break

                        if lineno % 10000 == 0:
                            print(lineno, "lines", end=" ", flush=True)

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

                        cur_date = file.split("/")[-1].replace(".json", "").replace(".gz", "")
                        reladate = int(date_diff(cur_date, START_DATE).days)
                        media_id = art["media_id"]
                        out.write(f"{reladate}\t{media_id}\n")


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