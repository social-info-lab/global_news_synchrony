from functools import partial
import os
#import multiprocessing
import concurrent.futures
import sys
import traceback
from functools import partial
import os
# import multiprocessing
import concurrent.futures
import sys
import json
# import spacy
import polyglot
from polyglot.text import Text, Word
import collections
import csv
# import gcld3
import time
from polyglot.downloader import downloader
downloader.download("embeddings2.ar")
downloader.download("embeddings2.en")
downloader.download("embeddings2.es")
downloader.download("embeddings2.fr")
downloader.download("embeddings2.th")
# downloader.download("embeddings2.ps")
# downloader.download("embeddings2.om")
# downloader.download("embeddings2.un")
downloader.download("ner2.ar")
downloader.download("ner2.en")
downloader.download("ner2.es")
downloader.download("ner2.fr")
downloader.download("ner2.th")
# downloader.download("ner2.ps")
# downloader.download("ner2.om")
# downloader.download("ner2.un")

def parse_polyglot_entity(entity):
    return {
        "text": json.loads(json.dumps(entity)),
        "type": entity.tag
    }


def parse_polyglot(text):
    try:
        return [parse_polyglot_entity(e) for e in Text(text).entities]
    except:
        traceback.print_exc()
        return []


def process_article(raw_article, model=None):
    text = ''.join([ch for ch in raw_article["story_text"] if ch.isprintable()])
    if text == None:
        return json.dumps(raw_article)
    # lang = raw_article["language"]  # detector.FindLanguage(text=text).language
    row = raw_article
    try:
        row["polyglot"] = parse_polyglot(text)
    except:
        row["polyglot"] = None

    publish_date = raw_article['publish_date']
    if publish_date == None:
        publish_date = "NAN"
    else:
        publish_date = publish_date[:10]

    return publish_date, json.dumps(row)


def process_file(filename, model):
    print(filename)
    story_id = filename.split('/')[-1].split('.')[0]
    articles = []
    with open(filename, "r") as fh:
        for line in fh.readlines():
            cur_art = json.loads(line)
            articles.append(cur_art)
    output = collections.defaultdict(list)
    for art in articles:
        art['story_id'] = story_id
        date, dat = process_article(art, model)
        output[date].append(dat)
    return output


PROCESSED_FILES = 0


def append_to_files(future, output_dir):
    global PROCESSED_FILES
    PROCESSED_FILES += 1
    data = future.result()
    for k, v in data.items():
        with open(f"{output_dir}/{k}.json", "a") as fh:
            for art in v:
                fh.write(art)
                fh.write("\n")
    print(f"{PROCESSED_FILES} completed")


def log_error(e, output_dir):
    with open(output_dir + "/ERR.txt", "a") as fh:
        print(e)
        fh.write(str(e))



import glob

if __name__ == "__main__":
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    # input_directory = "UK_RU/splitted_by_date"
    # output_directory = "UK_RU/splitted_by_date/ner"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    lang_list = ["ar"]

    # Assume all articles are in the correct language
    # will greatly reduce memory as we can load only ONE spacy model

    for lang in lang_list:
        cur_output_directory = output_directory + "/" + lang
        if not os.path.exists(cur_output_directory):
            os.makedirs(cur_output_directory)

        files = glob.glob(input_directory + f"/{lang}/*.json")
        print()
        print(len(files))


        pfile = partial(process_file, model=None)
        tofiles = partial(append_to_files, output_dir=cur_output_directory)
        logerr = partial(log_error, output_dir=cur_output_directory)
        # with multiprocessing.Pool(processes=1,maxtasksperchild=100) as pool: #2*os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for f in files:
                with open(f, "r") as fh:
                    future = executor.submit(pfile, f)
                    future.add_done_callback(tofiles)

        print("DONE")
